import random
from typing import Dict, List, Sequence, Tuple, Union

import jax
import jax.numpy as jnp
from flax import struct
from jax import Array

from navix.components import DISCARD_PILE_COORDS
from navix.entities import Entities
from navix.environments import Environment
from navix.environments.environment import Timestep
from navix.grid import positions_equal, translate
from navix.rendering.cache import RenderingCache
from navix.spaces import Discrete
from navix.states import State

PRIMITIVE = -1
"""`heading` value marking a row as a primitive action: an option of duration 1."""
NO_INTERACT = -1

NEEDS_NOTHING = 0
NEEDS_KEY = 1
NEEDS_OPENABLE_DOOR = 2
NEEDS_POCKET = 3

HEADINGS = (0, 1, 2, 3)
DIRECTIONS = ("east", "south", "west", "north")
INTERACTIONS = (None, "pickup", "toggle", "open", "drop")

_PRECONDITION = {
    "pickup": NEEDS_KEY,
    "toggle": NEEDS_OPENABLE_DOOR,
    "open": NEEDS_OPENABLE_DOOR,
    "drop": NEEDS_POCKET,
}


def action_names(env: Environment) -> List[str]:
    return [fn.__name__ for fn in env.action_set]


def _precondition(row: Tuple[int, ...], names: List[str]) -> int:
    heading, _, interact, _ = row
    if heading < 0 or interact < 0:
        return NEEDS_NOTHING
    return _PRECONDITION.get(names[interact], NEEDS_NOTHING)


def _cap(row: Tuple[int, ...]) -> int:
    """Hard bound on the option's duration, the last disjunct of `beta`.

    Worst case: two turns to face the heading, `reach` cells of progress, one
    turn per blocked cell when following, and the interaction.
    """
    heading, reach, interact, follow = row
    if heading < 0:
        return 1
    return 2 + reach + (reach if follow else 0) + (1 if interact >= 0 else 0)


def _nominal(row: Tuple[int, ...]) -> int:
    """Duration to expect rather than the worst case, for sizing the budget.

    One turn on average from a uniformly drawn facing, then the reach and the
    interaction. What the option actually takes is whatever `beta` fires at.
    """
    heading, reach, interact, _ = row
    if heading < 0:
        return 1
    return 1 + reach + (1 if interact >= 0 else 0)


def mean_nominal_duration(table: Sequence[Tuple[int, ...]]) -> float:
    """Cheap duration estimate, for reporting only.

    Systematically low, because it charges one turn per option and a closed-loop
    option turns again whenever it is blocked. Use `measure_duration` for
    anything that divides the budget.
    """
    return sum(_nominal(row) for row in table) / len(table)


def _label(row: Tuple[int, ...], names: List[str]) -> str:
    heading, reach, interact, follow = row
    if heading < 0:
        return names[interact]
    parts = [f"{DIRECTIONS[heading]}{reach}"]
    if follow:
        parts.append("follow")
    if interact >= 0:
        parts.append(names[interact])
    return "+".join(parts)


def _catalogue(max_forward: int, names: List[str]) -> List[Tuple[int, ...]]:
    """Every `(heading, reach, interact, follow)` controller, longest reach first.

    The sort is what keeps mean duration from collapsing as the option count
    grows: a run with n options gets the n longest-reaching controllers.
    """
    index = {name: i for i, name in enumerate(names)}
    rows = []
    for rank, interact in enumerate(INTERACTIONS):
        if interact is not None and interact not in index:
            continue
        code = NO_INTERACT if interact is None else index[interact]
        for follow in (1, 0):
            for heading in HEADINGS:
                for reach in range(1, max_forward + 1):
                    row = (heading, reach, code, follow)
                    rows.append((row, (-reach, -follow, rank, heading)))
    rows.sort(key=lambda item: item[1])
    return [row for row, _ in rows]


def grammar_options(
    n: int, max_forward: int, names: List[str]
) -> Tuple[List[Tuple[int, ...]], List[str]]:
    catalogue = _catalogue(max_forward, names)
    if n > len(catalogue):
        raise ValueError(
            f"asked for {n} options but the controller family only yields "
            f"{len(catalogue)} at max_forward={max_forward}; raise --max-forward"
        )
    chosen = catalogue[:n]
    return chosen, [_label(row, names) for row in chosen]


def random_options(
    n: int, max_forward: int, names: List[str], seed: int
) -> Tuple[List[Tuple[int, ...]], List[str]]:
    """Uniform draws from the same catalogue, without the longest-reach priority.

    Duration is deliberately not matched to the grammar family here. Every
    controller in the catalogue is one option, so matching the reach multiset
    would force the draw back onto the grammar's own rows once n approaches the
    per-reach pool size. The contrast is therefore the reach prior itself, which
    `mean_nominal_duration` logs per run rather than hiding.
    """
    catalogue = _catalogue(max_forward, names)
    if n > len(catalogue):
        raise ValueError(
            f"asked for {n} options but the controller family only yields "
            f"{len(catalogue)} at max_forward={max_forward}; raise --max-forward"
        )
    chosen = random.Random(seed).sample(catalogue, n)
    return chosen, [_label(row, names) for row in chosen]


def make_options(
    family: str, n: int, max_forward: int, names: List[str], seed: int
) -> Tuple[List[Tuple[int, ...]], List[str]]:
    if family == "grammar":
        return grammar_options(n, max_forward, names)
    if family == "random":
        return random_options(n, max_forward, names, seed)
    raise ValueError(f"unknown option family {family}")


def action_table(
    mode: str,
    names: List[str],
    options: List[Tuple[int, ...]],
    labels: List[str],
) -> Tuple[List[Tuple[int, ...]], List[str]]:
    primitives = [(PRIMITIVE, 0, i, 0) for i in range(len(names))]
    if mode == "action":
        return primitives, list(names)
    if mode == "option":
        return options, labels
    if mode == "both":
        return primitives + options, list(names) + labels
    raise ValueError(f"unknown action space {mode}")


class OptionSpec(struct.PyTreeNode):
    """The `(I, pi, beta)` triple of every option in the action table.

    One parameterised controller covers the whole table, so the three
    components are arrays indexed by option id rather than Python callables:
    the id is a traced value inside the rollout.
    """

    heading: Array
    """i32[n] compass direction `pi` steers to, -1 for a primitive action."""
    reach: Array
    """i32[n] cells `pi` advances before `beta` fires."""
    interact: Array
    """i32[n] primitive `pi` emits on arrival, -1 for none."""
    follow: Array
    """i32[n] 1 to turn and carry on when blocked, 0 to let `beta` fire."""
    requires: Array
    """i32[n] which precondition `I` checks, one of the NEEDS_* codes."""
    cap: Array
    """i32[n] hard duration bound, the last disjunct of `beta`."""
    forward: int = struct.field(pytree_node=False, default=0)
    cw: int = struct.field(pytree_node=False, default=0)
    ccw: int = struct.field(pytree_node=False, default=0)
    horizon: int = struct.field(pytree_node=False, default=1)
    """Longest `cap` in the table, the static length of the executor's scan."""

    @property
    def size(self) -> int:
        return int(self.heading.shape[0])

    @classmethod
    def create(
        cls, table: Sequence[Tuple[int, ...]], names: List[str]
    ) -> "OptionSpec":
        rows = [tuple(int(x) for x in row) for row in table]
        if not rows:
            raise ValueError("the action table is empty; --n-options must be >= 1")
        index = {name: i for i, name in enumerate(names)}
        missing = [a for a in ("forward", "rotate_cw", "rotate_ccw") if a not in index]
        if missing and any(row[0] >= 0 for row in rows):
            raise ValueError(
                f"the option controller steers with {missing}, which this "
                f"environment's action set does not provide: {names}"
            )
        return cls(
            heading=jnp.asarray([row[0] for row in rows], dtype=jnp.int32),
            reach=jnp.asarray([row[1] for row in rows], dtype=jnp.int32),
            interact=jnp.asarray([row[2] for row in rows], dtype=jnp.int32),
            follow=jnp.asarray([row[3] for row in rows], dtype=jnp.int32),
            requires=jnp.asarray(
                [_precondition(row, names) for row in rows], dtype=jnp.int32
            ),
            cap=jnp.asarray([_cap(row) for row in rows], dtype=jnp.int32),
            forward=index.get("forward", 0),
            cw=index.get("rotate_cw", 0),
            ccw=index.get("rotate_ccw", 0),
            horizon=max(_cap(row) for row in rows),
        )


def walkable(state: State, position: Array) -> Array:
    """Whether the player could stand on `position`.

    A read-only twin of `navix.actions._can_walk_there`: the upstream version
    records a wall_hit event, and neither `pi` nor `beta` may touch the state
    they read.
    """
    clear = jnp.equal(state.grid[tuple(position)], 0)
    for name in state.entities:
        entity = state.entities[name]
        here = positions_equal(entity.position, position)
        # logical_not, not ~: Door.walkable is the `open` field, which is an
        # integer, and a bitwise negation of 1 is truthy
        blocked = jnp.logical_and(here, jnp.logical_not(entity.walkable))
        clear = jnp.logical_and(clear, jnp.logical_not(jnp.any(blocked)))
    return clear


def initiation(spec: OptionSpec, state: State) -> Array:
    """`I`: bool[n], the options selectable from `state`.

    Primitives are available everywhere, so the `action` condition sees an
    unrestricted action space and stays comparable to a run without options.
    """
    player = state.get_player()
    ahead = jnp.stack(
        [
            walkable(state, translate(player.position, jnp.asarray(d)))
            for d in HEADINGS
        ]
    )[jnp.clip(spec.heading, 0, 3)]

    # an option that can neither turn, advance nor interact cannot change the
    # state, and a decision spent on it is a decision thrown away
    facing = (spec.heading - player.direction) % 4 == 0
    usable = ~facing | ahead | (spec.interact >= 0)

    key_loose = jnp.asarray(False)
    if Entities.KEY in state.entities:
        keys = state.get_keys()
        key_loose = jnp.any(~positions_equal(keys.position, DISCARD_PILE_COORDS))

    door_openable = jnp.asarray(False)
    if Entities.DOOR in state.entities:
        doors = state.get_doors()
        shut = ~doors.open.astype(jnp.bool_)
        unlocked = doors.requires == -1
        door_openable = jnp.any(shut & (unlocked | (doors.requires == player.pocket)))

    met = jnp.where(
        spec.requires == NEEDS_KEY,
        key_loose,
        jnp.where(
            spec.requires == NEEDS_OPENABLE_DOOR,
            door_openable,
            jnp.where(spec.requires == NEEDS_POCKET, player.pocket > 0, True),
        ),
    )

    available = (met & usable) | (spec.heading < 0)
    # an all-false mask sends every logit to -inf and the sample to NaN
    return jnp.where(jnp.any(available), available, jnp.ones_like(available))


def option_policy(
    spec: OptionSpec, option: Array, state: State, aim: Array, advanced: Array
) -> Tuple[Array, Array]:
    """`pi`: the primitive action the option takes from `state`.

    Also returns whether the option re-aimed, which is the one part of the
    decision the executor cannot recover from the action alone: a clockwise turn
    means both "turn toward the aim" and "the aim is blocked, try the next one".
    """
    player = state.get_player()
    spin = (aim - player.direction) % 4
    facing = spin == 0
    clear = walkable(state, translate(player.position, aim))
    advancing = advanced < spec.reach[option]

    keep_going = advancing & (clear | (spec.follow[option] == 1))
    walk = jnp.where(clear, spec.forward, spec.cw)
    turn = jnp.where(spin == 3, spec.ccw, spec.cw)
    tail = jnp.where(spec.interact[option] >= 0, spec.interact[option], spec.cw)

    move = jnp.where(facing, jnp.where(keep_going, walk, tail), turn)
    action = jnp.where(spec.heading[option] < 0, spec.interact[option], move)
    reaim = facing & advancing & ~clear & (spec.follow[option] == 1)
    return action.astype(jnp.int32), reaim


def termination(
    spec: OptionSpec,
    option: Array,
    state: State,
    aim: Array,
    advanced: Array,
    interacted: Array,
    steps: Array,
) -> Array:
    """`beta`: whether the option stops in `state`.

    Deterministic, so a stop probability in {0, 1}. It fires when `pi` has no
    work left in this state, which is why the duration is a property of the map
    and the agent's position rather than of the option.
    """
    player = state.get_player()
    facing = (aim - player.direction) % 4 == 0
    clear = walkable(state, translate(player.position, aim))
    arrived = advanced >= spec.reach[option]
    stuck = ~clear & (spec.follow[option] == 0)
    return (facing & (arrived | stuck) & interacted) | (steps >= spec.cap[option])


class OptionEnv(Environment):
    """Executes one already-selected option as a single decision.

    It does not select: the option id arrives as the action. Both `pi` and
    `beta` read the state at every primitive step, so the duration is whatever
    `beta` fires at rather than a length fixed at selection time. The scan runs
    for the table's static `horizon` and masks the iterations past termination,
    which is what keeps the whole thing jittable inside the PPO rollout.
    """

    env: Environment = None  # type: ignore[assignment]
    spec: OptionSpec = None  # type: ignore[assignment]

    @classmethod
    def create(cls, env: Environment, spec: OptionSpec) -> "OptionEnv":
        return cls(
            height=env.height,
            width=env.width,
            max_steps=env.max_steps,
            observation_space=env.observation_space,
            action_space=Discrete.create(spec.size),
            reward_space=env.reward_space,
            gamma=env.gamma,
            penality_coeff=env.penality_coeff,
            observation_fn=env.observation_fn,
            reward_fn=env.reward_fn,
            termination_fn=env.termination_fn,
            transitions_fn=env.transitions_fn,
            action_set=env.action_set,
            env=env,
            spec=spec,
        )

    def _reset(self, key: Array, cache: Union[RenderingCache, None] = None) -> Timestep:
        return self.env._reset(key, cache)

    def reset(self, key: Array, cache: Union[RenderingCache, None] = None) -> Timestep:
        timestep = self.env.reset(key, cache)
        return timestep.replace(
            info=self._info(
                timestep.info,
                timestep.state,
                steps=jnp.asarray(0, dtype=jnp.int32),
                decision_t=jnp.asarray(0, dtype=jnp.int32),
            )
        )

    def step(self, timestep: Timestep, action: Array) -> Timestep:
        # a done timestep must consume exactly one iteration (the inner
        # autoreset) and stop, or the option would keep stepping into the
        # fresh episode and leak reward across the boundary
        started_done = timestep.is_done()
        inner = timestep.replace(info={"return": timestep.info["return"]})
        interact = self.spec.interact[action]
        gamma = jnp.asarray(self.env.gamma, dtype=jnp.float32)

        def body(carry, _):
            current, aim, advanced, interacted, stop, total, steps = carry
            active = ~stop

            primitive, reaim = option_policy(
                self.spec, action, current.state, aim, advanced
            )
            before = current.state.get_player().position
            stepped = self.env.step(current, primitive)
            current = jax.tree.map(
                lambda a, b: jnp.where(active, a, b), stepped, current
            )

            total = total + jnp.where(active, gamma**steps * stepped.reward, 0.0)
            steps = steps + jnp.where(active, 1, 0)

            # progress read back off the state rather than assumed from the
            # action: a forward that was blocked did not advance the option
            moved = jnp.any(current.state.get_player().position != before)
            advanced = advanced + jnp.where(active & moved, 1, 0)
            interacted = interacted | (active & (primitive == interact))
            aim = jnp.where(active & reaim, (aim + 1) % 4, aim)

            fired = termination(
                self.spec, action, current.state, aim, advanced, interacted, steps
            )
            stop = stop | started_done | (active & (current.is_done() | fired))
            return (current, aim, advanced, interacted, stop, total, steps), None

        init = (
            inner,
            jnp.clip(self.spec.heading[action], 0, 3),
            jnp.asarray(0, dtype=jnp.int32),
            interact < 0,
            jnp.asarray(False),
            jnp.asarray(0.0, dtype=jnp.float32),
            jnp.asarray(0, dtype=jnp.int32),
        )
        (final, _, _, _, _, total, steps), _ = jax.lax.scan(
            body, init, None, self.spec.horizon
        )
        decision_t = jnp.where(started_done, 0, timestep.info["decision_t"] + 1)
        return final.replace(
            reward=total,
            info=self._info(
                final.info, final.state, steps=steps, decision_t=decision_t
            ),
        )

    def _info(
        self, info: Dict[str, Array], state: State, steps: Array, decision_t: Array
    ) -> Dict[str, Array]:
        return {
            **info,
            "primitive_steps": steps,
            # for SMDP discounting downstream: the decision spans `steps`
            # primitive steps, so the next state is that much further away
            "option_discount": jnp.asarray(self.env.gamma, dtype=jnp.float32) ** steps,
            # `I` at the state the policy over options selects from next
            "available": initiation(self.spec, state),
            "decision_t": decision_t,
        }


def measure_duration(
    env: OptionEnv, num_envs: int = 64, decisions: int = 64, seed: int = 0
) -> float:
    """Realised mean primitive steps per decision, under uniform selection from `I`.

    The training budget is spent in primitive steps, so `num_updates` divides by
    this number. A closed-loop option's duration depends on the map and on where
    the agent happens to stand, so it has to be measured rather than derived:
    `mean_nominal_duration` runs about 15% low on DoorKey, and undercounting
    hands the options condition more environment interaction than the baseline,
    which is exactly the comparison the primitive-step budget exists to prevent.
    """

    def step(carry, _):
        timestep, rng = carry
        rng, key = jax.random.split(rng)
        logits = jnp.where(timestep.info["available"], 0.0, -1e8)
        action = jax.random.categorical(key, logits)
        timestep = jax.vmap(env.step)(timestep, action)
        return (timestep, rng), timestep.info["primitive_steps"]

    rng = jax.random.PRNGKey(seed)
    timestep = jax.vmap(env.reset)(jax.random.split(rng, num_envs))
    _, steps = jax.lax.scan(step, (timestep, rng), None, decisions)
    return float(jnp.mean(steps))
