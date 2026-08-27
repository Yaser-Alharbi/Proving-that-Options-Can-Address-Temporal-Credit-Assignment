import random
from typing import Dict, List, Sequence, Tuple, Union

import jax
import jax.numpy as jnp
from flax import struct
from jax import Array

from navix.components import DISCARD_PILE_COORDS, Pickable
from navix.entities import Entities
from navix.environments import Environment
from navix.environments.environment import StepType, Timestep
from navix.grid import positions_equal, translate
from navix.rendering.cache import RenderingCache
from navix.spaces import Discrete
from navix.states import State

PRIMITIVE = -1
"""`heading` value marking a row as a primitive action: an option of duration 1."""
NO_INTERACT = -1

NEEDS_NOTHING = 0
NEEDS_PICKABLE = 1
NEEDS_OPENABLE_DOOR = 2
NEEDS_POCKET = 3

# what work `pi` has left, as returned by `_status`
STATUS_TURN = 0
STATUS_ADVANCE = 1
STATUS_DETOUR = 2
STATUS_INTERACT = 3
STATUS_DONE = 4

# which state change verifies a row's interaction
EFFECT_NONE = 0
EFFECT_TAKE = 1
EFFECT_PUT = 2
EFFECT_DOOR = 3
EFFECT_UNKNOWN = 4

HEADINGS = (0, 1, 2, 3)
DIRECTIONS = ("east", "south", "west", "north")
INTERACTIONS = (None, "pickup", "toggle", "open", "drop")

DEFAULT_OPTION_SEED = 0
"""Not the training seed: the option set is a property of the family."""

INITIAL_TURNS = 2
"""Worst-case rotations to face the heading, since `pi` turns the short way."""
DETOUR_EPISODE_COST = 5
"""Worst-case primitives in one detour episode: 1 + 1 + 1 to turn onto a
lateral, move and turn back, or 2 + 1 + 2 for the reverse."""

_PRECONDITION = {
    "pickup": NEEDS_PICKABLE,
    "toggle": NEEDS_OPENABLE_DOOR,
    "open": NEEDS_OPENABLE_DOOR,
    "drop": NEEDS_POCKET,
}

_EFFECT = {
    "pickup": EFFECT_TAKE,
    "drop": EFFECT_PUT,
    "toggle": EFFECT_DOOR,
    "open": EFFECT_DOOR,
}


def action_names(env: Environment) -> List[str]:
    return [fn.__name__ for fn in env.action_set]


def _precondition(row: Tuple[int, ...], names: List[str]) -> int:
    heading, _, interact, _ = row
    if heading < 0 or interact < 0:
        return NEEDS_NOTHING
    return _PRECONDITION.get(names[interact], NEEDS_NOTHING)


def _effect(row: Tuple[int, ...], names: List[str]) -> int:
    """Which state change verifies this row's interaction, as an EFFECT_* code."""
    _, _, interact, _ = row
    if interact < 0:
        return EFFECT_NONE
    return _EFFECT.get(names[interact], EFFECT_UNKNOWN)


def _max_duration(row: Tuple[int, ...]) -> int:
    """Hard bound on the option's duration, the last disjunct of `beta`: the
    turns onto the heading, `reach` advances, `reach` detours, one interaction."""
    heading, reach, interact, follow = row
    if heading < 0:
        return 1
    return (
        INITIAL_TURNS
        + reach
        + (DETOUR_EPISODE_COST * reach if follow else 0)
        + (1 if interact >= 0 else 0)
    )


def _nominal(row: Tuple[int, ...]) -> int:
    """Expected rather than worst-case duration: one turn, the reach, the act."""
    heading, reach, interact, _ = row
    if heading < 0:
        return 1
    return 1 + reach + (1 if interact >= 0 else 0)


def mean_nominal_duration(table: Sequence[Tuple[int, ...]]) -> float:
    """Mean `_nominal` over the table, for reporting only: systematically low."""
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
    """Every `(heading, reach, interact, follow)` controller, ordered by
    `(follow, -reach, rank, heading)`: a run with n options gets the first n."""
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
                    rows.append((row, (follow, -reach, rank, heading)))
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
    n: int,
    max_forward: int,
    names: List[str],
    option_seed: int = DEFAULT_OPTION_SEED,
) -> Tuple[List[Tuple[int, ...]], List[str]]:
    """Uniform draws from the catalogue, seeded by `option_seed`."""
    # sorted, not `_catalogue`'s priority order: `random.sample` reads its input
    # positionally, so the draw would change whenever the grammar's prior did
    catalogue = sorted(_catalogue(max_forward, names))
    if n > len(catalogue):
        raise ValueError(
            f"asked for {n} options but the controller family only yields "
            f"{len(catalogue)} at max_forward={max_forward}; raise --max-forward"
        )
    chosen = random.Random(option_seed).sample(catalogue, n)
    return chosen, [_label(row, names) for row in chosen]


def make_options(
    family: str,
    n: int,
    max_forward: int,
    names: List[str],
    option_seed: int = DEFAULT_OPTION_SEED,
) -> Tuple[List[Tuple[int, ...]], List[str]]:
    if family == "grammar":
        return grammar_options(n, max_forward, names)
    if family == "random":
        return random_options(n, max_forward, names, option_seed)
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


def missing_interactions(
    table: Sequence[Tuple[int, ...]], names: List[str]
) -> List[str]:
    """Interactions this action set provides that no row in `table` performs."""
    reachable = {names[row[2]] for row in table if row[2] >= 0}
    return [
        name
        for name in INTERACTIONS
        if name is not None and name in names and name not in reachable
    ]


class OptionSpec(struct.PyTreeNode):
    """The `(I, pi, beta)` triple of every option, as arrays by option id."""

    heading: Array
    """i32[n] compass direction `pi` steers to, -1 for a primitive action."""
    reach: Array
    """i32[n] directed moves `pi` makes, and the detour budget it may spend."""
    interact: Array
    """i32[n] primitive `pi` emits on arrival, -1 for none."""
    follow: Array
    """i32[n] 1 to route around a blocked aim, 0 to let `beta` fire."""
    requires: Array
    """i32[n] which precondition `I` checks, one of the NEEDS_* codes."""
    effect: Array
    """i32[n] which state change verifies the interaction, an EFFECT_* code."""
    max_duration: Array
    """i32[n] hard duration bound, the last disjunct of `beta`."""
    max_reach: int = struct.field(pytree_node=False, default=0)
    """Longest `reach` in the table: how far ahead `I` inspects the map."""
    forward: int = struct.field(pytree_node=False, default=0)
    cw: int = struct.field(pytree_node=False, default=0)
    ccw: int = struct.field(pytree_node=False, default=0)
    horizon: int = struct.field(pytree_node=False, default=1)
    """Longest `max_duration` in the table, the static bound on the executor's loop."""

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
            effect=jnp.asarray([_effect(row, names) for row in rows], dtype=jnp.int32),
            max_duration=jnp.asarray(
                [_max_duration(row) for row in rows], dtype=jnp.int32
            ),
            max_reach=max(row[1] for row in rows),
            forward=index.get("forward", 0),
            cw=index.get("rotate_cw", 0),
            ccw=index.get("rotate_ccw", 0),
            horizon=max(_max_duration(row) for row in rows),
        )


def walkable(state: State, position: Array) -> Array:
    """Whether the player could stand on `position`, read-only unlike
    `navix.actions._can_walk_there`, which records a wall_hit event."""
    clear = jnp.equal(state.grid[tuple(position)], 0)
    for name in state.entities:
        entity = state.entities[name]
        here = positions_equal(entity.position, position)
        # logical_not, not ~: Door.walkable is the `open` field, which is an
        # integer, and a bitwise negation of 1 is truthy
        blocked = jnp.logical_and(here, jnp.logical_not(entity.walkable))
        clear = jnp.logical_and(clear, jnp.logical_not(jnp.any(blocked)))
    return clear


def _neighbours(state: State) -> Array:
    """bool[4]: whether the cell in each compass direction is walkable."""
    player = state.get_player()
    return jnp.stack(
        [
            walkable(state, translate(player.position, jnp.asarray(d)))
            for d in HEADINGS
        ]
    )


def _detour_aim(neighbours: Array, home_aim: Array) -> Array:
    """Where `pi` re-aims when the heading is blocked: right, left, then reverse,
    chosen by inspection so a blocked candidate costs no primitive step."""
    right = (home_aim + 1) % 4
    left = (home_aim + 3) % 4
    reverse = (home_aim + 2) % 4
    # the reverse needs no test: `_status` reports STATUS_DETOUR only when some neighbour is clear
    return jnp.where(
        neighbours[right], right, jnp.where(neighbours[left], left, reverse)
    ).astype(jnp.int32)


def _ray(state: State, depth: int) -> Tuple[Array, Array, Array]:
    """Per heading over the cells `0..depth` ahead: a loose pickable, an openable
    shut door, and the distance to the first cell the player cannot walk into."""
    player = state.get_player()
    origin = jnp.zeros((2,), dtype=jnp.int32)
    steps = jnp.stack([translate(origin, jnp.asarray(d)) for d in HEADINGS])
    distances = jnp.arange(depth + 1, dtype=jnp.int32)
    cells = (
        player.position[None, None, :] + steps[:, None, :] * distances[None, :, None]
    )

    height, width = state.grid.shape

    inside = (
        (cells[..., 0] >= 0)
        & (cells[..., 0] < height)
        & (cells[..., 1] >= 0)
        & (cells[..., 1] < width)
    )
    clear = (
        jax.vmap(walkable, in_axes=(None, 0))(state, cells.reshape(-1, 2)).reshape(
            inside.shape
        )
        & inside
    )
    # one past the clear run: the option stops in front of that cell and acts
    first_blocked = 1 + jnp.sum(jnp.cumprod(clear[:, 1:], axis=1), axis=1)

    def occupied_by(positions: Array, mask: Array) -> Array:
        """bool[4, depth + 1]: a masked entity of this kind sits on the cell."""
        here = jnp.all(positions[:, None, None, :] == cells[None], axis=-1)
        return jnp.any(here & mask[:, None, None], axis=0) & inside

    pickable_at = jnp.zeros(inside.shape, dtype=jnp.bool_)
    for name in state.entities:
        entity = state.entities[name]
        if not isinstance(entity, Pickable):
            continue
        loose = ~positions_equal(entity.position, DISCARD_PILE_COORDS)
        pickable_at = pickable_at | occupied_by(entity.position, loose)

    door_openable_at = jnp.zeros(inside.shape, dtype=jnp.bool_)
    if Entities.DOOR in state.entities:
        doors = state.get_doors()
        shut = ~doors.open.astype(jnp.bool_)
        unlocked = doors.requires == -1
        door_openable_at = occupied_by(
            doors.position, shut & (unlocked | (doors.requires == player.pocket))
        )

    return pickable_at, door_openable_at, first_blocked.astype(jnp.int32)


def _pile_size(state: State) -> Array:
    """How many pickables are held, i.e. sit in the discard pile."""
    total = jnp.asarray(0, dtype=jnp.int32)
    for name in state.entities:
        entity = state.entities[name]
        if not isinstance(entity, Pickable):
            continue
        # DISCARD_PILE_COORDS is (0, -1), which `positions_equal` matches on the row alone
        in_pile = positions_equal(entity.position, DISCARD_PILE_COORDS)
        total = total + jnp.sum(in_pile.astype(jnp.int32))
    return total


def _open_doors(state: State) -> Array:
    """How many doors are open."""
    if Entities.DOOR not in state.entities:
        return jnp.asarray(0, dtype=jnp.int32)
    return jnp.sum(state.get_doors().open.astype(jnp.int32))


def _interaction_succeeded(
    spec: OptionSpec, option: Array, before: State, after: State
) -> Array:
    """Whether the interaction changed the world, read off the discard pile
    rather than `player.pocket`, which `drop` and `open` leave unchanged."""
    effect = spec.effect[option]
    pile_before, pile_after = _pile_size(before), _pile_size(after)
    took = pile_after > pile_before
    put = pile_after < pile_before
    opened = _open_doors(after) > _open_doors(before)
    return jnp.where(
        effect == EFFECT_TAKE,
        took,
        jnp.where(
            effect == EFFECT_PUT,
            put,
            # EFFECT_UNKNOWN is True: an unknown action set must not deadlock the executor into retrying forever
            jnp.where(effect == EFFECT_DOOR, opened, True),
        ),
    )


def _initial_aim(spec: OptionSpec, option: Array, state: State) -> Array:
    """The heading `pi` steers to at the start of an execution: for a primitive
    row, whatever the player already faces, so it is charged no turn."""
    heading = spec.heading[option]
    direction = state.get_player().direction
    return jnp.where(heading < 0, direction, heading).astype(jnp.int32)


def _status(
    spec: OptionSpec,
    option: Array,
    state: State,
    neighbours: Array,
    aim: Array,
    advanced: Array,
    detoured: Array,
    interacted: Array,
    attempted: Array,
) -> Array:
    """What work `pi` has left in `state`, one of the STATUS_* codes. `option`
    may be one id or an array of them, which is how `initiation` calls it."""
    player = state.get_player()
    # indexing, not a second walkable() call: `aim` is i32[n] in `initiation`
    clear_aim = neighbours[aim]

    advancing = advanced < spec.reach[option]
    # jnp.any: with every neighbour blocked there is nothing to route around
    can_detour = (
        (spec.follow[option] == 1)
        & jnp.any(neighbours)
        & (detoured < spec.reach[option])
    )
    move_work = advancing & (clear_aim | can_detour)
    pending = (spec.interact[option] >= 0) & ~interacted & ~attempted
    work = move_work | pending
    facing = (aim - player.direction) % 4 == 0

    return jnp.where(
        ~work,
        STATUS_DONE,
        jnp.where(
            ~facing,
            # turn before interacting: navix acts on the cell in front of `player.direction`, so acting unfaced would act on the wrong cell
            STATUS_TURN,
            jnp.where(
                move_work,
                jnp.where(clear_aim, STATUS_ADVANCE, STATUS_DETOUR),
                STATUS_INTERACT,
            ),
        ),
    ).astype(jnp.int32)


def initiation(spec: OptionSpec, state: State) -> Array:
    """`I`: bool[n], the options selectable from `state`."""
    player = state.get_player()
    options = jnp.arange(spec.size)
    zero = jnp.zeros((), dtype=jnp.int32)
    status = _status(
        spec,
        options,
        state,
        _neighbours(state),
        _initial_aim(spec, options, state),
        zero,
        zero,
        spec.interact < 0,
        jnp.asarray(False),
    )

    pickable_at, door_openable_at, first_blocked = _ray(state, spec.max_reach + 1)
    # a primitive row has no heading and no precondition, so its ray index only has to stay in bounds
    ray_heading = jnp.maximum(spec.heading, 0)
    reachable = spec.reach + 1
    arrival = jnp.minimum(reachable, first_blocked[ray_heading])
    # exact at `arrival` for a non-following row that only rotates and advances; a following
    # row may detour, so it takes the looser test of anywhere on the ray
    following = spec.follow == 1
    pickable_ahead = jnp.where(
        following,
        (jnp.cumsum(pickable_at, axis=1) > 0)[ray_heading, reachable],
        pickable_at[ray_heading, arrival],
    )
    door_ahead = jnp.where(
        following,
        (jnp.cumsum(door_openable_at, axis=1) > 0)[ray_heading, reachable],
        door_openable_at[ray_heading, arrival],
    )
    # navix `drop`s only onto a cell the player could walk into, so a blocked
    # arrival cell is a failed drop
    room_ahead = following | (arrival < first_blocked[ray_heading])

    met = jnp.where(
        spec.requires == NEEDS_PICKABLE,
        pickable_ahead,
        jnp.where(
            spec.requires == NEEDS_OPENABLE_DOOR,
            door_ahead,
            jnp.where(
                spec.requires == NEEDS_POCKET,
                # the pile, not `player.pocket`, which navix's `drop` leaves set
                (_pile_size(state) > 0) & room_ahead,
                True,
            ),
        ),
    )

    has_work = status != STATUS_DONE
    available = (met & has_work) | (spec.heading < 0)
    fallback = jnp.where(jnp.any(has_work), has_work, jnp.ones_like(available))
    return jnp.where(jnp.any(available), available, fallback)


def option_policy(
    spec: OptionSpec,
    option: Array,
    state: State,
    neighbours: Array,
    aim: Array,
    home_aim: Array,
    status: Array,
) -> Array:
    """`pi`: the primitive the option takes from `state`, one per STATUS_* code."""
    player = state.get_player()
    # the executor re-aims off these same `neighbours`, so the two agree without the detour target being carried
    target = jnp.where(
        status == STATUS_DETOUR, _detour_aim(neighbours, home_aim), aim
    )
    spin = (target - player.direction) % 4
    turn = jnp.where(spin == 3, spec.ccw, spec.cw)
    interact = jnp.where(spec.interact[option] >= 0, spec.interact[option], spec.cw)
    # STATUS_DONE falls through to `interact` only to type the action
    action = jnp.where(
        (status == STATUS_TURN) | (status == STATUS_DETOUR),
        turn,
        jnp.where(status == STATUS_ADVANCE, spec.forward, interact),
    )
    return action.astype(jnp.int32)


class Execution(struct.PyTreeNode):
    """One option's progress through its own execution: the executor's carry."""

    current: Timestep
    aim: Array
    advanced: Array
    detoured: Array
    interacted: Array
    attempted: Array
    status: Array
    neighbours: Array
    stop: Array
    total: Array
    steps: Array
    banked: Array
    hold: Array


class OptionEnv(Environment):
    """Executes one already-selected option as a single decision: the id arrives
    as the action, and the duration is whatever `beta` fires at."""

    env: Environment = None  # type: ignore[assignment]
    spec: OptionSpec = None  # type: ignore[assignment]
    executor: str = struct.field(pytree_node=False, default="scan")
    """`while_loop` stops each lane at its own `beta`; `scan` always runs
    `horizon` iterations and masks the rest. Same trajectories, different cost."""

    reward_delay: int = struct.field(pytree_node=False, default=0)
    """Primitive steps between earning the terminal reward and being paid it.
    0 is off exactly: the window opens and closes on the same step."""

    @classmethod
    def create(
        cls,
        env: Environment,
        spec: OptionSpec,
        executor: str = "scan",
        reward_delay: int = 0,
    ) -> "OptionEnv":
        if executor not in ("while_loop", "scan"):
            raise ValueError(f"unknown executor {executor}")
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
            executor=executor,
            reward_delay=reward_delay,
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
                interact_failed=jnp.asarray(False),
                banked=jnp.asarray(0.0, dtype=jnp.float32),
                hold=jnp.asarray(0, dtype=jnp.int32),
                episode_t=jnp.asarray(0, dtype=jnp.int32),
            )
        )

    def _hold(
        self, stepped: Timestep, banked: Array, hold: Array
    ) -> Tuple[Timestep, Array, Array]:
        """`stepped` with its terminal reward withheld, and the window's state."""
        # Reward and termination are state predicates; holding re-applies both. 
        # Dropped reward is restored from `info["return"]` at payout.
 
        holding = hold > 0
        earned = stepped.is_termination() & jnp.logical_not(holding)
        window = earned | holding
        remaining = jnp.where(
            earned, self.reward_delay, jnp.where(holding, hold - 1, 0)
        )
        # Payment by time, not step_type—window surviving episode is unpaid
 
        due = window & ((remaining == 0) | (stepped.t >= self.max_steps))
        banked = jnp.where(earned, stepped.reward, banked)
        dropped = jnp.where(window & stepped.is_termination(), stepped.reward, 0.0)
        paid = jnp.where(due, banked, 0.0)
        return (
            stepped.replace(
                reward=stepped.reward - dropped + paid,
                step_type=jnp.where(
                    window,
                    jnp.where(due, StepType.TERMINATION, StepType.TRANSITION),
                    stepped.step_type,
                ).astype(jnp.int32),
                info={
                    **stepped.info,
                    "return": stepped.info["return"] - dropped + paid,
                },
            ),
            jnp.where(due, 0.0, banked),
            jnp.where(due, 0, remaining).astype(jnp.int32),
        )

    def step(self, timestep: Timestep, action: Array) -> Timestep:
        # done must take one step, else reward leaks across episode boundary
 
        started_done = timestep.is_done()
        inner = timestep.replace(info={"return": timestep.info["return"]})
        gamma = jnp.asarray(self.env.gamma, dtype=jnp.float32)
        max_duration = self.spec.max_duration[action]
        home_aim = _initial_aim(self.spec, action, timestep.state)

        def body(execution: Execution) -> Execution:
            current, aim, status, neighbours, stop, steps = (
                execution.current,
                execution.aim,
                execution.status,
                execution.neighbours,
                execution.stop,
                execution.steps,
            )
            # masked rather than skipped, so `scan` can run the same body
            active = ~stop

            primitive = option_policy(
                self.spec, action, current.state, neighbours, aim, home_aim, status
            )
            before = current.state
            stepped, banked_next, hold_next = self._hold(
                self.env.step(current, primitive), execution.banked, execution.hold
            )
            current = jax.tree.map(
                lambda a, b: jnp.where(active, a, b), stepped, current
            )
            banked = jnp.where(active, banked_next, execution.banked)
            hold = jnp.where(active, hold_next, execution.hold)

            total = execution.total + jnp.where(
                active, gamma**steps * stepped.reward, 0.0
            )
            steps = steps + jnp.where(active, 1, 0)

            # an autoreset neither moved nor interacted: `env.step` discarded
            # the action and started a fresh episode
            live = active & ~started_done
            moved = live & jnp.any(
                current.state.get_player().position != before.get_player().position
            )
            # a blocked forward did not advance the option
            walked = moved & (status == STATUS_ADVANCE)
            directed = walked & (aim == home_aim)
            lateral = walked & (aim != home_aim)
            advanced = execution.advanced + jnp.where(directed, 1, 0)
            detoured = execution.detoured + jnp.where(lateral, 1, 0)
            # back to the heading after a lateral, so `reach` bounds directed moves
            aim = jnp.where(
                lateral,
                home_aim,
                jnp.where(
                    live & (status == STATUS_DETOUR),
                    _detour_aim(neighbours, home_aim),
                    aim,
                ),
            )

            acted = live & (status == STATUS_INTERACT)
            attempted = execution.attempted | acted
            interacted = execution.interacted | (
                acted & _interaction_succeeded(self.spec, action, before, current.state)
            )

            # carried, not recomputed next iteration: `pi` and `_detour_aim`
            # read the same cells, and each read is four `walkable` calls
            neighbours = _neighbours(current.state)
            status = _status(
                self.spec,
                action,
                current.state,
                neighbours,
                aim,
                advanced,
                detoured,
                interacted,
                attempted,
            )
            fired = (status == STATUS_DONE) | (steps >= max_duration)
            stop = stop | started_done | (active & (current.is_done() | fired))
            return Execution(
                current=current,
                aim=aim,
                advanced=advanced,
                detoured=detoured,
                interacted=interacted,
                attempted=attempted,
                status=status,
                neighbours=neighbours,
                stop=stop,
                total=total,
                steps=steps,
                banked=banked,
                hold=hold,
            )

        def cond(execution: Execution) -> Array:
            steps = execution.steps
            fired = (execution.status == STATUS_DONE) | (steps >= max_duration)
            # `started_done` survives only the first check, since the body sets `stop` unconditionally on that path
            running = ~execution.stop & (started_done | ~fired)
            return running & (steps < self.spec.horizon)

        zero = jnp.asarray(0, dtype=jnp.int32)
        false = jnp.asarray(False)
        interacted = self.spec.interact[action] < 0
        neighbours = _neighbours(inner.state)
        init_status = _status(
            self.spec,
            action,
            inner.state,
            neighbours,
            home_aim,
            zero,
            zero,
            interacted,
            false,
        )
        # Mask iteration 0 so a beta-fires-on-selection option costs zero steps,
        # as it does under `while_loop`. `started_done` is excluded: its one step
        # performs the autoreset.
        init_fired = (init_status == STATUS_DONE) & jnp.logical_not(started_done)
        init = Execution(
            current=inner,
            aim=home_aim,
            advanced=zero,
            detoured=zero,
            interacted=interacted,
            attempted=false,
            status=init_status,
            neighbours=neighbours,
            stop=init_fired,
            total=jnp.asarray(0.0, dtype=jnp.float32),
            steps=zero,
            banked=timestep.info["reward_banked"],
            hold=timestep.info["reward_hold"],
        )
        if self.executor == "scan":
            final, _ = jax.lax.scan(
                lambda c, _: (body(c), None), init, None, self.spec.horizon
            )
        else:
            final = jax.lax.while_loop(cond, body, init)

        # both zero on the autoreset decision, which belongs to neither episode
        decision_t = jnp.where(started_done, 0, timestep.info["decision_t"] + 1)
        episode_t = jnp.where(
            started_done, 0, timestep.info["episode_t"] + final.steps
        )
        return final.current.replace(
            reward=final.total,
            info=self._info(
                final.current.info,
                final.current.state,
                steps=final.steps,
                decision_t=decision_t,
                interact_failed=final.attempted & ~final.interacted,
                banked=final.banked,
                hold=final.hold,
                episode_t=episode_t,
            ),
        )

    def _info(
        self,
        info: Dict[str, Array],
        state: State,
        steps: Array,
        decision_t: Array,
        interact_failed: Array,
        banked: Array,
        hold: Array,
        episode_t: Array,
    ) -> Dict[str, Array]:
        available = initiation(self.spec, state)
        return {
            **info,
            "primitive_steps": steps,
            # for SMDP discounting: the next state is `steps` steps away
            "option_discount": jnp.asarray(self.env.gamma, dtype=jnp.float32) ** steps,
            # `I` at the state the policy over options selects from next
            "available": available,
            "available_frac": jnp.mean(available.astype(jnp.float32)),
            "decision_t": decision_t,
            # not `Timestep.t`, the truncation clock, which `stagger_envs`
            # offsets once at init
            "episode_t": episode_t,
            "interact_failed": interact_failed,
            # in the info because a reward-delay window outlives the option, and
            # often the episode's remaining decisions, that opened it
            "reward_banked": banked,
            "reward_hold": hold,
        }


def _duration_rollout(
    env: OptionEnv, num_envs: int, decisions: int, seed: int
) -> Array:
    """i32[decisions, num_envs] primitive steps, under uniform selection from `I`."""

    def step(carry: Tuple[Timestep, Array], _) -> Tuple[Tuple[Timestep, Array], Array]:
        timestep, rng = carry
        rng, key = jax.random.split(rng)
        logits = jnp.where(timestep.info["available"], 0.0, -1e8)
        action = jax.random.categorical(key, logits)
        timestep = jax.vmap(env.step)(timestep, action)
        return (timestep, rng), timestep.info["primitive_steps"]

    rng = jax.random.PRNGKey(seed)
    timestep = jax.vmap(env.reset)(jax.random.split(rng, num_envs))
    _, steps = jax.lax.scan(step, (timestep, rng), None, decisions)
    return steps


def measure_duration_stats(
    env: OptionEnv, num_envs: int = 64, decisions: int = 64, seed: int = 0
) -> Dict[str, float]:
    """Mean and worst-lane duration under uniform selection: the mean is what
    the primitive-step budget divides by, so it is measured, not derived."""
    steps = _duration_rollout(env, num_envs, decisions, seed)
    max_lane = jnp.max(steps, axis=1)
    return {
        "mean": float(jnp.mean(steps)),
        "max_lane_mean": float(jnp.mean(max_lane)),
        "max_lane_max": float(jnp.max(max_lane)),
    }
