"""Tests for the option controller family and its executor.

`cd navix && JAX_PLATFORMS=cpu python -m pytest test_options.py -q`.

Most tests hand-build a DoorKey-8x8 timestep with `build()`. Walls go into the
grid, not `Wall` entities; unused entities are parked on the corner cell.
"""

from typing import List, Sequence, Tuple

import jax
import jax.numpy as jnp
import pytest
from jax import Array

import navix as nx
from navix import actions, observations
from navix.components import DISCARD_PILE_COORDS, EMPTY_POCKET_ID
from navix.entities import Entities
from navix.environments.environment import StepType, Timestep
from navix.grid import positions_equal
from navix.states import State

from options import (
    NO_INTERACT,
    PRIMITIVE,
    STATUS_DETOUR,
    STATUS_DONE,
    OptionEnv,
    OptionSpec,
    _detour_aim,
    _interaction_succeeded,
    _max_duration,
    _neighbours,
    _status,
    action_names,
    grammar_options,
    initiation,
)

ENV = nx.make(
    "Navix-DoorKey-8x8-v0", max_steps=400, observation_fn=observations.symbolic
)
NAMES = action_names(ENV)
FORWARD = NAMES.index("forward")
PICKUP = NAMES.index("pickup")
DROP = NAMES.index("drop")
TOGGLE = NAMES.index("toggle")

EAST, SOUTH, WEST, NORTH = 0, 1, 2, 3
KEY_ID = 3
"""The key id DoorKey spawns, and the id its door requires."""
CORNER = (0, 0)
"""A grid wall in every layout below: where entities go to be out of the way."""

OPEN_ROOM = (
    "########",
    "#......#",
    "#......#",
    "#......#",
    "#......#",
    "#......#",
    "#......#",
    "########",
)
BLOCKED_EAST = (
    "########",
    "#.#....#",
    "#......#",
    "#......#",
    "#......#",
    "#......#",
    "#......#",
    "########",
)
ENCLOSED = (
    "########",
    "#.#....#",
    "##.....#",
    "#......#",
    "#......#",
    "#......#",
    "#......#",
    "########",
)
DOOR_CELL = (
    "########",
    "#......#",
    "##.....#",
    "#......#",
    "#......#",
    "#......#",
    "#......#",
    "########",
)
DEAD_END = (
    "########",
    "#..#...#",
    "#.#....#",
    "#......#",
    "#......#",
    "#......#",
    "#......#",
    "########",
)
RIGHT_LATERAL_BLOCKED = (
    "########",
    "#......#",
    "#..#...#",
    "#.##...#",
    "#......#",
    "#......#",
    "#......#",
    "########",
)
"""From (2, 2) facing east: heading and right lateral blocked, left lateral and reverse clear."""

_RESET = ENV.reset(jax.random.PRNGKey(0))
"""One real reset, used only as a source of correctly shaped entity structs."""


def build(
    layout: Sequence[str],
    player_position: Tuple[int, int],
    player_direction: int,
    key_position: Tuple[int, int] = CORNER,
    door_position: Tuple[int, int] = CORNER,
    door_open: bool = False,
    pocket: int = int(EMPTY_POCKET_ID),
    goal_position: Tuple[int, int] = CORNER,
) -> Timestep:
    """A DoorKey-8x8 timestep with the map replaced by `layout`.

    `layout` is eight strings of eight characters, `.` walkable and anything
    else a wall. Entities left at `CORNER` sit on a grid wall and are unreachable.
    """
    state = _RESET.state
    grid = jnp.asarray(
        [[0 if cell == "." else -1 for cell in row] for row in layout],
        dtype=state.grid.dtype,
    )
    # not state.get_walls(): its `entities.get(WALL, Wall())` default is
    # constructed eagerly, and Wall() is missing its required arguments
    walls = state.entities[Entities.WALL]
    player = state.get_player().replace(
        position=jnp.asarray(player_position, dtype=jnp.int32),
        direction=jnp.asarray(player_direction, dtype=jnp.int32),
        pocket=jnp.asarray(pocket, dtype=jnp.int32),
    )
    entities = {
        "player": player[None],
        "key": state.get_keys().replace(
            position=jnp.asarray([key_position], dtype=jnp.int32)
        ),
        "door": state.get_doors().replace(
            position=jnp.asarray([door_position], dtype=jnp.int32),
            open=jnp.asarray([door_open]),
        ),
        "goal": state.get_goals().replace(
            position=jnp.asarray([goal_position], dtype=jnp.int32)
        ),
        "wall": walls.replace(position=jnp.zeros_like(walls.position)),
    }
    state = state.replace(grid=grid, entities=entities)
    return _RESET.replace(
        t=jnp.asarray(0, dtype=jnp.int32),
        step_type=jnp.asarray(int(StepType.TRANSITION), dtype=jnp.int32),
        reward=jnp.asarray(0.0, dtype=jnp.float32),
        state=state,
        observation=ENV.observation_fn(state),
        info={
            "return": jnp.asarray(0.0),
            "decision_t": jnp.asarray(0, dtype=jnp.int32),
            "reward_banked": jnp.asarray(0.0, dtype=jnp.float32),
            "reward_hold": jnp.asarray(0, dtype=jnp.int32),
            "episode_t": jnp.asarray(0, dtype=jnp.int32),
        },
    )


def option_env(
    rows: List[Tuple[int, ...]],
    executor: str = "while_loop",
    reward_delay: int = 0,
) -> OptionEnv:
    """An `OptionEnv` over DoorKey-8x8 whose action table is exactly `rows`."""
    return OptionEnv.create(
        ENV,
        OptionSpec.create(rows, NAMES),
        executor=executor,
        reward_delay=reward_delay,
    )


def position_of(timestep: Timestep) -> Tuple[int, int]:
    """The player's `(row, col)` as Python ints, for readable assertions."""
    position = timestep.state.get_player().position
    return int(position[0]), int(position[1])


def held(state: State) -> int:
    """How many keys sit in the discard pile, i.e. are held by the player."""
    keys = state.get_keys()
    return int(jnp.sum(positions_equal(keys.position, DISCARD_PILE_COORDS)))


def test_a_blocked_option_with_no_work_never_spins() -> None:
    """A blocked option with no interaction and no follow is absent from `I`, and costs no steps if stepped anyway."""
    walk = (PRIMITIVE, 0, FORWARD, 0)
    blocked = (EAST, 1, NO_INTERACT, 0)
    env = option_env([walk, blocked])
    timestep = build(BLOCKED_EAST, player_position=(1, 1), player_direction=EAST)

    available = initiation(env.spec, timestep.state)
    assert bool(available[0]), "the primitive row must stay available"
    assert not bool(available[1]), "an option with no work is not in I"

    stepped = env.step(timestep, jnp.asarray(1))
    assert int(stepped.info["primitive_steps"]) == 0
    assert position_of(stepped) == (1, 1)


def test_scan_executor_matches_while_loop_on_a_beta_fires_on_selection_option() -> (
    None
):
    """A `beta`-fires-on-selection option costs zero steps under `scan`, not just `while_loop`.

    `initiation`'s graded fallback (`options.py`'s `initiation`) can select such
    an option when nothing else has work; this checks the executor, not `I`, by
    stepping it directly.
    """
    blocked = (EAST, 1, NO_INTERACT, 0)
    timestep = build(BLOCKED_EAST, player_position=(1, 1), player_direction=EAST)

    fast_stepped = option_env([blocked], executor="while_loop").step(
        timestep, jnp.asarray(0)
    )
    slow_stepped = option_env([blocked], executor="scan").step(
        timestep, jnp.asarray(0)
    )

    assert int(fast_stepped.info["primitive_steps"]) == 0
    assert int(slow_stepped.info["primitive_steps"]) == 0
    assert position_of(slow_stepped) == (1, 1)
    assert int(slow_stepped.state.get_player().direction) == EAST


@pytest.mark.parametrize("seed", range(10))
def test_no_available_option_is_zero_length(seed: int) -> None:
    """Any option present in `I` consumes at least one primitive step, checked over sampled states."""
    rows, _ = grammar_options(32, 2, NAMES)
    env = option_env(rows)
    num_envs = 8
    step = jax.jit(jax.vmap(env.step))

    rng = jax.random.PRNGKey(seed)
    timestep = jax.vmap(env.reset)(jax.random.split(rng, num_envs))
    for _ in range(6):
        for option in range(env.spec.size):
            result = step(timestep, jnp.full((num_envs,), option, dtype=jnp.int32))
            available = timestep.info["available"][:, option]
            steps = result.info["primitive_steps"]
            assert bool(jnp.all(jnp.where(available, steps >= 1, True))), (
                f"option {option} is available somewhere but takes 0 steps"
            )
        rng, key = jax.random.split(rng)
        logits = jnp.where(timestep.info["available"], 0.0, -1e8)
        timestep = step(timestep, jax.random.categorical(key, logits))


def test_enclosed_player_can_still_interact() -> None:
    """With every neighbour blocked, `south1+pickup` takes the key and `east1+toggle` opens the door.

    Both interactions unblock the cell they act on, so each execution then
    spends the move it still owed: the interaction is not necessarily the last step.
    """
    take_key = (SOUTH, 1, PICKUP, 1)
    open_door = (EAST, 1, TOGGLE, 0)
    env = option_env([take_key, open_door])
    timestep = build(
        OPEN_ROOM,
        player_position=(1, 1),
        player_direction=EAST,
        key_position=(2, 1),
        door_position=(1, 2),
    )
    assert held(timestep.state) == 0

    after_pickup = env.step(timestep, jnp.asarray(0))
    assert not bool(after_pickup.info["interact_failed"])
    assert held(after_pickup.state) == 1
    assert int(after_pickup.state.get_player().pocket) == KEY_ID
    assert int(after_pickup.info["primitive_steps"]) == 3
    assert position_of(after_pickup) == (2, 1)

    enclosed_with_key = build(
        DOOR_CELL,
        player_position=(1, 1),
        player_direction=EAST,
        key_position=tuple(int(x) for x in DISCARD_PILE_COORDS),
        door_position=(1, 2),
        pocket=KEY_ID,
    )
    after_toggle = env.step(enclosed_with_key, jnp.asarray(1))
    assert bool(after_toggle.state.get_doors().open[0]), "the door must be open"
    assert not bool(after_toggle.info["interact_failed"])
    assert position_of(after_toggle) == (1, 2)
    assert int(after_toggle.info["primitive_steps"]) == 2


def test_enclosed_player_with_nothing_to_do_terminates() -> None:
    """Fully enclosed with no interaction: option is absent from `I` and costs no steps."""
    walk = (PRIMITIVE, 0, FORWARD, 0)
    follower = (EAST, 1, NO_INTERACT, 1)
    env = option_env([walk, follower])
    timestep = build(ENCLOSED, player_position=(1, 1), player_direction=EAST)

    assert not bool(initiation(env.spec, timestep.state)[1])
    stepped = env.step(timestep, jnp.asarray(1))
    assert int(stepped.info["primitive_steps"]) == 0
    assert position_of(stepped) == (1, 1)


def test_reach_counts_directed_moves_after_a_detour() -> None:
    """After a lateral detour the aim resets to the heading, so reach is spent on heading moves, not the detour direction."""
    env = option_env([(EAST, 2, NO_INTERACT, 1)])
    start = (1, 1)
    timestep = build(BLOCKED_EAST, player_position=start, player_direction=EAST)

    stepped = env.step(timestep, jnp.asarray(0))
    row, col = position_of(stepped)
    along, lateral = col - start[1], row - start[0]
    assert (row, col) == (2, 3)
    assert along == 2, "the reach is spent on the heading, not on the detour"
    assert -2 <= along <= 2, "net displacement along the heading is within reach"
    assert abs(lateral) <= 2, "lateral drift is within the detour budget"
    assert int(stepped.info["primitive_steps"]) == 5


def test_follow_prefers_a_lateral_detour_over_reversing() -> None:
    """A detour tries both laterals before it turns back."""
    env = option_env([(EAST, 2, NO_INTERACT, 1)])
    start = (2, 2)
    timestep = build(
        RIGHT_LATERAL_BLOCKED, player_position=start, player_direction=EAST
    )
    neighbours = _neighbours(timestep.state)
    assert not bool(neighbours[EAST]), "the heading is blocked"
    assert not bool(neighbours[SOUTH]), "the right lateral is blocked"
    assert bool(neighbours[WEST]), "the reverse is clear, and must not be preferred"

    stepped = env.step(timestep, jnp.asarray(0))
    assert position_of(stepped) == (1, 4)
    assert int(stepped.info["primitive_steps"]) == 5


@pytest.mark.parametrize("home_aim", [EAST, SOUTH, WEST, NORTH])
def test_detour_aim_prefers_laterals_before_reversing(home_aim: int) -> None:
    """`_detour_aim` returns `heading+1` if clear, else `heading+3`, else `heading+2`."""
    right, reverse, left = (home_aim + 1) % 4, (home_aim + 2) % 4, (home_aim + 3) % 4
    aim = jnp.asarray(home_aim, dtype=jnp.int32)

    def clear(*directions: int) -> Array:
        """bool[4] with exactly `directions` walkable."""
        return jnp.asarray([d in directions for d in (EAST, SOUTH, WEST, NORTH)])

    assert int(_detour_aim(clear(right, left, reverse), aim)) == right
    assert int(_detour_aim(clear(right, reverse), aim)) == right
    assert int(_detour_aim(clear(left, reverse), aim)) == left
    assert int(_detour_aim(clear(left), aim)) == left
    assert int(_detour_aim(clear(reverse), aim)) == reverse


def test_follow_ping_pongs_in_a_dead_end_within_budget() -> None:
    """In a dead end the detour reverses repeatedly, but stops on the detour budget, not the duration bound."""
    reach = 2
    row = (EAST, reach, NO_INTERACT, 1)
    env = option_env([row])
    start = (1, 2)
    timestep = build(DEAD_END, player_position=start, player_direction=EAST)

    stepped = env.step(timestep, jnp.asarray(0))
    steps = int(stepped.info["primitive_steps"])
    assert position_of(stepped) in (start, (1, 1)), "net displacement is 0 or -1"
    assert steps < _max_duration(row), "the option stopped on a budget, not the bound"
    assert steps == 12

    spent = _status(
        env.spec,
        jnp.asarray(0),
        timestep.state,
        _neighbours(timestep.state),
        aim=jnp.asarray(EAST),
        advanced=jnp.asarray(0),
        detoured=jnp.asarray(reach),
        interacted=jnp.asarray(True),
        attempted=jnp.asarray(True),
    )
    assert int(spent) == STATUS_DONE
    unspent = _status(
        env.spec,
        jnp.asarray(0),
        timestep.state,
        _neighbours(timestep.state),
        aim=jnp.asarray(EAST),
        advanced=jnp.asarray(0),
        detoured=jnp.asarray(reach - 1),
        interacted=jnp.asarray(True),
        attempted=jnp.asarray(True),
    )
    assert int(unspent) == STATUS_DETOUR


# paired, not crossed: n=128 exceeds the 64-row catalogue at max_forward=2, and
# `_catalogue` orders follow rows last, so n=64 at max_forward=4 holds no follow row
@pytest.mark.parametrize("max_forward,n_options", ((2, 64), (4, 64), (4, 128), (8, 64)))
@pytest.mark.parametrize("seed", range(4))
def test_max_duration_bounds_duration_without_its_disjunct(
    seed: int, max_forward: int, n_options: int
) -> None:
    """Every execution finishes inside its row's `_max_duration`, even with `spec.max_duration` inflated so that disjunct never fires."""
    loose = 200
    rows, _ = grammar_options(n_options, max_forward, NAMES)
    spec = OptionSpec.create(rows, NAMES)
    spec = spec.replace(
        max_duration=jnp.full_like(spec.max_duration, loose), horizon=loose
    )
    env = OptionEnv.create(ENV, spec)
    bounds = jnp.asarray([_max_duration(row) for row in rows], dtype=jnp.int32)

    def decide(carry: Tuple[Timestep, Array], _) -> Tuple[Tuple[Timestep, Array], Tuple[Array, Array]]:
        timestep, rng = carry
        rng, key = jax.random.split(rng)
        logits = jnp.where(timestep.info["available"], 0.0, -1e8)
        action = jax.random.categorical(key, logits)
        timestep = jax.vmap(env.step)(timestep, action)
        return (timestep, rng), (action, timestep.info["primitive_steps"])

    rng = jax.random.PRNGKey(seed)
    timestep = jax.vmap(env.reset)(jax.random.split(rng, 16))
    _, (actions_taken, steps) = jax.lax.scan(decide, (timestep, rng), None, 32)

    assert bool(
        jnp.all(steps <= bounds[actions_taken])
    ), "an execution exceeded its max_duration"
    assert int(jnp.max(steps)) < loose, "an execution never terminated"


def test_failed_interaction_terminates_without_retrying() -> None:
    """A failed interaction terminates the option after one attempt, not a retry.

    `step` never consults `I`, so this holds even though the row (key on the
    ray but past reach + 1) is excluded from `I`.
    """
    env = option_env([(EAST, 1, PICKUP, 0), (EAST, 1, NO_INTERACT, 0)])
    timestep = build(
        OPEN_ROOM, player_position=(1, 1), player_direction=EAST, key_position=(1, 5)
    )
    available = initiation(env.spec, timestep.state)
    assert bool(available[1]), "the row without a precondition keeps I non-empty"
    assert not bool(available[0]), "the key is on the ray, but past reach + 1"

    stepped = env.step(timestep, jnp.asarray(0))
    assert int(stepped.info["primitive_steps"]) == 2, "one advance, one attempt"
    assert bool(stepped.info["interact_failed"])
    assert held(stepped.state) == 0
    assert int(stepped.state.get_player().pocket) == int(EMPTY_POCKET_ID)


def test_initiation_is_local() -> None:
    """A non-following row's precondition is exact: it checks the arrival cell `min(reach + 1, b)` ahead, not the whole ray or map."""
    take = (EAST, 2, PICKUP, 0)
    filler = (EAST, 1, NO_INTERACT, 0)
    env = option_env([take, filler])

    def available_at(key_position: Tuple[int, int]) -> bool:
        """Whether the pickup row is in `I` with the key at `key_position`."""
        timestep = build(
            OPEN_ROOM,
            player_position=(1, 1),
            player_direction=EAST,
            key_position=key_position,
        )
        return bool(initiation(env.spec, timestep.state)[0])

    assert available_at((1, 3)), "a key on the ray, inside the reach"
    assert available_at((1, 4)), "a key on the ray, at reach + 1"
    assert not available_at((1, 5)), "a key on the ray, past reach + 1"
    assert not available_at((2, 3)), "a key one cell off the ray"
    assert not available_at(CORNER), "a key parked out of the way"

    on_the_ray = build(
        OPEN_ROOM, player_position=(1, 1), player_direction=EAST, key_position=(1, 3)
    )
    stepped = env.step(on_the_ray, jnp.asarray(0))
    assert not bool(stepped.info["interact_failed"]), "the exact cell was predicted"
    assert held(stepped.state) == 1
    assert int(stepped.info["primitive_steps"]) == 3
    assert position_of(stepped) == (1, 3)


def test_a_following_row_takes_the_looser_ray_test() -> None:
    """A following row's precondition is the looser "anywhere on the ray within reach + 1", since its arrival cell isn't fixed."""
    env = option_env([(EAST, 2, PICKUP, 0), (EAST, 2, PICKUP, 1)])
    timestep = build(
        BLOCKED_EAST, player_position=(1, 1), player_direction=EAST, key_position=(1, 3)
    )
    assert not bool(_neighbours(timestep.state)[EAST]), "the ray is blocked at once"

    available = initiation(env.spec, timestep.state)
    assert bool(available[1]), "a following row may reach a key behind the wall"
    assert not bool(available[0]), "a non-following row would pick up the wall"


def test_drop_precondition_reads_the_pile_not_the_pocket() -> None:
    """Drop's precondition reads discard-pile membership, not the stale `player.pocket` that `drop` leaves set."""
    env = option_env([(EAST, 1, DROP, 0), (EAST, 1, NO_INTERACT, 0)])
    dropped = build(
        OPEN_ROOM,
        player_position=(1, 1),
        player_direction=EAST,
        key_position=(3, 3),
        pocket=KEY_ID,
    )
    assert held(dropped.state) == 0, "the key is on the floor, not in the pile"
    assert int(dropped.state.get_player().pocket) == KEY_ID, "the stale pocket"
    assert not bool(initiation(env.spec, dropped.state)[0])

    holding = build(
        OPEN_ROOM,
        player_position=(1, 1),
        player_direction=EAST,
        key_position=tuple(int(x) for x in DISCARD_PILE_COORDS),
        pocket=KEY_ID,
    )
    assert bool(initiation(env.spec, holding.state)[0])
    stepped = env.step(holding, jnp.asarray(0))
    assert not bool(stepped.info["interact_failed"])
    assert held(stepped.state) == 0, "the key left the pile"


def test_fallback_is_graded() -> None:
    """With nothing selectable, `I`'s fallback returns only rows with work, not every row."""
    attempt = (EAST, 1, PICKUP, 0)
    nothing_to_do = (EAST, 1, NO_INTERACT, 0)
    env = option_env([attempt, nothing_to_do])
    timestep = build(BLOCKED_EAST, player_position=(1, 1), player_direction=EAST)

    available = initiation(env.spec, timestep.state)
    assert bool(available[0]), "the fallback fired: no row met its precondition"
    assert not bool(available[1]), "and it did not readmit a row with no work"

    stepped = env.step(timestep, jnp.asarray(0))
    assert int(stepped.info["primitive_steps"]) >= 1, "no available option is empty"
    assert bool(stepped.info["interact_failed"]), "it attempted, on a wall"


def test_option_on_a_done_timestep_only_autoresets() -> None:
    """Stepping a done timestep only autoresets: exactly one step, no reward carried across."""
    env = option_env([(EAST, 4, NO_INTERACT, 1)])
    timestep = build(OPEN_ROOM, player_position=(1, 1), player_direction=EAST)
    done = timestep.replace(
        step_type=jnp.asarray(int(StepType.TERMINATION), dtype=jnp.int32),
        t=jnp.asarray(37, dtype=jnp.int32),
        info={**timestep.info, "decision_t": jnp.asarray(9, dtype=jnp.int32)},
    )

    stepped = env.step(done, jnp.asarray(0))
    assert int(stepped.info["primitive_steps"]) == 1
    assert float(stepped.reward) == 0.0
    assert int(stepped.t) == 0, "a fresh episode"
    assert int(stepped.step_type) == int(StepType.TRANSITION)
    assert int(stepped.info["decision_t"]) == 0


def test_reward_is_discounted_per_primitive_step() -> None:
    """The decision's reward is the sum of primitive rewards discounted by `gamma ** t` within the decision."""
    env = option_env([(EAST, 2, NO_INTERACT, 0)])
    timestep = build(
        OPEN_ROOM, player_position=(1, 1), player_direction=EAST, goal_position=(1, 3)
    )

    stepped = env.step(timestep, jnp.asarray(0))
    assert int(stepped.info["primitive_steps"]) == 2

    replay = timestep
    expected = 0.0
    for step in range(2):
        replay = ENV.step(replay, jnp.asarray(FORWARD))
        expected += ENV.gamma**step * float(replay.reward)

    assert expected == pytest.approx(ENV.gamma, abs=1e-6), "reward on the second step"
    assert float(stepped.reward) == pytest.approx(expected, abs=1e-6)


def test_reward_delay_holds_the_payout_without_ending_the_episode() -> None:
    """`reward_delay` withholds the terminal reward for that many primitive steps, without re-paying on the held steps in between."""
    delay = 3
    rows = [(PRIMITIVE, 0, FORWARD, 0), (PRIMITIVE, 0, PICKUP, 0)]
    walk, stand = jnp.asarray(0), jnp.asarray(1)
    start = build(
        OPEN_ROOM, player_position=(1, 1), player_direction=EAST, goal_position=(1, 2)
    )

    immediate = option_env(rows).step(start, walk)
    assert float(immediate.reward) == pytest.approx(1.0), "off means paid on arrival"
    assert int(immediate.step_type) == int(StepType.TERMINATION)
    assert float(immediate.info["return"]) == pytest.approx(1.0)

    env = option_env(rows, reward_delay=delay)
    timestep = env.step(start, walk)
    assert position_of(timestep) == (1, 2), "the goal was reached"
    assert float(timestep.reward) == 0.0, "banked, not paid"
    assert int(timestep.step_type) == int(StepType.TRANSITION), "the episode runs on"
    assert float(timestep.info["reward_banked"]) == pytest.approx(1.0)

    for step in range(1, delay):
        timestep = env.step(timestep, stand)
        assert position_of(timestep) == (1, 2), "still standing on the goal"
        assert float(timestep.reward) == 0.0, f"still held after {step} steps"
        assert int(timestep.step_type) == int(StepType.TRANSITION)
        assert float(timestep.info["return"]) == 0.0, "and not paid by re-firing"

    paid = env.step(timestep, stand)
    assert float(paid.reward) == pytest.approx(1.0)
    assert int(paid.step_type) == int(StepType.TERMINATION)
    assert float(paid.info["reward_banked"]) == 0.0
    assert float(paid.info["return"]) == pytest.approx(1.0), "paid exactly once"
    assert int(paid.t) == delay + 1, "the episode is the delay longer"


def test_episode_t_counts_the_episode_not_the_clock() -> None:
    """`episode_t` counts the current episode's own primitive steps, independent of `t`'s stagger offset, and resets to 0 on autoreset."""
    offset = 37
    env = option_env([(EAST, 2, NO_INTERACT, 0), (PRIMITIVE, 0, FORWARD, 0)])
    staggered = build(OPEN_ROOM, player_position=(1, 1), player_direction=EAST).replace(
        t=jnp.asarray(offset, dtype=jnp.int32)
    )

    timestep, taken = staggered, 0
    for option in (0, 1, 0):
        timestep = env.step(timestep, jnp.asarray(option))
        taken += int(timestep.info["primitive_steps"])
        assert int(timestep.info["episode_t"]) == taken
        assert int(timestep.t) == offset + taken, "the clock keeps its offset"
    assert taken == 5, "two advances, one primitive, two advances"

    done = timestep.replace(
        step_type=jnp.asarray(int(StepType.TERMINATION), dtype=jnp.int32)
    )
    restarted = env.step(done, jnp.asarray(0))
    assert int(restarted.info["primitive_steps"]) == 1, "the autoreset step"
    assert int(restarted.info["episode_t"]) == 0, "which belongs to no episode"


def test_interaction_success_is_falsifiable() -> None:
    """`_interaction_succeeded` reads discard-pile membership from the world, not `player.pocket`, which `drop` never clears."""
    put = option_env([(EAST, 1, DROP, 0)]).spec
    take = option_env([(EAST, 1, PICKUP, 0)]).spec
    option = jnp.asarray(0)
    # jit forces a copy: actions.drop/pickup write into the state's entity dict in place
    drop = jax.jit(actions.drop)
    pickup = jax.jit(actions.pickup)

    empty_pocket = build(
        OPEN_ROOM, player_position=(1, 1), player_direction=EAST, key_position=(5, 5)
    ).state
    assert not bool(
        _interaction_succeeded(put, option, empty_pocket, drop(empty_pocket))
    ), "dropping with an empty pocket changes nothing"

    into_wall = build(
        BLOCKED_EAST,
        player_position=(1, 1),
        player_direction=EAST,
        key_position=tuple(int(x) for x in DISCARD_PILE_COORDS),
        pocket=KEY_ID,
    ).state
    assert not bool(
        _interaction_succeeded(put, option, into_wall, drop(into_wall))
    ), "dropping into a wall changes nothing"

    holding = build(
        OPEN_ROOM,
        player_position=(1, 1),
        player_direction=EAST,
        key_position=tuple(int(x) for x in DISCARD_PILE_COORDS),
        pocket=KEY_ID,
    ).state
    assert bool(_interaction_succeeded(put, option, holding, drop(holding)))

    facing_key = build(
        OPEN_ROOM, player_position=(1, 1), player_direction=EAST, key_position=(1, 2)
    ).state
    assert bool(_interaction_succeeded(take, option, facing_key, pickup(facing_key)))
    assert not bool(
        _interaction_succeeded(take, option, empty_pocket, pickup(empty_pocket))
    ), "picking up nothing is not a successful pickup"


@pytest.mark.parametrize("seed", range(10))
def test_executors_agree(seed: int) -> None:
    """`scan` and `while_loop` executors produce identical decisions on the same trajectories."""
    rows, _ = grammar_options(32, 3, NAMES)
    while_loop = option_env(rows, executor="while_loop")
    scan = option_env(rows, executor="scan")

    def rollout(env: OptionEnv) -> Tuple[Array, Array, Array]:
        def decide(carry: Tuple[Timestep, Array], _):
            timestep, rng = carry
            rng, key = jax.random.split(rng)
            logits = jnp.where(timestep.info["available"], 0.0, -1e8)
            action = jax.random.categorical(key, logits)
            timestep = jax.vmap(env.step)(timestep, action)
            return (timestep, rng), (
                timestep.info["primitive_steps"],
                timestep.reward,
                timestep.state.get_player().position,
            )

        rng = jax.random.PRNGKey(seed)
        timestep = jax.vmap(env.reset)(jax.random.split(rng, 8))
        _, out = jax.lax.scan(decide, (timestep, rng), None, 24)
        return out

    fast_steps, fast_reward, fast_position = rollout(while_loop)
    slow_steps, slow_reward, slow_position = rollout(scan)
    assert bool(jnp.array_equal(fast_steps, slow_steps))
    assert bool(jnp.allclose(fast_reward, slow_reward))
    assert bool(jnp.array_equal(fast_position, slow_position))
