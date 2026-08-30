"""Catalogue enumeration and executor tests for `options.py`."""

from typing import Any, Dict, FrozenSet, List, Optional, Sequence, Tuple

import gymnasium as gym
import numpy as np
import pytest
from nle import nethack

from envs import OBSERVATION_KEYS
from options import (
    BOULDER,
    COMPASS,
    COMPASS_OFFSETS,
    CONCEDING_COMMANDS,
    DETOUR_EPISODE_COST,
    DRAIN_LIMIT,
    GROUP_ARG,
    GROUP_DIR,
    GROUP_MOVE,
    GROUP_SINGLE,
    MISC_IN_YN,
    MOVE_REPEATS,
    NEEDS_WALKABLE,
    TERM_BLOCKED,
    TERM_DETOUR,
    TERM_NO_HEADING,
    TERM_NO_PROMPT,
    TERM_SEQUENCE,
    OptionRow,
    OptionWrapper,
    _catalogue,
    catalogue_digest,
    grammar_depth_options,
    grammar_options,
    make_options,
    random_options,
    select_options,
)

ENV_ID = "NetHackChallenge-v0"
"""Host env for exp1/2/4."""

SHORT_EPISODE = 10
"""Legal horizon; no test here steps the env."""

CATALOGUE_SIZE = 227
GROUP_SIZES = {GROUP_MOVE: 80, GROUP_SINGLE: 15, GROUP_ARG: 100, GROUP_DIR: 32}
CATALOGUE_DIGEST = "6106edb3543445855b3d36ef13b7cc6a170d95d9aa9000f4a19f37c7c3599f0a"

GAMMA = 0.999
INVENTORY_LENGTH = 55
MESSAGE_LENGTH = 256
MISC_LENGTH = 3

START_X = 10
START_Y = 8
"""Stub hero cell. Away from the edges so all eight neighbours are on the map."""

FLOOR_GLYPH = nethack.GLYPH_CMAP_OFF + 19
WALL_GLYPH = nethack.GLYPH_CMAP_OFF + 1
DOORWAY_GLYPH = nethack.GLYPH_CMAP_OFF + 12
CLOSED_DOOR_GLYPH = nethack.GLYPH_CMAP_OFF + 15
OPEN_DOOR_GLYPH = nethack.GLYPH_CMAP_OFF + 13
"""cmap 19/1/12/15/13. Named here: `options` does not export them."""

MONSTER_GLYPH = nethack.GLYPH_MON_OFF + 1
"""Not a cmap, so `_is_walkable` reads the cell as open."""

BOULDER_GLYPH = nethack.GLYPH_OBJ_OFF + BOULDER
"""Object glyph: `glyph_is_cmap` cannot see it."""

DRAW_SIZE = 64
"""The n exp1, exp3, and exp4 use."""

ENV_ACTION_SET = 121
ACTION_TABLE = ENV_ACTION_SET - len(CONCEDING_COMMANDS)
"""Table drops conceding commands; the env action set is unchanged."""

KNOWN_TURN_ONE_ESCAPES = ("up",)
"""Free termination in the primitive table, not the catalogue."""

GRAMMAR_DIGEST = "c9c7555798bd30e9b50871a00916c2adeb948db22c57e6f75bb21d9cbae151bd"
GRAMMAR_DEPTH_DIGEST = "b973f18cb5e90d675b5da2ef849493f1c48a91430f9d1d821719a8eae14c74f8"
RANDOM_DIGESTS: Dict[int, str] = {
    0: "d5cf7e1ae76f9e4646be902e4a4b68acde6707f37a49a07878c351e14402b842",
    1: "dd89951dc083f809e0a9a72c2404fb04e5bfe422cd612a1533d7381292c8b054",
    2: "d2945b139d5f08956642a60cf122bf93496f69777d6e688786c93215eda7590d",
    3: "83df4c8bfe670426ec347308d4542884b0fbbede35c012d318846056c14a5a1d",
    4: "be5547d5d58b7ff7c5c8052f402dae6c7b48710f031c6a8975c26c508e65d830",
}
"""The five option seeds exp3 draws."""

GRAMMAR_PREFIX_AT_EIGHT = [
    "move_N_x16+follow",
    "down",
    "open_N",
    "move_S_x16+follow",
    "wait",
    "open_S",
    "move_E_x16+follow",
    "pickup",
]
"""Breadth-first prefix at n=8, spelled out."""

FIRST_INTERACTION_DEPTH = 80
"""`grammar_depth` movement rows before the first command."""


@pytest.fixture(scope="module")
def actions() -> Tuple[Any, ...]:
    """`ENV_ID` action set."""
    env = gym.make(
        ENV_ID, observation_keys=OBSERVATION_KEYS, max_episode_steps=SHORT_EPISODE
    )
    action_set = tuple(env.unwrapped.actions)
    env.close()
    return action_set


@pytest.fixture(scope="module")
def rows(actions: Tuple[Any, ...]) -> List[OptionRow]:
    """Full catalogue for `ENV_ID`."""
    return _catalogue(actions)


def digest_by_name(chosen: Sequence[OptionRow]) -> str:
    """Order-independent digest of the draw."""
    return catalogue_digest(sorted(chosen, key=lambda row: row.name))


class StubNetHack(gym.Env):
    """Minimal NLE: position, `misc`, and one map cell."""

    def __init__(
        self,
        actions: Tuple[Any, ...],
        *,
        advance: bool = False,
        walk: bool = False,
        obstacles: Optional[Dict[Tuple[int, int], int]] = None,
        prompt_keys: FrozenSet[int] = frozenset(),
        prompt_length: int = 0,
        background_glyph: int = FLOOR_GLYPH,
        neighbour_glyph: int = FLOOR_GLYPH,
        neighbour_heading: int = 0,
        opens_door: bool = False,
    ) -> None:
        self.actions = list(actions)
        self.advance = advance
        self.walk = walk
        self.obstacles = dict(obstacles or {})
        self.prompt_keys = prompt_keys
        self.prompt_length = prompt_length
        self.background_glyph = background_glyph
        self.neighbour_glyph = neighbour_glyph
        self.neighbour_heading = neighbour_heading
        self.opens_door = opens_door
        self.inv_letters = np.zeros(INVENTORY_LENGTH, dtype=np.uint8)
        self.keys: List[int] = []
        self.action_space = gym.spaces.Discrete(len(self.actions))
        self.offset_of_key = {
            self.actions.index(direction): COMPASS_OFFSETS[heading]
            for heading, direction in enumerate(COMPASS)
        }
        self._modal = 0
        self._x = START_X
        self._y = START_Y
        self._time = 1

    def _observation(self) -> Dict[str, np.ndarray]:
        """Observation keys the wrapper reads."""
        glyphs = np.full(nethack.DUNGEON_SHAPE, self.background_glyph, dtype=np.int16)
        dx, dy = COMPASS_OFFSETS[self.neighbour_heading]
        glyphs[self._y + dy, self._x + dx] = self.neighbour_glyph
        for (x, y), glyph in self.obstacles.items():
            glyphs[y, x] = glyph
        blstats = np.zeros(nethack.NLE_BLSTATS_SIZE, dtype=np.int64)
        blstats[nethack.NLE_BL_X] = self._x
        blstats[nethack.NLE_BL_Y] = self._y
        blstats[nethack.NLE_BL_TIME] = self._time
        misc = np.zeros(MISC_LENGTH, dtype=np.int32)
        misc[MISC_IN_YN] = int(self._modal > 0)
        return {
            "glyphs": glyphs,
            "blstats": blstats,
            "inv_letters": self.inv_letters,
            "misc": misc,
            "message": np.zeros(MESSAGE_LENGTH, dtype=np.uint8),
        }

    def reset(self, **kwargs: Any) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        """Fresh position and no prompt."""
        self._modal = 0
        self._x = START_X
        self._y = START_Y
        self._time = 1
        self.keys = []
        return self._observation(), {"end_status": 0}

    def step(
        self, action: int
    ) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict[str, Any]]:
        """One keystroke; never terminates."""
        self.keys.append(action)
        answering = self._modal > 0 and action not in self.prompt_keys
        if action in self.prompt_keys:
            self._modal = self.prompt_length
        else:
            self._modal = max(self._modal - 1, 0)
        if self.opens_door and answering:
            self.neighbour_glyph = OPEN_DOOR_GLYPH
        if self.advance:
            self._x += 1
        elif self.walk and action in self.offset_of_key:
            dx, dy = self.offset_of_key[action]
            target = (self._x + dx, self._y + dy)
            if target not in self.obstacles:
                self._x, self._y = target
        self._time += 1
        return self._observation(), 0.0, False, False, {"end_status": 0}


def test_catalogue_enumeration_is_pinned(rows: List[OptionRow]) -> None:
    """Canonical order matches the digest."""
    assert len(rows) == CATALOGUE_SIZE
    counts = {group: 0 for group in GROUP_SIZES}
    for row in rows:
        counts[row.group] += 1
    assert counts == GROUP_SIZES
    assert catalogue_digest(rows) == CATALOGUE_DIGEST, (
        "the enumeration order moved; current names are:\n"
        + "\n".join(row.name for row in rows)
    )


def test_movement_rows_carry_the_reach_ladder(rows: List[OptionRow]) -> None:
    """`reach` is the repeat count on movement rows and zero elsewhere."""
    reaches = {row.reach for row in rows if row.group == GROUP_MOVE}
    assert reaches == set(MOVE_REPEATS)
    assert all(row.reach == 0 for row in rows if row.group != GROUP_MOVE)


def test_random_draw_is_pinned(rows: List[OptionRow]) -> None:
    """Each option seed draws a pinned set."""
    for option_seed, expected in RANDOM_DIGESTS.items():
        chosen = random_options(DRAW_SIZE, rows, option_seed)
        assert len(chosen) == DRAW_SIZE
        assert digest_by_name(chosen) == expected, (
            f"option_seed {option_seed} now draws:\n"
            + "\n".join(sorted(row.name for row in chosen))
        )


def test_grammar_draws_are_pinned(rows: List[OptionRow]) -> None:
    """Both grammar catalogues at n=64 are pinned."""
    assert digest_by_name(grammar_options(DRAW_SIZE, rows)) == GRAMMAR_DIGEST
    assert (
        digest_by_name(grammar_depth_options(DRAW_SIZE, rows)) == GRAMMAR_DEPTH_DIGEST
    )


def test_the_breadth_prior_reaches_three_groups_at_eight_rows(
    rows: List[OptionRow],
) -> None:
    """`grammar` at n=8 reaches three groups."""
    assert [row.name for row in grammar_options(8, rows)] == GRAMMAR_PREFIX_AT_EIGHT


def test_the_depth_prior_holds_no_interaction_below_its_movement_block(
    rows: List[OptionRow],
) -> None:
    """`grammar_depth` places every movement row before every command."""
    ordered = grammar_depth_options(len(rows), rows)
    first_command = next(
        index for index, row in enumerate(ordered) if row.group != GROUP_MOVE
    )
    assert first_command == FIRST_INTERACTION_DEPTH
    assert all(row.group == GROUP_MOVE for row in ordered[:first_command])


def test_every_family_returns_the_whole_catalogue_at_full_size(
    rows: List[OptionRow],
) -> None:
    """At full catalogue size the families coincide."""
    everything = {row.name for row in rows}
    for chosen in (
        grammar_options(len(rows), rows),
        grammar_depth_options(len(rows), rows),
        random_options(len(rows), rows, 0),
    ):
        assert {row.name for row in chosen} == everything


def test_asking_for_more_options_than_the_catalogue_holds_is_an_error(
    rows: List[OptionRow],
) -> None:
    """Asking for more than the catalogue raises."""
    env = gym.make(
        ENV_ID, observation_keys=OBSERVATION_KEYS, max_episode_steps=SHORT_EPISODE
    )
    with pytest.raises(ValueError, match="catalogue rows"):
        make_options(env.unwrapped.actions, "option", len(rows) + 1, "grammar", 0)
    env.close()


def test_the_conceding_commands_are_absent_from_every_condition() -> None:
    """No condition can select `QUIT` or `SAVE`."""
    env = gym.make(
        ENV_ID, observation_keys=OBSERVATION_KEYS, max_episode_steps=SHORT_EPISODE
    )
    actions = env.unwrapped.actions
    tables = {
        condition: make_options(actions, condition, DRAW_SIZE, "grammar", 0)
        for condition in ("action", "option", "both")
    }
    env.close()

    assert len(actions) == ENV_ACTION_SET, "the env's own action set is untouched"
    banned = {command.name.lower() for command in CONCEDING_COMMANDS}
    for condition, (_, names, _) in tables.items():
        assert not banned & set(names), f"{condition} can still concede"
    assert len(tables["action"][1]) == ACTION_TABLE
    assert len(tables["both"][1]) == ACTION_TABLE + DRAW_SIZE


def test_no_new_action_ends_the_episode_within_two_steps() -> None:
    """Only the recorded escapes concede within two steps."""
    env = gym.make(
        ENV_ID, observation_keys=OBSERVATION_KEYS, max_episode_steps=SHORT_EPISODE
    )
    actions = env.unwrapped.actions
    sequences, names, _ = make_options(actions, "action", DRAW_SIZE, "grammar", 0)
    confirm = actions.index(nethack.CompassDirection.NW)

    conceded = []
    for sequence, name in zip(sequences, names):
        env.reset()
        _, _, terminated, _, _ = env.step(sequence[0])
        if not terminated:
            _, _, terminated, _, _ = env.step(confirm)
        if terminated:
            conceded.append(name)
    env.close()

    unrecorded = set(conceded) - set(KNOWN_TURN_ONE_ESCAPES)
    assert not unrecorded, (
        f"{sorted(unrecorded)} end the episode within two steps and are not in "
        "KNOWN_TURN_ONE_ESCAPES. Either the table regained a conceding command, "
        "or a new free termination exists and has to be decided on"
    )


def test_the_recorded_escape_still_exists() -> None:
    """`up` then northwest still concedes."""
    env = gym.make(
        ENV_ID, observation_keys=OBSERVATION_KEYS, max_episode_steps=SHORT_EPISODE
    )
    actions = env.unwrapped.actions
    env.reset()
    _, _, terminated, _, _ = env.step(actions.index(nethack.MiscDirection.UP))
    assert not terminated, "the climb prompt should not end the episode on its own"
    _, reward, terminated, _, info = env.step(
        actions.index(nethack.CompassDirection.NW)
    )
    env.close()

    assert terminated, "up then northwest should escape the dungeon"
    assert reward == 0.0, "the escape pays nothing, which is what makes it an exploit"
    assert "up" in KNOWN_TURN_ONE_ESCAPES


def test_both_is_the_primitive_table_followed_by_the_options(
    rows: List[OptionRow],
) -> None:
    """`both` is the primitive table followed by the options."""
    env = gym.make(
        ENV_ID, observation_keys=OBSERVATION_KEYS, max_episode_steps=SHORT_EPISODE
    )
    actions = env.unwrapped.actions
    primitive_sequences, primitive_names, _ = make_options(
        actions, "action", DRAW_SIZE, "grammar", 0
    )
    both_sequences, both_names, _ = make_options(
        actions, "both", DRAW_SIZE, "grammar", 0
    )
    option_sequences, _, _ = make_options(actions, "option", DRAW_SIZE, "grammar", 0)
    env.close()

    count = len(primitive_names)
    assert both_sequences[:count] == primitive_sequences
    assert both_names[:count] == primitive_names
    assert both_sequences[count:] == option_sequences


def primitive_table(actions: Tuple[Any, ...]) -> Tuple[List[OptionRow], List[str]]:
    """The `action` table and its names."""
    rows, names, _ = make_options(actions, "action", 0, "grammar", 0)
    return rows, names


def option_table(actions: Tuple[Any, ...]) -> Tuple[List[OptionRow], List[str]]:
    """The full catalogue as an option table."""
    rows, names, _ = make_options(actions, "option", CATALOGUE_SIZE, "grammar", 0)
    return rows, names


def key_index(actions: Tuple[Any, ...], command: Any) -> int:
    """Env action index of `command`."""
    return list(actions).index(command)


def monster_ring(heading_glyph: int, detour_glyph: int) -> Dict[Tuple[int, int], int]:
    """All eight neighbours occupied."""
    east = COMPASS.index(nethack.CompassDirection.E)
    return {
        (START_X + dx, START_Y + dy): (
            heading_glyph if heading == east else detour_glyph
        )
        for heading, (dx, dy) in enumerate(COMPASS_OFFSETS)
    }


def test_a_modal_state_surviving_a_decision_is_drained_and_charged(
    actions: Tuple[Any, ...],
) -> None:
    """Drain keystrokes are counted in `primitive_steps`."""
    rows, names = primitive_table(actions)
    stub = StubNetHack(
        actions,
        prompt_keys=frozenset({key_index(actions, nethack.Command.EAT)}),
        prompt_length=4,
    )
    env = OptionWrapper(stub, rows, gamma=GAMMA)
    env.reset()

    _, _, _, _, opened = env.step(names.index("eat"))
    assert opened["drain_steps"] == 0, "the decision that opens a prompt keeps it"
    assert opened["primitive_steps"] == 1

    _, _, _, _, drained = env.step(names.index("wait"))
    assert drained["drain_steps"] == 3, "the drain runs until misc is clear"
    assert drained["primitive_steps"] == 1 + drained["drain_steps"], (
        "the row's own keystroke plus the drain's, charged to this decision"
    )


def test_the_grace_decision_is_the_only_modal_one(actions: Tuple[Any, ...]) -> None:
    """A prompt survives exactly one decision, then is drained."""
    rows, names = primitive_table(actions)
    stub = StubNetHack(
        actions,
        prompt_keys=frozenset({key_index(actions, nethack.Command.EAT)}),
        prompt_length=2,
    )
    env = OptionWrapper(stub, rows, gamma=GAMMA)
    obs, _ = env.reset()

    modal_at_choice: List[bool] = []
    drained: List[int] = []
    for _ in range(6):
        modal_at_choice.append(bool(obs["misc"].any()))
        obs, _, _, _, info = env.step(names.index("eat"))
        drained.append(int(info["drain_steps"]))

    assert any(modal_at_choice), "the stub must go modal or this is vacuous"
    assert not any(
        earlier and later
        for earlier, later in zip(modal_at_choice, modal_at_choice[1:])
    ), "no two consecutive decisions may be chosen under a modal observation"
    assert [count > 0 for count in drained] == modal_at_choice, (
        "a decision drains exactly when it began modal"
    )


def test_drain_raises_when_misc_does_not_clear(actions: Tuple[Any, ...]) -> None:
    """The hang guard raises; `-O` would leave an unbounded loop."""
    rows, names = primitive_table(actions)
    stub = StubNetHack(
        actions,
        prompt_keys=frozenset({key_index(actions, nethack.Command.EAT)}),
        prompt_length=DRAIN_LIMIT + 2,
    )
    env = OptionWrapper(stub, rows, gamma=GAMMA)
    env.reset()

    _, _, _, _, opened = env.step(names.index("eat"))
    assert opened["drain_steps"] == 0, "the decision that opens the prompt keeps it"

    with pytest.raises(RuntimeError, match="no longer answers every modal state"):
        env.step(names.index("wait"))


def test_a_movement_row_stops_when_the_position_stops_changing(
    actions: Tuple[Any, ...],
) -> None:
    """A blocked 16-move costs one primitive step."""
    rows, names = option_table(actions)
    stub = StubNetHack(actions, advance=False)
    env = OptionWrapper(stub, rows, gamma=GAMMA)
    env.reset()

    _, _, _, _, info = env.step(names.index(f"move_N_x{max(MOVE_REPEATS)}"))
    assert info["primitive_steps"] == 1, "one keystroke establishes it is blocked"
    assert len(stub.keys) == 1


def test_a_movement_row_that_keeps_moving_spends_its_reach(
    actions: Tuple[Any, ...],
) -> None:
    """An unobstructed movement row spends its reach."""
    rows, names = option_table(actions)
    reach = max(MOVE_REPEATS)
    stub = StubNetHack(actions, advance=True)
    env = OptionWrapper(stub, rows, gamma=GAMMA)
    env.reset()

    _, _, _, _, info = env.step(names.index(f"move_N_x{reach}"))
    assert info["primitive_steps"] == reach
    assert len(stub.keys) == reach


def test_a_following_row_routes_around_a_wall_and_the_directed_twin_stops(
    actions: Tuple[Any, ...],
) -> None:
    """Following re-aims around a wall; the directed twin stops."""
    rows, names = option_table(actions)
    reach = max(MOVE_REPEATS)
    east = key_index(actions, nethack.CompassDirection.E)
    wall = {(START_X + 1, START_Y): WALL_GLYPH}

    directed_stub = StubNetHack(actions, walk=True, obstacles=wall)
    directed_env = OptionWrapper(directed_stub, rows, gamma=GAMMA)
    directed_env.reset()
    _, _, _, _, directed = directed_env.step(names.index(f"move_E_x{reach}"))

    following_stub = StubNetHack(actions, walk=True, obstacles=wall)
    following_env = OptionWrapper(following_stub, rows, gamma=GAMMA)
    following_env.reset()
    _, _, _, _, following = following_env.step(
        names.index(f"move_E_x{reach}+follow")
    )

    assert directed["primitive_steps"] == 1
    assert directed["term_cause"] == TERM_BLOCKED
    assert following["primitive_steps"] == reach + DETOUR_EPISODE_COST, (
        "one detour off the heading, then the reach spent along it"
    )
    assert following["term_cause"] == TERM_SEQUENCE, (
        "`reach` counts directed moves, so the detour did not eat the plan"
    )
    assert following_stub.keys[0] != east, "the first step leaves the blocked heading"
    assert east in following_stub.keys[1:], "and the heading is resumed after it"


def test_re_aim_takes_the_angularly_nearest_open_cell(
    actions: Tuple[Any, ...],
) -> None:
    """Re-aim takes the angularly nearest open cell."""
    rows, names = option_table(actions)
    northeast = key_index(actions, nethack.CompassDirection.NE)
    southeast = key_index(actions, nethack.CompassDirection.SE)
    east_cell = (START_X + 1, START_Y)
    northeast_cell = (START_X + 1, START_Y - 1)

    open_diagonals = StubNetHack(actions, walk=True, obstacles={east_cell: WALL_GLYPH})
    env = OptionWrapper(open_diagonals, rows, gamma=GAMMA)
    env.reset()
    env.step(names.index("move_E_x1+follow"))

    walled_northeast = StubNetHack(
        actions,
        walk=True,
        obstacles={east_cell: WALL_GLYPH, northeast_cell: WALL_GLYPH},
    )
    env = OptionWrapper(walled_northeast, rows, gamma=GAMMA)
    env.reset()
    env.step(names.index("move_E_x1+follow"))

    assert open_diagonals.keys == [northeast], (
        "NE and SE are both 45 degrees off east; the lower COMPASS index breaks it"
    )
    assert walled_northeast.keys == [southeast], (
        "with NE walled the nearest open cell is SE, which sorts second to last"
    )


def test_re_aim_does_not_steer_into_a_monster(actions: Tuple[Any, ...]) -> None:
    """Re-aim does not steer into a monster."""
    rows, names = option_table(actions)
    southeast = key_index(actions, nethack.CompassDirection.SE)
    stub = StubNetHack(
        actions,
        walk=True,
        obstacles={
            (START_X + 1, START_Y): WALL_GLYPH,
            (START_X + 1, START_Y - 1): MONSTER_GLYPH,
        },
    )
    env = OptionWrapper(stub, rows, gamma=GAMMA)
    env.reset()

    assert env._monsters.dtype == np.bool_, (
        "`~_monsters` is a logical not only while the array is bool; a uint8 array "
        "would make `can_follow` always true"
    )

    env.step(names.index("move_E_x1+follow"))
    assert stub.keys == [southeast], (
        "NE is nearer in angle and holds a monster, so the detour goes to SE"
    )


def test_a_following_row_in_a_sealed_cell_spends_no_keystroke(
    actions: Tuple[Any, ...], rows: List[OptionRow]
) -> None:
    """A following row in a sealed cell spends no keystroke."""
    entombed = StubNetHack(
        actions, walk=True, background_glyph=WALL_GLYPH, neighbour_glyph=WALL_GLYPH
    )
    draw = [row for row in rows if row.name == f"move_E_x{max(MOVE_REPEATS)}+follow"]
    env = OptionWrapper(entombed, draw, gamma=GAMMA)

    observation, opening = env.reset()
    assert not observation["misc"].any(), "a modal stub would drain and charge steps"
    assert opening["initiation_empty"], "nothing is walkable, so `I` falls back"

    _, _, _, _, sealed = env.step(0)
    assert entombed.keys == [], "the row read the map and stepped nowhere"
    assert sealed["drain_steps"] == 0
    assert sealed["primitive_steps"] == 0
    assert sealed["term_cause"] == TERM_NO_HEADING


def test_the_detour_budget_bounds_a_row_blocked_by_a_walkable_looking_cell(
    actions: Tuple[Any, ...],
) -> None:
    """The detour budget ends a row blocked by a walkable-looking cell."""
    rows, names = option_table(actions)
    reach = 4
    east = key_index(actions, nethack.CompassDirection.E)
    monster = StubNetHack(
        actions, walk=True, obstacles={(START_X + 1, START_Y): MONSTER_GLYPH}
    )
    env = OptionWrapper(monster, rows, gamma=GAMMA)
    _, opening = env.reset()
    assert opening["available"][names.index(f"move_E_x{reach}")], (
        "the monster's glyph hides the floor, so `I` cannot refuse the row"
    )

    _, _, _, _, spent = env.step(names.index(f"move_E_x{reach}+follow"))
    assert monster.keys == [east] * reach, "every keystroke went at the heading"
    assert spent["primitive_steps"] == reach, (
        "the detour budget stopped it, not the step limit of "
        f"{reach * (1 + DETOUR_EPISODE_COST)}"
    )
    assert spent["term_cause"] == TERM_DETOUR


def test_a_following_row_is_not_offered_when_every_detour_is_a_monster(
    actions: Tuple[Any, ...],
) -> None:
    """A following row is refused when every detour is a monster."""
    rows, names = option_table(actions)
    reach = max(MOVE_REPEATS)

    walled_heading = StubNetHack(
        actions, walk=True, obstacles=monster_ring(WALL_GLYPH, MONSTER_GLYPH)
    )
    env = OptionWrapper(walled_heading, rows, gamma=GAMMA)
    _, ringed = env.reset()

    assert ringed["available"][names.index(f"move_N_x{reach}")], (
        "a monster cell is walkable, so the directed north row is offered"
    )
    assert not ringed["initiation_empty"], (
        "that row keeps `available` non-empty, so the sealed-cell fallback cannot "
        "fire and mask what this test measures"
    )
    for repeat in MOVE_REPEATS:
        assert not ringed["available"][names.index(f"move_E_x{repeat}+follow")], (
            f"move_E_x{repeat}+follow has nowhere monster-free to steer"
        )
    assert ringed["available"][names.index(f"move_N_x{reach}+follow")], (
        "north is the monster itself, which its own heading may still walk into"
    )

    monster_heading = StubNetHack(
        actions, walk=True, obstacles=monster_ring(MONSTER_GLYPH, WALL_GLYPH)
    )
    env = OptionWrapper(monster_heading, rows, gamma=GAMMA)
    _, attacking = env.reset()

    assert attacking["available"][names.index(f"move_E_x{reach}+follow")], (
        "the heading is walkable, so the row is offered whatever the detours hold"
    )


def test_an_offered_following_row_always_sends_a_keystroke(
    actions: Tuple[Any, ...],
) -> None:
    """An offered following row always sends a keystroke."""
    rows, names = option_table(actions)
    northeast = COMPASS.index(nethack.CompassDirection.NE)
    dx, dy = COMPASS_OFFSETS[northeast]
    maps = {
        "walled heading": monster_ring(WALL_GLYPH, MONSTER_GLYPH),
        "monster heading": monster_ring(MONSTER_GLYPH, WALL_GLYPH),
        "one monster-free detour": {
            **monster_ring(WALL_GLYPH, MONSTER_GLYPH),
            (START_X + dx, START_Y + dy): FLOOR_GLYPH,
        },
    }

    for label, obstacles in maps.items():
        stub = StubNetHack(actions, walk=True, obstacles=obstacles)
        env = OptionWrapper(stub, rows, gamma=GAMMA)
        _, info = env.reset()
        offered = [
            index
            for index, row in enumerate(rows)
            if row.follow and info["available"][index]
        ]

        assert offered, f"no following row is offered on the {label} map"
        for index in offered:
            env.reset()
            _, _, _, _, spent = env.step(index)
            assert spent["primitive_steps"] >= 1, (
                f"{names[index]} was offered on the {label} map and spent nothing"
            )


def test_initiation_reads_the_map_and_the_inventory(actions: Tuple[Any, ...]) -> None:
    """`I` offers a directional row only when the heading cell fits."""
    rows, names = option_table(actions)
    stub = StubNetHack(actions, neighbour_glyph=FLOOR_GLYPH)
    env = OptionWrapper(stub, rows, gamma=GAMMA)

    _, floor = env.reset()
    assert not floor["available"][names.index("open_N")], "no door to open"
    assert not floor["available"][names.index("close_N")], "no door to close"
    assert not floor["available"][names.index("fight_N")], "no monster to fight"
    assert not floor["available"][names.index("eat_a")], "slot a is empty"
    assert floor["available"][names.index("move_N_x16")], "floor north is walkable"
    assert floor["available"][names.index("pickup")], (
        "pickup stays unconditional: the hero's own glyph hides what it would test"
    )

    stub.neighbour_glyph = CLOSED_DOOR_GLYPH
    stub.inv_letters[0] = ord("a")
    _, closed = env.reset()
    assert closed["available"][names.index("open_N")]
    assert not closed["available"][names.index("close_N")]
    assert closed["available"][names.index("eat_a")]
    assert not closed["available"][names.index("eat_b")], "slot b is still empty"

    stub.neighbour_glyph = OPEN_DOOR_GLYPH
    _, opened = env.reset()
    assert opened["available"][names.index("close_N")]
    assert not opened["available"][names.index("open_N")]


def test_a_movement_row_needs_a_walkable_first_cell(actions: Tuple[Any, ...]) -> None:
    """`I` refuses a movement row whose first cell is a wall."""
    rows, names = option_table(actions)
    stub = StubNetHack(actions, neighbour_glyph=WALL_GLYPH)
    env = OptionWrapper(stub, rows, gamma=GAMMA)

    _, walled = env.reset()
    for reach in MOVE_REPEATS:
        assert not walled["available"][names.index(f"move_N_x{reach}")], (
            f"a wall north blocks move_N_x{reach} whatever its reach"
        )
    assert walled["available"][names.index("move_S_x16")], "only north is walled"
    assert walled["available"][names.index("wait")], "SINGLE rows stay unconditional"
    assert walled["available"][names.index("move_N_x16+follow")], (
        "a following row needs somewhere to go rather than the heading cell, and "
        "every other neighbour is open"
    )
    assert 0.0 < walled["available_frac"] < 1.0

    stub.neighbour_glyph = FLOOR_GLYPH
    _, floor = env.reset()
    assert floor["available"][names.index("move_N_x16")]
    assert floor["available_frac"] > walled["available_frac"], (
        "gating movement on the first cell has to move available_frac"
    )


def test_a_diagonal_movement_row_is_refused_into_a_door(
    actions: Tuple[Any, ...],
) -> None:
    """No diagonal step into an open door; a doorway is allowed."""
    rows, names = option_table(actions)
    northeast = COMPASS.index(nethack.CompassDirection.NE)
    stub = StubNetHack(
        actions, neighbour_glyph=OPEN_DOOR_GLYPH, neighbour_heading=northeast
    )
    env = OptionWrapper(stub, rows, gamma=GAMMA)

    _, door = env.reset()
    assert not door["available"][names.index("move_NE_x16")], "no diagonal into a door"
    assert door["available"][names.index("move_N_x16")], "orthogonal is unaffected"

    stub.neighbour_glyph = DOORWAY_GLYPH
    _, doorway = env.reset()
    assert doorway["available"][names.index("move_NE_x16")], (
        "an empty doorway takes a diagonal move"
    )


def test_boulder_otyp_is_the_installed_nle_object() -> None:
    """`BOULDER` matches the installed nle object."""
    assert nethack.OBJ_NAME(nethack.objclass(BOULDER)) == "boulder"
    assert not nethack.glyph_is_cmap(BOULDER_GLYPH), (
        "an object glyph, which is why `_is_walkable` read it as open terrain"
    )
    assert nethack.glyph_is_object(BOULDER_GLYPH)
    assert nethack.glyph_to_obj(BOULDER_GLYPH) == BOULDER


def test_a_boulder_is_refused_as_a_first_cell(actions: Tuple[Any, ...]) -> None:
    """`I` refuses a movement row aimed at a boulder."""
    rows, names = option_table(actions)
    stub = StubNetHack(actions, neighbour_glyph=BOULDER_GLYPH)
    env = OptionWrapper(stub, rows, gamma=GAMMA)

    _, blocked = env.reset()
    for reach in MOVE_REPEATS:
        assert not blocked["available"][names.index(f"move_N_x{reach}")], (
            f"a boulder north blocks move_N_x{reach} whatever its reach"
        )
    assert blocked["available"][names.index("move_S_x16")], "only north is blocked"
    assert blocked["available"][names.index("move_N_x16+follow")], (
        "a following row needs somewhere monster-free to go, and every other "
        "neighbour is open floor"
    )


def test_a_following_row_routes_around_a_boulder(actions: Tuple[Any, ...]) -> None:
    """Following routes around a boulder; the directed twin stops."""
    rows, names = option_table(actions)
    reach = max(MOVE_REPEATS)
    east = key_index(actions, nethack.CompassDirection.E)
    boulder = {(START_X + 1, START_Y): BOULDER_GLYPH}

    directed_stub = StubNetHack(actions, walk=True, obstacles=boulder)
    directed_env = OptionWrapper(directed_stub, rows, gamma=GAMMA)
    directed_env.reset()
    _, _, _, _, directed = directed_env.step(names.index(f"move_E_x{reach}"))

    following_stub = StubNetHack(actions, walk=True, obstacles=boulder)
    following_env = OptionWrapper(following_stub, rows, gamma=GAMMA)
    following_env.reset()
    _, _, _, _, following = following_env.step(names.index(f"move_E_x{reach}+follow"))

    assert directed["primitive_steps"] == 1
    assert directed["term_cause"] == TERM_BLOCKED
    assert following["primitive_steps"] == reach + DETOUR_EPISODE_COST, (
        "one detour off the heading, then the reach spent along it"
    )
    assert following["term_cause"] == TERM_SEQUENCE
    assert following_stub.keys[0] != east, "the first step leaves the boulder's cell"
    assert east in following_stub.keys[1:], "and the heading is resumed after it"


def test_interact_failed_reads_the_target_glyph(actions: Tuple[Any, ...]) -> None:
    """A door row fails when its target glyph is unchanged."""
    rows, names = option_table(actions)
    for opens_door, expected_failure in ((True, False), (False, True)):
        stub = StubNetHack(
            actions,
            neighbour_glyph=CLOSED_DOOR_GLYPH,
            prompt_keys=frozenset({key_index(actions, nethack.Command.OPEN)}),
            prompt_length=1,
            opens_door=opens_door,
        )
        env = OptionWrapper(stub, rows, gamma=GAMMA)
        env.reset()

        _, _, _, _, info = env.step(names.index("open_N"))
        assert info["primitive_steps"] == 2, "the command and its argument"
        assert info["interact_failed"] is expected_failure


def test_a_prompt_that_never_opens_suppresses_the_argument(
    actions: Tuple[Any, ...],
) -> None:
    """Pi emits the argument only once `misc` waits for one."""
    rows, names = option_table(actions)
    stub = StubNetHack(actions, neighbour_glyph=CLOSED_DOOR_GLYPH)
    env = OptionWrapper(stub, rows, gamma=GAMMA)
    env.reset()

    _, _, _, _, info = env.step(names.index("open_N"))
    assert info["primitive_steps"] == 1, "beta fires rather than emitting the argument"
    assert info["term_cause"] == TERM_NO_PROMPT
    assert stub.keys == [key_index(actions, nethack.Command.OPEN)]


def test_every_row_carries_a_step_limit_and_the_movement_ladder(
    rows: List[OptionRow],
) -> None:
    """Every row carries a step limit; following adds the detour allowance."""
    assert all(row.step_limit >= 1 for row in rows), "every row emits a keystroke"
    movement = [row for row in rows if row.group == GROUP_MOVE]
    assert all(row.heading in range(len(COMPASS)) for row in movement)
    assert all(row.step_limit == row.reach for row in movement if not row.follow)
    assert all(
        row.step_limit == row.reach * (1 + DETOUR_EPISODE_COST)
        for row in movement
        if row.follow
    ), "a following row is allowed one detour per unit of reach on top of the reach"


def test_a_mask_with_nothing_in_it_offers_everything(
    actions: Tuple[Any, ...], rows: List[OptionRow]
) -> None:
    """An empty mask falls back to the whole table."""
    walled_in = StubNetHack(
        actions, background_glyph=WALL_GLYPH, neighbour_glyph=WALL_GLYPH
    )
    draw = select_options(rows, 8, "grammar_depth", 0)
    assert all(row.requires == NEEDS_WALKABLE for row in draw), (
        "the draw must hold no unconditional row or the fallback is untested"
    )
    env = OptionWrapper(walled_in, draw, gamma=GAMMA)

    _, info = env.reset()
    assert info["available"].all()
    assert info["available_frac"] == 1.0
    assert info["initiation_empty"], "the fallback has to be visible in the contract"

    walled_in.background_glyph = FLOOR_GLYPH
    walled_in.neighbour_glyph = FLOOR_GLYPH
    _, open_floor = env.reset()
    assert not open_floor["initiation_empty"], "the flag must not latch"


def test_the_option_table_sizes_the_action_space(actions: Tuple[Any, ...]) -> None:
    """`action_space` counts table rows."""
    rows, names = option_table(actions)
    stub = StubNetHack(actions)
    env = OptionWrapper(stub, rows, gamma=GAMMA)
    _, info = env.reset()

    assert env.action_space.n == CATALOGUE_SIZE == len(names)
    assert info["available"].shape == (CATALOGUE_SIZE,)
