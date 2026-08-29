"""Tests for `options.py`: the catalogue it enumerates and the controller it runs.

Catalogue order is load-bearing: `random.sample` is positional and both grammar
families take a prefix. The digests fail if that order moves. Executor tests use
`StubNetHack`: `NetHackChallenge.seed` raises, so the pinned env cannot be seeded.
"""

from typing import Any, Dict, FrozenSet, List, Optional, Sequence, Tuple

import gymnasium as gym
import numpy as np
import pytest
from nle import nethack

from envs import OBSERVATION_KEYS
from options import (
    COMPASS,
    COMPASS_OFFSETS,
    CONCEDING_COMMANDS,
    DETOUR_EPISODE_COST,
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
"""The env exp1/2/4 run on. A task env exposes a smaller action set."""

SHORT_EPISODE = 10
"""Legal horizon. No test here steps the env."""

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
"""cmap 19 lit room, 1 wall, 12 doorway, 15 closed door, 13 open. Named here: `options` does not export every cell."""

MONSTER_GLYPH = nethack.GLYPH_MON_OFF + 1
"""Not a cmap, so `_is_walkable` reads its cell as open however the hero fares against it."""

DRAW_SIZE = 64
"""The n exp1, exp3 and exp4 use."""

ENV_ACTION_SET = 121
ACTION_TABLE = ENV_ACTION_SET - len(CONCEDING_COMMANDS)
"""The env keeps its action set; the table drops the conceding commands from every condition."""

KNOWN_TURN_ONE_ESCAPES = ("up",)
"""Free termination in the primitive table, not the option catalogue.

`up` then northwest leaves level 1 for reward 0. `TASK_ACTIONS` contains `UP`;
the confirm is northwest (`y`). Reachable under `action`/`both` only.
"""

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
"""Spelled out rather than digested: the breadth-first prefix at n=8 is not
movement-only, and every movement row in it is the following twin."""

FIRST_INTERACTION_DEPTH = 80
"""`grammar_depth` rows before the first command: 40 following, then 40 directed.
n at or below this is movement only."""


@pytest.fixture(scope="module")
def actions() -> Tuple[Any, ...]:
    """`ENV_ID`'s action set, which is the one every digest is pinned against."""
    env = gym.make(
        ENV_ID, observation_keys=OBSERVATION_KEYS, max_episode_steps=SHORT_EPISODE
    )
    action_set = tuple(env.unwrapped.actions)
    env.close()
    return action_set


@pytest.fixture(scope="module")
def rows(actions: Tuple[Any, ...]) -> List[OptionRow]:
    """The full catalogue for `ENV_ID`'s action set."""
    return _catalogue(actions)


def digest_by_name(chosen: Sequence[OptionRow]) -> str:
    """The draw's digest, order-independent: names sorted, then hashed."""
    return catalogue_digest(sorted(chosen, key=lambda row: row.name))


class StubNetHack(gym.Env):
    """Enough of NLE to drive the executor: a position, `misc`, and one map cell.

    `prompt_keys` raise a modal prompt held for `prompt_length` further keystrokes.
    `obstacles` maps an absolute cell to the glyph drawn there, and the hero may
    not enter one whatever that glyph reads as: that is how a monster stays
    walkable to `I` and still stops the step, as it does in NetHack.
    """

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
        """The observation keys the wrapper reads."""
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
        """Fresh position and no prompt. `neighbour_glyph` and `inv_letters` are left alone."""
        self._modal = 0
        self._x = START_X
        self._y = START_Y
        self._time = 1
        self.keys = []
        return self._observation(), {"end_status": 0}

    def step(
        self, action: int
    ) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict[str, Any]]:
        """One keystroke. Never terminates: episode ends are not under test here."""
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
    """The canonical order is exactly what the digest records. Re-recording it redraws every family."""
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
    """`reach` is the repeat count on a movement row and zero elsewhere. Both sort keys read it."""
    reaches = {row.reach for row in rows if row.group == GROUP_MOVE}
    assert reaches == set(MOVE_REPEATS)
    assert all(row.reach == 0 for row in rows if row.group != GROUP_MOVE)


def test_random_draw_is_pinned(rows: List[OptionRow]) -> None:
    """Each option seed draws the set it has always drawn. `random.sample` is positional."""
    for option_seed, expected in RANDOM_DIGESTS.items():
        chosen = random_options(DRAW_SIZE, rows, option_seed)
        assert len(chosen) == DRAW_SIZE
        assert digest_by_name(chosen) == expected, (
            f"option_seed {option_seed} now draws:\n"
            + "\n".join(sorted(row.name for row in chosen))
        )


def test_grammar_draws_are_pinned(rows: List[OptionRow]) -> None:
    """Both grammar catalogues at n=64 are the ones exp1, exp2 and exp4 ran."""
    assert digest_by_name(grammar_options(DRAW_SIZE, rows)) == GRAMMAR_DIGEST
    assert (
        digest_by_name(grammar_depth_options(DRAW_SIZE, rows)) == GRAMMAR_DEPTH_DIGEST
    )


def test_the_breadth_prior_reaches_three_groups_at_eight_rows(
    rows: List[OptionRow],
) -> None:
    """`grammar` at n=8 can descend and open a door, not only walk."""
    assert [row.name for row in grammar_options(8, rows)] == GRAMMAR_PREFIX_AT_EIGHT


def test_the_depth_prior_holds_no_interaction_below_its_movement_block(
    rows: List[OptionRow],
) -> None:
    """`grammar_depth` places every movement row before every command row."""
    ordered = grammar_depth_options(len(rows), rows)
    first_command = next(
        index for index, row in enumerate(ordered) if row.group != GROUP_MOVE
    )
    assert first_command == FIRST_INTERACTION_DEPTH
    assert all(row.group == GROUP_MOVE for row in ordered[:first_command])


def test_every_family_returns_the_whole_catalogue_at_full_size(
    rows: List[OptionRow],
) -> None:
    """At n equal to the catalogue size the families coincide."""
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
    """A reachable mistake: exp2's n axis is written by hand."""
    env = gym.make(
        ENV_ID, observation_keys=OBSERVATION_KEYS, max_episode_steps=SHORT_EPISODE
    )
    with pytest.raises(ValueError, match="catalogue rows"):
        make_options(env.unwrapped.actions, "option", len(rows) + 1, "grammar", 0)
    env.close()


def test_the_conceding_commands_are_absent_from_every_condition() -> None:
    """No condition can select `QUIT` or `SAVE`. Two-step reward-0 exit."""
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
    """Only the recorded escapes concede. Subset, not equality: a two-step death by misfortune would fail equality."""
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
    """`up` then northwest still concedes, so the recorded exception is live."""
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
    """An action index means the same thing under `action` and under `both`. Prefix is table rows, not env indices."""
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
    """The `action` table and its row names."""
    rows, names, _ = make_options(actions, "action", 0, "grammar", 0)
    return rows, names


def option_table(actions: Tuple[Any, ...]) -> Tuple[List[OptionRow], List[str]]:
    """The whole catalogue as an option table, so every row name is present."""
    rows, names, _ = make_options(actions, "option", CATALOGUE_SIZE, "grammar", 0)
    return rows, names


def key_index(actions: Tuple[Any, ...], command: Any) -> int:
    """The env action index of `command`."""
    return list(actions).index(command)


def test_a_modal_state_surviving_a_decision_is_drained_and_charged(
    actions: Tuple[Any, ...],
) -> None:
    """The drain's keystrokes are counted in `primitive_steps`. Four prompt keys, so three drain steps."""
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
    """A prompt survives exactly one decision, and only then is it drained. Pins "survived", not "is set"."""
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


def test_a_movement_row_stops_when_the_position_stops_changing(
    actions: Tuple[Any, ...],
) -> None:
    """Beta reads the position, so a blocked 16-move costs one primitive step."""
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
    """The other side of the same beta: nothing stops an unobstructed row early."""
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
    """The lever re-aim buys. Both rows meet the same wall on the same map: the
    directed one pays a step to learn it is blocked, the following one steps past
    and spends its whole reach on the heading."""
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
    """Not a fixed ordering. `COMPASS` runs N, S, E, W, NE, NW, SE, SW, so any
    positional fallback would reach N or S first, and both are 90 degrees off a
    blocked easterly heading."""
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


def test_a_following_row_in_a_sealed_cell_spends_no_keystroke(
    actions: Tuple[Any, ...], rows: List[OptionRow]
) -> None:
    """Candidates are inspected, so with nothing open beta fires before `pi` pays
    anything. Asserted on the row's own keystrokes first: the drain also charges
    `primitive_steps`, so that reading is only sound once it is pinned at zero."""
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
    """A monster is not a cmap, so its glyph hides the terrain and both `I` and
    re-aim read the cell as open. Re-aim never fires and the row keeps walking into
    it; `detoured` is what ends that, short of the step limit."""
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


def test_initiation_reads_the_map_and_the_inventory(actions: Tuple[Any, ...]) -> None:
    """`I` offers a directional row only when the cell along its heading fits."""
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
    """`I` refuses a movement row that would spend its first step on a wall. Beta cannot refuse that without paying the step."""
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
    """NetHack takes no diagonal step into an open door (cmap 13/14). A doorway (cmap 12) is not refused."""
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


def test_interact_failed_reads_the_target_glyph(actions: Tuple[Any, ...]) -> None:
    """A door row fails when its cell still looks the way it did. Glyph, not message: both lock and open print something."""
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
    """Pi emits the argument only once `misc` says the game is waiting for one."""
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
    """`step_limit` is beta's bound: the reach ladder directed, and the detour allowance on top of it following."""
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
    """`I` falls back to the whole table rather than handing back all-False. `grammar_depth` n=8 is movement only."""
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
    """`action_space` counts table rows, which is what the policy's head sizes to."""
    rows, names = option_table(actions)
    stub = StubNetHack(actions)
    env = OptionWrapper(stub, rows, gamma=GAMMA)
    _, info = env.reset()

    assert env.action_space.n == CATALOGUE_SIZE == len(names)
    assert info["available"].shape == (CATALOGUE_SIZE,)
