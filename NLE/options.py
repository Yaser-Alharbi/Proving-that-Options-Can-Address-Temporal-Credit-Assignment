import hashlib
import random
from typing import Any, Dict, List, Literal, NamedTuple, Optional, Sequence, Tuple

import gymnasium as gym
import numpy as np
from nle import nethack
from nle.env.base import NLE

MOVE_REPEATS = [1, 2, 4, 8, 16]
"""Reach ladder. Module constant: catalogue size is a function of the action set."""

COMPASS = [
    nethack.CompassDirection.N,
    nethack.CompassDirection.S,
    nethack.CompassDirection.E,
    nethack.CompassDirection.W,
    nethack.CompassDirection.NE,
    nethack.CompassDirection.NW,
    nethack.CompassDirection.SE,
    nethack.CompassDirection.SW,
]

SINGLE_COMMANDS = [
    nethack.MiscDirection.DOWN,
    nethack.MiscDirection.WAIT,
    nethack.Command.PICKUP,
    nethack.Command.SEARCH,
    nethack.Command.KICK,
    nethack.Command.PRAY,
    nethack.Command.LOOT,
    nethack.Command.SIT,
    nethack.Command.FORCE,
    nethack.Command.TAKEOFFALL,
    nethack.Command.FIRE,
    nethack.Command.SWAP,
    nethack.Command.CHAT,
    nethack.Command.ENGRAVE,
    nethack.Command.AUTOPICKUP,
]

ARG_COMMANDS = [
    nethack.Command.EAT,
    nethack.Command.WIELD,
    nethack.Command.WEAR,
    nethack.Command.QUAFF,
    nethack.Command.READ,
    nethack.Command.APPLY,
    nethack.Command.DROP,
    nethack.Command.PUTON,
    nethack.Command.REMOVE,
    nethack.Command.TAKEOFF,
]

DIR_COMMANDS = [
    nethack.Command.OPEN,
    nethack.Command.CLOSE,
    nethack.Command.FIGHT,
    nethack.Command.UNTRAP,
]

INVENTORY_SLOTS = "abcdefghij"

CONCEDING_COMMANDS = (nethack.Command.QUIT, nethack.Command.SAVE)
"""Out of every condition's table, not the env action set (catalogue indices stay fixed).

`QUIT` then `y` is a two-step reward-0 exit (`allow_all_yn_questions=True`). `UP`
is out of the option catalogue only; `TASK_ACTIONS` contains it. See
`test_options.KNOWN_TURN_ONE_ESCAPES`.
"""

EMPTY_SLOT = 0
"""`inv_letters` pads empty slots with 0, not a letter."""

UNPROMPTED_COMMANDS = (nethack.Command.FIGHT,)
"""`FIGHT` takes a direction with no prompt, so its argument is unconditional."""

NEEDS_NOTHING = 0
NEEDS_SLOT = 1
NEEDS_CLOSED_DOOR = 2
NEEDS_OPEN_DOOR = 3
NEEDS_MONSTER = 4
NEEDS_TRAP = 5
NEEDS_WALKABLE = 6
"""What `I` tests. `NEEDS_WALKABLE` is the first cell of a movement row."""

EFFECT_UNKNOWN = 0
EFFECT_TAKE = 1
EFFECT_PUT = 2
EFFECT_DOOR = 3
"""What `interact_failed` diffs. `EFFECT_UNKNOWN` reports no failure."""

NO_HEADING = -1
"""`heading` when the row has no direction; otherwise an index into `COMPASS`."""

COMPASS_OFFSETS = [(0, -1), (0, 1), (1, 0), (-1, 0), (1, -1), (-1, -1), (1, 1), (-1, 1)]
"""`(dx, dy)` per `COMPASS` entry. North is `dy == -1`: row 0 of `glyphs` is the top."""

CMAP_OPEN_DOOR = (13, 14)
CMAP_CLOSED_DOOR = (15, 16)
"""Vertical then horizontal. cmap 12 is a doorway, not a door."""

CMAP_STONE = 0
CMAP_WALLS = range(1, 12)
CMAP_NOT_WALKABLE = frozenset({CMAP_STONE, *CMAP_WALLS, *CMAP_CLOSED_DOOR})
"""Stone (also unexplored), walls, closed doors. Diagonal into cmap 13/14 is
refused in `_is_walkable`; cmap 12 is not."""

DIR_REQUIREMENT = {
    nethack.Command.OPEN: NEEDS_CLOSED_DOOR,
    nethack.Command.CLOSE: NEEDS_OPEN_DOOR,
    nethack.Command.FIGHT: NEEDS_MONSTER,
    nethack.Command.UNTRAP: NEEDS_TRAP,
}

DIR_EFFECT = {
    nethack.Command.OPEN: EFFECT_DOOR,
    nethack.Command.CLOSE: EFFECT_DOOR,
}

ARGUMENT_STEP_LIMIT = 2

DRAIN_KEY = nethack.Command.ESC
"""Clears all three `misc` fields. Space reprints a repeating `--More--`."""

DRAIN_LIMIT = 128
"""Hang guard. The drain runs until `misc` is clear, no allowance."""

STATUS_ABORTED = int(NLE.StepStatus.ABORTED)
"""`end_status` when NLE abort (step limit / no-progress), not the game."""

TERM_NONE = -1
TERM_SEQUENCE = 0
TERM_ENV_ABORT = 1
TERM_EPISODE_END = 2
TERM_INTERRUPT = 3
TERM_CAUSE_NAMES = ("sequence", "env_abort", "episode_end", "interrupt")
"""`interrupt` is beta firing on the state. `TERM_NONE` is the NEXT_STEP phantom."""

MISC_IN_YN = 0
"""`misc[0]`. Prompted catalogue commands raise this."""

STATUS_KEY = 0
STATUS_ARGUMENT = 1
STATUS_DONE = 2

GROUP_PRIMITIVE = -1
GROUP_MOVE = 0
GROUP_SINGLE = 1
GROUP_ARG = 2
GROUP_DIR = 3
"""Catalogue group. Emission order; both grammar priors read it."""


class OptionRow(NamedTuple):
    """`(I, pi, beta)` controller. `key` is first: tests pass `row[0]` to `env.step`."""

    key: int
    name: str
    slot: Optional[int]
    """Inventory letter ASCII, or None."""
    reach: int
    """Movement reach; 0 for a command row."""
    group: int
    argument: Optional[int]
    """Slot letter or direction index, or None."""
    prompts: bool
    step_limit: int
    """Own keystrokes, not `primitive_steps`; the drain does not spend this."""
    requires: int
    effect: int
    heading: int
    """Index into `COMPASS`, or `NO_HEADING`."""

    @property
    def keystrokes(self) -> Tuple[int, ...]:
        """For reporting. The executor derives its own keys."""
        return (self.key,) if self.argument is None else (self.key, self.argument)


def _catalogue(env_actions: Sequence[Any]) -> List[OptionRow]:
    """Every option this action set admits, frozen order. Changing it redraws every family."""
    index = {a: i for i, a in enumerate(env_actions)}
    rows: List[OptionRow] = []

    for heading, direction in enumerate(COMPASS):
        if direction not in index:
            continue
        for repeat in MOVE_REPEATS:
            rows.append(
                OptionRow(
                    key=index[direction],
                    name=f"move_{direction.name}_x{repeat}",
                    slot=None,
                    reach=repeat,
                    group=GROUP_MOVE,
                    argument=None,
                    prompts=False,
                    step_limit=repeat,
                    requires=NEEDS_WALKABLE,
                    effect=EFFECT_UNKNOWN,
                    heading=heading,
                )
            )

    for command in SINGLE_COMMANDS:
        if command in index:
            rows.append(
                OptionRow(
                    key=index[command],
                    name=command.name.lower(),
                    slot=None,
                    reach=0,
                    group=GROUP_SINGLE,
                    argument=None,
                    prompts=command not in UNPROMPTED_COMMANDS,
                    step_limit=1,
                    requires=NEEDS_NOTHING,
                    effect=(
                        EFFECT_TAKE
                        if command == nethack.Command.PICKUP
                        else EFFECT_UNKNOWN
                    ),
                    heading=NO_HEADING,
                )
            )

    for command in ARG_COMMANDS:
        if command not in index:
            continue
        for slot in INVENTORY_SLOTS:
            if ord(slot) not in index:
                continue
            rows.append(
                OptionRow(
                    key=index[command],
                    name=f"{command.name.lower()}_{slot}",
                    slot=ord(slot),
                    reach=0,
                    group=GROUP_ARG,
                    argument=index[ord(slot)],
                    prompts=command not in UNPROMPTED_COMMANDS,
                    step_limit=ARGUMENT_STEP_LIMIT,
                    requires=NEEDS_SLOT,
                    effect=(
                        EFFECT_PUT if command == nethack.Command.DROP else EFFECT_UNKNOWN
                    ),
                    heading=NO_HEADING,
                )
            )

    for command in DIR_COMMANDS:
        if command not in index:
            continue
        for heading, direction in enumerate(COMPASS):
            if direction not in index:
                continue
            rows.append(
                OptionRow(
                    key=index[command],
                    name=f"{command.name.lower()}_{direction.name}",
                    slot=None,
                    reach=0,
                    group=GROUP_DIR,
                    argument=index[direction],
                    prompts=command not in UNPROMPTED_COMMANDS,
                    step_limit=ARGUMENT_STEP_LIMIT,
                    requires=DIR_REQUIREMENT[command],
                    effect=DIR_EFFECT.get(command, EFFECT_UNKNOWN),
                    heading=heading,
                )
            )

    return rows


def _class_rank(rows: Sequence[OptionRow]) -> Dict[str, int]:
    """Position inside the row's group. Longest reach first, then catalogue order."""
    keyed: Dict[int, List[Tuple[int, int, str]]] = {}
    for position, row in enumerate(rows):
        keyed.setdefault(row.group, []).append((-row.reach, position, row.name))
    rank: Dict[str, int] = {}
    for group in keyed.values():
        for place, (_, _, name) in enumerate(sorted(group)):
            rank[name] = place
    return rank


def grammar_options(n: int, rows: Sequence[OptionRow]) -> List[OptionRow]:
    """First n breadth-first: contingent last, then one per group, longest-reach first."""
    rank = _class_rank(rows)
    ordered = sorted(
        rows, key=lambda row: (row.slot is not None, rank[row.name], row.group)
    )
    return list(ordered[:n])


def grammar_depth_options(n: int, rows: Sequence[OptionRow]) -> List[OptionRow]:
    """First n longest-reach-first. No command row before n=41."""
    # `sorted` is stable, so catalogue order is the last tiebreak without an index in the key
    ordered = sorted(rows, key=lambda row: (row.slot is not None, -row.reach, row.group))
    return list(ordered[:n])


def random_options(
    n: int, rows: Sequence[OptionRow], option_seed: int
) -> List[OptionRow]:
    # by name: `random.sample` is positional, so catalogue order or action-set indices would redraw
    ordered = sorted(rows, key=lambda row: row.name)
    return random.Random(option_seed).sample(ordered, n)


def catalogue_digest(rows: Sequence[OptionRow]) -> str:
    """sha256 of the row names in order."""
    return hashlib.sha256("\n".join(row.name for row in rows).encode()).hexdigest()


def select_options(
    rows: Sequence[OptionRow],
    n_options: int,
    option_family: Literal["grammar", "grammar_depth", "random"],
    option_seed: int,
) -> List[OptionRow]:
    if n_options > len(rows):
        raise ValueError(
            f"asked for {n_options} options but this action set yields only "
            f"{len(rows)} catalogue rows"
        )
    if option_family == "grammar":
        return grammar_options(n_options, rows)
    if option_family == "grammar_depth":
        return grammar_depth_options(n_options, rows)
    if option_family == "random":
        return random_options(n_options, rows, option_seed)
    raise ValueError(f"unknown option family {option_family}")


def _table(
    rows: Sequence[OptionRow],
) -> Tuple[List[OptionRow], List[str], List[Optional[int]]]:
    """Rows, names, slots. Tests unpack three values."""
    return (
        list(rows),
        [row.name for row in rows],
        [row.slot for row in rows],
    )


def make_options(
    env_actions: Sequence[Any],
    condition: Literal["action", "option", "both"],
    n_options: int,
    option_family: Literal["grammar", "grammar_depth", "random"],
    option_seed: int,
) -> Tuple[List[OptionRow], List[str], List[Optional[int]]]:
    """Action table for `condition`. A primitive is `step_limit == 1`. `both` is primitives then options."""
    primitives = [
        OptionRow(
            key=i,
            name=getattr(action, "name", str(action)).lower(),
            slot=None,
            reach=1,
            group=GROUP_PRIMITIVE,
            argument=None,
            prompts=False,
            step_limit=1,
            requires=NEEDS_NOTHING,
            effect=EFFECT_UNKNOWN,
            heading=NO_HEADING,
        )
        for i, action in enumerate(env_actions)
        if action not in CONCEDING_COMMANDS
    ]
    if condition == "action":
        return _table(primitives)

    rows = _catalogue(env_actions)
    chosen = select_options(rows, n_options, option_family, option_seed)
    if condition == "both":
        return _table(primitives + chosen)
    return _table(chosen)


def _is_closed_door(glyph: int) -> bool:
    """Closed or locked. Glyph cannot tell them apart."""
    return nethack.glyph_is_cmap(glyph) and nethack.glyph_to_cmap(glyph) in CMAP_CLOSED_DOOR


def _is_open_door(glyph: int) -> bool:
    return nethack.glyph_is_cmap(glyph) and nethack.glyph_to_cmap(glyph) in CMAP_OPEN_DOOR


def _is_walkable(glyph: int, diagonal: bool) -> bool:
    """Diagonal into cmap 13/14 is refused; cmap 12 is not."""
    if not nethack.glyph_is_cmap(glyph):
        return True
    cmap = nethack.glyph_to_cmap(glyph)
    if cmap in CMAP_NOT_WALKABLE:
        return False
    return not (diagonal and cmap in CMAP_OPEN_DOOR)


MAP_PREDICATE = {
    NEEDS_CLOSED_DOOR: _is_closed_door,
    NEEDS_OPEN_DOOR: _is_open_door,
    NEEDS_MONSTER: nethack.glyph_is_monster,  # pets included
    NEEDS_TRAP: nethack.glyph_is_trap,
}


def _neighbour_glyphs(glyphs: np.ndarray, x: int, y: int) -> np.ndarray:
    """Eight neighbours in `COMPASS` order, `NO_GLYPH` off-map."""
    height, width = glyphs.shape
    return np.array(
        [
            glyphs[y + dy, x + dx]
            if 0 <= y + dy < height and 0 <= x + dx < width
            else nethack.NO_GLYPH
            for dx, dy in COMPASS_OFFSETS
        ]
    )


def _status(
    row: OptionRow, prompted: bool, keys_sent: int, moves: int, blocked: bool
) -> int:
    """pi and beta. `STATUS_DONE` is beta firing."""
    if keys_sent >= row.step_limit:
        return STATUS_DONE
    if keys_sent == 1 and row.argument is not None and (prompted or not row.prompts):
        return STATUS_ARGUMENT
    if row.reach:
        return STATUS_DONE if blocked or moves >= row.reach else STATUS_KEY
    if keys_sent == 0:
        return STATUS_KEY
    # no prompt: the argument would type at the map
    return STATUS_DONE


class OptionWrapper(gym.Wrapper):
    """Closed-loop `(I, pi, beta)` executor.

    Drain a modal state only once it has survived a whole decision. ESC clears
    all three `misc` fields. `NetHackChallenge` passes `allow_all_modes=True`,
    so NLE drains nothing.
    """

    def __init__(self, env: gym.Env, rows: Sequence[OptionRow], gamma: float) -> None:
        super().__init__(env)
        self.rows = list(rows)
        self.gamma = gamma
        self.action_space = gym.spaces.Discrete(len(self.rows))
        self.required_slots = np.array(
            [EMPTY_SLOT if row.slot is None else row.slot for row in self.rows],
            dtype=np.uint8,
        )
        self.requires = np.array([row.requires for row in self.rows])
        self.headings = np.array([row.heading for row in self.rows])
        self._drain_key = list(env.unwrapped.actions).index(DRAIN_KEY)
        self._neighbours = np.full(len(COMPASS), nethack.NO_GLYPH)
        self._occupied = 0
        """Neighbours and slot count `interact_failed` diffs against."""
        self._initiation_empty = False
        self._position = (0, 0)
        """Hero cell after the last primitive. First-keystroke baseline for beta."""
        self._episode_decisions = 0
        self._prompt_pending = False
        """Modal flag still set when the previous decision ended."""
        self._ingame_blstats = np.zeros(0, dtype=np.int64)
        self._ingame_message = np.zeros(0, dtype=np.uint8)
        """Last frame with a turn; NetHack zeroes blstats/message at game over."""
        self._begin_decision()

    def _begin_decision(self) -> None:
        self._steps = 0
        self._undiscounted = 0.0
        self._positive = 0.0
        self._negative = 0.0
        self._nonzero = 0
        self._first_offset = -1

    def _primitive(self, action: int) -> Tuple[Any, float, bool, bool, Dict[str, Any]]:
        """One env step. Drain keystrokes go through here too."""
        obs, reward, terminated, truncated, info = self.env.step(action)
        if reward > 0.0:
            self._positive += reward
        elif reward < 0.0:
            self._negative += reward
        if reward != 0.0:
            self._nonzero += 1
            if self._first_offset < 0:
                self._first_offset = self._steps
        self._steps += 1
        self._undiscounted += reward
        self._position = (
            int(obs["blstats"][nethack.NLE_BL_X]),
            int(obs["blstats"][nethack.NLE_BL_Y]),
        )
        if obs["blstats"][nethack.NLE_BL_TIME] > 0:
            # copies, not views: NLE reuses the same two buffers
            self._ingame_blstats = obs["blstats"].copy()
            self._ingame_message = obs["message"].copy()
        return obs, reward, terminated, truncated, info

    def _drain(
        self, obs: Dict[str, np.ndarray]
    ) -> Tuple[Any, float, bool, bool, Dict[str, Any], int]:
        """ESC until `misc` is clear. Called only on a modal `obs`."""
        reward_sum = 0.0
        discount = 1.0
        drained = 0
        while True:
            assert drained < DRAIN_LIMIT, (
                f"{drained} keystrokes have not cleared misc={tuple(obs['misc'])}; "
                f"{DRAIN_KEY!r} no longer answers every modal state"
            )
            obs, reward, terminated, truncated, info = self._primitive(self._drain_key)
            reward_sum += discount * reward
            discount *= self.gamma
            drained += 1
            if terminated or truncated or not obs["misc"].any():
                return obs, reward_sum, terminated, truncated, info, drained

    def _initiation(self, obs: Dict[str, np.ndarray]) -> np.ndarray:
        """`I` as a bool mask.

        Movement: first cell walkable (not stone/walls/closed door; diagonal
        refuses cmap 13/14, not doorway 12). Empty mask offers everything
        (`grammar_depth` n<=40 is movement only).
        """
        self._neighbours = _neighbour_glyphs(
            obs["glyphs"],
            int(obs["blstats"][nethack.NLE_BL_X]),
            int(obs["blstats"][nethack.NLE_BL_Y]),
        )
        self._occupied = int((obs["inv_letters"] != EMPTY_SLOT).sum())

        available = self.requires == NEEDS_NOTHING
        available |= (self.requires == NEEDS_SLOT) & np.isin(
            self.required_slots, obs["inv_letters"]
        )
        walkable = np.array(
            [
                _is_walkable(int(glyph), dx != 0 and dy != 0)
                for glyph, (dx, dy) in zip(self._neighbours, COMPASS_OFFSETS)
            ]
        )
        available |= (self.requires == NEEDS_WALKABLE) & walkable[self.headings]
        for code, predicate in MAP_PREDICATE.items():
            rows = self.requires == code
            if not rows.any():
                continue
            met = np.array([predicate(int(glyph)) for glyph in self._neighbours])
            # `NO_HEADING` indexes from the end; `rows` excludes those rows
            available |= rows & met[self.headings]
        self._initiation_empty = not available.any()
        if self._initiation_empty:
            available[:] = True
        return available

    def _interact_failed(self, row: OptionRow, obs: Dict[str, np.ndarray]) -> bool:
        """Whether the named effect did not happen. Runs before `_contract`."""
        if row.effect == EFFECT_DOOR:
            after = _neighbour_glyphs(obs["glyphs"], *self._position)
            return bool(after[row.heading] == self._neighbours[row.heading])
        if row.effect in (EFFECT_TAKE, EFFECT_PUT):
            return int((obs["inv_letters"] != EMPTY_SLOT).sum()) == self._occupied
        return False

    def _contract(
        self,
        info: Dict[str, Any],
        obs: Dict[str, np.ndarray],
        drain_steps: int,
        term_cause: int,
        interact_failed: bool,
    ) -> Dict[str, Any]:
        available = self._initiation(obs)
        info["primitive_steps"] = self._steps
        info["drain_steps"] = drain_steps
        info["option_discount"] = float(self.gamma**self._steps)
        info["available"] = available
        info["available_frac"] = float(available.mean())
        info["initiation_empty"] = self._initiation_empty
        info["term_cause"] = term_cause
        info["interact_failed"] = interact_failed
        info["undiscounted_reward"] = self._undiscounted
        info["sum_positive_clipped"] = self._positive
        info["sum_negative_clipped"] = self._negative
        info["nonzero_reward_steps"] = self._nonzero
        info["first_reward_offset"] = self._first_offset
        info["ingame_blstats"] = self._ingame_blstats
        info["ingame_message"] = self._ingame_message
        return info

    def reset(self, **kwargs: Any) -> Tuple[Any, Dict[str, Any]]:
        self._episode_decisions = 0
        self._prompt_pending = False
        obs, info = self.env.reset(**kwargs)
        self._begin_decision()
        self._position = (
            int(obs["blstats"][nethack.NLE_BL_X]),
            int(obs["blstats"][nethack.NLE_BL_Y]),
        )
        self._ingame_blstats = obs["blstats"].copy()
        self._ingame_message = obs["message"].copy()
        info = self._contract(info, obs, 0, TERM_NONE, False)
        info["decision_t"] = 0
        return obs, info

    def step(self, option_id: int) -> Tuple[Any, float, bool, bool, Dict[str, Any]]:
        row = self.rows[option_id]
        self._begin_decision()
        total_reward = 0.0
        discount = 1.0
        keys_sent = 0
        moves = 0
        blocked = False
        prompted = False
        # loop locals, not fields
        position = self._position
        while True:
            status = _status(row, prompted, keys_sent, moves, blocked)
            if status == STATUS_DONE:
                break
            key = row.key if status == STATUS_KEY else row.argument
            obs, reward, terminated, truncated, info = self._primitive(key)
            total_reward += discount * reward
            discount *= self.gamma
            keys_sent += 1
            if terminated or truncated:
                break
            prompted = bool(obs["misc"][MISC_IN_YN])
            if status == STATUS_KEY and row.reach:
                if self._position == position:
                    blocked = True
                else:
                    moves += 1
            position = self._position
        drained = 0
        modal = not (terminated or truncated) and bool(obs["misc"].any())
        if modal and self._prompt_pending:
            obs, reward, terminated, truncated, info, drained = self._drain(obs)
            total_reward += discount * reward
            modal = not (terminated or truncated) and bool(obs["misc"].any())
        # after the drain so next `available` matches the state the next decision begins in
        self._prompt_pending = modal
        self._episode_decisions += 1
        if keys_sent >= row.step_limit:
            term_cause = TERM_SEQUENCE
        elif not (terminated or truncated):
            term_cause = TERM_INTERRUPT
        elif int(info["end_status"]) == STATUS_ABORTED:
            term_cause = TERM_ENV_ABORT
        else:
            term_cause = TERM_EPISODE_END
        # before `_contract` overwrites the before-state
        interact_failed = self._interact_failed(row, obs)
        info = self._contract(info, obs, drained, term_cause, interact_failed)
        info["decision_t"] = self._episode_decisions
        if terminated or truncated:
            self._episode_decisions = 0
        return obs, total_reward, terminated, truncated, info
