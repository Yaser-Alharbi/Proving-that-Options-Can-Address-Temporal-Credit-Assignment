import hashlib
import random
from typing import Any, Dict, List, Literal, NamedTuple, Optional, Sequence, Tuple

import gymnasium as gym
import numpy as np
from nle import nethack

MOVE_REPEATS = [1, 2, 4, 8, 16]
"""The reach ladder. A module constant, not a per-cell argument: the catalogue's
size is a function of the action set, so there is no `max_forward` analogue to
sweep, and exp3's `duration_vs_cap` is skipped for want of a second cap."""

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
    nethack.MiscDirection.UP,
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
"""Kept out of the action table in every condition.

`QUIT` followed by `y` ends the episode in two primitive steps with reward 0,
and `NetHackChallenge` sets `allow_all_yn_questions=True`, so the confirmation
is answerable. Since the reward is a score delta plus a per-step time penalty,
conceding beats any trajectory whose future score does not cover the penalty,
and it is reachable from the first move.

Removing them is what NLE itself does whenever the env has a goal: the goal
tasks run on `TASK_ACTIONS`, 23 keys, which contains neither. `NetHackChallenge`
carries them because a competition entry has to be able to concede.

They are dropped from the table rather than from the env's action set, so the
keystroke indices a catalogue row holds stay the same on every env.

This does not leave the table free of concessions. `MiscDirection.UP` on the
first move escapes the dungeon for reward 0, and it is kept: it is in
`TASK_ACTIONS`, and the key that confirms its prompt is northwest movement, so
neither the command nor the confirmation can be removed. See
`test_options.KNOWN_TURN_ONE_ESCAPES`.
"""

EMPTY_SLOT = 0
"""`inv_letters` pads unoccupied slots with 0 rather than a letter."""

GROUP_PRIMITIVE = -1
GROUP_MOVE = 0
GROUP_SINGLE = 1
GROUP_ARG = 2
GROUP_DIR = 3
"""Which class of the enumeration a row came from. The order of the four
catalogue groups is the order `_catalogue` emits them in, and both grammar
priors read it."""


class OptionRow(NamedTuple):
    """One row of the action table: the keystrokes, its label, its sort fields."""

    sequence: Tuple[int, ...]
    name: str
    slot: Optional[int]
    """ASCII code of the inventory letter this row needs, or None if it needs none."""
    reach: int
    """Primitive moves a movement row makes; 0 for a command row."""
    group: int


def _catalogue(env_actions: Sequence[Any]) -> List[OptionRow]:
    """Every option this action set admits, in the frozen canonical order.

    That order is the contract both grammar prefixes and the random draw are
    taken from, so changing it redraws every family and invalidates every run
    that used one. Rows whose keys the action set lacks are skipped, so the
    length is a function of the action set rather than the 188 of the full
    keyboard.

    Every non-movement row ends with ESC so any prompt the command opened is
    cancelled before the next option runs. Without this an unanswered prompt
    swallows the following option's first keystroke.
    """
    index = {a: i for i, a in enumerate(env_actions)}
    esc = index.get(nethack.Command.ESC)
    tail = (esc,) if esc is not None else ()
    rows: List[OptionRow] = []

    for direction in COMPASS:
        if direction not in index:
            continue
        for repeat in MOVE_REPEATS:
            rows.append(
                OptionRow(
                    (index[direction],) * repeat,
                    f"move_{direction.name}_x{repeat}",
                    None,
                    repeat,
                    GROUP_MOVE,
                )
            )

    for command in SINGLE_COMMANDS:
        if command in index:
            rows.append(
                OptionRow(
                    (index[command],) + tail,
                    command.name.lower(),
                    None,
                    0,
                    GROUP_SINGLE,
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
                    (index[command], index[ord(slot)]) + tail,
                    f"{command.name.lower()}_{slot}",
                    ord(slot),
                    0,
                    GROUP_ARG,
                )
            )

    for command in DIR_COMMANDS:
        if command not in index:
            continue
        for direction in COMPASS:
            if direction not in index:
                continue
            rows.append(
                OptionRow(
                    (index[command], index[direction]) + tail,
                    f"{command.name.lower()}_{direction.name}",
                    None,
                    0,
                    GROUP_DIR,
                )
            )

    return rows


def _class_rank(rows: Sequence[OptionRow]) -> Dict[str, int]:
    """Each row's position inside its own group, keyed by name.

    One rule for all four groups: longest reach first, then canonical order.
    Where a group's rows all have reach 0 that reduces to canonical order, and on
    the movement rows it is reach-major, which is what both priors want.
    """
    keyed: Dict[int, List[Tuple[int, int, str]]] = {}
    for position, row in enumerate(rows):
        keyed.setdefault(row.group, []).append((-row.reach, position, row.name))
    rank: Dict[str, int] = {}
    for group in keyed.values():
        for place, (_, _, name) in enumerate(sorted(group)):
            rank[name] = place
    return rank


def grammar_options(n: int, rows: Sequence[OptionRow]) -> List[OptionRow]:
    """The first n rows breadth-first: contingent rows last, then one row from
    each group in turn, each group longest-reach-first.

    This is what `grammar` means in every experiment that names it. At n=8 it
    reaches three of the four groups, so the prefix can descend a staircase and
    open a door rather than only walk.
    """
    rank = _class_rank(rows)
    ordered = sorted(
        rows, key=lambda row: (row.slot is not None, rank[row.name], row.group)
    )
    return list(ordered[:n])


def grammar_depth_options(n: int, rows: Sequence[OptionRow]) -> List[OptionRow]:
    """The first n rows longest-reach-first, transliterating Navix's
    `(follow, -reach, rank, heading)`.

    Its prefix is every movement row before any command row, so it holds no
    interaction below n=41. exp3 is the only sweep that runs it.
    """
    # `sorted` is stable, so canonical order is the last tiebreak without the
    # enumeration index appearing in the key
    ordered = sorted(rows, key=lambda row: (row.slot is not None, -row.reach, row.group))
    return list(ordered[:n])


def random_options(
    n: int, rows: Sequence[OptionRow], option_seed: int
) -> List[OptionRow]:
    """Uniform draws from the catalogue, seeded by `option_seed`."""
    # by name, not by the canonical order and not by the index tuples:
    # `random.sample` reads its input positionally, so the draw would move
    # whenever the enumeration order changed, and an index tuple moves with the
    # action set. A name moves under neither.
    ordered = sorted(rows, key=lambda row: row.name)
    return random.Random(option_seed).sample(ordered, n)


def catalogue_digest(rows: Sequence[OptionRow]) -> str:
    """sha256 of the row names in order, for pinning the enumeration in a test."""
    return hashlib.sha256("\n".join(row.name for row in rows).encode()).hexdigest()


def select_options(
    rows: Sequence[OptionRow],
    n_options: int,
    option_family: Literal["grammar", "grammar_depth", "random"],
    option_seed: int,
) -> List[OptionRow]:
    """`n_options` rows of `rows` under the named prior."""
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
) -> Tuple[List[Tuple[int, ...]], List[str], List[Optional[int]]]:
    """The three parallel lists the wrapper and the trainer read."""
    return (
        [row.sequence for row in rows],
        [row.name for row in rows],
        [row.slot for row in rows],
    )


def make_options(
    env_actions: Sequence[Any],
    condition: Literal["action", "option", "both"],
    n_options: int,
    option_family: Literal["grammar", "grammar_depth", "random"],
    option_seed: int,
) -> Tuple[List[Tuple[int, ...]], List[str], List[Optional[int]]]:
    """Return the action table for `condition`: sequences, names, required slots.

    A primitive is a length-one sequence, so all three conditions are the same
    executor over a different table. `both` is the primitive rows followed by
    the options, so an index means the same action under `action` and under
    `both`. Table indices are not the env's own action indices, because
    `CONCEDING_COMMANDS` is dropped from every condition.

    `required_slots[i]` is the ASCII code of the inventory letter row `i` needs,
    or None when the row is selectable from every state.
    """
    primitives = [
        OptionRow((i,), getattr(action, "name", str(action)).lower(), None, 1, GROUP_PRIMITIVE)
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


class OptionWrapper(gym.Wrapper):
    """Execute one row of the action table open-loop and emit the info contract.

    The contract is `primitive_steps`, `option_discount`, `available` and
    `available_frac`, on every `step` and on `reset`. `reset` reports zero
    primitive steps: a vector env under NEXT_STEP autoreset routes the phantom
    decision through it, and that decision must cost the budget nothing.
    """

    def __init__(
        self,
        env: gym.Env,
        sequences: Sequence[Tuple[int, ...]],
        required_slots: Sequence[Optional[int]],
        gamma: float,
    ) -> None:
        super().__init__(env)
        self.sequences = list(sequences)
        self.gamma = gamma
        self.action_space = gym.spaces.Discrete(len(self.sequences))
        self.unconditional = np.array([slot is None for slot in required_slots])
        self.required_slots = np.array(
            [EMPTY_SLOT if slot is None else slot for slot in required_slots], dtype=np.uint8
        )
        self._episode_decisions = 0

    def _initiation(self, obs: Dict[str, np.ndarray]) -> np.ndarray:
        """`I` at the state `obs` describes, as a bool mask over the action table.

        A row naming an inventory letter is excluded when that slot is empty: the
        command would open a prompt with nothing to answer it and the trailing ESC
        would cancel it, so its beta fires on the spot. The remaining rows carry no
        precondition this wrapper can decide, so they are always offered.
        """
        return self.unconditional | np.isin(self.required_slots, obs["inv_letters"])

    def _contract(
        self, info: Dict[str, Any], obs: Dict[str, np.ndarray], primitive_steps: int
    ) -> Dict[str, Any]:
        available = self._initiation(obs)
        info["primitive_steps"] = primitive_steps
        info["option_discount"] = float(self.gamma**primitive_steps)
        info["available"] = available
        info["available_frac"] = float(available.mean())
        return info

    def reset(self, **kwargs: Any) -> Tuple[Any, Dict[str, Any]]:
        self._episode_decisions = 0
        obs, info = self.env.reset(**kwargs)
        info = self._contract(info, obs, 0)
        info["decision_t"] = 0
        return obs, info

    def step(self, option_id: int) -> Tuple[Any, float, bool, bool, Dict[str, Any]]:
        total_reward = 0.0
        steps = 0
        discount = 1.0
        for action in self.sequences[option_id]:
            obs, reward, terminated, truncated, info = self.env.step(action)
            total_reward += discount * reward
            discount *= self.gamma
            steps += 1
            if terminated or truncated:
                break
        self._episode_decisions += 1
        info = self._contract(info, obs, steps)
        info["decision_t"] = self._episode_decisions
        if terminated or truncated:
            self._episode_decisions = 0
        return obs, total_reward, terminated, truncated, info
