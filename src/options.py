from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple

import gymnasium as gym
import numpy as np
from nle import nethack

MOVE_REPEATS = [1, 2, 4, 8, 16]

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

EMPTY_SLOT = 0
"""`inv_letters` pads unoccupied slots with 0 rather than a letter."""


def make_options(
    env_actions: Sequence[Any],
    condition: Literal["action", "option", "both"],
) -> Tuple[List[Tuple[int, ...]], List[str], List[Optional[int]]]:
    """Return the action table for `condition`: sequences, names, required slots.

    A primitive is a length-one sequence, so all three conditions are the same
    executor over a different table. `both` is the primitives at their own
    indices followed by the options, so an index means the same action under
    `action` and under `both`.

    `required_slots[i]` is the ASCII code of the inventory letter row `i` needs,
    or None when the row is selectable from every state.

    Every non-movement option ends with ESC so any prompt the command opened
    is cancelled before the next option runs. Without this an unanswered
    prompt swallows the following option's first keystroke.
    """
    index = {a: i for i, a in enumerate(env_actions)}
    sequences: List[Tuple[int, ...]] = []
    names: List[str] = []
    required_slots: List[Optional[int]] = []

    def add(sequence: Tuple[int, ...], name: str, slot: Optional[int] = None) -> None:
        sequences.append(sequence)
        names.append(name)
        required_slots.append(slot)

    if condition in ("action", "both"):
        for i, action in enumerate(env_actions):
            add((i,), getattr(action, "name", str(action)).lower())

    if condition == "action":
        return sequences, names, required_slots

    esc = index.get(nethack.Command.ESC)
    tail = (esc,) if esc is not None else ()

    for direction in COMPASS:
        if direction not in index:
            continue
        for n in MOVE_REPEATS:
            add((index[direction],) * n, f"move_{direction.name}_x{n}")

    for command in SINGLE_COMMANDS:
        if command in index:
            add((index[command],) + tail, command.name.lower())

    for command in ARG_COMMANDS:
        if command not in index:
            continue
        for slot in INVENTORY_SLOTS:
            if ord(slot) not in index:
                continue
            add(
                (index[command], index[ord(slot)]) + tail,
                f"{command.name.lower()}_{slot}",
                ord(slot),
            )

    for command in DIR_COMMANDS:
        if command not in index:
            continue
        for direction in COMPASS:
            if direction not in index:
                continue
            add(
                (index[command], index[direction]) + tail,
                f"{command.name.lower()}_{direction.name}",
            )

    return sequences, names, required_slots


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
