import gymnasium as gym
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


def make_options(env_actions):
    """Build option sequences for this env's action list.

    Every non-movement option ends with ESC so any prompt the command opened
    is cancelled before the next option runs. Without this an unanswered
    prompt swallows the following option's first keystroke.
    """
    index = {a: i for i, a in enumerate(env_actions)}
    options, names = [], []

    esc = index.get(nethack.Command.ESC)
    tail = (esc,) if esc is not None else ()

    for direction in COMPASS:
        if direction not in index:
            continue
        for n in MOVE_REPEATS:
            options.append((index[direction],) * n)
            names.append(f"move_{direction.name}_x{n}")

    for command in SINGLE_COMMANDS:
        if command in index:
            options.append((index[command],) + tail)
            names.append(command.name.lower())

    for command in ARG_COMMANDS:
        if command not in index:
            continue
        for slot in INVENTORY_SLOTS:
            if ord(slot) not in index:
                continue
            options.append((index[command], index[ord(slot)]) + tail)
            names.append(f"{command.name.lower()}_{slot}")

    for command in DIR_COMMANDS:
        if command not in index:
            continue
        for direction in COMPASS:
            if direction not in index:
                continue
            options.append((index[command], index[direction]) + tail)
            names.append(f"{command.name.lower()}_{direction.name}")

    return options, names


class OptionWrapper(gym.Wrapper):
    """Execute an option open-loop, returning the SMDP-discounted reward sum.
    """

    def __init__(self, env, options, gamma=0.99):
        super().__init__(env)
        self.options = options
        self.gamma = gamma
        self.action_space = gym.spaces.Discrete(len(options))

    def step(self, option_id):
        total_reward = 0.0
        steps = 0
        discount = 1.0
        for action in self.options[option_id]:
            obs, reward, terminated, truncated, info = self.env.step(action)
            total_reward += discount * reward
            discount *= self.gamma
            steps += 1
            if terminated or truncated:
                break
        info["primitive_steps"] = steps
        info["option_discount"] = discount
        return obs, total_reward, terminated, truncated, info


class EpisodeAccounting(gym.Wrapper):

    def __init__(self, env):
        super().__init__(env)
        self._length = 0
        self._return = 0.0

    def reset(self, **kwargs):
        self._length = 0
        self._return = 0.0
        return self.env.reset(**kwargs)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._length += info.get("primitive_steps", 1)
        self._return += reward
        if terminated or truncated:
            info["primitive_length"] = self._length
            info["raw_return"] = self._return
            self._length = 0
            self._return = 0.0
        return obs, reward, terminated, truncated, info