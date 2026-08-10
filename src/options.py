import gymnasium as gym
from nle import nethack

MOVE_REPEATS = [1, 2, 4, 8]

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
    nethack.Command.PICKUP,
    nethack.Command.SEARCH,
    nethack.Command.OPEN,
    nethack.Command.KICK,
    nethack.Command.PRAY,
]

ARG_COMMANDS = [
    nethack.Command.EAT,
    nethack.Command.WIELD,
    nethack.Command.WEAR,
    nethack.Command.QUAFF,
    nethack.Command.READ,
    nethack.Command.APPLY,
]

INVENTORY_SLOTS = "abcdefghij"


def make_options(env_actions):

    index = {a: i for i, a in enumerate(env_actions)}
    options, names = [], []

    for direction in COMPASS:
        if direction not in index:
            continue
        for n in MOVE_REPEATS:
            options.append((index[direction],) * n)
            names.append(f"move_{direction.name}_x{n}")

    for command in SINGLE_COMMANDS:
        if command in index:
            options.append((index[command],))
            names.append(command.name.lower())

    for command in ARG_COMMANDS:
        if command not in index:
            continue
        for slot in INVENTORY_SLOTS:
            if ord(slot) not in index:
                continue
            options.append((index[command], index[ord(slot)]))
            names.append(f"{command.name.lower()}_{slot}")

    return options, names


class OptionWrapper(gym.Wrapper):
    """Execute an option open-loop and return the undiscounted reward sum"""

    def __init__(self, env, options):
        super().__init__(env)
        self.options = options
        self.action_space = gym.spaces.Discrete(len(options))

    def step(self, option_id):
        total_reward = 0.0
        steps = 0
        for action in self.options[option_id]:
            obs, reward, terminated, truncated, info = self.env.step(action)
            total_reward += reward
            steps += 1
            if terminated or truncated:
                break
        info["primitive_steps"] = steps
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