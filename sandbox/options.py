import itertools
import gymnasium as gym


def make_option_list(actions, max_len):
    options = []
    for n in range(1, max_len + 1):
        options.extend(itertools.product(actions, repeat=n))
    return options


class OptionWrapper(gym.Wrapper):
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