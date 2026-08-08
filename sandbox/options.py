import gymnasium as gym


def make_repeat_options(specs):
    """specs: list of (action, repeats). Returns list of action sequences."""
    return [(action,) * repeats for action, repeats in specs]


DOORKEY_SPECS = [
    (0, 1), (0, 2),                    # turn left
    (1, 1), (1, 2),                    # turn right
    (2, 1), (2, 2), (2, 4), (2, 8),    # forward
    (3, 1),                            # pickup
    (5, 1),                            # toggle
]


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