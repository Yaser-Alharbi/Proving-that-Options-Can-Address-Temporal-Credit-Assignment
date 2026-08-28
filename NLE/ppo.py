# based on https://docs.cleanrl.dev/rl-algorithms/ppo/#ppo_ataripy, edited in place.
# Deviations from upstream, carried over from navix/ppo.py:
#   - the budget is spent in primitive steps, not decisions, so the loop is a
#     while on frames and the learning rate anneals against the same count
#   - the policy over options is masked to the initiation sets containing s_t,
#     and the mask that gated the action is stored and reapplied in the loss
#   - gamma applies once per decision, or raised to the option's duration (SMDP)
#   - the transition a NEXT_STEP autoreset inserts is excluded from the loss and
#     from the diagnostics
import csv
import os
import pathlib
import random
import time
from dataclasses import dataclass
from typing import Dict, Literal, Optional, Tuple

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tyro
from nle import nethack
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter

from envs import TASK_SUCCESSFUL, make_env

MASKED_LOGIT = -1e8
"""Logit given to an unavailable action. Large and negative rather than -inf, so
`log_softmax` stays finite and the entropy term is 0 * finite, not 0 * -inf."""

HIDDEN_DIM = 512

CHECKPOINT_EVERY_FRAMES = 500_000
"""Write a checkpoint this many primitive steps. Against frames, not wall-clock,
so the interval is comparable across conditions. Resume is not offered: NLE
env state is not serialised, so a resumed run is not the continuation of a
killed one."""

IDENTITY_SENTINEL_STR = "-"
IDENTITY_SENTINEL_INT = 0
"""Navix identity columns with no NLE analogue: `executor`, because one Python
loop is the only executor, and `max_forward`, because no reach bound parameterises
the NLE catalogue. Kept so plot.py reads both tracks without joins."""

IDENTITY_COLUMNS = (
    "env_id",
    "condition",
    "family",
    "n_options",
    "option_seed",
    "budget",
    "max_forward",
    "max_steps",
    "reward_delay",
    "gamma",
    "discount",
    "executor",
    "tag",
)
EPISODE_COLUMNS = IDENTITY_COLUMNS + (
    "seed",
    "decision_step",
    "primitive_step",
    "episodic_return",
    "episodic_length",
    "terminated",
    # not the same question as `terminated`, which NLE also sets on death and on
    # its own step-limit abort. Navix's `terminated` means goal-reached; giving
    # this column that meaning and leaving `terminated` the gymnasium flag keeps
    # one name per question on both tracks.
    "solved",
    "mean_option_duration",
)

ENCODER_OBS_DTYPE = {
    "glyphs": torch.int16,
    # int64 in the observation space, but the encoder's first op is `.float()`,
    # and float32 halves the buffer. Exact only to 2**24, which NetHack score
    # and gold cannot reach at these budgets.
    "blstats": torch.float32,
}
"""What the network reads, and the storage dtype for each. The env emits more
keys than this — the option wrapper needs `inv_letters` to decide `available` —
and the rollout buffer holds only these, so it does not grow with the wrapper's
needs. Shapes still come from the observation space."""


@dataclass
class Args:
    env_id: str = "NetHackChallenge-v0"
    budget: int = 10_000_000
    """training budget in PRIMITIVE steps, not decisions"""
    condition: Literal["action", "option", "both"] = "action"
    """primitives only, options only, or the union of both"""
    n_options: int = 64
    """how many rows the catalogue is subsampled to; ignored when condition is `action`"""
    option_family: Literal["grammar", "grammar_depth", "random"] = "grammar"
    """`grammar` takes the catalogue breadth-first across its row classes,
    `grammar_depth` longest-reach-first, `random` a uniform draw seeded by
    `option_seed`. `grammar` is the same catalogue in every experiment that names
    it; only exp3 runs `grammar_depth`."""
    option_seed: int = 0
    """seed for the random option family, kept separate from the training seeds"""
    reward_delay: int = 0
    """primitive steps between earning the terminal reward and being paid it. 0 is off."""
    discount: Literal["decision", "primitive"] = "decision"
    """gamma once per decision, or raised to the option's duration (SMDP)"""
    max_episode_steps: int = 100_000
    """primitive steps before truncation. Set explicitly: NetHackChallenge
    defaults to 1e6, at which one episode can consume a whole run."""
    clip_reward: bool = True
    """clip reward to [-1, 1] per primitive step, before any option sums it"""
    directory: str = ""
    """cell directory written by main.py. Empty: a standalone run under results/"""
    tag: str = ""
    """optional label appended to the run directory name"""
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "cleanRL"
    """the wandb's project name"""
    wandb_entity: Optional[str] = None
    """the entity (team) of wandb's project"""

    # Algorithm specific arguments
    learning_rate: float = 1e-4
    """the learning rate of the optimizer"""
    num_envs: int = 16
    """the number of parallel game environments"""
    num_steps: int = 128
    """the number of decisions to run in each environment per policy rollout"""
    anneal_lr: bool = True
    """Toggle learning rate annealing for policy and value networks"""
    gamma: float = 0.999
    """the discount factor gamma"""
    gae_lambda: float = 0.95
    """the lambda for the general advantage estimation"""
    num_minibatches: int = 4
    """the number of mini-batches"""
    update_epochs: int = 4
    """the K epochs to update the policy"""
    norm_adv: bool = True
    """Toggles advantages normalization"""
    clip_coef: float = 0.1
    """the surrogate clipping coefficient"""
    clip_vloss: bool = True
    """Toggles whether or not to use a clipped loss for the value function, as per the paper."""
    ent_coef: float = 0.001
    """coefficient of the entropy"""
    vf_coef: float = 0.5
    """coefficient of the value function"""
    max_grad_norm: float = 0.5
    """the maximum norm for the gradient clipping"""
    target_kl: Optional[float] = None
    """the target KL divergence threshold"""

    # to be filled in runtime
    batch_size: int = 0
    """the batch size (computed in runtime)"""
    minibatch_size: int = 0
    """the mini-batch size (computed in runtime)"""


def layer_init(layer: nn.Linear, std: float = np.sqrt(2), bias_const: float = 0.0) -> nn.Linear:
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


def masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Mean of `x` over the entries where `mask` is 1."""
    return (x * mask).sum() / mask.sum().clamp(min=1.0)


def cell_identity(args: Args) -> Dict[str, object]:
    """Navix `Cell.identity` columns for one NLE episode row."""
    options = args.condition != "action"
    return {
        "env_id": args.env_id,
        "condition": args.condition,
        "family": args.option_family if options else IDENTITY_SENTINEL_STR,
        "n_options": args.n_options if options else IDENTITY_SENTINEL_INT,
        "option_seed": args.option_seed,
        "budget": args.budget,
        "max_forward": IDENTITY_SENTINEL_INT,
        "max_steps": args.max_episode_steps,
        "reward_delay": args.reward_delay,
        "gamma": args.gamma,
        "discount": args.discount,
        "executor": IDENTITY_SENTINEL_STR,
        "tag": args.tag,
    }


def save_checkpoint(
    path: pathlib.Path,
    agent: nn.Module,
    optimizer: optim.Adam,
    frames: int,
    decisions: int,
    iteration: int,
    updates: int,
    args: Args,
) -> None:
    """Write trainer state. Does not include env states; do not resume from this."""
    torch.save(
        {
            "model": agent.state_dict(),
            "optimizer": optimizer.state_dict(),
            "frames": frames,
            "decisions": decisions,
            "iteration": iteration,
            "updates": updates,
            "rng": {
                "python": random.getstate(),
                "numpy": np.random.get_state(),
                "torch": torch.get_rng_state(),
                "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            },
            "env_seeds": [args.seed + index for index in range(args.num_envs)],
            "args": vars(args),
        },
        path,
    )


def _step_to_range(delta: float, num_steps: int) -> torch.Tensor:
    """Range of `num_steps` integers with distance `delta` centered around zero."""
    return delta * torch.arange(-num_steps // 2, num_steps // 2)


class Crop(nn.Module):
    """Centred crop of the glyph map around the agent, from facebookresearch/nle."""

    def __init__(self, height: int, width: int, height_target: int, width_target: int) -> None:
        super(Crop, self).__init__()
        self.width = width
        self.height = height
        self.width_target = width_target
        self.height_target = height_target
        width_grid = _step_to_range(2 / (self.width - 1), self.width_target)[None, :].expand(
            self.height_target, -1
        )
        height_grid = _step_to_range(2 / (self.height - 1), height_target)[:, None].expand(
            -1, self.width_target
        )

        # "clone" necessary, https://github.com/pytorch/pytorch/issues/34880
        self.register_buffer("width_grid", width_grid.clone())
        self.register_buffer("height_grid", height_grid.clone())

    def forward(self, inputs: torch.Tensor, coordinates: torch.Tensor) -> torch.Tensor:
        """Crop [B x H x W] inputs, centred on the [B x 2] x,y coordinates."""
        assert inputs.shape[1] == self.height
        assert inputs.shape[2] == self.width

        inputs = inputs[:, None, :, :].float()

        x = coordinates[:, 0]
        y = coordinates[:, 1]

        x_shift = 2 / (self.width - 1) * (x.float() - self.width // 2)
        y_shift = 2 / (self.height - 1) * (y.float() - self.height // 2)

        grid = torch.stack(
            [
                self.width_grid[None, :, :] + x_shift[:, None, None],
                self.height_grid[None, :, :] + y_shift[:, None, None],
            ],
            dim=3,
        )

        return torch.round(F.grid_sample(inputs, grid, align_corners=True)).squeeze(1).long()


class NetHackEncoder(nn.Module):
    """The Küttler NetHackNet trunk, stripped to a 512-dim encoder over [B, ...]."""

    def __init__(
        self,
        observation_space: gym.spaces.Dict,
        embedding_dim: int = 32,
        crop_dim: int = 9,
        num_layers: int = 5,
    ) -> None:
        super(NetHackEncoder, self).__init__()

        self.height, self.width = observation_space["glyphs"].shape
        self.blstats_size = observation_space["blstats"].shape[0]
        self.k_dim = embedding_dim
        self.h_dim = HIDDEN_DIM
        self.crop_dim = crop_dim

        self.crop = Crop(self.height, self.width, self.crop_dim, self.crop_dim)

        # MAX_GLYPH + 1, not MAX_GLYPH as upstream: the glyph space is
        # Box(low=0, high=MAX_GLYPH) with an inclusive high, so MAX_GLYPH is a
        # legal glyph and upstream's table is one row short of indexing it.
        self.embed = nn.Embedding(nethack.MAX_GLYPH + 1, self.k_dim)

        kernel, stride, padding = 3, 1, 1
        intermediate_filters, output_filters = 16, 8
        in_channels = [self.k_dim] + [intermediate_filters] * (num_layers - 1)
        out_channels = [intermediate_filters] * (num_layers - 1) + [output_filters]

        def convnet() -> nn.Sequential:
            layers: list[nn.Module] = []
            for i in range(num_layers):
                layers.append(
                    nn.Conv2d(
                        in_channels=in_channels[i],
                        out_channels=out_channels[i],
                        kernel_size=(kernel, kernel),
                        stride=stride,
                        padding=padding,
                    )
                )
                layers.append(nn.ELU())
            return nn.Sequential(*layers)

        self.extract_representation = convnet()
        self.extract_crop_representation = convnet()

        out_dim = (
            self.k_dim
            + self.height * self.width * output_filters
            + self.crop_dim**2 * output_filters
        )

        self.embed_blstats = nn.Sequential(
            nn.Linear(self.blstats_size, self.k_dim),
            nn.ReLU(),
            nn.Linear(self.k_dim, self.k_dim),
            nn.ReLU(),
        )
        self.fc = nn.Sequential(
            nn.Linear(out_dim, self.h_dim),
            nn.ReLU(),
            nn.Linear(self.h_dim, self.h_dim),
            nn.ReLU(),
        )

    def _select(self, embed: nn.Embedding, x: torch.Tensor) -> torch.Tensor:
        # index_select rather than calling the module: nn.Embedding's backward
        # pass is slow, https://github.com/pytorch/pytorch/issues/24912
        out = embed.weight.index_select(0, x.reshape(-1))
        return out.reshape(x.shape + (-1,))

    def forward(self, glyphs: torch.Tensor, blstats: torch.Tensor) -> torch.Tensor:
        glyphs = glyphs.long()
        blstats = blstats.float()
        coordinates = blstats[:, :2]

        representations = [self.embed_blstats(blstats)]

        crop = self.crop(glyphs, coordinates)
        crop_emb = self._select(self.embed, crop).transpose(1, 3)
        representations.append(self.extract_crop_representation(crop_emb).flatten(1))

        glyphs_emb = self._select(self.embed, glyphs).transpose(1, 3)
        representations.append(self.extract_representation(glyphs_emb).flatten(1))

        return self.fc(torch.cat(representations, dim=1))


class Agent(nn.Module):
    def __init__(self, envs: gym.vector.VectorEnv) -> None:
        super().__init__()
        self.encoder = NetHackEncoder(envs.single_observation_space)
        self.actor = layer_init(
            nn.Linear(self.encoder.h_dim, envs.single_action_space.n), std=0.01
        )
        self.critic = layer_init(nn.Linear(self.encoder.h_dim, 1), std=1.0)

    def get_value(self, obs: Dict[str, torch.Tensor]) -> torch.Tensor:
        return self.critic(self.encoder(obs["glyphs"], obs["blstats"]))

    def get_action_and_value(
        self,
        obs: Dict[str, torch.Tensor],
        available: torch.Tensor,
        action: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample or score an action under the policy masked to `available`.

        `available` is positional and required: the loss must mask with the same
        mask that gated the stored action, or the importance ratio is taken
        between two differently normalised distributions.
        """
        hidden = self.encoder(obs["glyphs"], obs["blstats"])
        logits = self.actor(hidden)
        logits = torch.where(available, logits, torch.full_like(logits, MASKED_LOGIT))
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(hidden)


if __name__ == "__main__":
    args = tyro.cli(Args)
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)

    if args.directory:
        run_dir = pathlib.Path(args.directory)
    else:
        parts = [args.env_id, args.condition, f"seed{args.seed}", time.strftime("%m-%d_%H-%M")]
        if args.tag:
            parts.append(args.tag.strip().replace(" ", "-"))
        run_dir = pathlib.Path("results") / "__".join(parts)
    run_dir.mkdir(parents=True, exist_ok=True)
    writer_dir = run_dir / f"seed{args.seed}"
    writer_dir.mkdir(parents=True, exist_ok=True)

    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_dir.name,
            monitor_gym=True,
            save_code=True,
        )

    writer = SummaryWriter(str(writer_dir))
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    identity = cell_identity(args)
    # per seed, not per cell: main.py runs the seeds of one cell concurrently
    # into one directory, and two processes appending to one file race on the
    # header and interleave their rows. The `seed` column makes the union of a
    # cell's files the same table the single file used to be.
    csv_path = run_dir / f"episodes_seed{args.seed}.csv"
    csv_file = open(csv_path, "a", newline="")
    csv_writer = csv.DictWriter(csv_file, fieldnames=list(EPISODE_COLUMNS))
    if csv_path.stat().st_size == 0:
        csv_writer.writeheader()
    checkpoint_path = run_dir / f"checkpoint_seed{args.seed}.pt"

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    print(device)

    # env setup. Sync, not async: an NLE step is ~7 us against ~1 ms of
    # per-vector-step IPC, so async loses at the option durations this
    # catalogue produces. NEXT_STEP is explicit because the primitive-step
    # accounting below depends on the phantom transition landing where it does.
    envs = gym.vector.SyncVectorEnv(
        [
            make_env(
                args.env_id,
                args.seed,
                i,
                args.condition,
                args.gamma,
                args.max_episode_steps,
                args.clip_reward,
                args.n_options,
                args.option_family,
                args.option_seed,
                args.reward_delay,
            )
            for i in range(args.num_envs)
        ],
        autoreset_mode=gym.vector.AutoresetMode.NEXT_STEP,
    )
    assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"
    num_actions = int(envs.single_action_space.n)
    print("n actions:", num_actions)

    agent = Agent(envs).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    # ALGO Logic: Storage setup
    obs = {
        key: torch.zeros(
            (args.num_steps, args.num_envs) + envs.single_observation_space[key].shape,
            dtype=dtype,
            device=device,
        )
        for key, dtype in ENCODER_OBS_DTYPE.items()
    }
    actions = torch.zeros((args.num_steps, args.num_envs)).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    discounts = torch.zeros((args.num_steps, args.num_envs)).to(device)
    primitive_steps = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)
    available = torch.zeros((args.num_steps, args.num_envs, num_actions), dtype=torch.bool, device=device)

    def to_device(observation: Dict[str, np.ndarray]) -> Dict[str, torch.Tensor]:
        """Move the encoder's observation keys onto the device, dropping the rest."""
        return {
            key: torch.as_tensor(observation[key], dtype=dtype, device=device)
            for key, dtype in ENCODER_OBS_DTYPE.items()
        }

    # TRY NOT TO MODIFY: start the game
    frames = 0
    decisions = 0
    iteration = 0
    updates = 0
    last_checkpoint_frames = 0
    start_time = time.time()
    next_obs_np, next_infos = envs.reset(seed=args.seed)
    next_obs = to_device(next_obs_np)
    assert next_infos["_available"].all(), "the env must emit `available` from reset"
    next_available = torch.as_tensor(next_infos["available"], dtype=torch.bool, device=device)
    next_done = torch.zeros(args.num_envs).to(device)

    while frames < args.budget:
        iteration += 1
        # read before the rollout advances `frames`, so the rate is the one this
        # iteration starts on and is constant across its minibatches. Clamped at
        # zero: an overrun would otherwise turn into gradient ascent.
        if args.anneal_lr:
            frac = max(1.0 - frames / args.budget, 0.0)
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        for step in range(0, args.num_steps):
            for key, buffer in obs.items():
                buffer[step] = next_obs[key]
            dones[step] = next_done
            available[step] = next_available

            # ALGO LOGIC: action logic
            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs, next_available)
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs_np, reward, terminations, truncations, infos = envs.step(action.cpu().numpy())
            next_done_np = np.logical_or(terminations, truncations)
            rewards[step] = torch.as_tensor(reward, dtype=torch.float32, device=device).view(-1)

            # `_add_info` zero-fills a key an env did not report, so a missing
            # mask would silently read as zero primitive steps rather than fail
            assert infos["_primitive_steps"].all(), "the env must emit `primitive_steps` every step"
            steps_taken = infos["primitive_steps"]
            frames += int(steps_taken.sum())
            decisions += args.num_envs
            primitive_steps[step] = torch.as_tensor(steps_taken, dtype=torch.float32, device=device)

            if args.discount == "primitive":
                discounts[step] = torch.as_tensor(
                    args.gamma ** steps_taken.astype(np.float64), dtype=torch.float32, device=device
                )
            else:
                discounts[step] = args.gamma

            assert infos["_available"].all(), "the env must emit `available` every step"
            next_available = torch.as_tensor(infos["available"], dtype=torch.bool, device=device)
            next_obs = to_device(next_obs_np)
            next_done = torch.as_tensor(next_done_np, dtype=torch.float32, device=device)

            if "episode" in infos:
                finished = infos["_episode"]
                for env_index in np.flatnonzero(finished):
                    episodic_return = float(infos["episode"]["r"][env_index])
                    episodic_length = float(infos["episode"]["l"][env_index])
                    episode_decisions = int(infos["decision_t"][env_index])
                    print(f"frames={frames}, episodic_return={episodic_return}")
                    writer.add_scalar("charts/episodic_return", episodic_return, frames)
                    writer.add_scalar("charts/episodic_length", episodic_length, frames)
                    csv_writer.writerow(
                        {
                            **identity,
                            "seed": args.seed,
                            "decision_step": decisions,
                            "primitive_step": frames,
                            "episodic_return": episodic_return,
                            "episodic_length": episodic_length,
                            "terminated": int(terminations[env_index]),
                            "solved": int(
                                int(infos["end_status"][env_index]) == TASK_SUCCESSFUL
                            ),
                            "mean_option_duration": episodic_length / max(episode_decisions, 1),
                        }
                    )
                    csv_file.flush()

        # bootstrap value if not done
        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = rewards[t] + discounts[t] * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = delta + discounts[t] * args.gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + values

        # flatten the batch
        b_obs = {key: value.reshape((-1,) + value.shape[2:]) for key, value in obs.items()}
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape(-1)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)
        b_available = available.reshape(-1, num_actions)
        # a NEXT_STEP autoreset inserts one transition per episode holding the
        # terminal observation, an action the env discarded and zero reward. It
        # is exactly the transition whose `dones` is set, so no extra buffer is
        # needed to find it, and it is excluded from every reduction below.
        b_masks = (1.0 - dones).reshape(-1)

        # Optimizing the policy and value network
        b_inds = np.arange(args.batch_size)
        clipfracs = []
        grad_norms = []
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]
                mb_obs = {key: value[mb_inds] for key, value in b_obs.items()}
                mb_masks = b_masks[mb_inds]

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(
                    mb_obs, b_available[mb_inds], b_actions.long()[mb_inds]
                )
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = masked_mean(-logratio, mb_masks)
                    approx_kl = masked_mean((ratio - 1) - logratio, mb_masks)
                    clipfracs += [
                        masked_mean(((ratio - 1.0).abs() > args.clip_coef).float(), mb_masks).item()
                    ]

                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_mean = masked_mean(mb_advantages, mb_masks)
                    mb_var = masked_mean((mb_advantages - mb_mean) ** 2, mb_masks)
                    mb_advantages = (mb_advantages - mb_mean) / (mb_var.sqrt() + 1e-8)

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = masked_mean(torch.max(pg_loss1, pg_loss2), mb_masks)

                # Value loss
                newvalue = newvalue.view(-1)
                if args.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -args.clip_coef,
                        args.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * masked_mean(v_loss_max, mb_masks)
                else:
                    v_loss = 0.5 * masked_mean((newvalue - b_returns[mb_inds]) ** 2, mb_masks)

                entropy_loss = masked_mean(entropy, mb_masks)
                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                optimizer.zero_grad()
                loss.backward()
                grad_norms.append(
                    nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm).item()
                )
                optimizer.step()
                updates += 1

            if args.target_kl is not None and approx_kl > args.target_kl:
                break

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], frames)
        writer.add_scalar("losses/value_loss", v_loss.item(), frames)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), frames)
        writer.add_scalar("losses/entropy", entropy_loss.item(), frames)
        writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), frames)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), frames)
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), frames)
        writer.add_scalar("losses/explained_variance", explained_var, frames)
        print("SPS:", int(frames / (time.time() - start_time)))
        writer.add_scalar("charts/SPS", int(frames / (time.time() - start_time)), frames)

        # option diagnostics, over non-phantom transitions only: the autoreset
        # transition would otherwise contribute an action the env discarded and
        # a terminal state the policy never acted in, and its share of the
        # buffer is condition-dependent
        live = 1.0 - dones
        live_rows = live.bool()
        writer.add_scalar("option/steps_mean", masked_mean(primitive_steps, live).item(), frames)
        writer.add_scalar(
            "option/steps_max_lane",
            torch.where(live_rows, primitive_steps, torch.zeros_like(primitive_steps))
            .max(dim=1)
            .values.mean()
            .item(),
            frames,
        )
        available_live = b_available[live_rows.reshape(-1)]
        if available_live.numel() > 0:
            n_available = available_live.sum(-1).float()
            writer.add_scalar("option/available_frac", available_live.float().mean().item(), frames)
            writer.add_scalar("option/entropy_ceiling", n_available.log().mean().item(), frames)
            writer.add_scalar(
                "option/entropy_ceiling_std", n_available.log().std(unbiased=False).item(), frames
            )
            # histograms rather than scalars: both are length-num_actions
            # distributions over the action index, and `add_scalar` cannot carry
            # a vector. Samples, not counts, so tensorboard bins them itself.
            writer.add_histogram("option/selected", actions[live_rows].long(), frames)
            writer.add_histogram("option/offered", available_live.nonzero()[:, 1], frames)

        writer.add_scalar("iter/frames", frames, frames)
        writer.add_scalar("iter/decisions", decisions, frames)
        writer.add_scalar("iter/updates", updates, frames)
        writer.add_scalar("iter/learning_rate", optimizer.param_groups[0]["lr"], frames)
        # pre-clip norm, once per minibatch, so a single scalar per iteration
        # would be ambiguous. The max decides whether max_grad_norm clips always.
        writer.add_scalar("iter/grad_norm_mean", float(np.mean(grad_norms)), frames)
        writer.add_scalar("iter/grad_norm_max", float(np.max(grad_norms)), frames)

        if frames - last_checkpoint_frames >= CHECKPOINT_EVERY_FRAMES:
            save_checkpoint(
                checkpoint_path, agent, optimizer, frames, decisions, iteration, updates, args
            )
            last_checkpoint_frames = frames

    save_checkpoint(
        checkpoint_path, agent, optimizer, frames, decisions, iteration, updates, args
    )
    envs.close()
    writer.close()
    csv_file.close()
