# CleanRL ppo_atari.py. Budget/LR in primitive steps; initiation mask stored for the loss; SMDP gamma; NEXT_STEP phantom excluded.
import atexit
import csv
import json
import os
import pathlib
import random
import signal
import socket
import subprocess
import sys
import time
from dataclasses import dataclass
from typing import Dict, List, Literal, Optional, Sequence, Tuple

import gymnasium as gym
import nle
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tyro
from nle import nethack
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter

from envs import STATUS_RUNNING, TASK_SUCCESSFUL, make_env
from options import TERM_CAUSE_NAMES

MASKED_LOGIT = -1e8
"""Unavailable logit. Finite so entropy is 0*finite, not 0*-inf."""

HIDDEN_DIM = 512

CHECKPOINT_EVERY_FRAMES = 500_000
"""Checkpoint cadence in primitive steps. No resume: NLE env state is not serialised."""

LOG_SCHEMA_VERSION = 4
"""Bump on `EPISODE_COLUMNS`, `TRACE_SCHEMA`, or `TERM_CAUSE_NAMES`."""

BLSTATS_FIELDS = (
    ("score", nethack.NLE_BL_SCORE),
    ("dlvl", nethack.NLE_BL_DEPTH),
    ("xp_level", nethack.NLE_BL_XP),
    ("xp_points", nethack.NLE_BL_EXP),
    ("gold", nethack.NLE_BL_GOLD),
    ("hp", nethack.NLE_BL_HP),
    ("hp_max", nethack.NLE_BL_HPMAX),
    ("ac", nethack.NLE_BL_AC),
    ("energy", nethack.NLE_BL_ENE),
    ("hunger", nethack.NLE_BL_HUNGER),
    ("time", nethack.NLE_BL_TIME),
)
BLSTATS_INDEX = dict(BLSTATS_FIELDS)

BLSTATS_PEAK_FIELDS = ("score", "dlvl", "xp_level", "xp_points", "gold")
"""Episode maxima as `max_{name}`. Skip `time` (`turns_survived`) and fluctuating fields."""

TRACE_BLSTATS_FIELDS = ("dlvl", "score", "hp", "time")
"""Trace columns from the pre-step observation (aligned with value/logprob)."""

TRACE_DEVICE_FIELDS = (
    "action",
    "primitive_steps",
    "done",
    "reward",
    "value",
    "logprob",
) + TRACE_BLSTATS_FIELDS
"""Trace columns held on the device. Row order of `trace_device_columns`'s transfer."""

TRACE_SCHEMA = pa.schema(
    [
        pa.field(name, pa.int32())
        for name in ("global_step", "env_id", "action", "primitive_steps", "term_cause", "done")
    ]
    + [
        pa.field(name, pa.float32())
        for name in ("reward", "undiscounted_reward", "value", "logprob")
    ]
    + [pa.field(name, pa.int32()) for name in TRACE_BLSTATS_FIELDS]
)

MAX_TRACKED_DLVL = 64
"""Width of the per-episode depth mask. A deeper level is uncounted, not a resize."""

XLOGFILE_FIELDS = ("role", "race", "gender", "align")
"""From NetHack's log. `character="@"` so these vary per episode. Death reason is `terminal_message`."""

IDENTITY_SENTINEL_STR = "-"
IDENTITY_SENTINEL_INT = 0
"""Navix-only columns: one Python loop, no `max_forward` parameter. plot.py joins both tracks."""

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
    # goal-reached (Navix `terminated`); gymnasium `terminated` is the other question
    "solved",
    "paid",
    "mean_option_duration",
    # last in-game frame, not the zeroed terminal obs
    *(name for name, _ in BLSTATS_FIELDS),
    *(f"max_{name}" for name in BLSTATS_PEAK_FIELDS),
    "end_status",
    "is_ascended",
    "terminal_message",
    # sums of what the wrapper was handed (clipped if `clip_reward`); `episodic_return` is unclipped
    "steps_to_first_reward",
    "n_nonzero_reward_steps",
    "sum_positive_clipped",
    "sum_negative_clipped",
    "clipped_return",
    # both clocks, whichever trained
    "return_decision",
    "return_primitive",
    "decision_count",
    "primitive_count",
    "option_calls",
    "fallback_frac",
    "option_stats",
    "unique_dlvls",
    "turns_survived",
    "env_seed",
    *XLOGFILE_FIELDS,
    "wall_clock",
    "host",
    "sps",
)

ENCODER_OBS_DTYPE = {
    "glyphs": torch.int16,
    # int64 in the space; encoder `.float()`s it. float32 exact to 2**24.
    "blstats": torch.float32,
}
"""Network keys. Wrapper also needs `inv_letters`/map for `I` and `misc` for the drain."""


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
    """`grammar` breadth-first, `grammar_depth` longest-reach-first, `random` by `option_seed`. Only exp3 runs `grammar_depth`."""
    option_seed: int = 0
    """seed for the random option family, kept separate from the training seeds"""
    reward_delay: int = 0
    """primitive steps between earning the terminal reward and being paid it. 0 is off."""
    discount: Literal["decision", "primitive"] = "decision"
    """gamma once per decision, or raised to the option's duration (SMDP)"""
    max_episode_steps: int = 100_000
    """primitive steps before truncation. NetHackChallenge defaults to 1e6."""
    clip_reward: bool = True
    """clip reward to [-1, 1] per primitive step, before any option sums it"""
    log_trace: bool = True
    """one row per `envs.step` (decisions, not primitives)"""
    trace_flush_iterations: int = 64
    """rollout iterations per parquet row group. Not the optimizer-step `updates` counter."""
    checkpoint_keep: Literal["all", "endpoints"] = "endpoints"
    """`endpoints` is first, midpoint, last. `all` is every cadence hit."""
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

    batch_size: int = 0
    """the batch size (computed in runtime)"""
    minibatch_size: int = 0
    """the mini-batch size (computed in runtime)"""


def layer_init(layer: nn.Linear, std: float = np.sqrt(2), bias_const: float = 0.0) -> nn.Linear:
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


def masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return (x * mask).sum() / mask.sum().clamp(min=1.0)


def trace_device_columns(
    actions: torch.Tensor,
    primitive_steps: torch.Tensor,
    dones: torch.Tensor,
    rewards: torch.Tensor,
    values: torch.Tensor,
    logprobs: torch.Tensor,
    blstats: torch.Tensor,
    blstats_indices: Sequence[int],
) -> np.ndarray:
    """`TRACE_DEVICE_FIELDS` as rows flattened step-major, in one host transfer.

    Every column is float32, so all ten ride one `.cpu()` instead of costing a
    device sync each. The caller's `trace_buffer` dtype does the int cast.
    """
    stacked = torch.cat(
        [
            actions.unsqueeze(0),
            primitive_steps.unsqueeze(0),
            dones.unsqueeze(0),
            rewards.unsqueeze(0),
            values.unsqueeze(0),
            logprobs.unsqueeze(0),
            blstats[:, :, blstats_indices].permute(2, 0, 1),
        ]
    )
    assert stacked.dtype == torch.float32, (
        f"one stacked transfer requires one dtype; got {stacked.dtype}"
    )
    assert stacked.shape[0] == len(TRACE_DEVICE_FIELDS), (
        f"{stacked.shape[0]} rows for {len(TRACE_DEVICE_FIELDS)} named columns"
    )
    return stacked.reshape(len(TRACE_DEVICE_FIELDS), -1).cpu().numpy()


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


def git_sha() -> Optional[str]:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=pathlib.Path(__file__).resolve().parent,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except OSError:
        return None
    return result.stdout.strip() if result.returncode == 0 else None


def read_xlogfile(path: pathlib.Path, offset: int) -> Tuple[Dict[str, str], int]:
    """Newest complete record past `offset`. Truncation writes nothing, so an empty record is not some earlier episode's character."""
    with open(path, "rb") as handle:
        handle.seek(offset)
        new = handle.read()
    cut = new.rfind(b"\n")
    if cut < 0:
        return {}, offset
    lines = [line for line in new[:cut].split(b"\n") if line]
    if not lines:
        return {}, offset + cut + 1
    record = dict(
        field.split("=", 1)
        for field in lines[-1].decode("ascii", "replace").split("\t")
        if "=" in field
    )
    return record, offset + cut + 1


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

        # MAX_GLYPH + 1: Box high is inclusive, so MAX_GLYPH is a legal glyph
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
        # index_select: nn.Embedding backward is slow, https://github.com/pytorch/pytorch/issues/24912
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
        """The loss must reuse the mask that gated the stored action, or the ratio is between two distributions."""
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
    # per seed: concurrent seeds in one directory would race on a shared file
    csv_path = run_dir / f"episodes_seed{args.seed}.csv"
    csv_file = open(csv_path, "a", newline="")
    csv_writer = csv.DictWriter(csv_file, fieldnames=list(EPISODE_COLUMNS))
    if csv_path.stat().st_size == 0:
        csv_writer.writeheader()
    episode_rows: List[Dict[str, object]] = []
    """Flushed per update. A flush per episode is a syscall inside the rollout."""

    host = socket.gethostname()
    # not `meta.json`: main.py writes that only when every seed of the cell succeeded
    (run_dir / f"provenance_seed{args.seed}.json").write_text(
        json.dumps(
            {
                "git_sha": git_sha(),
                "nle_version": nle.__version__,
                "host": host,
                "log_schema_version": LOG_SCHEMA_VERSION,
            },
            indent=2,
        )
    )

    trace_part = 0
    trace_filled = 0
    trace_buffer: Dict[str, np.ndarray] = {}
    trace_writer: Optional[pq.ParquetWriter] = None
    if args.log_trace:
        trace_capacity = args.trace_flush_iterations * args.num_steps * args.num_envs
        trace_buffer = {
            field.name: np.zeros(
                trace_capacity,
                dtype=np.int32 if pa.types.is_int32(field.type) else np.float32,
            )
            for field in TRACE_SCHEMA
        }
        trace_writer = pq.ParquetWriter(
            run_dir / f"decisions_seed{args.seed}_part{trace_part:02d}.parquet",
            TRACE_SCHEMA,
            compression="zstd",
            compression_level=3,
        )

        def flush_trace_row_group() -> None:
            """Write the buffered decisions as one row group and empty the buffer."""
            global trace_filled, trace_writer
            if trace_filled == 0:
                return
            assert trace_writer is not None
            trace_writer.write_batch(
                pa.record_batch(
                    [
                        pa.array(trace_buffer[field.name][:trace_filled], type=field.type)
                        for field in TRACE_SCHEMA
                    ],
                    schema=TRACE_SCHEMA,
                )
            )
            trace_filled = 0

        def close_trace() -> None:
            """Flush the tail and write the parquet footer."""
            global trace_writer
            flush_trace_row_group()
            if trace_writer is not None and trace_writer.is_open:
                trace_writer.close()

        atexit.register(close_trace)
        signal.signal(signal.SIGTERM, lambda signum, frame: sys.exit(128 + signum))

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    print(device)

    # Sync: NLE step is cheaper than vector IPC. NEXT_STEP: phantom accounting depends on it.
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
    drain_steps = torch.zeros((args.num_steps, args.num_envs)).to(device)
    fallbacks = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)
    available = torch.zeros((args.num_steps, args.num_envs, num_actions), dtype=torch.bool, device=device)

    ep_return_decision = np.zeros(args.num_envs, dtype=np.float64)
    ep_return_primitive = np.zeros(args.num_envs, dtype=np.float64)
    ep_clipped_return = np.zeros(args.num_envs, dtype=np.float64)
    ep_sum_positive = np.zeros(args.num_envs, dtype=np.float64)
    ep_sum_negative = np.zeros(args.num_envs, dtype=np.float64)
    ep_nonzero_steps = np.zeros(args.num_envs, dtype=np.int64)
    ep_steps_to_first = np.full(args.num_envs, -1, dtype=np.int64)
    ep_decision_count = np.zeros(args.num_envs, dtype=np.int64)
    ep_primitive_count = np.zeros(args.num_envs, dtype=np.int64)
    ep_option_calls = np.zeros(args.num_envs, dtype=np.int64)
    ep_fallbacks = np.zeros(args.num_envs, dtype=np.int64)
    ep_peak_blstats = np.zeros(
        (args.num_envs,) + envs.single_observation_space["blstats"].shape, dtype=np.int64
    )
    ep_dlvl_seen = np.zeros((args.num_envs, MAX_TRACKED_DLVL), dtype=bool)
    ep_option_n = np.zeros((args.num_envs, num_actions), dtype=np.int64)
    ep_option_steps = np.zeros((args.num_envs, num_actions), dtype=np.int64)
    ep_option_peak = np.zeros((args.num_envs, num_actions), dtype=np.int64)
    ep_option_cause = np.zeros((args.num_envs, num_actions, len(TERM_CAUSE_NAMES)), dtype=np.int64)

    # `option_calls` is the tail: `make_options` emits primitives first
    if args.condition == "action":
        n_primitives = num_actions
    elif args.condition == "option":
        n_primitives = 0
    else:
        n_primitives = num_actions - args.n_options

    # one xlogfile per env; shared would mis-attribute character
    xlog_paths = [pathlib.Path(env.unwrapped.nethack._vardir) / "xlogfile" for env in envs.envs]
    assert len(set(xlog_paths)) == args.num_envs, f"envs share an xlogfile: {xlog_paths}"
    xlog_offsets = [0] * args.num_envs

    trace_global_step = np.zeros((args.num_steps, args.num_envs), dtype=np.int32)
    trace_undiscounted = np.zeros((args.num_steps, args.num_envs), dtype=np.float32)
    trace_cause = np.zeros((args.num_steps, args.num_envs), dtype=np.int32)
    trace_env_id = np.tile(np.arange(args.num_envs, dtype=np.int32), args.num_steps)
    trace_blstats_indices = [BLSTATS_INDEX[name] for name in TRACE_BLSTATS_FIELDS]

    def to_device(observation: Dict[str, np.ndarray]) -> Dict[str, torch.Tensor]:
        """Encoder keys only. The wrapper's extra obs keys stay on the host."""
        return {
            key: torch.as_tensor(observation[key], dtype=dtype, device=device)
            for key, dtype in ENCODER_OBS_DTYPE.items()
        }

    frames = 0
    decisions = 0
    iteration = 0
    updates = 0
    last_checkpoint_frames = 0
    wrote_first_checkpoint = False
    wrote_midpoint_checkpoint = False
    start_time = time.time()
    next_obs_np, next_infos = envs.reset(seed=args.seed)
    next_obs = to_device(next_obs_np)
    assert next_infos["_available"].all(), "the env must emit `available` from reset"
    next_available = torch.as_tensor(next_infos["available"], dtype=torch.bool, device=device)
    next_fallback_np = np.asarray(next_infos["initiation_empty"], dtype=bool)
    next_done = torch.zeros(args.num_envs).to(device)
    next_done_np = np.zeros(args.num_envs, dtype=bool)

    while frames < args.budget:
        iteration += 1
        # before `frames` advances; clamped so overrun is not ascent
        if args.anneal_lr:
            frac = max(1.0 - frames / args.budget, 0.0)
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        for step in range(0, args.num_steps):
            for key, buffer in obs.items():
                buffer[step] = next_obs[key]
            dones[step] = next_done
            available[step] = next_available
            fallbacks[step] = torch.as_tensor(
                next_fallback_np, dtype=torch.float32, device=device
            )

            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs, next_available)
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            # NEXT_STEP phantom: predecessor was done; read before step overwrites it
            live = np.logical_not(next_done_np)
            # previous-step flag: the mask this action was sampled under
            chose_on_fallback = next_fallback_np

            action_np = action.cpu().numpy()
            next_obs_np, reward, terminations, truncations, infos = envs.step(action_np)
            next_done_np = np.logical_or(terminations, truncations)
            rewards[step] = torch.as_tensor(reward, dtype=torch.float32, device=device).view(-1)

            # `_add_info` zero-fills a missing key, so a missing mask would read as 0 steps
            assert infos["_primitive_steps"].all(), "the env must emit `primitive_steps` every step"
            steps_taken = infos["primitive_steps"]
            frames += int(steps_taken.sum())
            decisions += args.num_envs
            primitive_steps[step] = torch.as_tensor(steps_taken, dtype=torch.float32, device=device)
            assert infos["_drain_steps"].all(), "the env must emit `drain_steps` every step"
            drain_steps[step] = torch.as_tensor(
                infos["drain_steps"], dtype=torch.float32, device=device
            )

            if args.discount == "primitive":
                discounts[step] = torch.as_tensor(
                    args.gamma ** steps_taken.astype(np.float64), dtype=torch.float32, device=device
                )
            else:
                discounts[step] = args.gamma

            assert infos["_available"].all(), "the env must emit `available` every step"
            next_available = torch.as_tensor(infos["available"], dtype=torch.bool, device=device)
            assert infos["_initiation_empty"].all(), (
                "the env must emit `initiation_empty` every step"
            )
            next_fallback_np = np.asarray(infos["initiation_empty"], dtype=bool)
            next_obs = to_device(next_obs_np)
            next_done = torch.as_tensor(next_done_np, dtype=torch.float32, device=device)

            # wrapper reward is intra-option discounted (primitive clock); decision clock needs the undiscounted sum
            undiscounted = infos["undiscounted_reward"]
            ingame_blstats = infos["ingame_blstats"]
            term_cause = infos["term_cause"]
            ep_return_primitive += live * args.gamma**ep_primitive_count * reward
            ep_return_decision += live * args.gamma**ep_decision_count * undiscounted
            ep_clipped_return += live * undiscounted
            ep_sum_positive += live * infos["sum_positive_clipped"]
            ep_sum_negative += live * infos["sum_negative_clipped"]
            ep_nonzero_steps += live * infos["nonzero_reward_steps"]

            # wrapper offset is inside the decision; freeze after first hit
            first_offset = infos["first_reward_offset"]
            first_hit = live & (ep_steps_to_first < 0) & (first_offset >= 0)
            ep_steps_to_first[first_hit] = ep_primitive_count[first_hit] + first_offset[first_hit]

            ep_decision_count += live
            ep_primitive_count += live * steps_taken
            ep_option_calls += live & (action_np >= n_primitives)
            ep_fallbacks += live & chose_on_fallback
            np.maximum(ep_peak_blstats, ingame_blstats, out=ep_peak_blstats, where=live[:, None])

            live_rows = np.flatnonzero(live)
            live_ids = action_np[live_rows]
            # `TERM_NONE` is -1 and would wrap
            assert (term_cause[live_rows] >= 0).all(), "a live decision reported no cause"
            # distinct row indices: scatters, not `np.add.at`
            ep_option_n[live_rows, live_ids] += 1
            ep_option_steps[live_rows, live_ids] += steps_taken[live_rows]
            ep_option_peak[live_rows, live_ids] = np.maximum(
                ep_option_peak[live_rows, live_ids], steps_taken[live_rows]
            )
            ep_option_cause[live_rows, live_ids, term_cause[live_rows]] += 1

            depth = ingame_blstats[:, nethack.NLE_BL_DEPTH]
            seen = live & (depth >= 0) & (depth < MAX_TRACKED_DLVL)
            ep_dlvl_seen[np.flatnonzero(seen), depth[seen]] = True

            if args.log_trace:
                trace_global_step[step] = frames
                trace_undiscounted[step] = undiscounted
                trace_cause[step] = term_cause

            if "episode" in infos:
                finished = infos["_episode"]
                for env_index in np.flatnonzero(finished):
                    episodic_return = float(infos["episode"]["r"][env_index])
                    episodic_length = float(infos["episode"]["l"][env_index])
                    episode_decisions = int(infos["decision_t"][env_index])
                    print(f"frames={frames}, episodic_return={episodic_return}")
                    writer.add_scalar("charts/episodic_return", episodic_return, frames)
                    writer.add_scalar("charts/episodic_length", episodic_length, frames)
                    end_status = int(infos["end_status"][env_index])
                    solved = int(end_status == TASK_SUCCESSFUL)
                    # if no `paid`, use `solved`
                    paid_mask = infos.get("_paid")
                    paid = (
                        int(infos["paid"][env_index])
                        if paid_mask is not None and paid_mask[env_index]
                        else solved
                    )
                    # truncation leaves the game running and unrecorded
                    character = dict.fromkeys(XLOGFILE_FIELDS, "")
                    if end_status != STATUS_RUNNING:
                        record, xlog_offsets[env_index] = read_xlogfile(
                            xlog_paths[env_index], xlog_offsets[env_index]
                        )
                        character.update(
                            {field: record.get(field, "") for field in XLOGFILE_FIELDS}
                        )

                    peak_frame = ep_peak_blstats[env_index]
                    last_frame = ingame_blstats[env_index]
                    # printable: a control byte in a CSV field is a new row to some readers
                    message = (
                        bytes(infos["ingame_message"][env_index])
                        .split(b"\0")[0]
                        .decode("ascii", "replace")
                    )
                    terminal_message = "".join(c for c in message if c.isprintable()).strip()
                    invoked = np.flatnonzero(ep_option_n[env_index])
                    option_stats = {
                        int(option_id): {
                            "n": int(ep_option_n[env_index, option_id]),
                            "mean_duration": round(
                                float(ep_option_steps[env_index, option_id])
                                / float(ep_option_n[env_index, option_id]),
                                3,
                            ),
                            "max_duration": int(ep_option_peak[env_index, option_id]),
                            **{
                                name: int(ep_option_cause[env_index, option_id, cause_index])
                                for cause_index, name in enumerate(TERM_CAUSE_NAMES)
                            },
                        }
                        for option_id in invoked
                    }
                    episode_rows.append(
                        {
                            **identity,
                            "seed": args.seed,
                            "decision_step": decisions,
                            "primitive_step": frames,
                            "episodic_return": episodic_return,
                            "episodic_length": episodic_length,
                            "terminated": int(terminations[env_index]),
                            "solved": solved,
                            "paid": paid,
                            "mean_option_duration": episodic_length / max(episode_decisions, 1),
                            **{name: int(last_frame[index]) for name, index in BLSTATS_FIELDS},
                            **{
                                f"max_{name}": int(peak_frame[BLSTATS_INDEX[name]])
                                for name in BLSTATS_PEAK_FIELDS
                            },
                            "end_status": end_status,
                            "is_ascended": int(infos["is_ascended"][env_index]),
                            "terminal_message": terminal_message,
                            "steps_to_first_reward": int(ep_steps_to_first[env_index]),
                            "n_nonzero_reward_steps": int(ep_nonzero_steps[env_index]),
                            "sum_positive_clipped": float(ep_sum_positive[env_index]),
                            "sum_negative_clipped": float(ep_sum_negative[env_index]),
                            "clipped_return": float(ep_clipped_return[env_index]),
                            "return_decision": float(ep_return_decision[env_index]),
                            "return_primitive": float(ep_return_primitive[env_index]),
                            "decision_count": int(ep_decision_count[env_index]),
                            "primitive_count": int(ep_primitive_count[env_index]),
                            "option_calls": int(ep_option_calls[env_index]),
                            "fallback_frac": (
                                float(ep_fallbacks[env_index])
                                / max(int(ep_decision_count[env_index]), 1)
                            ),
                            "option_stats": json.dumps(option_stats, separators=(",", ":")),
                            "unique_dlvls": int(ep_dlvl_seen[env_index].sum()),
                            "turns_survived": int(last_frame[nethack.NLE_BL_TIME]),
                            "env_seed": args.seed + env_index,
                            **character,
                            "wall_clock": time.strftime("%Y-%m-%dT%H:%M:%S"),
                            "host": host,
                            "sps": int(frames / max(time.time() - start_time, 1e-9)),
                        }
                    )

                    ep_return_decision[env_index] = 0.0
                    ep_return_primitive[env_index] = 0.0
                    ep_clipped_return[env_index] = 0.0
                    ep_sum_positive[env_index] = 0.0
                    ep_sum_negative[env_index] = 0.0
                    ep_nonzero_steps[env_index] = 0
                    ep_steps_to_first[env_index] = -1
                    ep_decision_count[env_index] = 0
                    ep_primitive_count[env_index] = 0
                    ep_option_calls[env_index] = 0
                    ep_fallbacks[env_index] = 0
                    ep_peak_blstats[env_index] = 0
                    ep_dlvl_seen[env_index] = False
                    ep_option_n[env_index] = 0
                    ep_option_steps[env_index] = 0
                    ep_option_peak[env_index] = 0
                    ep_option_cause[env_index] = 0

        if episode_rows:
            csv_writer.writerows(episode_rows)
            csv_file.flush()
            episode_rows.clear()

        if args.log_trace:
            start = trace_filled
            stop = start + args.num_steps * args.num_envs
            assert stop <= trace_buffer["global_step"].size
            transferred = trace_device_columns(
                actions,
                primitive_steps,
                dones,
                rewards,
                values,
                logprobs,
                obs["blstats"],
                trace_blstats_indices,
            )
            trace_buffer["global_step"][start:stop] = trace_global_step.reshape(-1)
            trace_buffer["env_id"][start:stop] = trace_env_id
            trace_buffer["term_cause"][start:stop] = trace_cause.reshape(-1)
            trace_buffer["undiscounted_reward"][start:stop] = trace_undiscounted.reshape(-1)
            for row, name in enumerate(TRACE_DEVICE_FIELDS):
                trace_buffer[name][start:stop] = transferred[row]
            trace_filled = stop
            if iteration % args.trace_flush_iterations == 0:
                flush_trace_row_group()

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

        b_obs = {key: value.reshape((-1,) + value.shape[2:]) for key, value in obs.items()}
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape(-1)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)
        b_available = available.reshape(-1, num_actions)
        # NEXT_STEP phantom is the `dones` transition; excluded from every reduction below
        b_masks = (1.0 - dones).reshape(-1)

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
                    # http://joschu.net/blog/kl-approx.html
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

                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = masked_mean(torch.max(pg_loss1, pg_loss2), mb_masks)

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

        # non-phantom only: share of the buffer is condition-dependent
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
        writer.add_scalar(
            "option/drain_frac", masked_mean((drain_steps > 0).float(), live).item(), frames
        )
        writer.add_scalar(
            "option/drain_steps_mean", masked_mean(drain_steps, live).item(), frames
        )
        writer.add_scalar(
            "option/fallback_frac", masked_mean(fallbacks, live).item(), frames
        )
        available_live = b_available[live_rows.reshape(-1)]
        if available_live.numel() > 0:
            n_available = available_live.sum(-1).float()
            writer.add_scalar("option/available_frac", available_live.float().mean().item(), frames)
            writer.add_scalar("option/entropy_ceiling", n_available.log().mean().item(), frames)
            writer.add_scalar(
                "option/entropy_ceiling_std", n_available.log().std(unbiased=False).item(), frames
            )
            # `add_scalar` cannot carry a vector; samples, not counts
            writer.add_histogram("option/selected", actions[live_rows].long(), frames)
            writer.add_histogram("option/offered", available_live.nonzero()[:, 1], frames)

        writer.add_scalar("iter/frames", frames, frames)
        writer.add_scalar("iter/decisions", decisions, frames)
        writer.add_scalar("iter/updates", updates, frames)
        writer.add_scalar("iter/learning_rate", optimizer.param_groups[0]["lr"], frames)
        # pre-clip, per minibatch
        writer.add_scalar("iter/grad_norm_mean", float(np.mean(grad_norms)), frames)
        writer.add_scalar("iter/grad_norm_max", float(np.max(grad_norms)), frames)

        if frames - last_checkpoint_frames >= CHECKPOINT_EVERY_FRAMES:
            last_checkpoint_frames = frames
            if args.log_trace:
                flush_trace_row_group()
                assert trace_writer is not None
                trace_writer.close()
                trace_part += 1
                trace_writer = pq.ParquetWriter(
                    run_dir / f"decisions_seed{args.seed}_part{trace_part:02d}.parquet",
                    TRACE_SCHEMA,
                    compression="zstd",
                    compression_level=3,
                )
            past_midpoint = frames >= args.budget // 2
            if (
                args.checkpoint_keep == "all"
                or not wrote_first_checkpoint
                or (past_midpoint and not wrote_midpoint_checkpoint)
            ):
                save_checkpoint(
                    run_dir / f"checkpoint_seed{args.seed}_f{frames:09d}.pt",
                    agent,
                    optimizer,
                    frames,
                    decisions,
                    iteration,
                    updates,
                    args,
                )
                wrote_first_checkpoint = True
                wrote_midpoint_checkpoint = wrote_midpoint_checkpoint or past_midpoint

    save_checkpoint(
        run_dir / f"checkpoint_seed{args.seed}_f{frames:09d}.pt",
        agent,
        optimizer,
        frames,
        decisions,
        iteration,
        updates,
        args,
    )
    envs.close()
    writer.close()
    csv_file.close()
