"""Terminal-reward delay for the NLE goal tasks, as a hold inside the env.

exp4 needs the +1 to arrive some number of primitive steps after the goal was
reached. Navix does this in a wrapper, banking the reward and overriding the step
type so the inner env keeps stepping. NLE cannot be wrapped that way: `NLE.step`
calls `_quit_game` as soon as `end_status` leaves RUNNING, and a further `step`
raises "Called step on finished NetHack". So the hold moves to the predicate that
decides the episode is over.

`HeldGoal` is written against `_is_episode_end`, which is the only thing the three
goal tasks differ in: `NetHackStaircase` reads the stairs_down flag,
`NetHackStaircasePet` also wants the pet adjacent, and `NetHackOracle` looks for
the oracle's glyph. All three inherit one `_reward_fn`, so one mixin covers them.
"""

from typing import Any, Dict, Optional, Tuple, Type

from nle import nethack
from nle.env.base import NLE
from nle.env.tasks import NetHackOracle, NetHackStaircase, NetHackStaircasePet


class HeldGoal:
    """Latches the goal, waits `reward_delay` primitive steps, then ends the episode.

    The delay compresses the payout rather than deleting it. If the horizon
    arrives with the latch still open, `_check_abort` fires ABORTED, which is
    terminal, and `_reward_fn` pays the bank on that step: the realised delay is
    `min(reward_delay, horizon - solve_step)`, which is the quantity exp4's
    `delay_slack` plots. The banked value is fixed at 1 when the goal fires, so
    it does not decay with the delay, and an episode that never reaches the goal
    never latches and keeps its own end status.
    """

    def __init__(self, *args: Any, reward_delay: int = 0, **kwargs: Any) -> None:
        self._reward_delay = reward_delay
        self._held: Optional[int] = None
        # the goal tasks default to TASK_ACTIONS, 23 keys, which enumerates to 49
        # catalogue rows and no directional row at all. `NetHackChallenge` uses
        # nethack.ACTIONS, so matching it is what makes `grammar` at n=64 the
        # same 64 options in exp4 as in exp1 and exp2, and it keeps the primitive
        # action indices aligned across the two envs
        kwargs.setdefault("actions", nethack.ACTIONS)
        super().__init__(*args, **kwargs)

    def reset(self, *args: Any, **kwargs: Any) -> Tuple[Any, Dict[str, Any]]:
        """Clear the latch, then reset as usual."""
        # before super(), because NLE.reset ends by calling _get_end_status, so a
        # game that starts the agent on a staircase would latch during it
        self._held = None
        return super().reset(*args, **kwargs)  # type: ignore[misc]

    def _is_episode_end(self, observation: Any) -> int:
        """RUNNING until `reward_delay` steps after the task was first solved."""
        status = super()._is_episode_end(observation)  # type: ignore[misc]
        if self._held is None:
            if status != self.StepStatus.TASK_SUCCESSFUL:  # type: ignore[attr-defined]
                return status
            self._held = self._reward_delay
        if self._held > 0:
            self._held -= 1
            return self.StepStatus.RUNNING  # type: ignore[attr-defined]
        return self.StepStatus.TASK_SUCCESSFUL  # type: ignore[attr-defined]

    def _reward_fn(
        self,
        last_observation: Any,
        action: int,
        observation: Any,
        end_status: int,
    ) -> float:
        """The time penalty every step, and the banked 1 once, on the last one."""
        del action
        time_penalty = self._get_time_penalty(  # type: ignore[attr-defined]
            last_observation, observation
        )
        if end_status == self.StepStatus.RUNNING:  # type: ignore[attr-defined]
            return time_penalty
        # any terminal step with the latch open pays, not only the countdown
        # reaching zero: the horizon and a death inside the hold are terminal
        # too, and both have to compress the payout rather than delete it, or
        # `episodic_return` stops being delay-invariant
        return time_penalty + (1.0 if self._held is not None else 0.0)


DELAYED_ENVS: Dict[str, Type[NLE]] = {
    f"Delayed{label}-v0": type(f"Delayed{label}", (HeldGoal, task), {})
    for label, task in (
        ("Staircase", NetHackStaircase),
        ("StaircasePet", NetHackStaircasePet),
        ("Oracle", NetHackOracle),
    )
}
"""Not registered with gymnasium, and constructed directly by `envs.make_env`.

`gymnasium.make` consumes `max_episode_steps` for its own `TimeLimit` and never
forwards it, so NLE's internal horizon would stay at the 5000-step default; and a
`TimeLimit` truncation happens above the env, where `_reward_fn` never runs and
the held reward is never flushed. Leaving these unregistered makes that trap
unreachable rather than documented.
"""
