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
    """On goal, banks reward and ends episode after `reward_delay` steps."""

    def __init__(self, *args: Any, reward_delay: int = 0, **kwargs: Any) -> None:
        self._reward_delay = reward_delay
        self._held: Optional[int] = None
        self._paid = False
        # Use nethack.ACTIONS for action alignment and catalogue consistency.
 
        kwargs.setdefault("actions", nethack.ACTIONS)
        # Remove per-step reward; dense reward buries the terminal payout.
 
        kwargs.setdefault("penalty_step", 0.0)
        super().__init__(*args, **kwargs)

    def reset(self, *args: Any, **kwargs: Any) -> Tuple[Any, Dict[str, Any]]:
        """Clear the latch, then reset as usual."""
        # before super(), because NLE.reset ends by calling _get_end_status, so a
        # game that starts the agent on a staircase would latch during it
        self._held = None
        self._paid = False
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
        """Terminal step pays banked 1 plus time penalty."""
   
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
        self._paid = self._held is not None
        return time_penalty + (1.0 if self._paid else 0.0)

    def _get_information(self, *args: Any, **kwargs: Any) -> Dict[str, Any]:
        """The base information, plus whether this step flushed the banked reward.

        Read off the latch `_reward_fn` set rather than recomputed here: NLE calls
        `_reward_fn` first, and only it is given the `end_status` that decides
        whether the payout happened. A flush at the horizon or on a death reports
        ABORTED or DEATH, so `end_status` alone cannot answer this.
        """
        information: Dict[str, Any] = super()._get_information(  # type: ignore[misc]
            *args, **kwargs
        )
        information["paid"] = self._paid
        return information


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
