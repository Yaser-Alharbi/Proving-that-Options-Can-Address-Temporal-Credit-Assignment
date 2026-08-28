"""Tests pinning the option catalogue, so a draw cannot move unnoticed.

`cd src && python -m pytest test_files/test_options.py -q`.

The catalogue is the input every option cell is defined against. Both grammar
families take a prefix of a total order and the random family reads its input
positionally, so an edit to the enumeration order or to a sort key silently
redraws the catalogue that some earlier run trained on. The digests below are the
guard: they fail loudly, and re-recording one is a decision about whether every
prior run of that family still counts, not a fix.
"""

from typing import Dict, List, Sequence

import gymnasium as gym
import pytest
from nle import nethack

from envs import OBSERVATION_KEYS
from options import (
    CONCEDING_COMMANDS,
    GROUP_ARG,
    GROUP_DIR,
    GROUP_MOVE,
    GROUP_SINGLE,
    MOVE_REPEATS,
    OptionRow,
    _catalogue,
    catalogue_digest,
    grammar_depth_options,
    grammar_options,
    make_options,
    random_options,
)

ENV_ID = "NetHackChallenge-v0"
"""The env exp1, exp2 and exp4 run on, and so the action set the digests pin.
A task env exposes a smaller action set and would yield a shorter catalogue."""

SHORT_EPISODE = 10
"""No test here steps the env; the horizon only has to be legal."""

CATALOGUE_SIZE = 188
GROUP_SIZES = {GROUP_MOVE: 40, GROUP_SINGLE: 16, GROUP_ARG: 100, GROUP_DIR: 32}
CATALOGUE_DIGEST = "3902aa692e84b2e70fa1fbf57578e02051f2cc01aa97ff75d97609beb33806fb"

DRAW_SIZE = 64
"""The n exp1, exp3 and exp4 all use, so it is the draw worth pinning."""

ENV_ACTION_SET = 121
ACTION_TABLE = ENV_ACTION_SET - len(CONCEDING_COMMANDS)
"""The env keeps its own action set; the table drops the conceding commands from
every condition, so the two differ by exactly those rows."""

KNOWN_TURN_ONE_ESCAPES = ("up",)
"""Free terminations kept in the table, as a decision rather than an oversight.

`up` on the first move climbs out of the dungeon for reward 0, because every game
starts on the up-staircase of level 1. It is kept for two reasons: `TASK_ACTIONS`
contains `MiscDirection.UP`, so removing it would not be what NLE does for a goal
env; and the key that answers its prompt is `CompassDirection.NW`, whose value
121 is ASCII `y`, so the confirmation cannot be removed without removing
northwest movement.

It is a confound, and it is not symmetric: the catalogue's `up` row is
`(UP, ESC)` and the ESC cancels the prompt, so `option` cannot take this route
while `action` and `both` can. It is reported rather than patched. The route is
measurable after the fact without extra instrumentation, because
`test_no_new_action_ends_the_episode_within_two_steps` establishes that nothing
else terminates in two steps from a fresh game: an `episodic_length` of 2 in
`episodes.csv` is this escape.
"""

GRAMMAR_DIGEST = "8f4fd14eb0fefcb8982f193963ded5587ec3c911bc3e581ab7440575e7d823a2"
GRAMMAR_DEPTH_DIGEST = "124ee31abd3f0d6c160230dd5219d6152aded894ac6e6664667128528fe93e98"
RANDOM_DIGESTS: Dict[int, str] = {
    0: "9b422918e2050b6857870504d5ed1958e737129148b2a0e687d7ab73e2a7e87f",
    1: "2a5e7fb6d1d2a7c388159c9b5c04b2574b91aecae0babad4f97238c7f03ca4eb",
    2: "c09ce916a73dab3d594bf17c496468a0d94ce286b4d5674aac609875a326fc8b",
    3: "50d6c3488fabe9bb40deb8ace025b68e5a2bfa2ca555e35d5d2e99c57bfdbc50",
    4: "ba1f8841334f992adc69d27d992343f14459c21c4de18c4aaf4edbc0e5f291cd",
}
"""The five option seeds exp3 draws."""

GRAMMAR_PREFIX_AT_EIGHT = [
    "move_N_x16",
    "down",
    "open_N",
    "move_S_x16",
    "up",
    "open_S",
    "move_E_x16",
    "wait",
]
"""Spelled out rather than digested: this is the prefix the breadth-first prior
was chosen for, and it is the readable evidence that a small n is not a
movement-only catalogue."""

FIRST_INTERACTION_DEPTH = 40
"""Rows of `grammar_depth` before the first non-movement row. Every movement row
precedes every command row, so n at or below this is a capability floor."""


@pytest.fixture(scope="module")
def rows() -> List[OptionRow]:
    """The full catalogue for `ENV_ID`'s action set."""
    env = gym.make(
        ENV_ID, observation_keys=OBSERVATION_KEYS, max_episode_steps=SHORT_EPISODE
    )
    catalogue = _catalogue(env.unwrapped.actions)
    env.close()
    return catalogue


def digest_by_name(chosen: Sequence[OptionRow]) -> str:
    """The draw's digest, order-independent: names sorted, then hashed."""
    return catalogue_digest(sorted(chosen, key=lambda row: row.name))


def test_catalogue_enumeration_is_pinned(rows: List[OptionRow]) -> None:
    """The canonical order is exactly what the digest records.

    Re-recording `CATALOGUE_DIGEST` redraws the random family and moves both
    grammar prefixes, so every prior option run is invalidated by it. Change the
    enumeration only if you are willing to discard those runs.
    """
    assert len(rows) == CATALOGUE_SIZE
    counts = {group: 0 for group in GROUP_SIZES}
    for row in rows:
        counts[row.group] += 1
    assert counts == GROUP_SIZES
    assert catalogue_digest(rows) == CATALOGUE_DIGEST, (
        "the enumeration order moved; current names are:\n"
        + "\n".join(row.name for row in rows)
    )


def test_movement_rows_carry_the_reach_ladder(rows: List[OptionRow]) -> None:
    """`reach` is the repeat count on a movement row and zero elsewhere.

    Both sort keys read `reach`, so a row that reports the wrong one is placed
    wrongly in every family without changing the catalogue's size.
    """
    reaches = {row.reach for row in rows if row.group == GROUP_MOVE}
    assert reaches == set(MOVE_REPEATS)
    assert all(row.reach == 0 for row in rows if row.group != GROUP_MOVE)


def test_random_draw_is_pinned(rows: List[OptionRow]) -> None:
    """Each option seed draws the set it has always drawn.

    `random.sample` reads its input positionally, so this fails if the
    name-sorted order changes for any reason, including a renamed row.
    """
    for option_seed, expected in RANDOM_DIGESTS.items():
        chosen = random_options(DRAW_SIZE, rows, option_seed)
        assert len(chosen) == DRAW_SIZE
        assert digest_by_name(chosen) == expected, (
            f"option_seed {option_seed} now draws:\n"
            + "\n".join(sorted(row.name for row in chosen))
        )


def test_grammar_draws_are_pinned(rows: List[OptionRow]) -> None:
    """Both grammar catalogues at n=64 are the ones exp1, exp2 and exp4 ran.

    Pinned for the same reason as the random draw: a sort-key edit moves a
    prefix without changing the catalogue it is taken from.
    """
    assert digest_by_name(grammar_options(DRAW_SIZE, rows)) == GRAMMAR_DIGEST
    assert (
        digest_by_name(grammar_depth_options(DRAW_SIZE, rows)) == GRAMMAR_DEPTH_DIGEST
    )


def test_the_breadth_prior_reaches_three_groups_at_eight_rows(
    rows: List[OptionRow],
) -> None:
    """`grammar` at n=8 can descend and open a door, not only walk.

    This is why `grammar` is the breadth-first prior: it makes exp2's small n a
    statement about catalogue size rather than about capability.
    """
    assert [row.name for row in grammar_options(8, rows)] == GRAMMAR_PREFIX_AT_EIGHT


def test_the_depth_prior_holds_no_interaction_below_its_movement_block(
    rows: List[OptionRow],
) -> None:
    """`grammar_depth` places every movement row before every command row.

    Asserts the floor itself, not a symptom of it: exp3 compares the two priors
    at n=64, above the floor, and this records where the floor is so the
    write-up's claim about small n under a depth-first prior is checkable.
    """
    ordered = grammar_depth_options(len(rows), rows)
    first_command = next(
        index for index, row in enumerate(ordered) if row.group != GROUP_MOVE
    )
    assert first_command == FIRST_INTERACTION_DEPTH
    assert all(row.group == GROUP_MOVE for row in ordered[:first_command])


def test_every_family_returns_the_whole_catalogue_at_full_size(
    rows: List[OptionRow],
) -> None:
    """At n equal to the catalogue size the families coincide.

    exp2's last n is the full catalogue, so its top point must not be a
    family-dependent draw; this is the same property that makes exp3 run at
    n=64 rather than at 188.
    """
    everything = {row.name for row in rows}
    for chosen in (
        grammar_options(len(rows), rows),
        grammar_depth_options(len(rows), rows),
        random_options(len(rows), rows, 0),
    ):
        assert {row.name for row in chosen} == everything


def test_asking_for_more_options_than_the_catalogue_holds_is_an_error(
    rows: List[OptionRow],
) -> None:
    """A reachable mistake, since exp2's n axis is written by hand.

    Raised in `make_options` rather than in the families, so all three report it
    the same way.
    """
    env = gym.make(
        ENV_ID, observation_keys=OBSERVATION_KEYS, max_episode_steps=SHORT_EPISODE
    )
    with pytest.raises(ValueError, match="catalogue rows"):
        make_options(env.unwrapped.actions, "option", len(rows) + 1, "grammar", 0)
    env.close()


def test_the_conceding_commands_are_absent_from_every_condition() -> None:
    """No condition can select `QUIT` or `SAVE`.

    `QUIT` then `y` ends an episode in two primitive steps for reward 0, which
    beats any trajectory whose future score does not cover the time penalty.
    They were reachable in `action` and `both` but never in `option`, because no
    catalogue row emits either key, so leaving them in gave two of the three
    conditions an exploit the third could not use — a confound pointing the same
    way as the hypothesis.
    """
    env = gym.make(
        ENV_ID, observation_keys=OBSERVATION_KEYS, max_episode_steps=SHORT_EPISODE
    )
    actions = env.unwrapped.actions
    tables = {
        condition: make_options(actions, condition, DRAW_SIZE, "grammar", 0)
        for condition in ("action", "option", "both")
    }
    env.close()

    assert len(actions) == ENV_ACTION_SET, "the env's own action set is untouched"
    banned = {command.name.lower() for command in CONCEDING_COMMANDS}
    for condition, (_, names, _) in tables.items():
        assert not banned & set(names), f"{condition} can still concede"
    assert len(tables["action"][1]) == ACTION_TABLE
    assert len(tables["both"][1]) == ACTION_TABLE + DRAW_SIZE


def test_no_new_action_ends_the_episode_within_two_steps() -> None:
    """Only the recorded escapes concede, so a new route cannot appear unnoticed.

    Asserts the property rather than the absence of two named commands: this is
    how `up` was found, after `QUIT` and `SAVE` had been removed. Each row is
    played from a fresh game and followed by northwest movement, which is the
    keystroke that answers a yes/no prompt.

    Subset rather than equality, because the character is random and a two-step
    death by misfortune would otherwise fail the run;
    `test_the_recorded_escape_still_exists` covers the other direction.
    """
    env = gym.make(
        ENV_ID, observation_keys=OBSERVATION_KEYS, max_episode_steps=SHORT_EPISODE
    )
    actions = env.unwrapped.actions
    sequences, names, _ = make_options(actions, "action", DRAW_SIZE, "grammar", 0)
    confirm = actions.index(nethack.CompassDirection.NW)

    conceded = []
    for sequence, name in zip(sequences, names):
        env.reset()
        _, _, terminated, _, _ = env.step(sequence[0])
        if not terminated:
            _, _, terminated, _, _ = env.step(confirm)
        if terminated:
            conceded.append(name)
    env.close()

    unrecorded = set(conceded) - set(KNOWN_TURN_ONE_ESCAPES)
    assert not unrecorded, (
        f"{sorted(unrecorded)} end the episode within two steps and are not in "
        "KNOWN_TURN_ONE_ESCAPES. Either the table regained a conceding command, "
        "or a new free termination exists and has to be decided on"
    )


def test_the_recorded_escape_still_exists() -> None:
    """`up` then northwest still concedes, so the recorded exception is live.

    Without this, an NLE change that closed the route would leave
    `KNOWN_TURN_ONE_ESCAPES` silently permitting something impossible, and the
    write-up would keep reporting a confound that no longer applies.
    """
    env = gym.make(
        ENV_ID, observation_keys=OBSERVATION_KEYS, max_episode_steps=SHORT_EPISODE
    )
    actions = env.unwrapped.actions
    env.reset()
    _, _, terminated, _, _ = env.step(actions.index(nethack.MiscDirection.UP))
    assert not terminated, "the climb prompt should not end the episode on its own"
    _, reward, terminated, _, info = env.step(
        actions.index(nethack.CompassDirection.NW)
    )
    env.close()

    assert terminated, "up then northwest should escape the dungeon"
    assert reward == 0.0, "the escape pays nothing, which is what makes it an exploit"
    assert "up" in KNOWN_TURN_ONE_ESCAPES


def test_both_is_the_primitive_table_followed_by_the_options(
    rows: List[OptionRow],
) -> None:
    """An action index means the same thing under `action` and under `both`.

    Cross-condition comparability rests on this: the two conditions share a
    prefix of the action table, so a logged action index is one quantity. The
    prefix is the table's rows, not the env's action indices, which differ from
    them wherever a conceding command was dropped.
    """
    env = gym.make(
        ENV_ID, observation_keys=OBSERVATION_KEYS, max_episode_steps=SHORT_EPISODE
    )
    actions = env.unwrapped.actions
    primitive_sequences, primitive_names, _ = make_options(
        actions, "action", DRAW_SIZE, "grammar", 0
    )
    both_sequences, both_names, _ = make_options(
        actions, "both", DRAW_SIZE, "grammar", 0
    )
    option_sequences, _, _ = make_options(actions, "option", DRAW_SIZE, "grammar", 0)
    env.close()

    count = len(primitive_names)
    assert both_sequences[:count] == primitive_sequences
    assert both_names[:count] == primitive_names
    assert both_sequences[count:] == option_sequences
