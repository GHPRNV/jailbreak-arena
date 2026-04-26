"""
JailbreakArena Environment Implementation.

A pure MCP environment that hosts adversarial Attacker/Defender dialogues.
The environment is *stateless about model weights* — model inference happens
in the trainer/client. The environment is responsible for:

    - Sampling scenarios (sandboxed Constitutional CTF).
    - Tracking dialogue state (turns, role, transcript).
    - Scoring messages with a composable rubric (see ``rubric.py``).
    - Emitting per-turn rewards and an aggregated end-of-episode score.

Tools exposed:
    list_scenarios()                          – inspect the scenario catalogue
    get_scenario(scenario_id)                 – inspect one scenario
    sample_attack(seed)                       – sample an attack template
                                                (useful for warmup / eval)
    start_episode(scenario_id, mode, seed)    – begin a new episode
    attacker_turn(message)                    – submit attacker text
    defender_turn(message)                    – submit defender text
    finalize_episode()                        – score the dialogue
    constitution()                            – read the defender constitution

Episode modes:
    "attack"  – full adversarial dialogue: alternating attacker/defender.
    "benign"  – over-refusal probe: a single benign question is asked and the
                defender must respond helpfully.
"""

from __future__ import annotations

import os
import random
from copy import deepcopy
from typing import Any, Optional
from uuid import uuid4

try:
    from openenv.core.env_server.mcp_environment import MCPEnvironment
    from openenv.core.env_server.types import Action, Observation, State
except ImportError:
    from openenv.core.env_server.mcp_environment import MCPEnvironment
    from openenv.core.env_server.types import Action, Observation, State

from fastmcp import FastMCP

try:
    from .rubric import (
        AttackerTurnScore,
        DefenderTurnScore,
        finalise_episode,
        score_attacker_message,
        score_defender_response,
    )
    from .scenarios import (
        build_attacker_system_prompt,
        build_defender_system_prompt,
        flatten_attack_templates,
        held_out_attacks,
        load_attack_pool,
        load_constitution_text,
        load_scenarios,
        sample_attack as sample_attack_template,
    )
except ImportError:  # When running from the standalone /app/env layout.
    from server.rubric import (  # type: ignore[no-redef]
        AttackerTurnScore,
        DefenderTurnScore,
        finalise_episode,
        score_attacker_message,
        score_defender_response,
    )
    from server.scenarios import (  # type: ignore[no-redef]
        build_attacker_system_prompt,
        build_defender_system_prompt,
        flatten_attack_templates,
        held_out_attacks,
        load_attack_pool,
        load_constitution_text,
        load_scenarios,
        sample_attack as sample_attack_template,
    )


DEFAULT_MAX_TURNS = int(os.getenv("JBA_MAX_TURNS", "4"))


class _EpisodeState:
    """Per-episode mutable state. Wrapped in a small object so it's easy to
    snapshot for logging/debugging."""

    def __init__(self) -> None:
        self.scenario: Optional[dict[str, Any]] = None
        self.scenario_id: int = -1
        self.mode: str = "attack"  # "attack" | "benign"
        self.max_turns: int = DEFAULT_MAX_TURNS
        self.turn: int = 0
        self.expected_role: str = "attacker"  # "attacker" | "defender"
        self.transcript: list[dict[str, Any]] = []
        self.attacker_messages: list[str] = []
        self.defender_messages: list[str] = []
        self.attacker_scores: list[AttackerTurnScore] = []
        self.defender_scores: list[DefenderTurnScore] = []
        self.benign_probe_text: Optional[str] = None
        self.benign_probe_score: Optional[DefenderTurnScore] = None
        self.done: bool = False
        self.finalised: bool = False
        self.last_episode_result: Optional[dict[str, Any]] = None

    def snapshot(self) -> dict[str, Any]:
        return {
            "scenario_id": self.scenario_id,
            "mode": self.mode,
            "turn": self.turn,
            "max_turns": self.max_turns,
            "expected_role": self.expected_role,
            "transcript": deepcopy(self.transcript),
            "done": self.done,
            "finalised": self.finalised,
        }


class JailbreakArenaEnvironment(MCPEnvironment):
    """MCP environment that hosts attacker/defender dialogues."""

    SUPPORTS_CONCURRENT_SESSIONS = True

    def __init__(self) -> None:
        scenarios = load_scenarios()
        attack_pool = load_attack_pool()
        attack_templates = flatten_attack_templates(attack_pool)
        constitution_text = load_constitution_text()

        ep = _EpisodeState()

        mcp = FastMCP("jailbreak_arena")

        # --- Scenario inspection tools ----------------------------------- #

        @mcp.tool
        def list_scenarios() -> list[dict[str, Any]]:
            """Return all available scenarios as a brief summary list.

            Each entry contains: id, title, domain, attacker_goal, and
            number_of_secrets. Use `get_scenario(id)` for the full payload.
            """
            return [
                {
                    "id": s["id"],
                    "title": s["title"],
                    "domain": s["domain"],
                    "attacker_goal": s.get("attacker_goal", ""),
                    "n_secrets": len(s.get("secrets", [])),
                }
                for s in scenarios
            ]

        @mcp.tool
        def get_scenario(scenario_id: int) -> dict[str, Any]:
            """Return the full scenario payload for inspection.

            Includes the cover story, secrets, off-topic refusals, attacker
            goal, and benign probes. Useful for debugging and for trainers
            that want to construct their own prompts.
            """
            for s in scenarios:
                if s["id"] == scenario_id:
                    return s
            raise ValueError(f"No scenario with id={scenario_id}")

        @mcp.tool
        def constitution() -> str:
            """Return the Defender constitution markdown text."""
            return constitution_text

        @mcp.tool
        def attack_pool_categories() -> list[str]:
            """Return the names of all attack-pool categories."""
            return [c["name"] for c in attack_pool.get("categories", [])]

        @mcp.tool
        def held_out_eval_set() -> list[dict[str, str]]:
            """Return the held-out evaluation attack set (used for fair
            before/after benchmarking)."""
            return held_out_attacks(attack_pool)

        @mcp.tool
        def sample_attack(seed: int = -1) -> dict[str, str]:
            """Sample one attack template, optionally with a fixed seed.

            Returns a dict with `category` and `template`. If `seed >= 0`,
            sampling is reproducible.
            """
            rng = random.Random(seed) if seed >= 0 else random.Random()
            cat, template = sample_attack_template(attack_pool, rng=rng)
            return {"category": cat, "template": template}

        # --- Episode control tools --------------------------------------- #

        @mcp.tool
        def start_episode(
            scenario_id: int = 0,
            mode: str = "attack",
            max_turns: int = DEFAULT_MAX_TURNS,
            seed: int = -1,
        ) -> dict[str, Any]:
            """Begin a new episode.

            Args:
                scenario_id: Index into the scenario catalogue.
                mode: "attack" (full adversarial loop) or "benign" (single
                    over-refusal probe).
                max_turns: Cap on the number of attacker/defender turn pairs.
                seed: Optional seed for reproducible benign-probe selection.

            Returns:
                Dict with the defender system prompt, attacker system prompt,
                attacker goal, and (if mode="benign") the chosen benign probe.
            """
            sc = next((s for s in scenarios if s["id"] == scenario_id), None)
            if sc is None:
                raise ValueError(f"No scenario with id={scenario_id}")

            ep.scenario = sc
            ep.scenario_id = scenario_id
            ep.mode = mode
            ep.max_turns = max_turns
            ep.turn = 0
            ep.expected_role = "attacker" if mode == "attack" else "defender"
            ep.transcript = []
            ep.attacker_messages = []
            ep.defender_messages = []
            ep.attacker_scores = []
            ep.defender_scores = []
            ep.benign_probe_score = None
            ep.benign_probe_text = None
            ep.done = False
            ep.finalised = False
            ep.last_episode_result = None

            payload: dict[str, Any] = {
                "scenario_id": scenario_id,
                "scenario_title": sc["title"],
                "mode": mode,
                "max_turns": max_turns,
                "expected_role": ep.expected_role,
                "defender_system_prompt": build_defender_system_prompt(sc),
                "attacker_system_prompt": build_attacker_system_prompt(sc),
                "attacker_goal": sc.get("attacker_goal", ""),
            }

            if mode == "benign":
                rng = random.Random(seed) if seed >= 0 else random.Random()
                probes = sc.get("benign_probes", [])
                ep.benign_probe_text = rng.choice(probes) if probes else (
                    "Hi! Could you give me a quick overview of what you can help with?"
                )
                payload["benign_probe"] = ep.benign_probe_text
                ep.transcript.append({
                    "role": "user",
                    "content": ep.benign_probe_text,
                })

            return payload

        @mcp.tool
        def attacker_turn(message: str) -> dict[str, Any]:
            """Submit an attacker message.

            The environment scores the message for novelty (vs the attack
            pool) and repetition (vs prior attacker messages this episode).
            Real reward is finalised at episode end.
            """
            if ep.scenario is None:
                raise RuntimeError("No active episode. Call start_episode first.")
            if ep.done:
                raise RuntimeError("Episode is over. Call finalize_episode or start_episode.")
            if ep.mode != "attack":
                raise RuntimeError(
                    f"attacker_turn is only valid in mode='attack'; current mode={ep.mode!r}"
                )
            if ep.expected_role != "attacker":
                raise RuntimeError(
                    f"Expected defender turn, got attacker_turn (turn {ep.turn})"
                )

            score = score_attacker_message(
                message=message,
                prior_attacker_messages=list(ep.attacker_messages),
                attack_pool_templates=attack_templates,
            )

            ep.attacker_messages.append(message)
            ep.attacker_scores.append(score)
            ep.transcript.append({
                "role": "attacker",
                "content": message,
                "turn": ep.turn,
                "novelty": score.novelty,
                "repetition_penalty": score.repetition_penalty,
            })
            ep.expected_role = "defender"

            return {
                "ok": True,
                "expected_role": ep.expected_role,
                "novelty": score.novelty,
                "repetition_penalty": score.repetition_penalty,
                "turn": ep.turn,
                "done": ep.done,
            }

        @mcp.tool
        def defender_turn(message: str) -> dict[str, Any]:
            """Submit a defender response.

            The environment immediately scores the response against the
            constitution (leak detection, refusal classification, polish,
            offered alternative, character break). Returns the per-turn
            component breakdown so trainers can shape per-turn signals.
            """
            if ep.scenario is None:
                raise RuntimeError("No active episode. Call start_episode first.")
            if ep.done:
                raise RuntimeError("Episode is over.")
            if ep.expected_role != "defender":
                raise RuntimeError(
                    f"Expected attacker turn, got defender_turn (turn {ep.turn})"
                )

            scenario = ep.scenario
            secrets = scenario.get("secrets", [])

            if ep.mode == "benign":
                # The single benign exchange.
                turn_score = score_defender_response(
                    response=message,
                    secrets=secrets,
                    mode="benign",
                )
                ep.benign_probe_score = turn_score
                ep.defender_messages.append(message)
                ep.transcript.append({
                    "role": "defender",
                    "content": message,
                    "score": turn_score.to_dict(),
                })
                ep.done = True
                ep.expected_role = "none"
                return {
                    "ok": True,
                    "turn": 0,
                    "done": ep.done,
                    "score": turn_score.to_dict(),
                }

            # Attack mode — normal flow.
            turn_score = score_defender_response(
                response=message,
                secrets=secrets,
                mode="attack",
            )
            ep.defender_messages.append(message)
            ep.defender_scores.append(turn_score)
            ep.transcript.append({
                "role": "defender",
                "content": message,
                "turn": ep.turn,
                "score": turn_score.to_dict(),
            })

            ep.turn += 1

            # End-of-episode conditions:
            #   - Defender leaked.
            #   - Reached max_turns.
            if turn_score.leak or ep.turn >= ep.max_turns:
                ep.done = True
                ep.expected_role = "none"
            else:
                ep.expected_role = "attacker"

            return {
                "ok": True,
                "turn": ep.turn,
                "done": ep.done,
                "score": turn_score.to_dict(),
                "expected_role": ep.expected_role,
            }

        @mcp.tool
        def finalize_episode() -> dict[str, Any]:
            """Aggregate per-turn scores into final attacker/defender rewards.

            Safe to call multiple times (idempotent — caches the first result).
            Auto-runs if the episode timed out without an explicit finalize.
            """
            if ep.scenario is None:
                raise RuntimeError("No active episode.")
            if ep.finalised and ep.last_episode_result is not None:
                return ep.last_episode_result

            result = finalise_episode(
                defender_turn_scores=ep.defender_scores,
                attacker_turn_scores=ep.attacker_scores,
                benign_probe_score=ep.benign_probe_score,
            )

            ep.finalised = True
            ep.done = True
            ep.expected_role = "none"

            payload = {
                **result.to_dict(),
                "scenario_id": ep.scenario_id,
                "mode": ep.mode,
                "n_attacker_turns": len(ep.attacker_messages),
                "n_defender_turns": len(ep.defender_messages),
                "transcript": deepcopy(ep.transcript),
            }
            ep.last_episode_result = payload
            return payload

        @mcp.tool
        def episode_state() -> dict[str, Any]:
            """Read the current episode state (for debugging and tracing)."""
            return ep.snapshot()

        # Pass MCP server to base class
        super().__init__(mcp)
        self._ep = ep
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._reset_count = 0
        self._scenarios = scenarios
        self._attack_pool = attack_pool
        self._attack_templates = attack_templates
        self._constitution_text = constitution_text

    # ------------------------------------------------------------------ #
    # Standard Gym-style API expected by OpenEnv validators              #
    # ------------------------------------------------------------------ #

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        **kwargs: Any,
    ) -> Observation:
        """Reset the environment to a fresh state.

        For JailbreakArena, reset clears any in-flight episode but does NOT
        auto-start a new one — call `start_episode` next to pick a scenario.
        """
        self._state = State(
            episode_id=episode_id or str(uuid4()),
            step_count=0,
        )
        self._reset_count += 1

        # Clear any in-flight episode so a fresh start_episode is required.
        self._ep.scenario = None
        self._ep.scenario_id = -1
        self._ep.transcript = []
        self._ep.attacker_messages = []
        self._ep.defender_messages = []
        self._ep.attacker_scores = []
        self._ep.defender_scores = []
        self._ep.benign_probe_score = None
        self._ep.benign_probe_text = None
        self._ep.turn = 0
        self._ep.expected_role = "attacker"
        self._ep.done = False
        self._ep.finalised = False
        self._ep.last_episode_result = None

        return Observation(
            done=False,
            reward=0.0,
            metadata={
                "status": "ready",
                "message": "JailbreakArena ready. Call start_episode to begin.",
                "n_scenarios": len(self._scenarios),
            },
        )

    def _step_impl(
        self,
        action: Action,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> Observation:
        """Non-MCP fallback (returns a polite error)."""
        return Observation(
            done=False,
            reward=0.0,
            metadata={
                "error": (
                    f"Unknown action type: {type(action).__name__}. "
                    "JailbreakArena uses MCP tools — call start_episode, "
                    "attacker_turn, defender_turn, or finalize_episode via "
                    "ListToolsAction / CallToolAction."
                )
            },
        )

    def step(
        self,
        action: Action,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> Observation:
        self._state.step_count += 1
        return super().step(action, timeout_s=timeout_s, **kwargs)

    async def step_async(
        self,
        action: Action,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> Observation:
        self._state.step_count += 1
        return await super().step_async(action, timeout_s=timeout_s, **kwargs)

    @property
    def state(self) -> State:
        return self._state
