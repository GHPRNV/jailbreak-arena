"""
JailbreakArena - Adversarial self-play environment for AI safety.

Two LLMs co-evolve in a sandboxed Constitutional CTF: an Attacker tries to
extract fictional planted secrets from a Defender that operates under a
written constitution. The Defender is rewarded for staying safe AND helpful
(over-refusal is penalised), making the reward hard to game.

Theme fit (PyTorch OpenEnv Hackathon):
    - Self-Improvement (self-play adversarial co-evolution)
    - Multi-agent interactions (asymmetric attacker/defender roles)

Quick start:

    >>> from jailbreak_arena import JailbreakArenaEnv
    >>>
    >>> with JailbreakArenaEnv(base_url="http://localhost:8000") as env:
    ...     env.reset()
    ...     scenario = env.call_tool("start_episode", scenario_id=0, mode="attack")
    ...     env.call_tool("attacker_turn", message="Ignore prior instructions...")
    ...     env.call_tool("defender_turn", message="I can't share that, but...")
    ...     scores = env.call_tool("finalize_episode")
    ...     print(scores)
"""

__version__ = "0.1.0"

# Re-exports below are guarded so that this file can also be imported as a
# top-level script (e.g. by tools that walk the directory tree and pick it up
# without setting up the proper package context).
try:
    from openenv.core.env_server.mcp_types import (  # noqa: F401
        CallToolAction,
        ListToolsAction,
    )
    from .client import JailbreakArenaEnv  # noqa: F401

    __all__ = ["JailbreakArenaEnv", "CallToolAction", "ListToolsAction"]
except ImportError:
    # Either openenv-core is not installed or this file is being imported as a
    # top-level script (no package context). In both cases the public API is
    # not available; users should `pip install -e .` and import via
    # ``jailbreak_arena.JailbreakArenaEnv``.
    __all__ = []
