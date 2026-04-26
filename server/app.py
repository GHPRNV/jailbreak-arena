"""
FastAPI application for the JailbreakArena Environment.

This module creates an HTTP/WebSocket server exposing the
JailbreakArenaEnvironment, compatible with the standard OpenEnv MCP client.

Run locally:
    uvicorn server.app:app --reload --host 0.0.0.0 --port 8000

Run via the OpenEnv CLI:
    openenv serve jailbreak_arena

Run inside the Docker container:
    docker run -p 8000:8000 jailbreak-arena:latest
"""

from __future__ import annotations

import os

try:
    from openenv.core.env_server.http_server import create_app
    from openenv.core.env_server.mcp_types import CallToolAction, CallToolObservation

    from .jailbreak_environment import JailbreakArenaEnvironment
except ImportError:
    from openenv.core.env_server.http_server import create_app
    from openenv.core.env_server.mcp_types import CallToolAction, CallToolObservation
    from server.jailbreak_environment import JailbreakArenaEnvironment

max_concurrent = int(os.getenv("MAX_CONCURRENT_ENVS", "8"))

app = create_app(
    JailbreakArenaEnvironment,
    CallToolAction,
    CallToolObservation,
    env_name="jailbreak_arena",
    max_concurrent_envs=max_concurrent,
)


def main() -> None:
    """Entry point so users can run `uv run --project . server` or
    `python -m jailbreak_arena.server.app`."""
    import uvicorn

    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
