# SPDX-License-Identifier: AGPL-3.0-or-later

import os
import sys

from fastapi import Depends, FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .auth import get_configured_api_key, require_api_key
from .routes import router

CORS_ORIGINS_ENV_VAR = "HERETIC_CORS_ORIGINS"


def _cors_origins() -> list[str]:
    """Returns allowed CORS origins.

    Configured via the ``HERETIC_CORS_ORIGINS`` environment variable as a
    comma-separated list. Defaults to ``["*"]`` (allow all) for convenience in
    trusted/local deployments.
    """

    raw = os.environ.get(CORS_ORIGINS_ENV_VAR, "").strip()
    if not raw:
        return ["*"]
    return [origin.strip() for origin in raw.split(",") if origin.strip()]


def create_app() -> FastAPI:
    for stream in (sys.stdout, sys.stderr):
        if (
            hasattr(stream, "reconfigure")
            and (getattr(stream, "encoding", "") or "").lower() != "utf-8"
        ):
            stream.reconfigure(encoding="utf-8")  # ty:ignore[call-non-callable]

    app = FastAPI(
        title="Heretic API",
        description="REST API for Heretic - automatic censorship removal for language models",
        version="1.4.0",
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # Authentication is header-based (Bearer token), not cookie-based, so
    # credentialed CORS is not needed. Enabling it alongside a wildcard origin
    # is both invalid per the CORS spec and a security risk, so it stays off.
    app.add_middleware(
        CORSMiddleware,  # ty:ignore[invalid-argument-type]
        allow_origins=_cors_origins(),
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # When an API key is configured, enforce Bearer authentication on every
    # HTTP route under the router. WebSocket routes are guarded separately
    # inside their handlers because HTTP dependencies do not run for them.
    dependencies = [Depends(require_api_key)] if get_configured_api_key() else []
    app.include_router(router, prefix="/api/v1", dependencies=dependencies)

    return app


app = create_app()
