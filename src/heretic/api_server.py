# SPDX-License-Identifier: AGPL-3.0-or-later

import argparse
import os

import uvicorn

from .api.auth import API_KEY_ENV_VAR, HF_TOKEN_ENV_VAR


def main() -> None:
    parser = argparse.ArgumentParser(description="Heretic API server")
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Host to bind the server to (default: 127.0.0.1)",
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help=(
            "If set, require clients to authenticate with an "
            "'Authorization: Bearer <key>' header matching this value. "
            "Can also be set via the HERETIC_API_KEY environment variable."
        ),
    )
    parser.add_argument(
        "--hf-token",
        default=None,
        help=(
            "Hugging Face access token. If set (here or via the HF_TOKEN "
            "environment variable), the server permits uploading exported "
            "models to the Hugging Face Hub. Without it, only local export is "
            "allowed."
        ),
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to bind the server to (default: 8000)",
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable auto-reload for development",
    )
    parser.add_argument(
        "--log-level",
        default="info",
        choices=["debug", "info", "warning", "error", "critical"],
        help="Logging level (default: info)",
    )

    args = parser.parse_args()

    # Propagate secrets via the environment so they survive into the app,
    # including the reload subprocess uvicorn spawns when --reload is used.
    # An explicit CLI flag takes precedence over a pre-existing env value.
    if args.api_key is not None:
        os.environ[API_KEY_ENV_VAR] = args.api_key
    if args.hf_token is not None:
        os.environ[HF_TOKEN_ENV_VAR] = args.hf_token

    print(f"Starting Heretic API server on http://{args.host}:{args.port}")
    print(f"API documentation available at http://{args.host}:{args.port}/docs")
    if os.environ.get(API_KEY_ENV_VAR):
        print("Bearer-token authentication is ENABLED.")
    else:
        print("Bearer-token authentication is DISABLED (no --api-key provided).")
    if os.environ.get(HF_TOKEN_ENV_VAR):
        print("Hugging Face uploads are ENABLED.")
    else:
        print("Hugging Face uploads are DISABLED (no --hf-token / HF_TOKEN).")

    uvicorn.run(
        "heretic.api.app:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level=args.log_level,
    )


if __name__ == "__main__":
    main()
