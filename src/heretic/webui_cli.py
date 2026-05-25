# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025-2026  Philipp Emanuel Weidmann <pew@worldwidemann.com> + contributors

"""Lightweight CLI entry point for the Heretic web UI."""

import argparse
import sys


def main() -> None:
    """CLI entry point for ``heretic-webui``."""
    parser = argparse.ArgumentParser(
        prog="heretic-webui",
        description="Launch the Heretic web UI",
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Server host (default: 127.0.0.1)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=7860,
        help="Server port (default: 7860)",
    )
    parser.add_argument(
        "--share",
        action="store_true",
        help="Create a temporary public share link via Gradio's tunnel",
    )
    args = parser.parse_args()

    try:
        from .webui import WEBUI_CSS, create_app
    except ImportError as exc:
        print(
            "The web UI dependencies are not available. Install them with "
            "`pip install heretic-llm[webui]`.",
            file=sys.stderr,
        )
        raise SystemExit(1) from exc

    try:
        app = create_app()
    except RuntimeError as exc:
        print(exc, file=sys.stderr)
        raise SystemExit(1) from exc
    app.queue()
    app.launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        css=WEBUI_CSS,
    )
