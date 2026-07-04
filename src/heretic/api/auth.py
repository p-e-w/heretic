# SPDX-License-Identifier: AGPL-3.0-or-later

"""Optional Bearer-token authentication for the Heretic API.

Authentication is enabled by setting the ``HERETIC_API_KEY`` environment
variable (the ``heretic-api`` CLI does this when ``--api-key`` is passed).
When enabled, every request must present an ``Authorization: Bearer <key>``
header whose value matches the configured key; otherwise the request is
rejected with HTTP 401.

When no key is configured, authentication is disabled and all requests are
allowed, preserving the previous behaviour.
"""

import hmac
import os

from fastapi import HTTPException, WebSocket, status
from starlette.requests import HTTPConnection

API_KEY_ENV_VAR = "HERETIC_API_KEY"

# Hugging Face reads the token from HF_TOKEN by convention. The server only
# permits uploads to the Hub when a token is available here.
HF_TOKEN_ENV_VAR = "HF_TOKEN"


def get_configured_api_key() -> str | None:
    """Returns the configured API key, or ``None`` if authentication is off."""

    key = os.environ.get(API_KEY_ENV_VAR)
    # An empty string is treated as "not configured" so that an accidental
    # empty value does not silently enable an unauthenticatable server.
    return key or None


def get_hf_token() -> str | None:
    """Returns the Hugging Face token, or ``None`` if uploads are disabled.

    The token is sourced from the ``HF_TOKEN`` environment variable (which the
    ``heretic-api`` CLI also populates from ``--hf-token``). When no token is
    configured, the server refuses uploads to the Hub; local export is always
    permitted.
    """

    token = os.environ.get(HF_TOKEN_ENV_VAR)
    return token or None


def _extract_bearer_token(authorization: str | None) -> str | None:
    if not authorization:
        return None

    scheme, _, credentials = authorization.partition(" ")
    if scheme.lower() != "bearer" or not credentials:
        return None

    return credentials


def _is_authorized(authorization: str | None) -> bool:
    expected = get_configured_api_key()
    if expected is None:
        # Authentication disabled.
        return True

    token = _extract_bearer_token(authorization)
    if token is None:
        return False

    # Constant-time comparison to avoid leaking the key via timing.
    return hmac.compare_digest(token, expected)


async def require_api_key(connection: HTTPConnection) -> None:
    """FastAPI dependency enforcing Bearer authentication.

    A router-level dependency is applied to every route under the router,
    including WebSocket routes. ``HTTPConnection`` is the common base of both
    ``Request`` and ``WebSocket``, so this resolves correctly in both cases.

    WebSocket connections are intentionally not rejected here (raising an
    ``HTTPException`` during a WebSocket handshake does not produce a clean
    close); they are authenticated explicitly inside the WebSocket handler
    via :func:`websocket_authorized`.
    """

    if connection.scope.get("type") != "http":
        return

    if not _is_authorized(connection.headers.get("Authorization")):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key.",
            headers={"WWW-Authenticate": "Bearer"},
        )


async def websocket_authorized(websocket: WebSocket) -> bool:
    """Returns whether a WebSocket connection presents valid credentials.

    WebSocket handshakes do not run HTTP route dependencies, so this must be
    called explicitly from the WebSocket handler. Browsers cannot set custom
    headers on WebSocket connections, so a ``token`` query parameter is also
    accepted as a fallback.
    """

    if get_configured_api_key() is None:
        return True

    authorization = websocket.headers.get("Authorization")
    if authorization is None:
        token = websocket.query_params.get("token")
        if token is not None:
            authorization = f"Bearer {token}"

    return _is_authorized(authorization)
