"""Origin parsing shared by the browser backend and the tool validator."""

from __future__ import annotations

from urllib.parse import urlparse

_DEFAULT_PORTS = {"http": 80, "https": 443}


def origin_of(url: str) -> str | None:
    """
    "scheme://host:port" for an http(s) URL, with the default port filled in.

    Returns None for anything else (other schemes, no host, embedded
    credentials, an invalid port). Never raises.
    """

    try:
        parsed = urlparse(url)
        port = parsed.port
    except ValueError:
        return None

    if (
        parsed.scheme not in _DEFAULT_PORTS
        or not parsed.hostname
        or parsed.username
        or parsed.password
    ):
        return None

    return f"{parsed.scheme}://{parsed.hostname}:{port or _DEFAULT_PORTS[parsed.scheme]}"