"""
Probes for the local shop.

ShopSessionProbe  - in-loop verifier. Reads only pages a user could open.
ShopOracleProbe   - evaluation oracle. Reads an authoritative snapshot.

The two never share a data path: the session probe goes over HTTP and parses
what the shop renders; the oracle probe reads a snapshot taken directly from
server memory. If the shop ever renders something different from what it
actually stored, the two will disagree, and that disagreement is exactly what
a false completion looks like.
"""

from __future__ import annotations

import urllib.error
import urllib.request
from collections.abc import Callable
from html.parser import HTMLParser
from urllib.parse import urlparse

from environments.base import Snapshot, validate_snapshot
from environments.sites.shop.server import PROBE_HEADER, SESSION_COOKIE
from verifier.probe import ProbeCapabilityError, ProbeError

_LOOPBACK_HOSTS = {"127.0.0.1", "localhost"}


class _CartPageParser(HTMLParser):
    """Extract {product_id: quantity} from the rendered /cart page."""

    def __init__(self) -> None:
        super().__init__()
        self.found_cart = False
        self.items: dict[str, int] = {}
        self._inside = False

    def handle_starttag(self, tag, attrs):
        attributes = dict(attrs)

        if tag == "ul" and attributes.get("data-testid") == "cart":
            self.found_cart = True
            self._inside = True
            return

        if tag == "li" and self._inside:
            product_id = attributes.get("data-product-id")
            quantity = attributes.get("data-quantity")

            if product_id is None or quantity is None:
                raise ProbeError(
                    "cart row is missing data-product-id/data-quantity"
                )

            try:
                parsed = int(quantity)
            except ValueError:
                raise ProbeError(
                    f"non-numeric cart quantity {quantity!r}"
                )

            self.items[product_id] = self.items.get(product_id, 0) + parsed

    def handle_endtag(self, tag):
        if tag == "ul" and self._inside:
            self._inside = False


def parse_cart_page(page: str) -> dict[str, int]:
    """Parse a rendered cart page. An empty cart is {}; no cart list is an error."""

    parser = _CartPageParser()
    parser.feed(page)
    parser.close()

    if not parser.found_cart:
        raise ProbeError("cart page has no cart list")

    return parser.items


class ShopSessionProbe:
    """
    Observable-state probe.

    It opens its own connection and presents the agent's session cookie, so
    it sees the same cart the agent's session sees, but it shares no client
    object with the agent and therefore cannot alter the agent's state
    (current URL, cookies, history).

    `current_url` is a callable supplied by the harness that returns the URL
    the agent is currently on (a scripted client's last URL, or a browser
    page's URL).
    """

    supported_checks = frozenset({"cart_contains", "url_matches"})

    def __init__(
        self,
        base_url: str,
        session_id: str,
        current_url: Callable[[], str],
        probe_token: str,
        timeout: float = 5.0,
    ) -> None:
        parsed = urlparse(base_url)

        if parsed.scheme != "http" or parsed.hostname not in _LOOPBACK_HOSTS:
            raise ValueError("ShopSessionProbe accepts only loopback HTTP origins")

        self._base_url = base_url.rstrip("/")
        self._session_id = session_id
        self._current_url = current_url
        self._probe_token = probe_token
        self._timeout = timeout

    def _get(self, path: str) -> str:
        request = urllib.request.Request(
            self._base_url + path,
            headers={
                "Cookie": f"{SESSION_COOKIE}={self._session_id}",
                PROBE_HEADER: self._probe_token,
            },
            method="GET",
        )

        try:
            with urllib.request.urlopen(request, timeout=self._timeout) as response:
                return response.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            raise ProbeError(f"GET {path} returned HTTP {exc.code}")
        except (urllib.error.URLError, TimeoutError) as exc:
            raise ProbeError(f"GET {path} failed: {exc}")

    def current_url(self) -> str:
        return self._current_url()

    def cart(self) -> dict[str, int]:
        return parse_cart_page(self._get("/cart"))

    def element_text(self, selector: str) -> str | None:
        raise ProbeCapabilityError(
            "ShopSessionProbe does not support element_text"
        )

    def submitted_form(self) -> dict[str, str] | None:
        raise ProbeCapabilityError(
            "ShopSessionProbe does not support form_submitted"
        )


class ShopOracleProbe:
    """
    Authoritative-state probe.

    Built from ONE validated snapshot, so every check in an oracle
    evaluation sees the same consistent state. Never constructed or
    reachable from agent code.
    """

    supported_checks = frozenset({"cart_contains", "url_matches"})

    def __init__(self, snapshot: Snapshot, final_url: str) -> None:
        self.snapshot: Snapshot = validate_snapshot(snapshot)
        self._final_url = final_url

    def current_url(self) -> str:
        return self._final_url

    def cart(self) -> dict[str, int]:
        return dict(self.snapshot["session"].get("cart", {}))

    def element_text(self, selector: str) -> str | None:
        raise ProbeCapabilityError(
            "ShopOracleProbe does not support element_text"
        )

    def submitted_form(self) -> dict[str, str] | None:
        raise ProbeCapabilityError(
            "ShopOracleProbe does not support form_submitted"
        )