"""
Local shop site for VeriPilot.

A tiny deterministic web shop served on a loopback port. One
`ShopEnvironment` is created per benchmark run, so every run starts from a
fresh state.

Routes
------
GET  /, /products   product list with an "Add to cart" form per product
GET  /cart          the session's cart (what a user sees)
POST /cart/add      add `quantity` of `product_id` to the session's cart
                    (303 redirect to /cart on success)

There is deliberately NO JSON/API endpoint exposing the cart. Anything the
in-loop verifier can learn, it must learn from pages a user could see.
Authoritative state is exposed only through `ShopEnvironment.snapshot()`,
which is for the evaluation oracle and is never reachable over HTTP.

Faults
------
Every request is registered with the FaultInjector as the event
"<METHOD> <path>". If a rule fires:

    fail_before_commit  -> error response, state untouched
    fail_after_commit   -> the request is processed normally (state
                           changes), THEN the error response is sent

(`delay_seconds` delays the response, so a short client timeout turns a
committed request into a timeout with unknown outcome.)

Requests carrying the harness probe token in `X-VP-Probe` bypass the
injector entirely. The token is random per environment and is known only to
the harness; the agent never sees it.
"""

from __future__ import annotations

import hmac
import html
import http.server
import secrets
import threading
import time
from dataclasses import dataclass
from urllib.parse import parse_qs, urlparse

from environments.base import Snapshot
from environments.faults import FaultInjector, FaultRule

PROBE_HEADER = "X-VP-Probe"
SESSION_COOKIE = "vp_session"

MAX_QUANTITY = 99
MAX_BODY_BYTES = 64 * 1024

_LOOPBACK_HOSTS = {"127.0.0.1", "localhost"}

PRODUCTS = {
    "Laptop_A": {"name": "Laptop A", "price": 899},
    "Laptop_B": {"name": "Laptop B", "price": 949},
    "Mouse_A": {"name": "Mouse A", "price": 29},
}


# --------------------------------------------------------------------------
# Authoritative state
# --------------------------------------------------------------------------


class ShopState:
    """Server-side truth. All access goes through a lock."""

    def __init__(self) -> None:
        self.carts: dict[str, dict[str, int]] = {}
        self._lock = threading.RLock()

    def ensure_session(self, session_id: str) -> None:
        with self._lock:
            self.carts.setdefault(session_id, {})

    def add_to_cart(
        self,
        session_id: str,
        product_id: str,
        quantity: int,
    ) -> None:
        if product_id not in PRODUCTS:
            raise ValueError("unknown product")

        if not 1 <= quantity <= MAX_QUANTITY:
            raise ValueError(
                f"quantity must be between 1 and {MAX_QUANTITY}"
            )

        with self._lock:
            cart = self.carts.setdefault(session_id, {})
            cart[product_id] = cart.get(product_id, 0) + quantity

    def get_cart(self, session_id: str) -> dict[str, int]:
        with self._lock:
            return dict(self.carts.get(session_id, {}))

    def snapshot(self, session_id: str) -> Snapshot:
        with self._lock:
            cart = self.carts.get(session_id, {})

            return {
                "session": {"cart": dict(sorted(cart.items()))},
                # The shop has no cross-session state yet. The key exists
                # because the Environment contract requires it.
                "shared": {},
            }


# --------------------------------------------------------------------------
# Rendering (pure functions)
# --------------------------------------------------------------------------


def render_products() -> str:
    items = []

    for product_id, product in PRODUCTS.items():
        items.append(
            f"""
            <article data-product-id="{html.escape(product_id)}">
              <h2>{html.escape(product["name"])}</h2>
              <p>${product["price"]}</p>

              <form method="post" action="/cart/add">
                <input type="hidden" name="product_id"
                       value="{html.escape(product_id)}">
                <input name="quantity" value="1">
                <button type="submit">Add to cart</button>
              </form>
            </article>
            """
        )

    return (
        "<!doctype html><html><head>"
        "<title>VeriPilot Local Shop</title></head><body>"
        "<h1>VeriPilot Local Shop</h1>"
        '<a href="/cart">Cart</a>'
        + "".join(items)
        + "</body></html>"
    )


def render_cart(cart: dict[str, int]) -> str:
    rows = []

    for product_id, quantity in cart.items():
        rows.append(
            f"""
            <li data-product-id="{html.escape(product_id)}"
                data-quantity="{quantity}">
              {html.escape(PRODUCTS[product_id]["name"])} x {quantity}
            </li>
            """
        )

    return (
        "<!doctype html><html><body><h1>Your Cart</h1>"
        '<a href="/products">Products</a>'
        '<ul data-testid="cart">'
        + "".join(rows)
        + "</ul></body></html>"
    )


# --------------------------------------------------------------------------
# HTTP layer
# --------------------------------------------------------------------------

_HTML = "text/html; charset=utf-8"
_TEXT = "text/plain; charset=utf-8"


@dataclass(frozen=True)
class _Response:
    status: int
    body: str = ""
    content_type: str = _HTML
    location: str | None = None


class _BadRequest(Exception):
    pass


class ShopEnvironment:
    """Implements `environments.base.Environment`."""

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 0,
        state: ShopState | None = None,
        fault_injector: FaultInjector | None = None,
    ) -> None:
        if host not in _LOOPBACK_HOSTS:
            raise ValueError(
                "ShopEnvironment binds to loopback addresses only"
            )

        self.host = host
        self.port = port
        self.state = state if state is not None else ShopState()
        self.fault_injector = (
            fault_injector
            if fault_injector is not None
            else FaultInjector()
        )

        # Known only to the harness. Requests that present it bypass
        # fault injection (used by verifier probes).
        self.probe_token = secrets.token_hex(16)

        # Unexpected exceptions inside request handling. A non-empty list
        # means a harness bug, not an injected fault; runners assert on it.
        self.internal_errors: list[str] = []

        self._server: http.server.ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None

    # -- lifecycle ---------------------------------------------------------

    def start(self) -> "ShopEnvironment":
        if self._server is not None:
            raise RuntimeError("environment already started")

        handler = type(
            "BoundShopHandler",
            (ShopRequestHandler,),
            {"environment": self},
        )

        self._server = http.server.ThreadingHTTPServer(
            (self.host, self.port),
            handler,
        )
        self.port = self._server.server_address[1]

        self._thread = threading.Thread(
            target=self._server.serve_forever,
            daemon=True,
        )
        self._thread.start()

        return self

    def stop(self) -> None:
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
            self._server = None

        if self._thread is not None:
            self._thread.join(timeout=2)
            self._thread = None

    def __enter__(self) -> "ShopEnvironment":
        return self.start()

    def __exit__(self, exc_type, exc, tb) -> None:
        self.stop()

    @property
    def base_url(self) -> str:
        if self._server is None:
            raise RuntimeError("environment is not started")

        return f"http://{self.host}:{self.port}"

    # -- oracle access -----------------------------------------------------

    def snapshot(self, session_id: str) -> Snapshot:
        """Authoritative state. Oracle use only; never served over HTTP."""
        return self.state.snapshot(session_id)


class ShopRequestHandler(http.server.BaseHTTPRequestHandler):
    environment: ShopEnvironment

    def log_message(self, format: str, *args) -> None:
        return

    def do_GET(self) -> None:
        self._handle("GET")

    def do_POST(self) -> None:
        self._handle("POST")

    # -- plumbing ----------------------------------------------------------

    def _session(self) -> tuple[str, bool]:
        for item in self.headers.get("Cookie", "").split(";"):
            name, _, value = item.strip().partition("=")
            if name == SESSION_COOKIE and value:
                return value, False

        return secrets.token_hex(16), True

    def _is_probe(self) -> bool:
        supplied = self.headers.get(PROBE_HEADER, "")
        return bool(supplied) and hmac.compare_digest(
            supplied, self.environment.probe_token
        )

    def _read_body(self) -> bytes:
        """Always consume the request body, even if we then reject it."""

        raw = self.headers.get("Content-Length", "0")

        try:
            length = int(raw)
        except ValueError:
            raise _BadRequest("invalid Content-Length")

        if length < 0 or length > MAX_BODY_BYTES:
            raise _BadRequest("unacceptable Content-Length")

        return self.rfile.read(length)

    def _send(
        self,
        response: _Response,
        set_cookie: str | None,
    ) -> None:
        payload = response.body.encode("utf-8")

        try:
            self.send_response(response.status)
            self.send_header("Content-Type", response.content_type)
            self.send_header("Content-Length", str(len(payload)))

            if response.location is not None:
                self.send_header("Location", response.location)

            if set_cookie is not None:
                self.send_header("Set-Cookie", set_cookie)

            self.end_headers()
            self.wfile.write(payload)

        except (BrokenPipeError, ConnectionResetError, TimeoutError):
            # The client gave up (e.g. timed out on a delayed fault).
            # That is an expected outcome, not a server error.
            pass

    # -- request handling --------------------------------------------------

    def _handle(self, method: str) -> None:
        env = self.environment
        path = urlparse(self.path).path

        try:
            body = self._read_body() if method == "POST" else b""
        except _BadRequest as exc:
            self._send(_Response(400, str(exc), _TEXT), None)
            return

        session_id, is_new = self._session()
        env.state.ensure_session(session_id)

        cookie = (
            f"{SESSION_COOKIE}={session_id}; "
            "Path=/; HttpOnly; SameSite=Lax"
            if is_new
            else None
        )

        fault = env.fault_injector.check(
            f"{method} {path}",
            bypass=self._is_probe(),
        )

        if fault is not None and fault.mode == "fail_before_commit":
            self._send_fault(fault, cookie)
            return

        try:
            response = self._route(method, path, body, session_id)
        except Exception as exc:  # harness bug, not an injected fault
            env.internal_errors.append(
                f"{method} {path}: {type(exc).__name__}: {exc}"
            )
            response = _Response(500, "internal error", _TEXT)

        if fault is not None:  # fail_after_commit: state already changed
            self._send_fault(fault, cookie)
            return

        self._send(response, cookie)

    def _send_fault(
        self,
        fault: FaultRule,
        cookie: str | None,
    ) -> None:
        if fault.delay_seconds > 0:
            time.sleep(fault.delay_seconds)

        self._send(
            _Response(
                fault.response_status,
                fault.response_body,
                _TEXT,
            ),
            cookie,
        )

    def _route(
        self,
        method: str,
        path: str,
        body: bytes,
        session_id: str,
    ) -> _Response:
        state = self.environment.state

        if method == "GET":
            if path in {"/", "/products"}:
                return _Response(200, render_products())

            if path == "/cart":
                return _Response(
                    200,
                    render_cart(state.get_cart(session_id)),
                )

        if method == "POST" and path == "/cart/add":
            try:
                form = parse_qs(
                    body.decode("utf-8"),
                    keep_blank_values=True,
                )
                product_id = form.get("product_id", [""])[0]
                quantity = int(form.get("quantity", ["1"])[0])
                state.add_to_cart(session_id, product_id, quantity)
            except ValueError:
                return _Response(
                    400,
                    "invalid cart request",
                    _TEXT,
                )

            return _Response(303, location="/cart")

        return _Response(404, "not found", _TEXT)