from __future__ import annotations

import html
import http.server
import json
import threading
import uuid
from dataclasses import dataclass, field
from urllib.parse import parse_qs, urlparse

from environments.faults import FaultInjector


PRODUCTS = {
    "Laptop_A": {
        "name": "Laptop A",
        "price": 899,
    },
    "Laptop_B": {
        "name": "Laptop B",
        "price": 949,
    },
    "Mouse_A": {
        "name": "Mouse A",
        "price": 29,
    },
}


@dataclass
class ShopState:
    carts: dict[str, dict[str, int]] = field(
        default_factory=dict
    )

    def ensure_session(
        self,
        session_id: str,
    ) -> None:
        self.carts.setdefault(
            session_id,
            {},
        )

    def add_to_cart(
        self,
        session_id: str,
        product_id: str,
        quantity: int,
    ) -> None:

        if product_id not in PRODUCTS:
            raise ValueError("unknown product")

        if quantity < 1:
            raise ValueError(
                "quantity must be positive"
            )

        self.ensure_session(session_id)

        cart = self.carts[session_id]

        cart[product_id] = (
            cart.get(product_id, 0)
            + quantity
        )

    def get_cart(
        self,
        session_id: str,
    ) -> dict[str, int]:

        return dict(
            self.carts.get(session_id, {})
        )


@dataclass
class ShopEnvironment:
    host: str = "127.0.0.1"
    port: int = 0

    state: ShopState = field(
        default_factory=ShopState
    )

    fault_injector: FaultInjector = field(
        default_factory=FaultInjector
    )

    _server: (
        http.server.ThreadingHTTPServer | None
    ) = field(
        default=None,
        init=False,
    )

    _thread: threading.Thread | None = field(
        default=None,
        init=False,
    )

    def start(self) -> "ShopEnvironment":
        environment = self

        class Handler(ShopRequestHandler):
            pass

        Handler.environment = environment

        self._server = (
            http.server.ThreadingHTTPServer(
                (
                    self.host,
                    self.port,
                ),
                Handler,
            )
        )

        self.port = (
            self._server.server_address[1]
        )

        self._thread = threading.Thread(
            target=self._server.serve_forever,
            daemon=True,
        )

        self._thread.start()

        return self

    @property
    def base_url(self) -> str:
        if self._server is None:
            raise RuntimeError(
                "environment is not started"
            )

        return (
            f"http://{self.host}:{self.port}"
        )

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

    def __exit__(
        self,
        exc_type,
        exc,
        tb,
    ) -> None:
        self.stop()


class ShopRequestHandler(
    http.server.BaseHTTPRequestHandler
):
    environment: ShopEnvironment

    def log_message(
        self,
        format: str,
        *args,
    ) -> None:
        return

    def _session_id(
        self,
        create: bool = True,
    ) -> tuple[str | None, bool]:

        raw_cookie = self.headers.get(
            "Cookie",
            "",
        )

        session_id = None

        for item in raw_cookie.split(";"):
            name, _, value = (
                item.strip().partition("=")
            )

            if (
                name == "vp_session"
                and value
            ):
                session_id = value
                break

        if session_id is None and create:
            session_id = uuid.uuid4().hex
            return session_id, True

        return session_id, False

    def _send(
        self,
        status: int,
        body: str,
        content_type: str = (
            "text/html; charset=utf-8"
        ),
        set_cookie: str | None = None,
    ) -> None:

        payload = body.encode("utf-8")

        self.send_response(status)

        self.send_header(
            "Content-Type",
            content_type,
        )

        self.send_header(
            "Content-Length",
            str(len(payload)),
        )

        if set_cookie is not None:
            self.send_header(
                "Set-Cookie",
                set_cookie,
            )

        self.end_headers()

        self.wfile.write(payload)

    def _redirect(
        self,
        location: str,
        set_cookie: str | None = None,
    ) -> None:

        self.send_response(303)

        self.send_header(
            "Location",
            location,
        )

        if set_cookie is not None:
            self.send_header(
                "Set-Cookie",
                set_cookie,
            )

        self.send_header(
            "Content-Length",
            "0",
        )

        self.end_headers()

    def do_GET(self) -> None:
        parsed = urlparse(self.path)

        session_id, new_session = (
            self._session_id()
        )

        assert session_id is not None

        self.environment.state.ensure_session(
            session_id
        )

        cookie = (
            f"vp_session={session_id}; Path=/"
            if new_session
            else None
        )

        if parsed.path in {
            "/",
            "/products",
        }:

            items = []

            for (
                product_id,
                product,
            ) in PRODUCTS.items():

                items.append(
                    f"""
                    <article
                      data-product-id="{html.escape(product_id)}"
                    >
                      <h2>{html.escape(product["name"])}</h2>
                      <p>${product["price"]}</p>

                      <form
                        method="post"
                        action="/cart/add"
                      >
                        <input
                          type="hidden"
                          name="product_id"
                          value="{html.escape(product_id)}"
                        >

                        <input
                          name="quantity"
                          value="1"
                        >

                        <button type="submit">
                          Add to cart
                        </button>
                      </form>
                    </article>
                    """
                )

            body = """
            <!doctype html>
            <html>
              <head>
                <title>VeriPilot Local Shop</title>
              </head>
              <body>
                <h1>VeriPilot Local Shop</h1>

                <a href="/cart">
                  Cart
                </a>

                {items}
              </body>
            </html>
            """.format(
                items="".join(items)
            )

            self._send(
                200,
                body,
                set_cookie=cookie,
            )

            return

        if parsed.path == "/cart":
            cart = (
                self.environment.state.get_cart(
                    session_id
                )
            )

            rows = []

            for (
                product_id,
                quantity,
            ) in cart.items():

                rows.append(
                    f'''
                    <li
                      data-product-id="{html.escape(product_id)}"
                      data-quantity="{quantity}"
                    >
                      {html.escape(PRODUCTS[product_id]["name"])}
                      x {quantity}
                    </li>
                    '''
                )

            body = """
            <!doctype html>
            <html>
              <body>
                <h1>Your Cart</h1>

                <a href="/products">
                  Products
                </a>

                <ul data-testid="cart">
                  {rows}
                </ul>
              </body>
            </html>
            """.format(
                rows="".join(rows)
            )

            self._send(
                200,
                body,
                set_cookie=cookie,
            )

            return

        if parsed.path == "/api/cart":
            self._send(
                200,
                json.dumps(
                    self.environment.state.get_cart(
                        session_id
                    )
                ),
                "application/json",
                cookie,
            )

            return

        self._send(
            404,
            "not found",
            "text/plain; charset=utf-8",
            cookie,
        )

    def do_POST(self) -> None:
        parsed = urlparse(self.path)

        session_id, new_session = (
            self._session_id()
        )

        assert session_id is not None

        self.environment.state.ensure_session(
            session_id
        )

        cookie = (
            f"vp_session={session_id}; Path=/"
            if new_session
            else None
        )

        if parsed.path != "/cart/add":
            self._send(
                404,
                "not found",
                "text/plain; charset=utf-8",
                cookie,
            )
            return

        fault = (
            self.environment.fault_injector
            .maybe_inject("POST /cart/add")
        )

        if fault is not None:
            self._send(
                fault.response_status,
                fault.response_body,
                "text/plain; charset=utf-8",
                cookie,
            )
            return

        length = int(
            self.headers.get(
                "Content-Length",
                "0",
            )
        )

        raw = self.rfile.read(
            length
        ).decode("utf-8")

        form = parse_qs(raw)

        product_id = form.get(
            "product_id",
            [""],
        )[0]

        quantity_raw = form.get(
            "quantity",
            ["1"],
        )[0]

        try:
            quantity = int(quantity_raw)

            self.environment.state.add_to_cart(
                session_id,
                product_id,
                quantity,
            )

        except (ValueError, KeyError):
            self._send(
                400,
                "invalid cart request",
                "text/plain; charset=utf-8",
                cookie,
            )
            return

        self._redirect(
            "/cart",
            cookie,
        )
