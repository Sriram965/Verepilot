"""
State-access interfaces for VeriPilot verification.

A probe provides read-only access to the state that a verifier is permitted
to inspect. It is responsible for obtaining state; it does not decide whether
the task succeeded.

Different verification layers use different probes:
- the in-loop verifier uses a session/agent-observable probe;
- the evaluation oracle uses an authoritative hidden-state probe.

The same SuccessSpec and check implementations should be usable with either
probe, while the probe determines where the observed data comes from.

Probes must be read-only and must not perform browser actions or otherwise
modify the agent's environment or task state while obtaining evidence.

Keep state-access logic separate from check logic:
- probe: "What is the relevant state?"
- check: "Does that state satisfy this condition?"
- verifier/oracle: "Does the required set of conditions pass?"
"""

from __future__ import annotations

import json

from environments.client import ShopClient
from environments.sites.shop.server import ShopState


class ShopSessionProbe:
    """
    Phase 1 session-visible probe for the local shop.

    The shop currently exposes cart state and URL.
    ElementText and FormSubmitted become available
    once the corresponding environment/browser
    capabilities are introduced.
    """

    def __init__(
        self,
        client: ShopClient,
    ):
        self.client = client

    def current_url(self) -> str:
        return self.client.last_url

    def cart(self) -> dict[str, int]:
        status, payload = (
            self.client.get("/api/cart")
        )

        if status != 200:
            raise RuntimeError(
                "session cart probe failed "
                f"with HTTP {status}"
            )

        data = json.loads(payload)

        return {
            str(key): int(value)
            for key, value in data.items()
        }

    def element_text(
        self,
        selector: str,
    ) -> str | None:
        raise NotImplementedError(
            "ElementText probing is not part "
            "of the Phase 1 shop environment."
        )

    def submitted_form(
        self,
    ) -> dict[str, str] | None:
        raise NotImplementedError(
            "FormSubmitted probing is not part "
            "of the Phase 1 shop environment."
        )


class HiddenShopProbe:
    """
    Evaluation-only probe backed by
    authoritative hidden environment state.
    """

    def __init__(
        self,
        state: ShopState,
        session_id: str,
        last_url: str,
    ):
        self.state = state
        self.session_id = session_id
        self._last_url = last_url

    def current_url(self) -> str:
        return self._last_url

    def cart(self) -> dict[str, int]:
        return self.state.get_cart(
            self.session_id
        )

    def element_text(
        self,
        selector: str,
    ) -> str | None:
        raise NotImplementedError(
            "ElementText oracle support is not "
            "part of the Phase 1 shop environment."
        )

    def submitted_form(
        self,
    ) -> dict[str, str] | None:
        raise NotImplementedError(
            "FormSubmitted oracle support is not "
            "part of the Phase 1 shop environment."
        )
