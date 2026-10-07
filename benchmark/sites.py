"""
Site adapters: the glue between the generic runner and a concrete site.

The runner knows nothing about carts, forms or HTTP. For each environment a
task can name (`TaskSpec.environment`), a SiteAdapter tells the runner how to
build that environment, how to give the agent a client, and how to build the
two probes:

- the SESSION probe (in-loop verifier): observable state only;
- the ORACLE probe (evaluation): an authoritative snapshot.

Adding a new site means adding one adapter here and registering it in SITES.
"""

from __future__ import annotations

from typing import Any, Protocol

from environments.base import Environment, take_snapshot
from environments.client import ShopClient
from environments.faults import FaultInjector
from environments.sites.shop import ShopEnvironment
from environments.sites.shop.probes import ShopOracleProbe, ShopSessionProbe
from verifier.oracle import OracleProbe
from verifier.probe import Probe


class SiteAdapter(Protocol):
    name: str
    session_supported: frozenset[str]
    oracle_supported: frozenset[str]

    def make_environment(self, faults: FaultInjector) -> Environment:
        ...

    def make_client(self, environment: Environment) -> Any:
        ...

    def session_id(self, client: Any) -> str:
        ...

    def current_url(self, client: Any) -> str:
        ...

    def session_probe(self, environment: Environment, client: Any) -> Probe:
        ...

    def oracle_probe(
        self, environment: Environment, client: Any
    ) -> OracleProbe:
        ...


class ShopAdapter:
    name = "shop"
    session_supported = ShopSessionProbe.supported_checks
    oracle_supported = ShopOracleProbe.supported_checks

    def make_environment(self, faults: FaultInjector) -> ShopEnvironment:
        return ShopEnvironment(fault_injector=faults)

    def make_client(self, environment: ShopEnvironment) -> ShopClient:
        return ShopClient(environment.base_url)

    def session_id(self, client: ShopClient) -> str:
        return client.session_id()

    def current_url(self, client: ShopClient) -> str:
        return client.last_url

    def session_probe(
        self, environment: ShopEnvironment, client: ShopClient
    ) -> ShopSessionProbe:
        return ShopSessionProbe(
            base_url=environment.base_url,
            session_id=client.session_id(),
            current_url=lambda: client.last_url,
            probe_token=environment.probe_token,
        )

    def oracle_probe(
        self, environment: ShopEnvironment, client: ShopClient
    ) -> ShopOracleProbe:
        session_id = client.session_id()

        return ShopOracleProbe(
            take_snapshot(environment, session_id),
            client.last_url,
        )


SITES: dict[str, SiteAdapter] = {"shop": ShopAdapter()}


def get_site(name: str) -> SiteAdapter:
    try:
        return SITES[name]
    except KeyError:
        raise ValueError(
            f"unknown environment {name!r}; known: {sorted(SITES)}"
        )