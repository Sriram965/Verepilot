import pytest

from environments.client import ShopClient
from environments.sites.shop import ShopEnvironment


def test_shop_environment_uses_loopback_origin():
    with ShopEnvironment() as environment:
        assert environment.base_url.startswith(
            "http://127.0.0.1:"
        )


def test_shop_client_rejects_non_loopback_origin():
    with pytest.raises(ValueError):
        ShopClient(
            "https://example.com"
        )
