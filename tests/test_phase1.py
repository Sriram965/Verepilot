from pathlib import Path

from benchmark.runner import (
    load_task,
    run_scripted_task,
)
from environments.client import ShopClient


TASK_PATH = (
    Path(__file__).parents[1]
    / "benchmark"
    / "tasks"
    / "cart_001.yaml"
)


def test_correct_scripted_solution_passes_verifier_and_oracle():
    task = load_task(TASK_PATH)

    def solution(
        client: ShopClient,
        _start_url: str,
    ) -> None:

        status, _ = client.post(
            "/cart/add",
            {
                "product_id": "Laptop_A",
                "quantity": "1",
            },
        )

        assert status == 303

    verifier_result, oracle_result = (
        run_scripted_task(
            task,
            solution,
        )
    )

    assert verifier_result.passed is True
    assert oracle_result.passed is True
    assert (
        oracle_result.collateral_passed
        is True
    )


def test_wrong_product_fails_verifier_and_oracle():
    task = load_task(TASK_PATH)

    def solution(
        client: ShopClient,
        _start_url: str,
    ) -> None:

        status, _ = client.post(
            "/cart/add",
            {
                "product_id": "Laptop_B",
                "quantity": "1",
            },
        )

        assert status == 303

    verifier_result, oracle_result = (
        run_scripted_task(
            task,
            solution,
        )
    )

    assert verifier_result.passed is False
    assert oracle_result.passed is False


def test_oracle_catches_collateral_change_even_when_in_loop_check_passes():
    task = load_task(TASK_PATH)

    def solution(
        client: ShopClient,
        _start_url: str,
    ) -> None:

        status, _ = client.post(
            "/cart/add",
            {
                "product_id": "Laptop_A",
                "quantity": "1",
            },
        )

        assert status == 303

        status, _ = client.post(
            "/cart/add",
            {
                "product_id": "Laptop_B",
                "quantity": "1",
            },
        )

        assert status == 303

    verifier_result, oracle_result = (
        run_scripted_task(
            task,
            solution,
        )
    )

    assert verifier_result.passed is True

    assert (
        oracle_result.success_check.passed
        is True
    )

    assert (
        oracle_result.collateral_passed
        is False
    )

    assert oracle_result.passed is False

    assert (
        oracle_result.observed_cart
        == {
            "Laptop_A": 1,
            "Laptop_B": 1,
        }
    )


def test_event_triggered_fault_is_deterministic():
    from benchmark.schema import TaskSpec

    task = TaskSpec.model_validate(
        {
            "task_id": "cart_fault_001",
            "instruction": (
                "Add Laptop_A to the cart "
                "despite one injected server failure."
            ),
            "start_url": "/products",
            "max_steps": 6,
            "success": {
                "type": "cart_contains",
                "product_id": "Laptop_A",
                "quantity": 1,
            },
            "fault_config": [
                {
                    "event": "POST /cart/add",
                    "occurrence": 1,
                    "response_status": 500,
                    "response_body": (
                        "first cart add fails"
                    ),
                }
            ],
            "oracle": {
                "no_collateral_changes": True,
                "expected_cart": {
                    "Laptop_A": 1
                },
            },
        }
    )

    def solution(
        client: ShopClient,
        _start_url: str,
    ) -> None:

        status, body = client.post(
            "/cart/add",
            {
                "product_id": "Laptop_A",
                "quantity": "1",
            },
        )

        assert status == 500
        assert (
            body
            == "first cart add fails"
        )

        status, _ = client.post(
            "/cart/add",
            {
                "product_id": "Laptop_A",
                "quantity": "1",
            },
        )

        assert status == 303

    verifier_result, oracle_result = (
        run_scripted_task(
            task,
            solution,
        )
    )

    assert verifier_result.passed is True
    assert oracle_result.passed is True


def test_fresh_run_has_fresh_state():
    task = load_task(TASK_PATH)

    def solution(
        client: ShopClient,
        _start_url: str,
    ) -> None:

        status, _ = client.post(
            "/cart/add",
            {
                "product_id": "Laptop_A",
                "quantity": "1",
            },
        )

        assert status == 303

    _, first_oracle = run_scripted_task(
        task,
        solution,
    )

    _, second_oracle = run_scripted_task(
        task,
        lambda _client, _url: None,
    )

    assert (
        first_oracle.observed_cart
        == {"Laptop_A": 1}
    )

    assert (
        second_oracle.observed_cart
        == {}
    )
