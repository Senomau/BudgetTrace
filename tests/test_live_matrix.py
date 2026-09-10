import json
from dataclasses import replace

from test_live_smoke import _config, _fake_response, _fake_task


def _action_response(action):
    response = _fake_response()
    response["choices"][0]["message"]["content"] = json.dumps(action)
    return response


def test_live_matrix_reports_scored_success_rate_and_unit_cost():
    from budgettrace.live_matrix import LiveRoute, run_live_matrix

    route = LiveRoute(
        route_id="fake-model",
        config=_config(),
    )
    responses = [
        _action_response({"kind": "tool_call", "tool": "noop", "args": {}}),
        _action_response({"kind": "finish", "score": 1.0}),
        _action_response({"kind": "finish", "score": 1.0}),
        dict(_fake_response(), choices=[{"message": {"role": "assistant", "content": "not-json"}, "finish_reason": "stop"}]),
    ]

    def transport(config, payload):
        return responses.pop(0)

    result = run_live_matrix(
        [route],
        [replace(_fake_task("70"), actions=[{"action_id": "1", "name": "noop", "arguments": {}}])],
        trials=2,
        max_steps=4,
        budget_window_steps=2,
        scorer_mode="reference-actions",
        transport=transport,
    )

    summary = result["summary"]
    assert summary["episodes_planned"] == 2
    assert summary["episodes_completed"] == 2
    assert summary["success_count"] == 1
    assert summary["success_rate"] == 0.5
    assert summary["false_stop_count"] == 1
    assert summary["unit_success_cost_micros"] == summary["cost_micros_total"]
    assert summary["scored"] is True


def test_load_route_manifest_uses_api_key_env_only(tmp_path, monkeypatch):
    from budgettrace.live_routes import load_route_manifest

    manifest = tmp_path / "routes.json"
    manifest.write_text(
        json.dumps(
            {
                "routes": [
                    {
                        "route_id": "fake-model",
                        "base_url": "https://example.com/v1",
                        "model": "fake-model",
                        "api_key_env": "FAKE_ROUTE_KEY",
                        "price_input_per_mtok": 1.0,
                        "price_output_per_mtok": 2.0,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("FAKE_ROUTE_KEY", "unit-test-route-key")

    routes = load_route_manifest(manifest)

    assert routes[0].route_id == "fake-model"
    assert routes[0].config.model == "fake-model"
    assert routes[0].config.api_key == "unit-test-route-key"
