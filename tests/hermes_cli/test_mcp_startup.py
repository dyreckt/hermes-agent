"""Regression tests for bounded/lazy CLI MCP startup."""

from __future__ import annotations

from argparse import Namespace
from contextlib import nullcontext
import sys
import threading
import time
import types

import pytest

import cli as cli_mod
from hermes_cli import main as main_mod
from hermes_cli import mcp_startup


@pytest.fixture(autouse=True)
def _reset_mcp_startup_state():
    saved_started = mcp_startup._mcp_discovery_started
    saved_thread = mcp_startup._mcp_discovery_thread
    try:
        mcp_startup._mcp_discovery_started = False
        mcp_startup._mcp_discovery_thread = None
        yield
    finally:
        thread = mcp_startup._mcp_discovery_thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=1.0)
        mcp_startup._mcp_discovery_started = saved_started
        mcp_startup._mcp_discovery_thread = saved_thread


def _agent_args(**overrides) -> Namespace:
    base = {
        "accept_hooks": False,
        "command": "chat",
        "cron_command": None,
        "gateway_command": None,
        "mcp_action": None,
        "tui": False,
    }
    base.update(overrides)
    return Namespace(**base)


def test_prepare_agent_startup_backgrounds_blocking_mcp_for_chat(monkeypatch):
    stop = threading.Event()
    calls = {"mcp": 0}

    def _blocking_discover():
        calls["mcp"] += 1
        stop.wait()

    monkeypatch.setitem(
        sys.modules,
        "hermes_cli.plugins",
        types.SimpleNamespace(discover_plugins=lambda: None),
    )
    monkeypatch.setitem(
        sys.modules,
        "hermes_cli.config",
        types.SimpleNamespace(
            read_raw_config=lambda: {"mcp_servers": {"demo": {"transport": "stdio"}}},
            load_config=lambda: {},
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "agent.shell_hooks",
        types.SimpleNamespace(register_from_config=lambda *_a, **_k: None),
    )
    # Stub mcp_oauth so the background thread doesn't pay the real (cold,
    # ~0.75s) ``tools.mcp_oauth`` import before calling discovery. This test
    # asserts the *backgrounding contract* (main thread returns fast, discovery
    # runs off-thread), not OAuth suppression — the unrelated import latency
    # would otherwise blow the polling deadline on a loaded CI runner.
    monkeypatch.setitem(
        sys.modules,
        "tools.mcp_oauth",
        types.SimpleNamespace(suppress_interactive_oauth=lambda: nullcontext()),
    )
    monkeypatch.setitem(
        sys.modules,
        "tools.mcp_tool",
        types.SimpleNamespace(discover_mcp_tools=_blocking_discover),
    )

    try:
        start = time.monotonic()
        main_mod._prepare_agent_startup(_agent_args())
        elapsed = time.monotonic() - start
        assert elapsed < 0.2
        deadline = time.monotonic() + 3.0
        while calls["mcp"] == 0 and time.monotonic() < deadline:
            time.sleep(0.01)
        assert calls["mcp"] == 1
        assert mcp_startup._mcp_discovery_thread is not None
        assert mcp_startup._mcp_discovery_thread.is_alive()
    finally:
        stop.set()


def test_background_mcp_discovery_suppresses_interactive_oauth(monkeypatch):
    state = {"active": False, "during_discover": None}

    class SuppressInteractiveOAuth:
        def __enter__(self):
            state["active"] = True

        def __exit__(self, *_exc):
            state["active"] = False

    def _discover():
        state["during_discover"] = state["active"]

    monkeypatch.setitem(
        sys.modules,
        "hermes_cli.config",
        types.SimpleNamespace(
            read_raw_config=lambda: {"mcp_servers": {"demo": {"url": "https://mcp.example.test/mcp"}}},
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "tools.mcp_oauth",
        types.SimpleNamespace(
            suppress_interactive_oauth=lambda: SuppressInteractiveOAuth(),
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "tools.mcp_tool",
        types.SimpleNamespace(discover_mcp_tools=_discover),
    )

    mcp_startup.start_background_mcp_discovery(
        logger=types.SimpleNamespace(debug=lambda *_a, **_k: None),
        thread_name="test-mcp-discovery",
    )
    assert mcp_startup._mcp_discovery_thread is not None
    mcp_startup._mcp_discovery_thread.join(timeout=1.0)

    assert state["during_discover"] is True
    assert state["active"] is False


def test_portable_only_mcp_configuration_opens_startup_gate(monkeypatch):
    monkeypatch.setitem(
        sys.modules,
        "hermes_cli.config",
        types.SimpleNamespace(read_raw_config=lambda: {}),
    )
    monkeypatch.setitem(
        sys.modules,
        "hermes_cli.agent_plugins",
        types.SimpleNamespace(
            has_enabled_agent_plugin_mcp=lambda _config: True,
        ),
    )

    assert mcp_startup._has_configured_mcp_servers() is True


@pytest.mark.parametrize("enabled", [False, 0, "false", "0", "no", "off"])
def test_disabled_native_mcp_configuration_keeps_startup_gate_closed(
    monkeypatch, enabled
):
    raw_config = {"mcp_servers": {"demo": {"enabled": enabled}}}
    monkeypatch.setitem(
        sys.modules,
        "hermes_cli.config",
        types.SimpleNamespace(read_raw_config=lambda: raw_config),
    )
    monkeypatch.setitem(
        sys.modules,
        "hermes_cli.agent_plugins",
        types.SimpleNamespace(
            has_enabled_agent_plugin_mcp=lambda _config: False,
        ),
    )

    assert mcp_startup._has_configured_mcp_servers() is False


def test_enabled_native_mcp_configuration_opens_startup_gate(monkeypatch):
    raw_config = {
        "mcp_servers": {
            "disabled": {"enabled": False},
            "enabled": {"transport": "stdio"},
        }
    }
    monkeypatch.setitem(
        sys.modules,
        "hermes_cli.config",
        types.SimpleNamespace(read_raw_config=lambda: raw_config),
    )

    assert mcp_startup._has_configured_mcp_servers() is True


def test_disabled_native_mcp_still_allows_enabled_portable_mcp(monkeypatch):
    raw_config = {"mcp_servers": {"disabled": {"enabled": False}}}
    monkeypatch.setitem(
        sys.modules,
        "hermes_cli.config",
        types.SimpleNamespace(read_raw_config=lambda: raw_config),
    )
    monkeypatch.setitem(
        sys.modules,
        "hermes_cli.agent_plugins",
        types.SimpleNamespace(
            has_enabled_agent_plugin_mcp=lambda _config: True,
        ),
    )

    assert mcp_startup._has_configured_mcp_servers() is True


@pytest.mark.parametrize(
    "raw_config",
    [
        {},
        {"mcp_servers": {}},
        {"mcp_servers": {"disabled": {"enabled": False}}},
    ],
)
def test_no_enabled_mcp_does_not_emit_retry_warning(monkeypatch, raw_config):
    monkeypatch.setitem(
        sys.modules,
        "hermes_cli.config",
        types.SimpleNamespace(read_raw_config=lambda: raw_config),
    )
    monkeypatch.setitem(
        sys.modules,
        "hermes_cli.agent_plugins",
        types.SimpleNamespace(
            has_enabled_agent_plugin_mcp=lambda _config: False,
        ),
    )
    warnings = []
    logger = types.SimpleNamespace(
        debug=lambda *_a, **_k: None,
        warning=lambda message, *_a, **_k: warnings.append(message),
    )

    mcp_startup.start_background_mcp_discovery(
        logger=logger,
        thread_name="disabled-mcp-discovery",
    )
    mcp_startup.start_background_mcp_discovery(
        logger=logger,
        thread_name="disabled-mcp-discovery",
    )

    assert warnings == []
    assert mcp_startup._mcp_discovery_started is False
    assert mcp_startup._mcp_discovery_thread is None








def _retry_logger():
    return types.SimpleNamespace(
        debug=lambda *_a, **_k: None,
        warning=lambda *_a, **_k: None,
    )


def _install_retry_stubs(monkeypatch, *, connected: bool, calls: dict):
    monkeypatch.setitem(
        sys.modules,
        "hermes_cli.config",
        types.SimpleNamespace(
            read_raw_config=lambda: {"mcp_servers": {"demo": {"transport": "stdio"}}},
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "tools.mcp_oauth",
        types.SimpleNamespace(suppress_interactive_oauth=lambda: nullcontext()),
    )
    monkeypatch.setitem(
        sys.modules,
        "tools.mcp_tool",
        types.SimpleNamespace(
            discover_mcp_tools=lambda: calls.__setitem__("mcp", calls["mcp"] + 1),
            get_mcp_status=lambda: [{"connected": connected}],
        ),
    )


def test_configured_zero_connected_discovery_retries(monkeypatch):
    calls = {"mcp": 0}
    warnings = []
    _install_retry_stubs(monkeypatch, connected=False, calls=calls)
    logger = types.SimpleNamespace(
        debug=lambda *_a, **_k: None,
        warning=lambda message, *_a, **_k: warnings.append(message),
    )

    mcp_startup.start_background_mcp_discovery(
        logger=logger,
        thread_name="zero-connected-mcp-discovery",
    )
    first_thread = mcp_startup._mcp_discovery_thread
    assert first_thread is not None
    first_thread.join(timeout=1.0)
    mcp_startup.start_background_mcp_discovery(
        logger=logger,
        thread_name="zero-connected-mcp-discovery",
    )
    second_thread = mcp_startup._mcp_discovery_thread
    assert second_thread is not None
    second_thread.join(timeout=1.0)

    assert calls["mcp"] == 2
    assert any("retrying discovery thread" in message for message in warnings)


def test_connected_discovery_is_not_restarted(monkeypatch):
    calls = {"mcp": 0}
    _install_retry_stubs(monkeypatch, connected=True, calls=calls)

    mcp_startup.start_background_mcp_discovery(
        logger=_retry_logger(),
        thread_name="connected-mcp-discovery",
    )
    thread = mcp_startup._mcp_discovery_thread
    assert thread is not None
    thread.join(timeout=1.0)
    mcp_startup.start_background_mcp_discovery(
        logger=_retry_logger(),
        thread_name="connected-mcp-discovery",
    )

    assert calls["mcp"] == 1
    assert mcp_startup._mcp_discovery_thread is None
