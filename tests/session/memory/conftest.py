# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0

import json

import pytest_asyncio

from openviking_cli.utils.config import OPENVIKING_CONFIG_ENV
from openviking_cli.utils.config.open_viking_config import OpenVikingConfigSingleton
from openviking_cli.utils.config.vlm_config import VLMConfig


def _write_test_config(tmp_path):
    config_path = tmp_path / "ov.conf"
    config_path.write_text(
        json.dumps(
            {
                "storage": {
                    "workspace": str(tmp_path / "workspace"),
                    "agfs": {"backend": "local", "mode": "binding-client"},
                    "vectordb": {"backend": "local"},
                },
                "embedding": {
                    "dense": {
                        "provider": "openai",
                        "model": "test-embedder",
                        "api_base": "http://127.0.0.1:11434/v1",
                        "dimension": 1024,
                    }
                },
                "encryption": {"enabled": False},
            }
        ),
        encoding="utf-8",
    )
    return config_path


@pytest_asyncio.fixture(autouse=True)
async def _isolate_test_config(monkeypatch, tmp_path):
    class FakeVLM:
        model = "fake-vlm"

        def get_completion(self, prompt="", thinking=False, tools=None, messages=None):
            return "{}"

        async def get_completion_async(
            self, prompt="", thinking=False, tools=None, tool_choice=None, messages=None
        ):
            return "{}"

        def get_vision_completion(
            self, prompt="", images=None, thinking=False, tools=None, messages=None
        ):
            return "fake image description"

        async def get_vision_completion_async(
            self, prompt="", images=None, thinking=False, tools=None, messages=None
        ):
            return "fake image description"

    config_path = _write_test_config(tmp_path)
    OpenVikingConfigSingleton.reset_instance()
    monkeypatch.setenv(OPENVIKING_CONFIG_ENV, str(config_path))
    monkeypatch.setattr(VLMConfig, "get_vlm_instance", lambda self: FakeVLM())
    monkeypatch.setattr(VLMConfig, "is_available", lambda self: True)
    yield
    OpenVikingConfigSingleton.reset_instance()


@pytest_asyncio.fixture(autouse=True)
async def _drain_background_tasks():
    yield
