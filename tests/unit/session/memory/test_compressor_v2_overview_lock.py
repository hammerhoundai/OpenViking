# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0

from types import SimpleNamespace
from typing import List
from unittest.mock import AsyncMock, patch

import pytest

from openviking.message import Message
from openviking.server.identity import RequestContext, Role
from openviking.session.compressor_v2 import SessionCompressorV2
from openviking.session.memory.memory_updater import MemoryUpdateResult
from openviking_cli.session.user_id import UserIdentifier


class TestCompressorV2OverviewLock:
    @pytest.mark.asyncio
    async def test_extract_phase_passes_handle_to_extract_loop_without_main_prelock(self):
        compressor = SessionCompressorV2(vikingdb=None)
        user = UserIdentifier.the_default_user()
        ctx = RequestContext(user=user, role=Role.ROOT)
        messages = [Message.create_user("test")]
        events: List[str] = []
        main_handle = SimpleNamespace(id="main-handle", locks=[])

        class FakeVikingFS:
            agfs = object()

            def _uri_to_path(self, uri: str, ctx=None) -> str:
                return uri

        class DummyProvider:
            def get_memory_schemas(self, _ctx):
                return []

            def _get_registry(self):
                return object()

        class DummyExtractLoop:
            def __init__(self, **kwargs):
                self._transaction_handle = None

            async def run(self):
                assert self._transaction_handle is main_handle
                events.append("run")
                return SimpleNamespace(upsert_operations=[], delete_file_contents=[]), []

        config = SimpleNamespace(
            vlm=SimpleNamespace(get_vlm_instance=lambda: object()),
            memory=SimpleNamespace(
                enable_role_id_memory_isolate=False,
                v2_lock_max_retries=1,
                v2_lock_retry_interval_seconds=0.0,
            ),
        )

        async def release(handle):
            events.append(f"release:{handle.id}")

        lock_manager = SimpleNamespace(
            create_handle=lambda: main_handle,
            acquire_exact_tree_batch=AsyncMock(side_effect=AssertionError("unexpected prelock")),
            release=AsyncMock(side_effect=release),
        )

        with (
            patch("openviking.session.compressor_v2.get_viking_fs", return_value=FakeVikingFS()),
            patch("openviking.session.compressor_v2.get_openviking_config", return_value=config),
            patch(
                "openviking.session.memory.memory_isolation_handler.get_openviking_config",
                return_value=config,
            ),
            patch("openviking.session.compressor_v2.ExtractLoop", DummyExtractLoop),
            patch("openviking.storage.transaction.init_lock_manager"),
            patch("openviking.storage.transaction.get_lock_manager", return_value=lock_manager),
            patch.object(compressor, "_resolve_supersedes", AsyncMock(return_value={})),
        ):
            result = await compressor._run_extract_phase(
                provider=DummyProvider(),
                messages=messages,
                ctx=ctx,
                strict_extract_errors=True,
                phase_label="experience(test)",
            )

        assert result == ([], [], [], {}, [])
        assert events == ["run", "release:main-handle"]

    @pytest.mark.asyncio
    async def test_extract_phase_generates_overview_after_main_lock_release(self):
        compressor = SessionCompressorV2(vikingdb=None)
        user = UserIdentifier.the_default_user()
        ctx = RequestContext(user=user, role=Role.ROOT)
        messages = [Message.create_user("test")]
        overview_dir = "viking://agent/default/memories/experiences"
        events: List[str] = []

        class FakeVikingFS:
            agfs = object()

            def _uri_to_path(self, uri: str, ctx=None) -> str:
                return uri

        class DummyProvider:
            def get_memory_schemas(self, _ctx):
                return []

            def _get_registry(self):
                return object()

        class DummyExtractLoop:
            def __init__(self, **kwargs):
                pass

            async def run(self):
                return SimpleNamespace(upsert_operations=[], delete_file_contents=[]), []

        class DummyUpdater:
            async def apply_operations(self, operations, ctx, **kwargs):
                events.append("apply")
                result = MemoryUpdateResult()
                result.written_uris = [f"{overview_dir}/debug.md"]
                result.overview_directories = {overview_dir: "experiences"}
                return result

            async def generate_overview(self, memory_type, directory, ctx, extract_context=None):
                assert memory_type == "experiences"
                assert directory == overview_dir
                events.append("overview")

        config = SimpleNamespace(
            vlm=SimpleNamespace(get_vlm_instance=lambda: object()),
            memory=SimpleNamespace(
                enable_role_id_memory_isolate=False,
                v2_lock_max_retries=1,
                v2_lock_retry_interval_seconds=0.0,
            ),
        )
        main_handle = SimpleNamespace(id="main-handle", locks=[])
        overview_handle = SimpleNamespace(id="overview-handle", locks=[])
        handle_iter = iter([main_handle, overview_handle])

        async def acquire_tree(handle, path, timeout=None):
            assert handle is overview_handle
            assert path == overview_dir
            assert timeout is None
            events.append("overview_acquire")
            return True

        async def release(handle):
            events.append(f"release:{handle.id}")

        lock_manager = SimpleNamespace(
            create_handle=lambda: next(handle_iter),
            acquire_exact_tree_batch=AsyncMock(side_effect=AssertionError("unexpected prelock")),
            acquire_tree=AsyncMock(side_effect=acquire_tree),
            release=AsyncMock(side_effect=release),
        )

        updater = DummyUpdater()
        split_result = (
            SimpleNamespace(
                upsert_operations=[SimpleNamespace()], delete_file_contents=[], errors=[]
            ),
            SimpleNamespace(upsert_operations=[]),
            [],
        )

        with (
            patch("openviking.session.compressor_v2.get_viking_fs", return_value=FakeVikingFS()),
            patch("openviking.session.compressor_v2.get_openviking_config", return_value=config),
            patch(
                "openviking.session.memory.memory_isolation_handler.get_openviking_config",
                return_value=config,
            ),
            patch("openviking.session.compressor_v2.ExtractLoop", DummyExtractLoop),
            patch("openviking.storage.transaction.init_lock_manager"),
            patch("openviking.storage.transaction.get_lock_manager", return_value=lock_manager),
            patch.object(compressor, "_get_or_create_updater", return_value=updater),
            patch.object(compressor, "_resolve_supersedes", AsyncMock(return_value={})),
            patch.object(compressor, "_split_operations_by_memory_type", return_value=split_result),
        ):
            result = await compressor._run_extract_phase(
                provider=DummyProvider(),
                messages=messages,
                ctx=ctx,
                strict_extract_errors=True,
                phase_label="experience(test)",
            )

        assert result[0] == [f"{overview_dir}/debug.md"]
        assert events == [
            "apply",
            "release:main-handle",
            "overview_acquire",
            "overview",
            "release:overview-handle",
        ]


class TestCompressorV2LongTermMemoryLock:
    @pytest.mark.asyncio
    async def test_extract_long_term_memories_passes_handle_without_main_prelock(self):
        compressor = SessionCompressorV2(vikingdb=None)
        user = UserIdentifier.the_default_user()
        ctx = RequestContext(user=user, role=Role.ROOT)
        messages = [Message.create_user("test")]
        events: List[str] = []
        main_handle = SimpleNamespace(id="main-handle", locks=[])
        memory_uri = "viking://user/default/memories/profile.md"

        class FakeVikingFS:
            agfs = object()

        class DummyIsolationHandler:
            def __init__(self, *_args, **_kwargs):
                pass

            def prepare_messages(self):
                events.append("prepare")

            def get_read_scope(self):
                return SimpleNamespace(user_ids=[ctx.user.user_id], agent_ids=[ctx.user.agent_id])

        class DummyOrchestrator:
            def __init__(self):
                self._transaction_handle = None
                self.context_provider = SimpleNamespace(get_memory_schemas=lambda _ctx: [])

            async def run(self):
                assert self._transaction_handle is main_handle
                events.append("run")
                return SimpleNamespace(upsert_operations=[], delete_file_contents=[]), []

        class DummyUpdater:
            async def apply_operations(self, operations, ctx, **kwargs):
                events.append("apply")
                result = MemoryUpdateResult()
                result.written_uris = [memory_uri]
                result.edited_uris = []
                result.deleted_uris = []
                result.errors = []
                return result

        config = SimpleNamespace(
            memory=SimpleNamespace(
                enable_role_id_memory_isolate=False,
                v2_lock_max_retries=1,
                v2_lock_retry_interval_seconds=0.0,
            )
        )
        telemetry = SimpleNamespace(set=lambda *args, **kwargs: None)
        orchestrator = DummyOrchestrator()

        async def release(handle):
            events.append(f"release:{handle.id}")

        lock_manager = SimpleNamespace(
            create_handle=lambda: main_handle,
            acquire_exact_tree_batch=AsyncMock(side_effect=AssertionError("unexpected prelock")),
            release=AsyncMock(side_effect=release),
        )

        with (
            patch("openviking.session.compressor_v2.get_openviking_config", return_value=config),
            patch("openviking.storage.viking_fs.get_viking_fs", return_value=FakeVikingFS()),
            patch("openviking.session.compressor_v2.MemoryIsolationHandler", DummyIsolationHandler),
            patch("openviking.storage.transaction.init_lock_manager"),
            patch("openviking.storage.transaction.get_lock_manager", return_value=lock_manager),
            patch(
                "openviking.session.memory.memory_type_registry.create_default_registry",
                return_value=SimpleNamespace(initialize_memory_files=AsyncMock()),
            ),
            patch("openviking.session.compressor_v2.get_current_telemetry", return_value=telemetry),
            patch.object(compressor, "_get_or_create_react", return_value=orchestrator),
            patch.object(compressor, "_get_or_create_updater", return_value=DummyUpdater()),
        ):
            result = await compressor.extract_long_term_memories(
                messages=messages,
                ctx=ctx,
                strict_extract_errors=True,
            )

        assert [context.uri for context in result] == [memory_uri]
        assert events == ["prepare", "run", "apply", "release:main-handle"]
