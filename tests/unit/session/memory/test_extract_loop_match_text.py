# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0

from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import pytest

from openviking.session.memory.dataclass import (
    MemoryField,
    MemoryFile,
    MemoryTypeSchema,
    ResolvedOperation,
    ResolvedOperations,
    WikiLink,
)
from openviking.session.memory.extract_loop import ExtractLoop
from openviking.session.memory.merge_op import FieldType, MergeOp
from openviking.session.memory.page_id_map import PageIdMap


class AttrDict(dict):
    __getattr__ = dict.get


class TestResolveOperations:
    @pytest.mark.asyncio
    async def test_existing_page_id_keeps_existing_uri_and_identity_fields(self):
        schema = MemoryTypeSchema(
            memory_type="entities",
            description="entity memory",
            directory="viking://user/{{ user_space }}/memories/entities",
            filename_template="{{ name }}.md",
            fields=[
                MemoryField(name="name", field_type=FieldType.STRING, merge_op=MergeOp.REPLACE),
                MemoryField(name="content", field_type=FieldType.STRING, merge_op=MergeOp.PATCH),
            ],
        )
        existing_uri = "viking://user/alice/memories/entities/Melanie.md"
        old_file = MemoryFile(
            uri=existing_uri,
            content="old content",
            memory_type="entities",
            extra_fields={"name": "Melanie"},
        )

        context_provider = Mock()
        context_provider.get_memory_schemas.return_value = [schema]
        context_provider._get_registry.return_value = Mock(get=Mock(return_value=schema))
        context_provider.read_file_contents = {existing_uri: old_file}

        isolation_handler = Mock()
        isolation_handler.get_read_scope.return_value = None
        isolation_handler.fill_role_ids.side_effect = lambda item, role_scope=None: item

        loop = ExtractLoop(
            vlm=Mock(model="test-model"),
            viking_fs=Mock(),
            context_provider=context_provider,
            isolation_handler=isolation_handler,
        )
        loop._extract_context = SimpleNamespace(
            page_id_map=SimpleNamespace(resolve=lambda page_id: existing_uri)
        )

        operations, _ = await loop.resolve_operations(
            AttrDict(
                entities=[{"name": "WrongName", "content": "new content", "page_id": 7}],
                delete_uris=[],
            )
        )

        operation = operations.upsert_operations[0]
        assert operation.uris == [existing_uri]
        assert operation.old_memory_file_content is old_file
        assert operation.memory_fields["name"] == "Melanie"
        assert operation.memory_fields["content"] == "new content"
        isolation_handler.calculate_memory_uris.assert_not_called()

    def test_unresolved_page_ids_logs_at_info(self):
        loop = ExtractLoop(vlm=Mock(model="test-model"), viking_fs=Mock(), context_provider=Mock())
        loop._extract_context = Mock()
        loop._extract_context.page_id_map = Mock()
        loop._extract_context.page_id_map._id_to_uri = {
            100: "viking://agent/agent_sample_0/memories/trajectories/a.md"
        }
        loop._extract_context.page_id_map.resolve.side_effect = lambda page_id: {
            100: "viking://agent/agent_sample_0/memories/trajectories/a.md"
        }.get(page_id)
        loop._extract_context.page_id_map.register_new_page_id = Mock()

        raw_links = [WikiLink(f=100, t=102, match_text="trip")]

        with (
            patch("openviking.session.memory.extract_loop.tracer.info") as mock_info,
            patch("openviking.session.memory.extract_loop.tracer.error") as mock_error,
        ):
            resolved = loop._resolve_links(raw_links, upsert_operations=[])

        assert resolved == []
        mock_error.assert_not_called()
        mock_info.assert_any_call(
            "Skipping link with unresolved page_ids: f=100, t=102, "
            "from_uri=viking://agent/agent_sample_0/memories/trajectories/a.md, to_uri=None, "
            "op_page_map_keys=[]"
        )


class TestResolveLinksMultiUri:
    def test_shared_page_id_pairs_matching_user_uris_only(self):
        loop = ExtractLoop(vlm=Mock(model="test-model"), viking_fs=Mock(), context_provider=Mock())
        loop._extract_context = Mock()
        loop._extract_context.page_id_map = Mock()
        loop._extract_context.page_id_map._id_to_uri = {}
        loop._extract_context.page_id_map.resolve.return_value = None
        loop._extract_context.page_id_map.register_new_page_id = Mock()

        raw_links = [WikiLink(f=100, t=101, match_text="trip")]
        upsert_operations = [
            ResolvedOperation(
                memory_fields={},
                memory_type="experiences",
                uris=[
                    "viking://user/a/memories/experiences/source.md",
                    "viking://user/b/memories/experiences/source.md",
                ],
                page_id=100,
            ),
            ResolvedOperation(
                memory_fields={},
                memory_type="experiences",
                uris=[
                    "viking://user/a/memories/experiences/target.md",
                    "viking://user/b/memories/experiences/target.md",
                ],
                page_id=101,
            ),
        ]

        resolved = loop._resolve_links(raw_links, upsert_operations=upsert_operations)

        assert {(link.from_uri, link.to_uri) for link in resolved} == {
            (
                "viking://user/a/memories/experiences/source.md",
                "viking://user/a/memories/experiences/target.md",
            ),
            (
                "viking://user/b/memories/experiences/source.md",
                "viking://user/b/memories/experiences/target.md",
            ),
        }


class TestPageIdInstruction:
    @pytest.mark.asyncio
    async def test_run_always_includes_page_id_rules_when_links_disabled(self):
        context_provider = Mock()
        context_provider.get_memory_schemas.return_value = [
            SimpleNamespace(
                memory_type="experiences",
                description="experience memory",
                fields=[],
                directory="",
                filename_template="",
            )
        ]
        context_provider.get_output_language.return_value = "zh-CN"
        context_provider.get_tools.return_value = []
        extract_context = Mock()
        extract_context.page_id_map = PageIdMap()
        context_provider.get_extract_context.return_value = extract_context
        context_provider.prefetch = AsyncMock(return_value=[])
        context_provider.read_file_contents = {}
        context_provider.instruction.return_value = "base instruction"
        context_provider._get_registry.return_value = Mock()

        isolation_handler = Mock()
        isolation_handler.get_read_scope.return_value = None
        isolation_handler.fill_role_ids.side_effect = lambda item, role_scope=None: item
        isolation_handler.calculate_memory_uris.return_value = [
            "viking://user/alice/memories/experiences/chat.md"
        ]

        loop = ExtractLoop(
            vlm=Mock(model="test-model"),
            viking_fs=Mock(),
            context_provider=context_provider,
            isolation_handler=isolation_handler,
        )
        loop._mark_cache_breakpoint = AsyncMock()
        loop._call_llm = AsyncMock(
            return_value=(
                [],
                AttrDict(
                    experiences=[{"experience_name": "chat", "content": "updated", "page_id": 100}]
                ),
            )
        )
        loop._check_unread_existing_files = AsyncMock(return_value=[])
        loop.finalize_operations = AsyncMock()

        captured_messages = []

        def capture_messages(messages):
            captured_messages.extend(messages)

        with (
            patch("openviking.session.memory.extract_loop.get_openviking_config") as mock_config,
            patch("openviking.session.memory.extract_loop.pretty_print_messages", capture_messages),
            patch(
                "openviking.session.memory.extract_loop.SchemaModelGenerator.generate_all_models"
            ),
            patch(
                "openviking.session.memory.extract_loop.SchemaModelGenerator.create_structured_operations_model"
            ) as mock_create_model,
        ):
            mock_config.return_value = SimpleNamespace(memory=SimpleNamespace(link_enabled=False))
            mock_create_model.return_value = SimpleNamespace(model_json_schema=lambda: {})

            await loop.run()

        system_content = captured_messages[0]["content"]
        assert "## Page ID Rules" in system_content
        assert "## Read Format Rules" in system_content
        assert 'Every memory item you create or edit MUST include "page_id".' in system_content
        assert (
            "The read tool accepts `uri`, optional `offset` (0-indexed), and optional `limit`."
            in system_content
        )
        assert "each visible line is prefixed with `line_number<TAB>`" in system_content
        assert (
            "Never include the line-number prefix itself in `search` or `replace`."
            in system_content
        )
        assert "For existing items, use the page_id shown in read/search results." in system_content
        assert "For new items, assign a unique page_id >= 100." in system_content
        assert "When editing an existing item, reuse its existing page_id." in system_content
        assert "Link fields" not in system_content

    @pytest.mark.asyncio
    async def test_run_includes_link_page_id_rule_when_links_enabled(self):
        context_provider = Mock()
        context_provider.get_memory_schemas.return_value = [
            SimpleNamespace(
                memory_type="experiences",
                description="experience memory",
                fields=[],
                directory="",
                filename_template="",
            )
        ]
        context_provider.get_output_language.return_value = "zh-CN"
        context_provider.get_tools.return_value = []
        extract_context = Mock()
        extract_context.page_id_map = PageIdMap()
        context_provider.get_extract_context.return_value = extract_context
        context_provider.prefetch = AsyncMock(return_value=[])
        context_provider.read_file_contents = {}
        context_provider.instruction.return_value = "base instruction"
        context_provider._get_registry.return_value = Mock()

        isolation_handler = Mock()
        isolation_handler.get_read_scope.return_value = None
        isolation_handler.fill_role_ids.side_effect = lambda item, role_scope=None: item
        isolation_handler.calculate_memory_uris.return_value = [
            "viking://user/alice/memories/experiences/chat.md"
        ]

        loop = ExtractLoop(
            vlm=Mock(model="test-model"),
            viking_fs=Mock(),
            context_provider=context_provider,
            isolation_handler=isolation_handler,
        )
        loop._mark_cache_breakpoint = AsyncMock()
        loop._call_llm = AsyncMock(
            return_value=(
                [],
                AttrDict(
                    experiences=[{"experience_name": "chat", "content": "updated", "page_id": 100}],
                    links=[],
                ),
            )
        )
        loop._check_unread_existing_files = AsyncMock(return_value=[])
        loop.finalize_operations = AsyncMock()

        captured_messages = []

        def capture_messages(messages):
            captured_messages.extend(messages)

        with (
            patch("openviking.session.memory.extract_loop.get_openviking_config") as mock_config,
            patch("openviking.session.memory.extract_loop.pretty_print_messages", capture_messages),
            patch(
                "openviking.session.memory.extract_loop.SchemaModelGenerator.generate_all_models"
            ),
            patch(
                "openviking.session.memory.extract_loop.SchemaModelGenerator.create_structured_operations_model"
            ) as mock_create_model,
        ):
            mock_config.return_value = SimpleNamespace(memory=SimpleNamespace(link_enabled=True))
            mock_create_model.return_value = SimpleNamespace(model_json_schema=lambda: {})

            await loop.run()

        system_content = captured_messages[0]["content"]
        assert "## Page ID Rules" in system_content
        assert "## Read Format Rules" in system_content
        assert "## Link Rules" in system_content
        assert "Link fields `f` and `t` must reference these page_id values." in system_content
        assert "each visible line is prefixed with `line_number<TAB>`" in system_content
        assert "Only create links when the relationship is meaningful" in system_content


class TestFinalOperationsHydration:
    @pytest.mark.asyncio
    async def test_run_logs_final_operations_after_old_memory_file_is_hydrated(self):
        old_file = MemoryFile(
            uri="viking://user/Caroline/memories/experiences/chat.md", content="old"
        )

        context_provider = Mock()
        schema = SimpleNamespace(
            memory_type="experiences",
            fields=[],
            description="experience memory",
            directory="",
            filename_template="",
        )
        context_provider.get_memory_schemas.return_value = [schema]
        context_provider.get_output_language.return_value = "zh-CN"
        context_provider.get_tools.return_value = []
        extract_context = Mock()
        extract_context.page_id_map = PageIdMap()
        extract_context.page_id_map.get_page_id(old_file.uri)
        context_provider.get_extract_context.return_value = extract_context
        context_provider.prefetch = AsyncMock(return_value=[])
        context_provider.read_file_contents = {old_file.uri: old_file}
        context_provider.instruction.return_value = "test instruction"
        context_provider._get_registry.return_value = Mock()

        isolation_handler = Mock()
        isolation_handler.get_read_scope.return_value = "user://Caroline"
        isolation_handler.fill_role_ids.side_effect = lambda item, role_scope=None: item

        loop = ExtractLoop(
            vlm=Mock(model="test-model"),
            viking_fs=Mock(),
            context_provider=context_provider,
            isolation_handler=isolation_handler,
        )
        loop._mark_cache_breakpoint = AsyncMock()
        loop._call_llm = AsyncMock(
            return_value=(
                [],
                AttrDict(
                    experiences=[{"experience_name": "chat", "content": "updated", "page_id": 1}]
                ),
            )
        )
        loop._check_unread_existing_files = AsyncMock(return_value=[])
        loop.finalize_operations = AsyncMock()

        with (
            patch("openviking.session.memory.extract_loop.get_openviking_config") as mock_config,
            patch(
                "openviking.session.memory.extract_loop.SchemaModelGenerator.generate_all_models"
            ),
            patch(
                "openviking.session.memory.extract_loop.SchemaModelGenerator.create_structured_operations_model"
            ) as mock_create_model,
            patch("openviking.session.memory.extract_loop.tracer.info") as mock_tracer_info,
        ):
            mock_config.return_value = SimpleNamespace(memory=SimpleNamespace(link_enabled=False))
            mock_create_model.return_value = SimpleNamespace(model_json_schema=lambda: {})

            final_operations, _ = await loop.run()

        assert extract_context.page_id_map.resolve(1) == old_file.uri

        op = final_operations.upsert_operations[0]
        assert op.page_id == 1
        assert op.old_memory_file_content is old_file
        assert final_operations.resolved_links == []
        logged_messages = [call.args[0] for call in mock_tracer_info.call_args_list]
        assert any(message.startswith("final_operations=") for message in logged_messages)


class TestFinalOperationRelock:
    @pytest.mark.asyncio
    async def test_run_locks_operation_uris_before_return(self):
        existing_uri = "viking://user/alice/memories/preferences/existing.md"
        new_uri = "viking://user/alice/memories/preferences/new.md"

        context_provider = Mock()
        schema = SimpleNamespace(
            memory_type="preferences",
            fields=[],
            description="preference memory",
            directory="",
            filename_template="",
        )
        context_provider.get_memory_schemas.return_value = [schema]
        context_provider.get_output_language.return_value = "zh-CN"
        context_provider.get_tools.return_value = []
        extract_context = Mock()
        extract_context.page_id_map = PageIdMap()
        context_provider.get_extract_context.return_value = extract_context
        context_provider.prefetch = AsyncMock(return_value=[])
        context_provider.read_file_contents = {}
        tracked_uris_list = [existing_uri]
        context_provider.memory_file_tracker = SimpleNamespace(
            tracked_uris=tracked_uris_list,
            track=lambda uri: tracked_uris_list.append(uri)
            if uri not in tracked_uris_list
            else None,
        )
        context_provider.read_file = AsyncMock(return_value=None)
        context_provider.instruction.return_value = "test instruction"
        context_provider._get_registry.return_value = Mock()

        isolation_handler = Mock()
        isolation_handler.get_read_scope.return_value = None
        isolation_handler.fill_role_ids.side_effect = lambda item, role_scope=None: item
        isolation_handler.calculate_memory_uris.return_value = [new_uri]

        events = []
        viking_fs_mock = Mock()
        viking_fs_mock._uri_to_path = lambda uri, ctx=None: uri

        async def lock(paths, timeout=None):
            events.append(("lock", list(paths), timeout))

        loop = ExtractLoop(
            vlm=Mock(model="test-model"),
            viking_fs=viking_fs_mock,
            context_provider=context_provider,
            isolation_handler=isolation_handler,
        )
        loop._lock_scope = SimpleNamespace(lock=AsyncMock(side_effect=lock), release=AsyncMock())
        loop._mark_cache_breakpoint = AsyncMock()
        loop._call_llm = AsyncMock(
            return_value=(
                [],
                AttrDict(preferences=[{"name": "new", "content": "updated", "page_id": 100}]),
            )
        )
        loop._check_unread_existing_files = AsyncMock(return_value=[])

        with (
            patch("openviking.session.memory.extract_loop.get_openviking_config") as mock_config,
            patch(
                "openviking.session.memory.extract_loop.SchemaModelGenerator.generate_all_models"
            ),
            patch(
                "openviking.session.memory.extract_loop.SchemaModelGenerator.create_structured_operations_model"
            ) as mock_create_model,
        ):
            mock_config.return_value = SimpleNamespace(memory=SimpleNamespace(link_enabled=False))
            mock_create_model.return_value = SimpleNamespace(model_json_schema=lambda: {})

            final_operations, _ = await loop.run()

        assert final_operations.upsert_operations[0].uris == [new_uri]
        # Events: lock after prefetch, lock after refresh, lock after expand with operations
        assert len(events) >= 2
        # The final lock call includes both existing_uri and new_uri
        last_lock_event = events[-1]
        assert last_lock_event[0] == "lock"
        assert existing_uri in last_lock_event[1]
        assert last_lock_event[2] is None

    @pytest.mark.asyncio
    async def test_run_initializes_lock_scope_from_transaction_handle(self):
        new_uri = "viking://user/alice/memories/preferences/new.md"

        context_provider = Mock()
        schema = SimpleNamespace(
            memory_type="preferences",
            fields=[],
            description="preference memory",
            directory="",
            filename_template="",
        )
        context_provider.get_memory_schemas.return_value = [schema]
        context_provider.get_output_language.return_value = "zh-CN"
        context_provider.get_tools.return_value = []
        extract_context = Mock()
        extract_context.page_id_map = PageIdMap()
        context_provider.get_extract_context.return_value = extract_context
        context_provider.prefetch = AsyncMock(return_value=[])
        context_provider.read_file_contents = {}
        context_provider.memory_file_tracker = SimpleNamespace(
            tracked_uris=[new_uri], track=lambda uri: None
        )
        context_provider.read_file = AsyncMock(return_value=None)
        context_provider.instruction.return_value = "test instruction"
        context_provider._get_registry.return_value = Mock()

        isolation_handler = Mock()
        isolation_handler.get_read_scope.return_value = None
        isolation_handler.fill_role_ids.side_effect = lambda item, role_scope=None: item
        isolation_handler.calculate_memory_uris.return_value = [new_uri]

        handle = SimpleNamespace(id="handle-1", locks=[])
        events = []

        async def acquire_exact_path_batch(handle_arg, paths, timeout=None):
            assert handle_arg is handle
            events.append(("acquire_batch", list(paths), timeout))
            for p in paths:
                handle_arg.locks.append(f"lock:{p}")
            return True

        lock_manager = SimpleNamespace(
            acquire_exact_path_batch=AsyncMock(side_effect=acquire_exact_path_batch),
            release_selected=AsyncMock(),
            release=AsyncMock(),
        )

        loop = ExtractLoop(
            vlm=Mock(model="test-model"),
            viking_fs=Mock(agfs=object(), _uri_to_path=lambda uri, ctx=None: uri),
            context_provider=context_provider,
            isolation_handler=isolation_handler,
        )
        loop._transaction_handle = handle
        loop._mark_cache_breakpoint = AsyncMock()
        loop._call_llm = AsyncMock(
            return_value=(
                [],
                AttrDict(preferences=[{"name": "new", "content": "updated", "page_id": 100}]),
            )
        )
        loop._check_unread_existing_files = AsyncMock(return_value=[])

        with (
            patch("openviking.session.memory.extract_loop.get_openviking_config") as mock_config,
            patch(
                "openviking.session.memory.extract_loop.SchemaModelGenerator.generate_all_models"
            ),
            patch(
                "openviking.session.memory.extract_loop.SchemaModelGenerator.create_structured_operations_model"
            ) as mock_create_model,
            patch(
                "openviking.session.memory.extract_loop.get_lock_manager", return_value=lock_manager
            ),
        ):
            mock_config.return_value = SimpleNamespace(memory=SimpleNamespace(link_enabled=False))
            mock_create_model.return_value = SimpleNamespace(model_json_schema=lambda: {})

            final_operations, _ = await loop.run()

        assert final_operations.upsert_operations[0].uris == [new_uri]
        assert len(events) >= 1
        assert events[0][0] == "acquire_batch"
        assert set(events[0][1]) == {new_uri}
        assert events[0][2] is None
        assert loop._lock_scope is not None


class TestToolCallRelock:
    @pytest.mark.asyncio
    async def test_execute_tool_calls_executes_reads_in_parallel(self):
        first_uri = "viking://user/alice/memories/preferences/first.md"
        second_uri = "viking://user/alice/memories/preferences/second.md"

        context_provider = Mock()
        context_provider.memory_file_tracker = SimpleNamespace(tracked_uris=[first_uri])
        events = []

        async def execute_tool(tool_call):
            events.append(("execute", tool_call.arguments["uri"]))
            return {"uri": tool_call.arguments["uri"]}

        context_provider.execute_tool = AsyncMock(side_effect=execute_tool)

        loop = ExtractLoop(
            vlm=Mock(model="test-model"), viking_fs=Mock(), context_provider=context_provider
        )
        loop._lock_scope = SimpleNamespace()

        await loop._execute_tool_calls(
            messages=[],
            tool_calls=[
                SimpleNamespace(id="call-1", name="read", arguments={"uri": first_uri}),
                SimpleNamespace(id="call-2", name="read", arguments={"uri": second_uri}),
            ],
            tools_used=[],
        )

        assert events == [("execute", first_uri), ("execute", second_uri)]


class TestUnreadExistingFileRelock:
    @pytest.mark.asyncio
    async def test_check_unread_existing_files_reads_missing_files(self):
        unread_uri = "viking://user/alice/memories/preferences/unread.md"

        context_provider = Mock()
        context_provider.read_file_contents = {}
        context_provider.memory_file_tracker = SimpleNamespace(tracked_uris=[])
        events = []

        async def execute_tool(tool_call):
            events.append(("execute", tool_call.arguments["uri"]))
            return {"uri": tool_call.arguments["uri"]}

        context_provider.execute_tool = AsyncMock(side_effect=execute_tool)

        loop = ExtractLoop(
            vlm=Mock(model="test-model"), viking_fs=Mock(), context_provider=context_provider
        )
        loop._lock_scope = SimpleNamespace()

        refetch_uris = await loop._check_unread_existing_files(
            ResolvedOperations(
                upsert_operations=[
                    ResolvedOperation(
                        memory_fields={}, memory_type="preferences", uris=[unread_uri]
                    )
                ],
                delete_file_contents=[],
                errors=[],
            )
        )

        assert refetch_uris == {unread_uri: {"uri": unread_uri}}
        assert events == [("execute", unread_uri)]
