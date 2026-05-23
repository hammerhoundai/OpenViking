# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0

from unittest.mock import AsyncMock, MagicMock

import pytest

from openviking.server.identity import RequestContext, Role
from openviking.session.memory.dataclass import (
    MemoryTypeSchema,
    ResolvedOperation,
    ResolvedOperations,
)
from openviking.session.memory.memory_updater import MemoryUpdater
from openviking_cli.session.user_id import UserIdentifier


class TestMemoryUpdaterOverviewTargets:
    @pytest.mark.asyncio
    async def test_apply_operations_reports_overview_directory_without_generating_it(self):
        uri = "viking://user/alice/agent/bot/memories/preferences/theme.md"
        directory = "viking://user/alice/agent/bot/memories/preferences"

        schema = MemoryTypeSchema(
            memory_type="preferences",
            description="preferences memory",
            directory="viking://user/{{ user_space }}/memories/preferences",
            filename_template="{{ name }}.md",
            fields=[],
            overview_template="overview",
        )
        registry = MagicMock()
        registry.get.return_value = schema

        updater = MemoryUpdater(registry=registry)
        updater._get_viking_fs = MagicMock(return_value=MagicMock())
        updater._apply_upsert = AsyncMock(return_value=False)
        updater._vectorize_memories = AsyncMock()
        updater.generate_overview = AsyncMock()

        operations = ResolvedOperations(
            upsert_operations=[
                ResolvedOperation(
                    memory_fields={"name": "theme"},
                    memory_type="preferences",
                    uris=[uri],
                )
            ],
            delete_file_contents=[],
            errors=[],
        )
        ctx = RequestContext(user=UserIdentifier("acme", "alice", "bot"), role=Role.USER)

        result = await updater.apply_operations(operations=operations, ctx=ctx)

        assert result.written_uris == [uri]
        assert result.overview_directories == {directory: "preferences"}
        updater.generate_overview.assert_not_awaited()
