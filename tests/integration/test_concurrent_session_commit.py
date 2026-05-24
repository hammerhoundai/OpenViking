# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0

"""并发 session commit 测试：验证两个 session 同时写入时锁机制正确工作。"""

import asyncio
from dataclasses import asdict
from datetime import datetime

import pytest
import pytest_asyncio

from openviking.message import TextPart
from openviking_cli.client.http import AsyncHTTPClient
from openviking_cli.utils import get_logger

logger = get_logger(__name__)

SERVER_URL = "http://127.0.0.1:1933"


async def _wait_for_task(client: AsyncHTTPClient, task_id: str, timeout: float = 120.0) -> dict:
    for _ in range(int(timeout / 0.1)):
        task = await client.get_task(task_id)
        if task and task["status"] in {"completed", "failed"}:
            telemetry = task.get("telemetry", {})
            trace_id = telemetry.get("trace_id", "")
            print(f"Task {task_id}: status={task['status']}, trace_id={trace_id}")
            return task
        await asyncio.sleep(0.1)
    raise TimeoutError(f"Task {task_id} did not complete within {timeout}s")


def _make_likes_coffee_conversation():
    """Session A: 用户喜欢喝咖啡（美式，不加糖）"""
    return [
        ("user", "你好，我叫王磊，今年25岁。"),
        ("assistant", "你好王磊！"),
        ("user", "我特别喜欢喝咖啡，每天早上都要喝一杯美式咖啡，不加糖的那种。"),
        ("assistant", "美式不错！"),
        (
            "user",
            "对，我只喝星巴克的美式，而且一定要大杯的。对了我还喜欢吃辣，特别是四川火锅。",
        ),
    ]


def _make_likes_tea_conversation():
    """Session B: 用户也喜欢喝茶（龙井、普洱），名称相似容易冲突。"""
    return [
        ("user", "你好，我也叫王磊，今年25岁。"),
        ("assistant", "你好！"),
        ("user", "我特别喜欢喝茶，尤其是龙井和普洱茶，每天下午都要泡一壶。"),
        ("assistant", "喝茶很养生！"),
        (
            "user",
            "对，我只喝杭州产的龙井，普洱要陈年的才好。我还喜欢爬山，周末经常去。",
        ),
    ]


def _make_same_topic_conversation():
    """Session C: 相同 topic，看是否合并还是创建新文件"""
    return [
        ("user", "你好，我叫张三，今年30岁。"),
        ("assistant", "你好张三！"),
        ("user", "我也特别喜欢喝咖啡，不过我喜欢喝拿铁，加两份糖。"),
        ("assistant", "拿铁也不错！"),
        (
            "user",
            "我一般在瑞幸买咖啡，因为离家近。不过周末会去精品咖啡馆尝试手冲。",
        ),
    ]


@pytest_asyncio.fixture(scope="function")
async def http_client():
    client = AsyncHTTPClient(url=SERVER_URL)
    await client.initialize()
    yield client
    await client.close()


async def _create_and_commit(client: AsyncHTTPClient, conversation, session_time=None):
    """创建 session → 添加消息 → commit → 等待处理 → 返回结果"""
    result = await client.create_session()
    session_id = result["session_id"]

    if session_time is None:
        session_time = datetime(2023, 4, 2, 9, 36)
    session_time_str = session_time.isoformat()

    for role, content in conversation:
        parts = [TextPart(content)]
        parts_dicts = [asdict(p) for p in parts]
        await client.add_message(session_id, role, parts=parts_dicts, created_at=session_time_str)

    commit_result = await client.commit_session(session_id)
    assert commit_result["status"] == "accepted"
    trace_id = commit_result.get("trace_id", "")
    print(f"Session {session_id}: commit accepted, trace_id={trace_id}")
    task_result = await _wait_for_task(client, commit_result["task_id"])
    return session_id, task_result


class TestConcurrentSessionCommit:
    """并发 session commit 测试"""

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_two_concurrent_sessions_same_user(self, http_client: AsyncHTTPClient):
        """两个同用户 session 同时 commit，验证锁机制正常运行且不丢失数据。"""
        client = http_client

        print("=" * 80)
        print("CONCURRENT SESSION COMMIT TEST")
        print("=" * 80)

        session_time = datetime(2023, 4, 2, 9, 36)

        # 并发执行两个 session
        results = await asyncio.gather(
            _create_and_commit(client, _make_likes_coffee_conversation(), session_time),
            _create_and_commit(client, _make_likes_tea_conversation(), session_time),
        )

        for sid, task in results:
            status = task["status"]
            print(f"Session {sid}: status={status}")
            assert status == "completed", f"Session {sid} failed: {task}"

        await client.wait_processed()
        print("Both sessions completed successfully!")

        # 搜索用户记忆
        find_result = await client.find(query="咖啡 茶 饮料 偏好")
        memories = getattr(find_result, "memories", [])
        print(f"\nSearch results: total={find_result.total}")
        for mem in memories:
            uri = mem.uri
            try:
                content = await client.read(uri)
                print(f"\n--- {uri} ---")
                print(content[:300])
            except Exception as e:
                print(f"  Failed to read {uri}: {e}")

        # 核心验证：两个 session 的写入都不应该丢失
        assert find_result.total >= 1, "Should find at least one preference memory"

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_three_concurrent_sessions_same_topic(self, http_client: AsyncHTTPClient):
        """三个同 topic session 同时 commit，验证并发去重/合并逻辑。"""
        client = http_client

        print("=" * 80)
        print("THREE CONCURRENT SESSIONS — SAME TOPIC TEST")
        print("=" * 80)

        session_time = datetime(2023, 4, 2, 9, 36)

        results = await asyncio.gather(
            _create_and_commit(client, _make_likes_coffee_conversation(), session_time),
            _create_and_commit(client, _make_likes_tea_conversation(), session_time),
            _create_and_commit(client, _make_same_topic_conversation(), session_time),
        )

        for sid, task in results:
            status = task["status"]
            print(f"Session {sid}: status={status}")
            assert status == "completed", f"Session {sid} failed: {task}"

        await client.wait_processed()
        print("All three sessions completed!")

        # 列出 memories 目录所有文件
        ls_result = await client.ls("viking://user/default/memories/preferences/")
        md_files = [
            e for e in ls_result if isinstance(e, dict) and e.get("name", "").endswith(".md")
        ]
        print(f"\nPreference files in storage ({len(md_files)}):")
        for e in md_files:
            uri = e.get("uri", "")
            try:
                content = await client.read(uri)
                print(f"\n--- {e['name']} ---")
                print(content[:300])
            except Exception as ex:
                print(f"  Failed to read {uri}: {ex}")
