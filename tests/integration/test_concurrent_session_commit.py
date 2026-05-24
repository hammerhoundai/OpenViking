# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0

"""并发 session commit 测试：验证多个 session 同时写入时锁机制正确工作。"""

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
CONCURRENCY = 20


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


_CONVERSATIONS = [
    (
        "用户喜欢喝咖啡",
        [
            ("user", "你好，我叫王磊，今年25岁。"),
            ("assistant", "你好王磊！"),
            ("user", "我特别喜欢喝咖啡，每天早上都要喝一杯美式咖啡，不加糖的那种。"),
            ("assistant", "美式不错！"),
            ("user", "我只喝星巴克的美式，而且一定要大杯的。对了我还喜欢吃辣，特别是四川火锅。"),
        ],
    ),
    (
        "用户喜欢喝茶",
        [
            ("user", "你好，我也叫王磊，今年25岁。"),
            ("assistant", "你好！"),
            ("user", "我特别喜欢喝茶，尤其是龙井和普洱茶，每天下午都要泡一壶。"),
            ("assistant", "喝茶很养生！"),
            ("user", "我只喝杭州产的龙井，普洱要陈年的才好。我还喜欢爬山，周末经常去。"),
        ],
    ),
    (
        "用户喜欢运动",
        [
            ("user", "你好，我叫李华，今年28岁。"),
            ("assistant", "你好李华！"),
            ("user", "我每天坚持跑步10公里，已经跑了三年了，参加过五次马拉松。"),
            ("assistant", "太厉害了！"),
            ("user", "跑步让我精力充沛，我还喜欢游泳和骑行，铁人三项是我的目标。"),
        ],
    ),
    (
        "用户喜欢音乐",
        [
            ("user", "你好，我是小美。"),
            ("assistant", "你好小美！"),
            ("user", "我学了十年钢琴，最近在学吉他，音乐是我生活中最重要的一部分。"),
            ("assistant", "音乐确实能治愈人心。"),
            ("user", "我最喜欢肖邦的夜曲，吉他的话喜欢弹唱民谣，周杰伦的歌我都会弹。"),
        ],
    ),
    (
        "用户喜欢读书",
        [
            ("user", "我叫大刘，是一名程序员。"),
            ("assistant", "你好大刘！"),
            ("user", "业余时间我喜欢读科幻小说，刘慈欣的三体我看了三遍。"),
            ("assistant", "三体确实是经典。"),
            ("user", "除了科幻，我也喜欢读历史书，最近在看《明朝那些事儿》，觉得明朝很有意思。"),
        ],
    ),
    (
        "用户喜欢旅行",
        [
            ("user", "你好，叫我阿飞就行。"),
            ("assistant", "阿飞你好！"),
            ("user", "我最大的爱好是旅行，已经走遍了全国所有省份。"),
            ("assistant", "好羡慕！"),
            ("user", "我最喜欢云南和新疆，那里的自然风光太美了。下一个目标是去西藏徒步。"),
        ],
    ),
    (
        "用户喜欢编程",
        [
            ("user", "我是小明，今年22岁，刚入行的程序员。"),
            ("assistant", "小明加油！"),
            ("user", "我主要写 Python 和 Go，最近在学 Rust，感觉内存管理好难。"),
            ("assistant", "Rust 上手确实有门槛。"),
            ("user", "编程让我着迷的是解决问题的快感，周末我也会参与一些开源项目。"),
        ],
    ),
    (
        "用户喜欢美食",
        [
            ("user", "大家好，我叫胖虎，是个吃货。"),
            ("assistant", "胖虎你好！"),
            ("user", "我的人生信条是：没有什么是一顿火锅解决不了的。"),
            ("assistant", "如果有呢？"),
            ("user", "那就两顿！我吃过全国各地的火锅，重庆老火锅最对胃口，毛肚必点。"),
        ],
    ),
    (
        "用户喜欢宠物",
        [
            ("user", "你好，我叫小芳，今年24岁。"),
            ("assistant", "你好小芳！"),
            ("user", "我养了一只金毛和两只猫，每天回家看到它们就觉得特别幸福。"),
            ("assistant", "养宠物确实很治愈。"),
            (
                "user",
                "金毛叫豆豆特别粘人，两只猫一个叫花花一个叫咪咪。我还经常去流浪动物救助站做义工。",
            ),
        ],
    ),
    (
        "用户喜欢摄影",
        [
            ("user", "我是一名摄影爱好者，叫我老陈就行。"),
            ("assistant", "老陈你好！"),
            ("user", "我玩摄影十年了，从佳能到索尼，拍过的照片有几万张。"),
            ("assistant", "真是资深玩家了。"),
            ("user", "我最喜欢拍风景和人文纪实，西藏和新疆是我去过最适合拍照的地方，光影太美了。"),
        ],
    ),
]


@pytest_asyncio.fixture(scope="function")
async def http_client():
    client = AsyncHTTPClient(url=SERVER_URL)
    await client.initialize()
    yield client
    await client.close()


async def _create_and_commit(client: AsyncHTTPClient, conversation, session_time=None):
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
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_concurrent_sessions(self, http_client: AsyncHTTPClient):
        client = http_client

        print("=" * 80)
        print(f"CONCURRENT SESSION COMMIT TEST — {CONCURRENCY} sessions")
        print("=" * 80)

        start_time = datetime.now()

        coros = []
        for i in range(CONCURRENCY):
            _, conv = _CONVERSATIONS[i % len(_CONVERSATIONS)]
            session_time = datetime(2023, 4, 2, 9, i // 60, i % 60)
            coros.append(_create_and_commit(client, conv, session_time))

        results = await asyncio.gather(*coros)

        elapsed = (datetime.now() - start_time).total_seconds()
        print(f"\nAll {CONCURRENCY} sessions finished in {elapsed:.1f}s")

        failed = []
        for sid, task in results:
            status = task["status"]
            if status != "completed":
                failed.append((sid, status, task))
                print(f"  FAILED: {sid}: status={status}")
            else:
                print(f"  OK: {sid}")

        assert len(failed) == 0, f"{len(failed)}/{CONCURRENCY} sessions failed: {failed}"

        await client.wait_processed()
        print(f"\nAll {CONCURRENCY} sessions completed successfully!")
