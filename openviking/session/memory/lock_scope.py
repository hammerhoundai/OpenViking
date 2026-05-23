# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Helpers for tracking file-scope memory locks."""

from __future__ import annotations

from typing import Iterable, Optional

from openviking.storage.transaction.lock_handle import LockHandle
from openviking.storage.transaction.lock_manager import LockManager


class LockScope:
    def __init__(self, lock_manager: LockManager, handle: LockHandle):
        self._lock_manager = lock_manager
        self._handle = handle
        self._target_paths: list[str] = []
        self._lock_paths_by_target: dict[str, list[str]] = {}

    @property
    def handle(self) -> LockHandle:
        return self._handle

    @property
    def target_paths(self) -> list[str]:
        return list(self._target_paths)

    def has_target(self, path: str) -> bool:
        normalized = str(path or "").strip()
        return bool(normalized) and normalized in self._lock_paths_by_target

    def lock_paths_for(self, path: str) -> list[str]:
        normalized = str(path or "").strip()
        if not normalized:
            return []
        return list(self._lock_paths_by_target.get(normalized, []))

    async def relock_to(self, target_paths: Iterable[str], timeout: Optional[float] = None) -> None:
        desired_paths = self._normalize_target_paths(target_paths)
        desired_set = set(desired_paths)

        removed_targets = [path for path in self._target_paths if path not in desired_set]
        if removed_targets:
            await self._release_targets(removed_targets)

        added_targets = [path for path in desired_paths if path not in self._lock_paths_by_target]
        if added_targets:
            await self._acquire_targets(added_targets, timeout=timeout)

        self._target_paths = desired_paths

    async def release(self) -> None:
        await self._lock_manager.release(self._handle)
        self._target_paths = []
        self._lock_paths_by_target.clear()

    async def _release_targets(self, targets: Iterable[str]) -> None:
        lock_paths: list[str] = []
        for target in targets:
            lock_paths.extend(self._lock_paths_by_target.pop(target, []))
        if lock_paths:
            await self._lock_manager.release_selected(self._handle, lock_paths)

    async def _acquire_targets(self, targets: Iterable[str], timeout: Optional[float]) -> None:
        acquired_by_target: dict[str, list[str]] = {}
        acquired_lock_paths: list[str] = []

        try:
            for target in self._normalize_target_paths(targets):
                locks_before = set(self._handle.locks)
                acquired = await self._lock_manager.acquire_exact_path(
                    self._handle,
                    target,
                    timeout=timeout,
                )
                if not acquired:
                    raise RuntimeError(f"Failed to acquire exact lock for path: {target}")
                new_lock_paths = [
                    lock_path for lock_path in self._handle.locks if lock_path not in locks_before
                ]
                acquired_by_target[target] = new_lock_paths
                acquired_lock_paths.extend(new_lock_paths)
        except Exception:
            if acquired_lock_paths:
                await self._lock_manager.release_selected(self._handle, acquired_lock_paths)
            raise

        self._lock_paths_by_target.update(acquired_by_target)

    @staticmethod
    def _normalize_target_paths(target_paths: Iterable[str]) -> list[str]:
        seen = set()
        normalized: list[str] = []
        for path in target_paths or []:
            value = str(path or "").strip()
            if not value or value in seen:
                continue
            seen.add(value)
            normalized.append(value)
        return sorted(normalized, key=lambda value: (len(value), value))
