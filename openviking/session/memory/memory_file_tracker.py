# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Helpers for tracking concrete memory file URIs during extraction."""

from __future__ import annotations

from typing import Iterable


class MemoryFileTracker:
    def __init__(self):
        self._tracked_uris: dict[str, None] = {}
        self._read_uris: dict[str, None] = {}

    @property
    def tracked_uris(self) -> list[str]:
        return list(self._tracked_uris)

    @property
    def read_uris(self) -> list[str]:
        return list(self._read_uris)

    def track(self, uri: str) -> None:
        normalized = self._normalize_uri(uri)
        if normalized:
            self._tracked_uris.setdefault(normalized, None)

    def track_many(self, uris: Iterable[str]) -> None:
        for uri in uris or []:
            self.track(str(uri))

    def mark_read(self, uri: str) -> None:
        normalized = self._normalize_uri(uri)
        if not normalized:
            return
        self._tracked_uris.setdefault(normalized, None)
        self._read_uris.setdefault(normalized, None)

    def mark_read_many(self, uris: Iterable[str]) -> None:
        for uri in uris or []:
            self.mark_read(str(uri))

    def is_tracked(self, uri: str) -> bool:
        normalized = self._normalize_uri(uri)
        return bool(normalized) and normalized in self._tracked_uris

    def is_read(self, uri: str) -> bool:
        normalized = self._normalize_uri(uri)
        return bool(normalized) and normalized in self._read_uris

    def unread(self, uris: Iterable[str]) -> list[str]:
        unread_uris: list[str] = []
        seen = set()
        for uri in uris or []:
            normalized = self._normalize_uri(str(uri))
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            if normalized not in self._read_uris:
                unread_uris.append(normalized)
        return unread_uris

    @staticmethod
    def _normalize_uri(uri: str) -> str:
        return str(uri or "").strip()
