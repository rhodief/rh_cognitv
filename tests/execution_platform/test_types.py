"""Tests for types.py — ID generation, timestamps, type aliases."""

import re
from datetime import datetime, timezone

from rh_cognitv.execution_platform.types import (
    generate_ulid,
    now_timestamp,
    parse_timestamp,
)


class TestULID:
    def test_generate_ulid_returns_26_chars(self):
        ulid = generate_ulid()
        assert len(ulid) == 26

    def test_generate_ulid_is_crockford_base32(self):
        ulid = generate_ulid()
        assert re.match(r"^[0-9A-HJKMNP-TV-Z]{26}$", ulid)

    def test_generate_ulid_uniqueness(self):
        ids = {generate_ulid() for _ in range(1000)}
        assert len(ids) == 1000

    def test_generate_ulid_lexicographic_ordering(self):
        """ULIDs generated later should sort after earlier ones."""
        import time

        id1 = generate_ulid()
        time.sleep(0.002)  # small delay to ensure different ms
        id2 = generate_ulid()
        assert id1 < id2


class TestTimestamp:
    def test_now_timestamp_is_iso8601(self):
        ts = now_timestamp()
        # Should parse without error
        dt = datetime.fromisoformat(ts)
        assert dt.tzinfo is not None  # must be timezone-aware

    def test_now_timestamp_is_utc(self):
        ts = now_timestamp()
        dt = parse_timestamp(ts)
        assert dt.tzinfo == timezone.utc

    def test_parse_roundtrip(self):
        ts = now_timestamp()
        dt = parse_timestamp(ts)
        # Re-serialize and compare
        ts2 = dt.isoformat()
        assert ts == ts2
