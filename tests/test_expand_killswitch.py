"""Expand loop respects the owner kill switch (default on, fail closed)."""
from __future__ import annotations

import expand


def test_owner_stop_fail_closed_without_source(monkeypatch):
    monkeypatch.delenv("ARTEMIS_KILLSWITCH_DB", raising=False)
    monkeypatch.delenv("ARTEMIS_KILLSWITCH_STATE_URL", raising=False)
    monkeypatch.setenv("ARTEMIS_KILLSWITCH", "1")
    assert expand.owner_stop_active() is True


def test_owner_stop_can_be_disabled(monkeypatch):
    monkeypatch.setenv("ARTEMIS_KILLSWITCH", "0")
    assert expand.owner_stop_active() is False


def test_owner_stop_reads_sqlite(tmp_path, monkeypatch):
    import sqlite3
    db = tmp_path / "ks.db"
    conn = sqlite3.connect(db)
    conn.execute("CREATE TABLE killswitch_state (id INTEGER PRIMARY KEY, stopped INTEGER NOT NULL, updated REAL, request_id TEXT, device_id TEXT)")
    conn.execute("INSERT INTO killswitch_state VALUES (1, 0, 0, NULL, NULL)")
    conn.commit(); conn.close()
    monkeypatch.setenv("ARTEMIS_KILLSWITCH", "1")
    monkeypatch.setenv("ARTEMIS_KILLSWITCH_DB", str(db))
    assert expand.owner_stop_active() is False
    conn = sqlite3.connect(db)
    conn.execute("UPDATE killswitch_state SET stopped = 1")
    conn.commit(); conn.close()
    assert expand.owner_stop_active() is True
