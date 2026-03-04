"""Database infrastructure for Talking Rock projects.

Provides a thread-safe SQLite connection manager with WAL mode,
foreign keys, and optional SQLCipher encryption. Includes shared
tables (app_state, audit_log, repos) used by all agents. Projects
subclass Database and override migrate() to add their own schema.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Generator
from uuid import uuid4

from . import db_crypto
from .settings import settings


class Database:
    """Thread-safe SQLite database with WAL mode and optional encryption."""

    db_path: Path | str

    def __init__(self, db_path: Path | str | None = None) -> None:
        if db_path == ":memory:":
            self.db_path = ":memory:"
        elif db_path is None:
            self.db_path = settings.data_dir / "reos.db"
        elif isinstance(db_path, str):
            self.db_path = Path(db_path)
        else:
            self.db_path = db_path
        self._local = threading.local()

    def connect(self) -> sqlite3.Connection:
        """Open or return an existing connection."""
        conn: sqlite3.Connection | None = getattr(self._local, "conn", None)
        if conn is not None:
            return conn
        if isinstance(self.db_path, Path):
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = db_crypto.connect(
            str(self.db_path),
            timeout=5.0,
            check_same_thread=False,
        )
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute("PRAGMA journal_mode = WAL")
        conn.execute("PRAGMA busy_timeout = 5000")
        self._local.conn = conn
        return conn

    @contextmanager
    def transaction(self) -> Generator[sqlite3.Connection, None, None]:
        """Execute operations within an explicit transaction."""
        conn = self.connect()
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise

    def close(self) -> None:
        """Close the database connection."""
        conn = getattr(self._local, "conn", None)
        if conn is not None:
            conn.close()
            self._local.conn = None

    def _execute(self, query: str, params: tuple[object, ...] | None = None) -> sqlite3.Cursor:
        """Execute a query and return the cursor."""
        conn = self.connect()
        if params is None:
            return conn.execute(query)
        return conn.execute(query, params)

    def migrate(self) -> None:
        """Create shared tables. Override to add project-specific schema."""
        conn = self.connect()

        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS app_state (
                key TEXT PRIMARY KEY,
                value TEXT,
                updated_at TEXT NOT NULL
            )
            """
        )

        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS audit_log (
                id TEXT PRIMARY KEY,
                action TEXT NOT NULL,
                resource_type TEXT,
                resource_id TEXT,
                before_state TEXT,
                after_state TEXT,
                timestamp TEXT NOT NULL
            )
            """
        )

        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS events (
                id TEXT PRIMARY KEY,
                source TEXT NOT NULL,
                kind TEXT,
                ts TEXT NOT NULL,
                payload_metadata TEXT,
                note TEXT,
                created_at TEXT NOT NULL,
                ingested_at TEXT NOT NULL
            )
            """
        )

        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS repos (
                id TEXT PRIMARY KEY,
                path TEXT NOT NULL UNIQUE,
                remote_summary TEXT,
                first_seen_at TEXT NOT NULL,
                last_seen_at TEXT NOT NULL,
                created_at TEXT NOT NULL,
                ingested_at TEXT NOT NULL
            )
            """
        )

        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS agent_personas (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL UNIQUE,
                system_prompt TEXT NOT NULL,
                default_context TEXT NOT NULL,
                temperature REAL NOT NULL,
                top_p REAL NOT NULL,
                tool_call_limit INTEGER NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                ingested_at TEXT NOT NULL
            )
            """
        )

        conn.commit()

    # -----------------------------------------------------------------
    # App state (key-value)
    # -----------------------------------------------------------------

    def set_state(self, *, key: str, value: str | None) -> None:
        """Set a small piece of app state."""
        now = datetime.now(UTC).isoformat()
        self._execute(
            """
            INSERT INTO app_state (key, value, updated_at)
            VALUES (?, ?, ?)
            ON CONFLICT(key) DO UPDATE SET value = excluded.value, updated_at = excluded.updated_at
            """,
            (key, value, now),
        )
        self.connect().commit()

    def get_state(self, *, key: str) -> str | None:
        """Get a small piece of app state."""
        row = self._execute("SELECT value FROM app_state WHERE key = ?", (key,)).fetchone()
        if row is None:
            return None
        if isinstance(row, sqlite3.Row):
            value = row["value"]
            return str(value) if value is not None else None
        return str(row[0]) if row[0] is not None else None

    # -----------------------------------------------------------------
    # Events
    # -----------------------------------------------------------------

    def insert_event(
        self,
        event_id: str,
        source: str,
        kind: str | None,
        ts: str,
        payload_metadata: str | None,
        note: str | None,
    ) -> None:
        """Insert an event."""
        now = datetime.now(UTC).isoformat()
        self._execute(
            """
            INSERT INTO events
            (id, source, kind, ts, payload_metadata, note, created_at, ingested_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (event_id, source, kind, ts, payload_metadata, note, now, now),
        )
        self.connect().commit()

    def iter_events_recent(self, limit: int | None = None) -> list[dict[str, object]]:
        """Retrieve recent events."""
        if limit is None:
            limit = 1000
        rows = self._execute(
            "SELECT * FROM events ORDER BY ts DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [dict(row) for row in rows]

    # -----------------------------------------------------------------
    # Audit log
    # -----------------------------------------------------------------

    def insert_audit_event(
        self,
        event_type: str,
        timestamp: str,
        details: dict[str, object] | None = None,
        user: str | None = None,
        session_id: str | None = None,
        success: bool = True,
    ) -> None:
        """Insert a security audit event."""
        self._execute(
            """
            INSERT INTO audit_log
            (id, action, resource_type, resource_id, before_state, after_state, timestamp)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                str(uuid4()),
                event_type,
                user,
                session_id,
                json.dumps(details) if details is not None else None,
                json.dumps({"success": success}),
                timestamp,
            ),
        )
        self.connect().commit()

    # -----------------------------------------------------------------
    # Repos
    # -----------------------------------------------------------------

    def upsert_repo(self, *, repo_id: str, path: str, remote_summary: str | None = None) -> None:
        """Insert or update a discovered repo by path."""
        now = datetime.now(UTC).isoformat()
        row = self._execute("SELECT id, first_seen_at FROM repos WHERE path = ?", (path,)).fetchone()
        if row is None:
            self._execute(
                """
                INSERT INTO repos
                (id, path, remote_summary, first_seen_at, last_seen_at, created_at, ingested_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (repo_id, path, remote_summary, now, now, now, now),
            )
        else:
            existing_id = str(row["id"]) if isinstance(row, sqlite3.Row) else str(row[0])
            self._execute(
                """
                UPDATE repos
                SET remote_summary = COALESCE(?, remote_summary),
                    last_seen_at = ?,
                    ingested_at = ?
                WHERE id = ?
                """,
                (remote_summary, now, now, existing_id),
            )
        self.connect().commit()

    def iter_repos(self) -> list[dict[str, object]]:
        """Return discovered repos (most recently seen first)."""
        rows = self._execute(
            "SELECT * FROM repos ORDER BY last_seen_at DESC"
        ).fetchall()
        return [dict(row) for row in rows]

    def get_repo_path(self, *, repo_id: str) -> str | None:
        """Resolve a discovered repo's path by id."""
        row = self._execute("SELECT path FROM repos WHERE id = ?", (repo_id,)).fetchone()
        if row is None:
            return None
        if isinstance(row, sqlite3.Row):
            val = row["path"]
            return str(val) if val is not None else None
        return str(row[0]) if row[0] is not None else None

    # -----------------------------------------------------------------
    # Agent personas
    # -----------------------------------------------------------------

    def set_active_persona_id(self, *, persona_id: str | None) -> None:
        """Set the active agent persona id."""
        self.set_state(key="active_persona_id", value=persona_id)

    def get_active_persona_id(self) -> str | None:
        """Get the active agent persona id."""
        return self.get_state(key="active_persona_id")

    def upsert_agent_persona(
        self,
        *,
        persona_id: str,
        name: str,
        system_prompt: str,
        default_context: str,
        temperature: float,
        top_p: float,
        tool_call_limit: int,
    ) -> None:
        """Insert or update an agent persona."""
        now = datetime.now(UTC).isoformat()
        self._execute(
            """
            INSERT INTO agent_personas
            (id, name, system_prompt, default_context, temperature, top_p, tool_call_limit,
             created_at, updated_at, ingested_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                name = excluded.name,
                system_prompt = excluded.system_prompt,
                default_context = excluded.default_context,
                temperature = excluded.temperature,
                top_p = excluded.top_p,
                tool_call_limit = excluded.tool_call_limit,
                updated_at = excluded.updated_at,
                ingested_at = excluded.ingested_at
            """,
            (
                persona_id, name, system_prompt, default_context,
                float(temperature), float(top_p), int(tool_call_limit),
                now, now, now,
            ),
        )
        self.connect().commit()

    def get_agent_persona(self, *, persona_id: str) -> dict[str, object] | None:
        row = self._execute(
            "SELECT * FROM agent_personas WHERE id = ?",
            (persona_id,),
        ).fetchone()
        return dict(row) if row is not None else None

    def iter_agent_personas(self) -> list[dict[str, object]]:
        rows = self._execute(
            "SELECT * FROM agent_personas ORDER BY name ASC"
        ).fetchall()
        return [dict(row) for row in rows]


_db_instance: Database | None = None


def get_db() -> Database:
    """Get or create the global database instance."""
    global _db_instance
    if _db_instance is None:
        _db_instance = Database()
        _db_instance.migrate()
    return _db_instance
