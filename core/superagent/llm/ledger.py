"""Transactional attempt accounting and resumable artifacts, scoped to one run."""

from contextlib import contextmanager
import hashlib
import json
import sqlite3
import time
from pathlib import Path

from ..errors import LLMUnavailableError
from ..utils import new_id


def fingerprint(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False, default=str).encode()
    ).hexdigest()


class DeferredCall(LLMUnavailableError):
    """No send: the route needs to cool down or the operation has no retry budget."""


class PermanentCall(LLMUnavailableError):
    """The provider rejected this request; regenerating the same operation is futile."""


class Ledger:
    def __init__(self, path, extras=2):
        self.path, self.extras = Path(path), extras
        self.limit = None
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.db() as db:
            db.executescript("""
                CREATE TABLE IF NOT EXISTS ops(id TEXT PRIMARY KEY, signature TEXT, status TEXT, available REAL);
                CREATE TABLE IF NOT EXISTS stages(op TEXT, name TEXT, sig TEXT, data TEXT, PRIMARY KEY(op,name,sig));
                CREATE TABLE IF NOT EXISTS attempts(id TEXT PRIMARY KEY, op TEXT, role TEXT, call_id TEXT, fingerprint TEXT,
                  status TEXT, reserve REAL, cost REAL, usage TEXT, started REAL, finished REAL);
            """)

    @contextmanager
    def db(self):
        db = sqlite3.connect(self.path, timeout=10)
        try:
            with db:
                db.execute("BEGIN IMMEDIATE")
                yield db
        finally:
            db.close()

    def claim(self, sig):
        with self.db() as db:
            row = db.execute(
                "SELECT id,available FROM ops WHERE signature=? AND status='suspended' ORDER BY available LIMIT 1",
                (sig,),
            ).fetchone()
            if row and row[1] > time.time():
                raise DeferredCall("operation cooling down")
            op = row[0] if row else new_id()
            db.execute(
                "INSERT OR REPLACE INTO ops VALUES(?,?,?,0)", (op, sig, "active")
            )
            return op

    def state(self, op, status, delay=0):
        with self.db() as db:
            db.execute(
                "UPDATE ops SET status=?,available=? WHERE id=?",
                (status, time.time() + delay, op),
            )

    def get(self, op, name, sig):
        with self.db() as db:
            row = db.execute(
                "SELECT data FROM stages WHERE op=? AND name=? AND sig=?",
                (op, name, sig),
            ).fetchone()
        return json.loads(row[0]) if row else None

    def put(self, op, name, sig, value):
        with self.db() as db:
            db.execute(
                "INSERT OR REPLACE INTO stages VALUES(?,?,?,?)",
                (op, name, sig, json.dumps(value)),
            )

    def begin(self, op, role, call_id, sig, reserve):
        attempt = new_id()
        with self.db() as db:
            # Each first role invocation buys one send; all additional sends share one wallet.
            if op:
                sends, roles, used = db.execute(
                    "SELECT count(*),count(DISTINCT role),count(CASE WHEN role=? THEN 1 END) FROM attempts WHERE op=?",
                    (role, op),
                ).fetchone()
                if used and sends - roles >= self.extras:
                    raise DeferredCall("operation recovery budget exhausted")
            committed = db.execute(
                "SELECT COALESCE(sum(COALESCE(cost,reserve)),0) FROM attempts"
            ).fetchone()[0]
            if self.limit is not None and committed + reserve > self.limit:
                raise DeferredCall("cost limit including in-flight and unknown usage")
            db.execute(
                "INSERT INTO attempts VALUES(?,?,?,?,?,'in_flight',?,NULL,NULL,?,NULL)",
                (attempt, op, role, call_id, sig, reserve, time.time()),
            )
        return attempt

    def finish(self, attempt, cost, usage, status):
        with self.db() as db:
            db.execute(
                "UPDATE attempts SET status=?,cost=?,usage=?,finished=? WHERE id=? AND (cost IS NULL OR ? IS NOT NULL)",
                (status, cost, json.dumps(usage), time.time(), attempt, cost),
            )

    def sends(self, call_id):
        with self.db() as db:
            return db.execute("SELECT count(*) FROM attempts WHERE call_id=?", (call_id,)).fetchone()[0]

    def charges(self, operations):
        """Known cost plus pending/unknown reservations per requested operation."""
        operations = list(dict.fromkeys(operations))
        if not operations:
            return {}
        marks = ",".join("?" for _ in operations)
        with self.db() as db:
            return dict(db.execute(
                f"SELECT op,sum(coalesce(cost,reserve)) FROM attempts WHERE op IN ({marks}) GROUP BY op",
                operations,
            ))

    def summary(self):
        with self.db() as db:
            known, unknown, reserved = db.execute(
                "SELECT COALESCE(sum(cost),0),count(CASE WHEN cost IS NULL THEN 1 END),COALESCE(sum(CASE WHEN cost IS NULL THEN reserve ELSE 0 END),0) FROM attempts"
            ).fetchone()
            pending = db.execute(
                "SELECT id,status,available FROM ops WHERE status IN ('active','suspended','generated')"
            ).fetchall()
        return dict(
            known_cost=known,
            unknown_or_inflight=unknown,
            risk_reserve=reserved,
            unfinished_operations=pending,
        )
