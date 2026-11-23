"""Transactional attempt accounting and resumable artifacts, scoped to one run."""

from contextlib import closing, contextmanager
import hashlib
import json
import sqlite3
import time
from pathlib import Path

from ..errors import LLMUnavailableError
from ..utils import new_id, now_iso

NON_QUOTA_STATUSES = ("not_sent", "rate_limited", "overloaded")
NO_BILLING_STATUSES = ("not_sent", "rate_limited", "overloaded", "http_rejected")


def fingerprint(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False, default=str).encode()
    ).hexdigest()


class DeferredCall(LLMUnavailableError):
    """No send: the route needs to cool down or the operation has no retry budget."""

    def __init__(self, reason, resume_at=None):
        super().__init__(reason)
        self.resume_at = resume_at


class PermanentCall(LLMUnavailableError):
    """The provider rejected this request; regenerating the same operation is futile."""


class CancelledCall(LLMUnavailableError):
    """The local caller cancelled this request."""


class Ledger:
    def __init__(self, path, extras=2, *, rate_limit_retries=8, run_id=None):
        self.path, self.extras = Path(path), extras
        self.rate_limit_retries = int(rate_limit_retries)
        self.run_id = run_id
        self.limit = None
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if run_id:
            with closing(sqlite3.connect(self.path, timeout=10)) as conn:
                conn.execute("PRAGMA journal_mode=WAL").fetchone()
        with self.db() as db:
            db.executescript("""
                CREATE TABLE IF NOT EXISTS ops(id TEXT PRIMARY KEY, signature TEXT, status TEXT, available REAL);
                CREATE TABLE IF NOT EXISTS stages(op TEXT, name TEXT, sig TEXT, data TEXT, PRIMARY KEY(op,name,sig));
                CREATE TABLE IF NOT EXISTS attempts(id TEXT PRIMARY KEY, op TEXT, role TEXT, call_id TEXT, fingerprint TEXT,
                  status TEXT, reserve REAL, cost REAL, usage TEXT, started REAL, finished REAL);
            """)
            if run_id:
                db.executescript("""
                    BEGIN IMMEDIATE;
                    CREATE TABLE IF NOT EXISTS run_identity(run_id TEXT PRIMARY KEY);
                    CREATE TABLE IF NOT EXISTS receipts(
                      receipt_seq INTEGER PRIMARY KEY, event_type TEXT NOT NULL, event_at TEXT NOT NULL,
                      attempt_id TEXT NOT NULL, payload_json TEXT NOT NULL,
                      UNIQUE(attempt_id,event_type,payload_json));
                    CREATE VIEW IF NOT EXISTS api_usage_events_v1 AS
                    SELECT run_id,receipt_seq,event_type,event_at,attempt_id,payload_json
                    FROM receipts CROSS JOIN run_identity;
                    CREATE VIEW IF NOT EXISTS api_usage_run_v1 AS SELECT run_id FROM run_identity;
                    CREATE TABLE IF NOT EXISTS experiment_facts(
                      seq INTEGER PRIMARY KEY, operation_id TEXT UNIQUE NOT NULL, data TEXT NOT NULL);
                """)
                existing = db.execute("SELECT run_id FROM run_identity").fetchone()
                if existing and existing[0] != run_id:
                    raise ValueError("operations.sqlite belongs to a different run")
                db.execute("INSERT OR IGNORE INTO run_identity VALUES (?)", (run_id,))

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

    def register_operation(self, op, sig, normal_roles):
        with self.db() as db:
            db.execute("INSERT INTO ops VALUES(?,?,'claimed',0)", (op, sig))
            db.execute("INSERT INTO stages VALUES(?,'normal_roles','v1',?)", (op, json.dumps(list(normal_roles))))

    def record_experiment(self, op, data, terminal):
        with self.db() as db:
            previous = db.execute("SELECT seq FROM experiment_facts WHERE operation_id=?", (op,)).fetchone()
            if previous:
                return previous[0]
            seq = db.execute("SELECT COALESCE(MAX(seq),0)+1 FROM experiment_facts").fetchone()[0]
            db.execute("INSERT INTO experiment_facts VALUES (?,?,?)", (seq, op, json.dumps({**data, "seq": seq})))
            db.execute("UPDATE ops SET status=?,available=0 WHERE id=?", (terminal, op))
            return seq

    def experiments(self, as_of_seq=None, *, operation_id=None):
        with self.db() as db:
            rows = db.execute(
                "SELECT data FROM experiment_facts WHERE (? IS NULL OR seq<=?) AND "
                "(? IS NULL OR operation_id=?) ORDER BY seq",
                (as_of_seq, as_of_seq, operation_id, operation_id),
            ).fetchall()
        return [json.loads(row[0]) for row in rows]

    @staticmethod
    def _event(db, attempt, event_type, payload):
        db.execute(
            "INSERT OR IGNORE INTO receipts(event_type,event_at,attempt_id,payload_json) VALUES(?,?,?,?)",
            (event_type, now_iso(), attempt, json.dumps(payload, sort_keys=True)),
        )

    def begin_request(self, op, role, call_id, sig, options):
        attempt = new_id()
        with self.db() as db:
            graph = db.execute("SELECT data FROM stages WHERE op=? AND name='normal_roles'", (op,)).fetchone()
            normal_roles = set(json.loads(graph[0])) if graph else {role}
            rows = db.execute("SELECT role, status FROM attempts WHERE op=?", (op,)).fetchall()
            non_quota = sum(1 for _, st in rows if st in NON_QUOTA_STATUSES)
            prior_non_quota = sum(1 for r, st in rows if r == role and st in NON_QUOTA_STATUSES)
            if prior_non_quota and non_quota > self.rate_limit_retries:
                raise PermanentCall("rate_limit_retry_exhausted")
            sent = [r for r, st in rows if st not in NON_QUOTA_STATUSES]
            extras = len(sent) - len(set(sent) & normal_roles)
            if (role not in normal_roles or role in sent) and extras >= self.extras:
                raise PermanentCall("retry_exhausted")
            if role in {"script_coder", "draft_integrator"} and role not in normal_roles and role in sent:
                # Additional recovery of the same fallback is allowed under its logical call ID.
                marks = ",".join("?" for _ in NON_QUOTA_STATUSES)
                first = db.execute(
                    f"SELECT call_id FROM attempts WHERE op=? AND role=? AND status NOT IN ({marks}) ORDER BY started LIMIT 1",
                    (op, role, *NON_QUOTA_STATUSES),
                ).fetchone()
                if first and first[0] != call_id:
                    raise PermanentCall("full_script_fallback_already_used")
            db.execute(
                "INSERT INTO attempts VALUES(?,?,?,?,?,'in_flight',0,NULL,NULL,?,NULL)",
                (attempt, op, role, call_id, sig, time.time()),
            )
            self._event(db, attempt, "attempt_started", {
                "operation_id": op, "logical_call_id": call_id, "role": role,
                "model": options["model"], "thinking": options["thinking"],
                "reasoning_effort": options["reasoning_effort"], "max_output_tokens": options["max_tokens"],
                "prompt_bytes": sum(len(m["content"].encode()) for m in options["messages"]),
            })
        return attempt

    def receive_usage(self, attempt, usage):
        with self.db() as db:
            row = db.execute("SELECT usage FROM attempts WHERE id=?", (attempt,)).fetchone()
            if row is None:
                raise ValueError(f"usage has no attempt {attempt}")
            merged = json.loads(row[0]) if row[0] else {}
            merged.update({key: value for key, value in usage.items() if value is not None})
            db.execute("UPDATE attempts SET usage=? WHERE id=?", (json.dumps(merged), attempt))
            self._event(db, attempt, "usage_received", merged)

    def finish_request(self, attempt, status, reason, latency, **timing):
        cost_val = 0.0 if status in NO_BILLING_STATUSES else None
        with self.db() as db:
            if cost_val is not None:
                db.execute("UPDATE attempts SET status=?,cost=?,finished=? WHERE id=? AND status='in_flight'",
                           (status, cost_val, time.time(), attempt))
            else:
                db.execute("UPDATE attempts SET status=?,finished=? WHERE id=? AND status='in_flight'",
                           (status, time.time(), attempt))
            self._event(db, attempt, "attempt_finished", {
                "status": status, "finish_reason": reason, "latency_seconds": latency,
                "not_sent": status == "not_sent", **timing,
            })

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
            # Non-quota transport/rate-limit statuses are bounded separately by rate_limit_retries.
            if op:
                rows = db.execute("SELECT role, status FROM attempts WHERE op=?", (op,)).fetchall()
                non_quota = sum(1 for _, st in rows if st in NON_QUOTA_STATUSES)
                prior_non_quota = sum(1 for r, st in rows if r == role and st in NON_QUOTA_STATUSES)
                if prior_non_quota and non_quota > self.rate_limit_retries:
                    raise DeferredCall("operation rate-limit retry budget exhausted")
                quota_roles = [r for r, st in rows if st not in NON_QUOTA_STATUSES]
                sends = len(quota_roles)
                roles = len(set(quota_roles))
                used = sum(1 for r in quota_roles if r == role)
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
        if status in NO_BILLING_STATUSES:
            cost = 0.0
        with self.db() as db:
            db.execute(
                "UPDATE attempts SET status=?,cost=?,usage=?,finished=? WHERE id=? AND (cost IS NULL OR ? IS NOT NULL)",
                (status, cost, json.dumps(usage), time.time(), attempt, cost),
            )

    def sends(self, call_id):
        with self.db() as db:
            marks = ",".join("?" for _ in NON_QUOTA_STATUSES)
            return db.execute(
                f"SELECT count(*) FROM attempts WHERE call_id=? AND status NOT IN ({marks})",
                (call_id, *NON_QUOTA_STATUSES),
            ).fetchone()[0]

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
