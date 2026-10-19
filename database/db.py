

import sqlite3
from pathlib import Path
import json
from datetime import datetime

# =========================================================
# DATABASE PATH
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATABASE_DIR = PROJECT_ROOT / "data" / "database"
DATABASE_PATH = DATABASE_DIR / "support_tickets.db"


# =========================================================
# DATABASE CONNECTION
# =========================================================

def get_connection():
    """Create and return a SQLite database connection."""

    DATABASE_DIR.mkdir(parents=True, exist_ok=True)

    connection = sqlite3.connect(str(DATABASE_PATH))
    connection.row_factory = sqlite3.Row

    # Enforce foreign-key relationships
    connection.execute("PRAGMA foreign_keys = ON")

    return connection


# =========================================================
# CREATE DATABASE TABLES
# =========================================================

def initialize_database():
    """Create the ticket and agent-log tables if needed."""

    with get_connection() as connection:

        connection.execute("""
            CREATE TABLE IF NOT EXISTS tickets (
                ticket_id TEXT PRIMARY KEY,
                ticket_text TEXT NOT NULL,

                intent TEXT,
                intent_confidence REAL,

                sentiment TEXT,
                sentiment_confidence REAL,

                priority TEXT,
                priority_confidence REAL,

                retrieval_top_score REAL,
                response TEXT,

                escalated INTEGER NOT NULL DEFAULT 0,
                escalation_reason TEXT,

                status TEXT NOT NULL,
                created_at TEXT NOT NULL,
                logged_at TEXT NOT NULL
            )
        """)

        
        # Add human-review fields to existing databases.
        # Existing ticket records are preserved.
        existing_columns = {
            row["name"]
            for row in connection.execute(
                "PRAGMA table_info(tickets)"
            ).fetchall()
        }

        review_columns = {
            "human_review_required": (
                "INTEGER NOT NULL DEFAULT 0"
            ),
            "review_decision": "TEXT",
            "review_notes": "TEXT",
            "reviewed_at": "TEXT",
            "reviewed_by": "TEXT",
        }

        for column_name, column_type in review_columns.items():
            if column_name not in existing_columns:
                connection.execute(
                    f"ALTER TABLE tickets ADD COLUMN "
                    f"{column_name} {column_type}"
                )


        connection.execute("""
            CREATE TABLE IF NOT EXISTS agent_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ticket_id TEXT NOT NULL,
                agent_name TEXT NOT NULL,
                action TEXT,
                details_json TEXT,
                timestamp TEXT,

                FOREIGN KEY (ticket_id)
                    REFERENCES tickets(ticket_id)
                    ON DELETE CASCADE
            )
        """)

        connection.execute("""
            CREATE INDEX IF NOT EXISTS idx_tickets_created_at
            ON tickets(created_at)
        """)

        connection.execute("""
            CREATE INDEX IF NOT EXISTS idx_agent_logs_ticket_id
            ON agent_logs(ticket_id)
        """)

    print("SQLite database initialized successfully.")
    print(f"Database location: {DATABASE_PATH}")



def save_ticket(state: dict) -> None:
    """
    Save a processed ticket and its agent logs to SQLite.

    Parameters
    ----------
    state : dict
        Final state returned by the LangGraph workflow.
    """

    ticket_id = state.get("ticket_id")
    ticket_text = state.get("ticket_text")

    if not ticket_id or not ticket_text:
        raise ValueError(
            "ticket_id and ticket_text are required."
        )

    now = datetime.now().isoformat()

    # Use the original intake timestamp when available.
    logs = state.get("agent_logs", [])

    created_at = next(
        (
            log["timestamp"]
            for log in logs
            if log.get("agent") == "Intake Agent"
            and log.get("timestamp")
        ),
        now,
    )

    processing_record = state.get("processing_record", {})

    ticket_values = {
        "ticket_id": ticket_id,
        "ticket_text": ticket_text,
        "intent": state.get("intent"),
        "intent_confidence": state.get("intent_confidence"),
        "sentiment": state.get("sentiment"),
        "sentiment_confidence": state.get("sentiment_confidence"),
        "priority": state.get("priority"),
        "priority_confidence": state.get("priority_confidence"),
        "retrieval_top_score": state.get("retrieval_top_score"),
        "response": state.get("response"),
        "escalated": int(bool(state.get("escalated", False))),
        "escalation_reason": state.get("escalation_reason"),
        "status": state.get("status", "completed"),
        "created_at": created_at,
        "logged_at": processing_record.get("logged_at", now),
    }

    # Ensure tables exist before saving.
    initialize_database()

    with get_connection() as connection:

        # Insert or update the ticket without deleting
        # its existing database row.
        connection.execute(
            """
            INSERT INTO tickets (
                ticket_id, ticket_text,
                intent, intent_confidence,
                sentiment, sentiment_confidence,
                priority, priority_confidence,
                retrieval_top_score, response,
                escalated, escalation_reason,
                status, created_at, logged_at
            )
            VALUES (
                :ticket_id, :ticket_text,
                :intent, :intent_confidence,
                :sentiment, :sentiment_confidence,
                :priority, :priority_confidence,
                :retrieval_top_score, :response,
                :escalated, :escalation_reason,
                :status, :created_at, :logged_at
            )
            ON CONFLICT(ticket_id) DO UPDATE SET
                ticket_text = excluded.ticket_text,
                intent = excluded.intent,
                intent_confidence = excluded.intent_confidence,
                sentiment = excluded.sentiment,
                sentiment_confidence = excluded.sentiment_confidence,
                priority = excluded.priority,
                priority_confidence = excluded.priority_confidence,
                retrieval_top_score = excluded.retrieval_top_score,
                response = excluded.response,
                escalated = excluded.escalated,
                escalation_reason = excluded.escalation_reason,
                status = excluded.status,
                logged_at = excluded.logged_at
            """,
            ticket_values,
        )

        # Replace the logs for this ticket so repeated saves
        # do not duplicate its agent-log entries.
        connection.execute(
            "DELETE FROM agent_logs WHERE ticket_id = ?",
            (ticket_id,),
        )

        for log in logs:
            connection.execute(
                """
                INSERT INTO agent_logs (
                    ticket_id,
                    agent_name,
                    action,
                    details_json,
                    timestamp
                )
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    ticket_id,
                    log.get("agent", "Unknown Agent"),
                    log.get("action", log.get("status", "")),
                    json.dumps(log, default=str),
                    log.get("timestamp"),
                ),
            )

    print(f"Ticket {ticket_id} saved to SQLite.")



def save_human_review(
    ticket_id: str,
    decision: str,
    reviewed_by: str,
    review_notes: str = "",
) -> dict:
    """Save a human review decision for a pending ticket."""

    decision = decision.strip().lower()
    reviewed_by = reviewed_by.strip()
    review_notes = review_notes.strip()

    if decision not in {"approved", "rejected"}:
        raise ValueError(
            "Decision must be 'approved' or 'rejected'."
        )

    if not reviewed_by:
        raise ValueError("Reviewer identity is required.")

    now = datetime.now().isoformat()

    initialize_database()

    with get_connection() as connection:
        ticket = connection.execute(
            """
            SELECT ticket_id, status
            FROM tickets
            WHERE ticket_id = ?
            """,
            (ticket_id,),
        ).fetchone()

        if ticket is None:
            raise ValueError("Ticket not found.")

        if ticket["status"] != "pending_human_review":
            raise ValueError(
                "Ticket is not awaiting human review."
            )

        new_status = (
            "review_approved"
            if decision == "approved"
            else "review_rejected"
        )

        connection.execute(
            """
            UPDATE tickets
            SET review_decision = ?,
                review_notes = ?,
                reviewed_at = ?,
                reviewed_by = ?,
                human_review_required = 0,
                status = ?
            WHERE ticket_id = ?
            """,
            (
                decision,
                review_notes,
                now,
                reviewed_by,
                new_status,
                ticket_id,
            ),
        )

        connection.execute(
            """
            INSERT INTO agent_logs (
                ticket_id,
                agent_name,
                action,
                details_json,
                timestamp
            )
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                ticket_id,
                "Human Reviewer",
                f"Ticket response {decision}",
                json.dumps({
                    "decision": decision,
                    "reviewed_by": reviewed_by,
                    "review_notes": review_notes,
                    "reviewed_at": now,
                }),
                now,
            ),
        )

    return {
        "ticket_id": ticket_id,
        "decision": decision,
        "reviewed_by": reviewed_by,
        "review_notes": review_notes,
        "reviewed_at": now,
        "status": new_status,
    }



if __name__ == "__main__":
    initialize_database()
