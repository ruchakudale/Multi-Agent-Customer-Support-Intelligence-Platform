
import json

from database.db import get_connection


def get_ticket(ticket_id: str):
    """Return one ticket by its ID, or None if not found."""

    with get_connection() as connection:
        row = connection.execute(
            """
            SELECT *
            FROM tickets
            WHERE ticket_id = ?
            """,
            (ticket_id,),
        ).fetchone()

    return dict(row) if row else None


def get_all_tickets(limit: int = 100):
    """Return the most recently created tickets first."""

    limit = max(1, min(int(limit), 1000))

    with get_connection() as connection:
        rows = connection.execute(
            """
            SELECT *
            FROM tickets
            ORDER BY created_at DESC, ticket_id DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()

    return [dict(row) for row in rows]


def get_agent_logs(ticket_id: str):
    """Return the processing logs for a ticket."""

    with get_connection() as connection:
        rows = connection.execute(
            """
            SELECT *
            FROM agent_logs
            WHERE ticket_id = ?
            ORDER BY id
            """,
            (ticket_id,),
        ).fetchall()

    logs = []

    for row in rows:
        log = dict(row)

        # Restore the complete log dictionary
        # stored in details_json.
        try:
            log["details"] = json.loads(
                log["details_json"]
            )
        except (TypeError, json.JSONDecodeError):
            log["details"] = {}

        logs.append(log)

    return logs


def get_ticket_statistics():
    """Return basic ticket and escalation statistics."""

    with get_connection() as connection:

        total = connection.execute(
            "SELECT COUNT(*) FROM tickets"
        ).fetchone()[0]

        escalated = connection.execute(
            "SELECT COUNT(*) FROM tickets WHERE escalated = 1"
        ).fetchone()[0]


        pending_reviews = connection.execute(
            """
            SELECT COUNT(*)
            FROM tickets
            WHERE status = 'pending_human_review'
            """
        ).fetchone()[0]

        approved_reviews = connection.execute(
            """
            SELECT COUNT(*)
            FROM tickets
            WHERE status = 'review_approved'
            """
        ).fetchone()[0]

        rejected_reviews = connection.execute(
            """
            SELECT COUNT(*)
            FROM tickets
            WHERE status = 'review_rejected'
            """
        ).fetchone()[0]

        priority_rows = connection.execute(
            """
            SELECT priority, COUNT(*) AS count
            FROM tickets
            GROUP BY priority
            """
        ).fetchall()

        intent_rows = connection.execute(
            """
            SELECT intent, COUNT(*) AS count
            FROM tickets
            GROUP BY intent
            """
        ).fetchall()

    return {
        "total_tickets": total,
        "escalated_tickets": escalated,
        "escalation_rate": (
            round(escalated / total * 100, 2)
            if total else 0.0
        ),
        "priority_distribution": {
            row["priority"]: row["count"]
            for row in priority_rows
        },
        "intent_distribution": {
            row["intent"]: row["count"]
            for row in intent_rows
        },
        "pending_human_reviews": pending_reviews,
        "approved_reviews": approved_reviews,
        "rejected_reviews": rejected_reviews,
    }


def get_pending_human_review_tickets(limit: int = 100):
    """Return tickets currently awaiting human review."""

    limit = max(1, min(int(limit), 1000))

    with get_connection() as connection:
        rows = connection.execute(
            """
            SELECT *
            FROM tickets
            WHERE status = ?
            ORDER BY created_at ASC, ticket_id ASC
            LIMIT ?
            """,
            ("pending_human_review", limit),
        ).fetchall()

    return [dict(row) for row in rows]
