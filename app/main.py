
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from typing import Literal

from graph.workflow import process_ticket
from database.db import save_human_review

from database.queries import (
    get_ticket,
    get_all_tickets,
    get_agent_logs,
    get_ticket_statistics,
    get_pending_human_review_tickets,
)
# =========================================================
# FASTAPI APPLICATION
# =========================================================

app = FastAPI(
    title="AI Multi-Agent Customer Support API",
    description=(
        "REST API for an AI-powered customer support system "
        "using LangGraph, ML models, FAISS, and SQLite."
    ),
    version="1.0.0",
)


# =========================================================
# REQUEST SCHEMA
# =========================================================

class TicketRequest(BaseModel):
    ticket_text: str = Field(
        ...,
        min_length=1,
        max_length=5000,
        description="Customer's support ticket text",
    )


# =========================================================
# HEALTH CHECK
# =========================================================

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "service": "AI Customer Support API",
    }


# =========================================================
# PROCESS A NEW TICKET
# =========================================================

@app.post("/tickets", status_code=201)
def create_ticket(request: TicketRequest):

    ticket_text = request.ticket_text.strip()

    if not ticket_text:
        raise HTTPException(
            status_code=422,
            detail="Ticket text cannot be empty or whitespace.",
        )

    try:
        result = process_ticket(ticket_text)

        return {
            "ticket_id": result.get("ticket_id"),
            "ticket_text": result.get("ticket_text"),
            "intent": result.get("intent"),
            "intent_confidence": result.get("intent_confidence"),
            "sentiment": result.get("sentiment"),
            "sentiment_confidence": result.get(
                "sentiment_confidence"
            ),
            "priority": result.get("priority"),
            "priority_confidence": result.get(
                "priority_confidence"
            ),
            "retrieval_top_score": result.get(
                "retrieval_top_score"
            ),
            "response": result.get("response"),
            "escalated": result.get("escalated"),
            "escalation_reason": result.get(
                "escalation_reason"
            ),
            "status": result.get("status"),
            "agent_logs": result.get("agent_logs", []),
        }

    except Exception as exc:
        # Log the full exception in the terminal during development.
        import logging

        logging.exception("Ticket processing failed")

        raise HTTPException(
            status_code=500,
            detail="Ticket processing failed. Check the API server logs.",
        ) from exc


# =========================================================
# GET A SAVED TICKET
# =========================================================

@app.get("/tickets/{ticket_id}")
def read_ticket(ticket_id: str):

    ticket = get_ticket(ticket_id)

    if ticket is None:
        raise HTTPException(
            status_code=404,
            detail="Ticket not found.",
        )

    # Convert SQLite integer to a JSON boolean.
    ticket["escalated"] = bool(ticket["escalated"])

    return ticket



# =========================================================
# LIST RECENT TICKETS
# =========================================================

@app.get("/tickets")
def list_tickets(limit: int = 100):

    if not 1 <= limit <= 1000:
        raise HTTPException(
            status_code=422,
            detail="Limit must be between 1 and 1000.",
        )

    tickets = get_all_tickets(limit=limit)

    for ticket in tickets:
        ticket["escalated"] = bool(ticket["escalated"])

    return {
        "count": len(tickets),
        "tickets": tickets,
    }


# =========================================================
# GET AGENT LOGS FOR A TICKET
# =========================================================

@app.get("/tickets/{ticket_id}/logs")
def read_ticket_logs(ticket_id: str):

    # Verify that the ticket exists first.
    ticket = get_ticket(ticket_id)

    if ticket is None:
        raise HTTPException(
            status_code=404,
            detail="Ticket not found.",
        )

    logs = get_agent_logs(ticket_id)

    return {
        "ticket_id": ticket_id,
        "count": len(logs),
        "agent_logs": logs,
    }



@app.get("/human-review-queue")
def human_review_queue(limit: int = 100):
    """Return tickets awaiting human review."""

    limit = max(1, min(limit, 1000))

    tickets = get_pending_human_review_tickets(limit=limit)

    return {
        "count": len(tickets),
        "tickets": [
            {
                **ticket,
                "escalated": bool(ticket["escalated"]),
            }
            for ticket in tickets
        ],
    }


# =========================================================
# TICKET ANALYTICS
# =========================================================

@app.get("/analytics")
def read_analytics():

    return get_ticket_statistics()




class HumanReviewRequest(BaseModel):
    decision: Literal["approved", "rejected"]
    reviewed_by: str = Field(..., min_length=1, max_length=100)
    review_notes: str = Field(default="", max_length=2000)


@app.post("/tickets/{ticket_id}/review")
def review_ticket(ticket_id: str, request: HumanReviewRequest):
    """Record a human review decision."""

    try:
        result = save_human_review(
            ticket_id=ticket_id,
            decision=request.decision,
            reviewed_by=request.reviewed_by.strip(),
            review_notes=request.review_notes,
        )
        return result

    except ValueError as exc:
        from fastapi import HTTPException
        raise HTTPException(
            status_code=404 if str(exc) == "Ticket not found."
            else 409,
            detail=str(exc),
        )
