from datetime import datetime

from graph.state import TicketState


def logging_agent(state: TicketState) -> TicketState:

    # ---------------------------------------------------------
    # Create final ticket processing record
    # ---------------------------------------------------------

    processing_record = {
        "ticket_id": state.get("ticket_id"),
        "ticket_text": state.get("ticket_text"),

        "intent": state.get("intent"),
        "intent_confidence": state.get(
            "intent_confidence"
        ),

        "sentiment": state.get("sentiment"),
        "sentiment_confidence": state.get(
            "sentiment_confidence"
        ),

        "priority": state.get("priority"),
        "priority_confidence": state.get(
            "priority_confidence"
        ),

        "retrieval_top_score": state.get(
            "retrieval_top_score"
        ),

        "response": state.get("response"),

        "escalated": state.get(
            "escalated",
            False
        ),

        "escalation_reason": state.get(
            "escalation_reason"
        ),

        "agent_logs": state.get(
            "agent_logs",
            []
        ),

        "logged_at": datetime.now().isoformat(),

        "status": "completed"
    }

    # ---------------------------------------------------------
    # Add Logging Agent entry
    # ---------------------------------------------------------

    log_entry = {
        "agent": "Logging Agent",
        "action": "Final ticket processing record created",
        "timestamp": datetime.now().isoformat()
    }

    existing_logs = state.get(
        "agent_logs",
        []
    )

    existing_logs.append(log_entry)

    # ---------------------------------------------------------
    # Update state
    # ---------------------------------------------------------

    state["agent_logs"] = existing_logs

    state["processing_record"] = processing_record

    state["status"] = (
        "pending_human_review"
        if state.get("human_review_required", False)
        else "completed"
    )

    return state