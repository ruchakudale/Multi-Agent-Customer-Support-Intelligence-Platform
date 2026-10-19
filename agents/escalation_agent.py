
from graph.state import TicketState


def escalation_agent(state: TicketState) -> TicketState:

    priority = state.get("priority", "")
    sentiment = state.get("sentiment", "")
    intent_confidence = state.get("intent_confidence", 0.0)
    retrieval_score = state.get("retrieval_top_score", 0.0)

    # Initial thresholds
    INTENT_CONFIDENCE_THRESHOLD = 0.40
    RETRIEVAL_SCORE_THRESHOLD = 0.60

    escalated = False
    escalation_reason = "No escalation required"
    human_review_required = False

    # Rule 1: Low intent confidence — human review
    if intent_confidence < INTENT_CONFIDENCE_THRESHOLD:
        escalated = True
        human_review_required = True
        escalation_reason = (
            f"Human review required: intent confidence "
            f"({intent_confidence:.2%}) is below the "
            f"{INTENT_CONFIDENCE_THRESHOLD:.0%} threshold"
        )

    # Rule 2: High priority and negative sentiment
    elif priority == "High" and sentiment == "NEGATIVE":
        escalated = True
        human_review_required = True
        escalation_reason = (
            "Human review required: high-priority ticket "
            "with negative customer sentiment"
        )

    # Rule 3: Weak FAQ retrieval
    elif retrieval_score < RETRIEVAL_SCORE_THRESHOLD:
        escalated = True
        human_review_required = True
        escalation_reason = (
            f"Human review required: FAQ retrieval score "
            f"({retrieval_score:.3f}) is below the "
            f"{RETRIEVAL_SCORE_THRESHOLD:.2f} threshold"
        )

    # Record the decision
    log_entry = {
        "agent": "Escalation Agent",
        "action": "Evaluated escalation and human-review requirement",
        "escalated": escalated,
        "human_review_required": human_review_required,
        "reason": escalation_reason,
        "intent_confidence": intent_confidence,
        "intent_confidence_threshold": INTENT_CONFIDENCE_THRESHOLD,
        "retrieval_score": retrieval_score,
        "retrieval_score_threshold": RETRIEVAL_SCORE_THRESHOLD,
    }

    existing_logs = list(state.get("agent_logs", []))
    existing_logs.append(log_entry)

    # Update workflow state
    state["escalated"] = escalated
    state["escalation_reason"] = escalation_reason
    state["human_review_required"] = human_review_required
    state["agent_logs"] = existing_logs
    state["status"] = (
        "pending_human_review"
        if human_review_required
        else "escalation_completed"
    )

    return state
