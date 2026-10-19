from datetime import datetime

from graph.state import TicketState


def intake_agent(state: TicketState) -> TicketState:
    """
    Intake Agent

    Validates and initializes an incoming customer ticket.
    """

    ticket_text = state.get("ticket_text", "").strip()

    if not ticket_text:
        raise ValueError(
            "Ticket text cannot be empty."
        )

    ticket_id = state.get("ticket_id")

    if not ticket_id:
        ticket_id = (
            "TICKET_"
            + datetime.now().strftime("%Y%m%d%H%M%S")
        )

    agent_logs = state.get(
        "agent_logs",
        []
    )

    agent_logs.append({
        "agent": "Intake Agent",
        "action": "Ticket received and validated",
        "timestamp": datetime.now().isoformat()
    })

    return {
        **state,
        "ticket_id": ticket_id,
        "ticket_text": ticket_text,
        "status": "intake_completed",
        "agent_logs": agent_logs
    }