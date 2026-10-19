
from langgraph.graph import StateGraph, START, END

from graph.state import TicketState
from database.db import save_ticket

from graph.nodes import (
    intake_node,
    classification_node,
    priority_node,
    retrieval_node,
    response_node,
    escalation_node,
    logging_node,
)


def build_workflow():
    """Build and compile the customer support workflow."""

    graph = StateGraph(TicketState)

    # Register nodes
    graph.add_node("intake", intake_node)
    graph.add_node("classification", classification_node)
    graph.add_node("priority", priority_node)
    graph.add_node("retrieval", retrieval_node)
    graph.add_node("response", response_node)
    graph.add_node("escalation", escalation_node)
    graph.add_node("logging", logging_node)

    # Define execution order
    graph.add_edge(START, "intake")
    graph.add_edge("intake", "classification")
    graph.add_edge("classification", "priority")
    graph.add_edge("priority", "retrieval")
    graph.add_edge("retrieval", "response")
    graph.add_edge("response", "escalation")
    graph.add_edge("escalation", "logging")
    graph.add_edge("logging", END)

    return graph.compile()


# Compile workflow once
ticket_workflow = build_workflow()



def process_ticket(ticket_text: str) -> TicketState:
    """
    Process a customer ticket through LangGraph
    and persist the completed result to SQLite.
    """

    if not ticket_text or not ticket_text.strip():
        raise ValueError("Ticket text cannot be empty.")

    initial_state: TicketState = {
        "ticket_text": ticket_text.strip()
    }

    # Run all seven agents through LangGraph
    final_state = ticket_workflow.invoke(initial_state)

    # Save the completed ticket and agent logs
    save_ticket(final_state)

    return final_state
