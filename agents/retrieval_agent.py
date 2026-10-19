from rag.retriever import retrieve_documents
from graph.state import TicketState


def retrieval_agent(state: TicketState) -> TicketState:

    ticket_text = state["ticket_text"]

    # Retrieve relevant FAQ documents
    retrieved_documents = retrieve_documents(
        ticket_text,
        top_k=5
    )

    # Extract top similarity score
    if retrieved_documents:
        retrieval_top_score = float(
            retrieved_documents[0]["score"]
        )
    else:
        retrieval_top_score = 0.0

    # Add agent log
    log_entry = {
        "agent": "Retrieval Agent",
        "action": "Retrieved relevant FAQ documents",
        "top_k": len(retrieved_documents),
        "top_score": round(retrieval_top_score, 4)
    }

    existing_logs = state.get("agent_logs", [])
    existing_logs.append(log_entry)

    # Update state
    state["retrieved_documents"] = retrieved_documents
    state["retrieval_top_score"] = round(
        retrieval_top_score,
        4
    )

    state["agent_logs"] = existing_logs
    state["status"] = "retrieval_completed"

    return state