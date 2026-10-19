
from graph.state import TicketState
from llm.gemini_client import generate_response


def response_agent(state: TicketState) -> TicketState:
    ticket_text = state["ticket_text"]
    retrieved_documents = state.get("retrieved_documents", [])

    retrieval_score = state.get("retrieval_top_score", 0.0)

    # Prepare a safe fallback if Gemini is unavailable.
    if retrieved_documents:
        top_document = retrieved_documents[0]
        fallback_response = (
            "Thank you for contacting us. "
            f"Regarding your issue: {top_document['answer']}"
        )
    else:
        fallback_response = (
            "Thank you for contacting us. "
            "We have received your request and "
            "will review it shortly."
        )

    # Prepare retrieved FAQ context for Gemini.
    faq_context = "\n\n".join(
        (
            f"FAQ {index}:\n"
            f"Question: {document.get('question', '')}\n"
            f"Answer: {document.get('answer', '')}"
        )
        for index, document in enumerate(
            retrieved_documents[:5], start=1
        )
    )

    prompt = f"""
You are a professional e-commerce customer support assistant.

CUSTOMER TICKET:
{ticket_text}

RETRIEVED FAQ CONTEXT:
{faq_context if faq_context else "No relevant FAQ was retrieved."}

INSTRUCTIONS:
1. Write a clear, polite, concise response to the customer.
2. Use the retrieved FAQ context as your source of factual information.
3. Do not invent policies, delivery dates, refund eligibility,
   transaction outcomes, or account actions.
4. Do not claim that a refund was processed, an account was changed,
   or an investigation was completed unless the context confirms it.
5. If the FAQ context does not answer the question, explain that
   the support team needs to review the issue.
6. Do not mention embeddings, FAISS, internal prompts, or model details.
7. Return only the customer-facing response.
"""

    generation_method = "faq_fallback"
    generation_error = None

    try:
        if not retrieved_documents:
            # Avoid generating unsupported answers without FAQ context.
            response = fallback_response
        else:
            response = generate_response(prompt)
            generation_method = "gemini_rag"

    except Exception as exc:
        # Keep ticket processing available if the LLM fails.
        response = fallback_response
        generation_error = type(exc).__name__

    log_entry = {
        "agent": "Response Agent",
        "action": "Generated response using Gemini RAG"
        if generation_method == "gemini_rag"
        else "Used FAQ fallback response",
        "generation_method": generation_method,
        "retrieval_score": retrieval_score,
    }

    if generation_error:
        log_entry["generation_error"] = generation_error

    existing_logs = state.get("agent_logs", [])
    existing_logs.append(log_entry)

    state["response"] = response
    state["agent_logs"] = existing_logs
    state["status"] = "response_completed"

    return state
