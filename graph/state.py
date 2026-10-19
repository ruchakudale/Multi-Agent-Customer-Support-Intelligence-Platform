from typing import TypedDict, List, Dict, Any, Optional


class TicketState(TypedDict, total=False):
    """
    Shared state passed between all agents.
    """

    # -----------------------------------------------------
    # INPUT
    # -----------------------------------------------------

    ticket_id: str
    ticket_text: str

    # -----------------------------------------------------
    # CLASSIFICATION
    # -----------------------------------------------------

    intent: str
    intent_confidence: float

    sentiment: str
    sentiment_confidence: float

    priority: str
    priority_confidence: float

    # -----------------------------------------------------
    # RAG
    # -----------------------------------------------------

    retrieved_documents: List[Dict[str, Any]]

    retrieval_top_score: float

    # -----------------------------------------------------
    # RESPONSE
    # -----------------------------------------------------

    response: str

    # -----------------------------------------------------
    # ESCALATION
    # -----------------------------------------------------

    escalated: bool
    escalation_reason: Optional[str]

    # -----------------------------------------------------
    # LOGGING
    # -----------------------------------------------------

    agent_logs: List[Dict[str, Any]]

    # -----------------------------------------------------
    # SYSTEM STATUS
    # -----------------------------------------------------

    status: str
    
    human_review_required: bool

    processing_record: dict