from pathlib import Path
from datetime import datetime

import joblib
from transformers import pipeline

from graph.state import TicketState


# =========================================================
# PROJECT PATH
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

INTENT_MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "intent"
    / "intent_model.pkl"
)

INTENT_VECTORIZER_PATH = (
    PROJECT_ROOT
    / "models"
    / "intent"
    / "tfidf_vectorizer.pkl"
)


# =========================================================
# LOAD INTENT MODEL
# =========================================================

intent_model = joblib.load(
    INTENT_MODEL_PATH
)

intent_vectorizer = joblib.load(
    INTENT_VECTORIZER_PATH
)


# =========================================================
# LOAD PRETRAINED SENTIMENT MODEL
# =========================================================

sentiment_pipeline = pipeline(
    "sentiment-analysis",
    model="distilbert-base-uncased-finetuned-sst-2-english"
)


# =========================================================
# CLASSIFICATION AGENT
# =========================================================

def classification_agent(
    state: TicketState
) -> TicketState:
    """
    Classification Agent

    Predicts:
    - Intent
    - Sentiment
    """

    ticket_text = state["ticket_text"]

    # -----------------------------------------------------
    # INTENT PREDICTION
    # -----------------------------------------------------

    intent_vector = intent_vectorizer.transform(
        [ticket_text]
    )

    predicted_intent = intent_model.predict(
        intent_vector
    )[0]

    # Logistic Regression provides probability estimates.
    intent_probabilities = (
        intent_model.predict_proba(
            intent_vector
        )[0]
    )

    intent_confidence = float(
        intent_probabilities.max()
    )

    # -----------------------------------------------------
    # SENTIMENT PREDICTION
    # -----------------------------------------------------

    sentiment_result = sentiment_pipeline(
        ticket_text,
        truncation=True
    )[0]

    predicted_sentiment = sentiment_result[
        "label"
    ]

    sentiment_confidence = float(
        sentiment_result["score"]
    )

    # -----------------------------------------------------
    # LOGGING
    # -----------------------------------------------------

    agent_logs = state.get(
        "agent_logs",
        []
    )

    agent_logs.append({
        "agent": "Classification Agent",
        "action": "Intent and sentiment predicted",
        "timestamp": datetime.now().isoformat(),
        "intent": predicted_intent,
        "intent_confidence": intent_confidence,
        "sentiment": predicted_sentiment,
        "sentiment_confidence": sentiment_confidence
    })

    # -----------------------------------------------------
    # UPDATE STATE
    # -----------------------------------------------------

    return {
        **state,

        "intent": predicted_intent,
        "intent_confidence": intent_confidence,

        "sentiment": predicted_sentiment,
        "sentiment_confidence": sentiment_confidence,

        "status": "classification_completed",

        "agent_logs": agent_logs
    }