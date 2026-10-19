import os
import joblib
import numpy as np
from scipy.sparse import hstack

from graph.state import TicketState


# ---------------------------------------------------------
# Model paths
# ---------------------------------------------------------

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

MODEL_DIR = os.path.join(
    PROJECT_ROOT,
    "models",
    "priority"
)

MODEL_PATH = os.path.join(
    MODEL_DIR,
    "priority_model.pkl"
)

TFIDF_PATH = os.path.join(
    MODEL_DIR,
    "tfidf_vectorizer.pkl"
)

ENCODER_PATH = os.path.join(
    MODEL_DIR,
    "categorical_encoder.pkl"
)


# ---------------------------------------------------------
# Load artifacts
# ---------------------------------------------------------

priority_model = joblib.load(MODEL_PATH)
priority_tfidf = joblib.load(TFIDF_PATH)
priority_encoder = joblib.load(ENCODER_PATH)


# ---------------------------------------------------------
# Priority Agent
# ---------------------------------------------------------

def priority_agent(state: TicketState) -> TicketState:

    ticket_text = state["ticket_text"]
    intent = state["intent"]
    sentiment = state["sentiment"]

    # -----------------------------------------------------
    # Convert sentiment to the same binary representation
    # used during model training
    # -----------------------------------------------------

    if sentiment == "POSITIVE":
        sentiment_binary = "POSITIVE"
    else:
        sentiment_binary = "NEGATIVE"

    # -----------------------------------------------------
    # TF-IDF features
    # -----------------------------------------------------

    text_features = priority_tfidf.transform(
        [ticket_text]
    )

    # -----------------------------------------------------
    # Categorical features
    # -----------------------------------------------------
    import pandas as pd

    categorical_data = pd.DataFrame(
        [[intent, sentiment_binary]],
        columns=["ticket_category", "sentiment_binary"]
    )

    categorical_features = priority_encoder.transform(
        categorical_data
    )

    # -----------------------------------------------------
    # Combine features
    # -----------------------------------------------------

    final_features = hstack([
        text_features,
        categorical_features
    ])

    # -----------------------------------------------------
    # Prediction
    # -----------------------------------------------------

    prediction = priority_model.predict(
        final_features
    )[0]

    # -----------------------------------------------------
    # Confidence
    # -----------------------------------------------------

    if hasattr(priority_model, "predict_proba"):

        probabilities = priority_model.predict_proba(
            final_features
        )[0]

        confidence = float(
            np.max(probabilities)
        )

    else:
        confidence = 0.0

    # -----------------------------------------------------
    # Update agent log
    # -----------------------------------------------------

    log_entry = {
        "agent": "PriorityAgent",
        "status": "completed",
        "priority": prediction,
        "confidence": round(confidence, 4)
    }

    existing_logs = state.get("agent_logs", [])

    existing_logs.append(log_entry)

    # -----------------------------------------------------
    # Update state
    # -----------------------------------------------------

    state["priority"] = prediction
    state["priority_confidence"] = round(
        confidence,
        4
    )

    state["agent_logs"] = existing_logs
    state["status"] = "priority_completed"

    return state