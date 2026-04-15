from fastapi import FastAPI, HTTPException, Query, Depends
from typing import Optional
import uuid
import pandas as pd
import numpy as np
import services.Recommendation as RecommendationService
from services.Recommendation import get_recommendation_service
from services.IntentClassifier import get_intent_classifier_service, IntentClassifierService
from services.ChatbotService import AppointmentChatbot
import joblib

# Initialize the FastAPI instance
app = FastAPI()


_sessions: dict[str, AppointmentChatbot] = {}

def _get_chatbot(session_id: str) -> AppointmentChatbot:
    if session_id not in _sessions:
        _sessions[session_id] = AppointmentChatbot(IntentClassifierService())
    return _sessions[session_id]



@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.post("/recommend")
def recommend(
    customer_id: str = Query(default="CUST00001", required=True), 
    num_recommendations:int=Query(default=3, required=True),
    recommendation_service: RecommendationService = Depends(get_recommendation_service)
):
    try:
        recommendations = recommendation_service.top_n_recommendations(customer_id, num_recommendations)
        return {"recommendations": recommendations}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/classify-intent")
def classify_intent(text: str = Query(default="Hello, I'd like to ask you about your pricings.", required=True)):
    intent_classifier_service = IntentClassifierService()
    intent = intent_classifier_service.predict_intent(text)
    return {"intent": intent}

@app.post("/chat")
def chat(
    text: str = Query(default="Hello, I'd like to ask you about your pricings."),
    session_id: Optional[str] = Query(default=None),
):
    if session_id is None:
        session_id = str(uuid.uuid4())

    chatbot = _get_chatbot(session_id)
    response = chatbot.chat(text)

    # Auto-reset: tool just executed, conversation is complete — drop the session
    # so the next call with this session_id starts fresh.
    conversation_complete = chatbot._conversation_done
    if conversation_complete:
        del _sessions[session_id]

    return {
        "session_id": session_id,
        "response": response,
        "conversation_complete": conversation_complete,
    }


@app.post("/chat/reset")
def reset_chat(session_id: str = Query()):
    """Explicitly end a session before the conversation is naturally complete."""
    if session_id in _sessions:
        del _sessions[session_id]
    return {"status": "reset", "session_id": session_id}