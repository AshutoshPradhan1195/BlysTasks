# ── Conversation constants ────────────────────────────────────────────────────
import random
import string
from ollama import chat as ollama_chat

# ── OpenAI alternative ────────────────────────────────────────────────────────
# To use OpenAI instead of Ollama:
#   1. pip install openai
#   2. Set OPENAI_API_KEY in your environment (or pass api_key= below)
#   3. Comment out the `from ollama import ...` line above
#   4. Uncomment the two lines below
#
# import os
# from openai import OpenAI as _OpenAIClient; _openai_client = _OpenAIClient(api_key=os.environ["OPENAI_API_KEY"])

_SYSTEM_PROMPT = (
    "You are a helpful customer service assistant for Blys, "
    "a premium on-demand beauty and wellness service. "
    "Be concise, warm, and professional. Answer only what the customer asked."
)

# ── Price list (used for general / pricing queries) ──────────────────────────
PRICE_LIST = {
    "Massage":              "$80",
    "Deep Tissue Massage":  "$100",
    "Aromatherapy":         "$70",
    "Spa":                  "$120",
    "Facial":               "$65",
    "Makeup":               "$60",
    "Hair Coloring":        "$90",
    "Haircut":              "$30",
    "Pedicure":             "$40",
    "Waxing":               "$35",
}

def _price_list_text() -> str:
    lines = ["Our current service prices:\n"]
    for svc, price in PRICE_LIST.items():
        lines.append(f"  • {svc}: {price}")
    return "\n".join(lines)

# ── Tool definitions ──────────────────────────────────────────────────────────
def cancel_appointment(appointment_id: str) -> dict:
    """Cancel an existing appointment by ID."""
    return {
        "status":         "success",
        "message":        f"Appointment {appointment_id} has been successfully cancelled.",
        "appointment_id": appointment_id,
    }

def reschedule_appointment(appointment_id: str, new_date: str) -> dict:
    """Reschedule an existing appointment to a new date."""
    return {
        "status":         "success",
        "message":        f"Appointment {appointment_id} has been rescheduled to {new_date}.",
        "appointment_id": appointment_id,
        "new_date":       new_date,
    }

def book_appointment(booking_date: str) -> dict:
    """Book a new appointment on the requested date."""
    new_id = "APT" + "".join(random.choices(string.digits, k=6))
    return {
        "status":         "success",
        "message":        f"Your appointment has been booked for {booking_date}. Appointment ID: {new_id}.",
        "appointment_id": new_id,
        "booking_date":   booking_date,
    }

print("Tools and price list ready.")

_CONFIRM_INTENT = {
    "cancellation": (
        "I can help you cancel your booking. Would you like me to proceed?"
    ),
    "reschedule": (
        "Yes, you can reschedule your booking through the Blys app. "
        "Would you like me to assist you?"
    ),
    "booking": (
        "I'd be happy to help you book an appointment. "
        "Would you like me to go ahead?"
    ),
}

# Params required per intent, in the order they will be collected
_PARAM_SEQUENCE = {
    "cancellation": ["appointment_id"],
    "reschedule":   ["appointment_id", "new_date"],
    "booking":      ["booking_date"],
}

# Question asked when collecting each individual param
_PARAM_QUESTIONS = {
    "appointment_id": "Please provide your appointment ID (e.g. APT123456).",
    "new_date":       "Please provide the new date and time you would like to reschedule your booking at.",
    "booking_date":   "What date and time would you like to book your appointment for?",
}

# Final message shown after a tool runs successfully
_SUCCESS_MSG = {
    "cancellation": "Your booking has been successfully cancelled. You will receive a confirmation shortly.",
    "reschedule":   "Sent reschedule information to pro, you will get notified once it's confirmed.",
    "booking":      "Your appointment has been booked! You will receive a confirmation with your appointment ID shortly.",
}

_AFFIRMATIVE = {"yes", "yeah", "yep", "sure", "please", "ok", "okay", "yup",
                "go ahead", "proceed", "absolutely", "of course", "do it"}
_NEGATIVE    = {"no", "nope", "never mind", "nevermind", "cancel", "stop",
                "don't", "not now"}


class AppointmentChatbot:
    """
    State-machine chatbot: BERT intent classification -> Ollama for responses
    and parameter extraction.

    States
    ------
    None         -- fresh message: classify with BERT
    'confirming' -- intent detected, asked 'Would you like me to assist?'
                    waiting for yes/no
    'collecting' -- confirmed, collecting required params one-by-one via Ollama
    """

    def __init__(self, intent_classifier_service):
        self.intent_classifier_service = intent_classifier_service
        self.history            = []
        self.state              = None   # None | "confirming" | "collecting"
        self.active_intent      = None
        self.collected          = {}     # params gathered so far
        self.queue              = []     # params still needed, in order
        self._conversation_done = False  # True after a tool executes successfully

    # -- Ollama helpers --------------------------------------------------------

    def _llm(self, extra_system: str = "") -> str:
        """Call Ollama with the full conversation history as context."""
        system = _SYSTEM_PROMPT + ("\n\n" + extra_system if extra_system else "")
        msgs   = [{"role": "system", "content": system}] + self.history
        return ollama_chat(
            model="gemma4:e4b", messages=msgs, stream=False, think=False
        ).message.content.strip()
        # ── OpenAI alternative ─────────────────────────────────────────────────
        # return _openai_client.chat.completions.create(
        #     model="gpt-4o-mini",   # swap for any OpenAI model you prefer
        #     messages=msgs,
        # ).choices[0].message.content.strip()

    def _extract_value(self, user_input: str, param: str) -> str:
        """Ask Ollama to pull one specific param value out of the user message."""
        prompt = (
            f"Extract the value of '{param}' from this message. "
            f"Return ONLY the extracted value, no explanation.\n"
            f"Message: \"{user_input}\"\n"
            f"If you cannot find a specific value, return the message as-is."
        )
        return ollama_chat(
            model="gemma4:e4b",
            messages=[{"role": "user", "content": prompt}],
            stream=False, think=False
        ).message.content.strip()
        # ── OpenAI alternative ─────────────────────────────────────────────────
        # return _openai_client.chat.completions.create(
        #     model="gpt-4o-mini",
        #     messages=[{"role": "user", "content": prompt}],
        # ).choices[0].message.content.strip()

    # -- Tool execution --------------------------------------------------------

    def _run_tool(self) -> str:
        p = self.collected
        if   self.active_intent == "cancellation":
            cancel_appointment(p["appointment_id"])
        elif self.active_intent == "reschedule":
            reschedule_appointment(p["appointment_id"], p["new_date"])
        elif self.active_intent == "booking":
            book_appointment(p["booking_date"])
        return _SUCCESS_MSG[self.active_intent]

    def _reset(self):
        self.state = None; self.active_intent = None
        self.collected = {}; self.queue = []

    # -- Public interface ------------------------------------------------------

    def chat(self, user_input: str) -> str:
        self.history.append({"role": "user", "content": user_input})
        reply = self._route(user_input)
        self.history.append({"role": "assistant", "content": reply})
        return reply

    def _route(self, user_input: str) -> str:
        low = user_input.lower().strip()

        # State: waiting for yes/no after confirm message ----------------------
        if self.state == "confirming":
            if any(w in low for w in _NEGATIVE):
                self._reset()
                return "No problem! Let me know if there's anything else I can help with."
            # Anything that is not a clear negative -> treat as yes
            self.state = "collecting"
            return _PARAM_QUESTIONS[self.queue[0]]

        # State: collecting required params one-by-one -------------------------
        if self.state == "collecting":
            param                  = self.queue.pop(0)
            self.collected[param]  = self._extract_value(user_input, param)

            if self.queue:                      # more params still needed
                return _PARAM_QUESTIONS[self.queue[0]]
            else:                               # all done -> execute tool
                reply = self._run_tool()
                self._reset()
                self._conversation_done = True  # signal: this chat is over
                return reply

        # State: fresh message -- classify intent ------------------------------
        intent = self.intent_classifier_service.predict_intent(user_input)

        if intent in ("general_query", "pricing"):
            return self._llm(extra_system=_price_list_text())

        if intent in _PARAM_SEQUENCE:
            self.active_intent = intent
            self.collected     = {}
            self.queue         = list(_PARAM_SEQUENCE[intent])
            self.state         = "confirming"
            return _CONFIRM_INTENT[intent]

        return self._llm()      # fallback

    def reset(self):
        """Clear history and all state to start a fresh conversation."""
        self.history = []
        self._conversation_done = False
        self._reset()



