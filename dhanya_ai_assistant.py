import os
import time
import streamlit as st

from google import genai


# =========================================================
# DHANYA AI ASSISTANT
# =========================================================

MODEL_NAME = "gemini-3.5-flash-lite"


# =========================================================
# SYSTEM INSTRUCTION
# =========================================================

SYSTEM_INSTRUCTION = """
You are Dhanya AI, an educational food assistant inside an
Indian millets, pulses and grains identification application.

Your role is to answer questions about the food detected by the
computer-vision model.

You can explain:

- Food identity
- Food category
- Scientific name
- Common names
- General nutrition
- Culinary uses
- Traditional Indian food preparations
- Ways to include the food in a balanced diet
- General age-group food information
- General food comparisons
- General agriculture and food information

SAFETY RULES:

- Do not diagnose diseases.
- Do not claim food cures diseases, pain or medical conditions.
- Do not claim that a food guarantees skin glow, hair growth,
  weight loss, pain relief or any other specific health outcome.
- Never present food as medicine.
- Nutrition information is general educational information.
- For medical conditions, allergies, pregnancy, medications or
  therapeutic diets, recommend consulting an appropriate
  healthcare professional.
- Do not invent exact prices, nutrition values or production
  statistics.
- Clearly distinguish general information from verified external
  information.
- Keep answers simple, useful and easy to understand.

When a detected food is supplied, use it as the main context
for the user's question.
"""


# =========================================================
# GET API KEY
# =========================================================

def get_api_key():

    # -----------------------------------------------------
    # Try environment variable
    # -----------------------------------------------------

    api_key = os.getenv(
        "GEMINI_API_KEY"
    )

    if api_key:
        return api_key

    # -----------------------------------------------------
    # Try Streamlit secrets
    # -----------------------------------------------------

    try:

        api_key = st.secrets.get(
            "GEMINI_API_KEY"
        )

    except Exception:

        api_key = None

    if api_key:
        return api_key

    raise RuntimeError(
        "GEMINI_API_KEY is not available."
    )


# =========================================================
# GEMINI CLIENT
# =========================================================

client = genai.Client(
    api_key=get_api_key()
)


# =========================================================
# CHAT STATE OBJECT
# =========================================================

class DhanyaChat:

    def __init__(self):

        self.previous_interaction_id = None


# =========================================================
# CREATE CHAT
# =========================================================

def create_chat():

    return DhanyaChat()


# =========================================================
# SEND MESSAGE WITH RETRY
# =========================================================

def ask_assistant(
    chat,
    question,
    detected_food=None,
    confidence=None,
    category=None
):

    # -----------------------------------------------------
    # Build food context
    # -----------------------------------------------------

    context = []

    if detected_food:

        context.append(
            f"Detected food: {detected_food}"
        )

    if confidence is not None:

        context.append(
            f"Prediction confidence: "
            f"{confidence * 100:.2f}%"
        )

    if category:

        context.append(
            f"Food category: {category}"
        )

    if context:

        food_context = "\n".join(
            context
        )

    else:

        food_context = (
            "No food prediction is currently available."
        )

    # -----------------------------------------------------
    # Build prompt
    # -----------------------------------------------------

    prompt = f"""
Current Dhanya AI context:

{food_context}

User question:
{question}

Answer clearly and safely.
"""

    # -----------------------------------------------------
    # Retry strategy
    # -----------------------------------------------------

    max_attempts = 3

    for attempt in range(
        max_attempts
    ):

        try:

            request_arguments = {
                "model": MODEL_NAME,
                "system_instruction":
                    SYSTEM_INSTRUCTION,
                "input": prompt
            }

            # -------------------------------------------------
            # Continue previous conversation
            # -------------------------------------------------

            if chat.previous_interaction_id:

                request_arguments[
                    "previous_interaction_id"
                ] = (
                    chat.previous_interaction_id
                )

            # -------------------------------------------------
            # Gemini Interactions API
            # -------------------------------------------------

            interaction = (
                client.interactions.create(
                    **request_arguments
                )
            )

            # -------------------------------------------------
            # Save interaction state
            # -------------------------------------------------

            chat.previous_interaction_id = (
                interaction.id
            )

            # -------------------------------------------------
            # Return answer
            # -------------------------------------------------

            return interaction.output_text

        except Exception as error:

            error_text = str(
                error
            )

            # Retry only transient server errors
            if (
                "503" not in error_text
                and "429" not in error_text
                and "500" not in error_text
                and "502" not in error_text
                and "504" not in error_text
            ):

                raise

            # Last attempt
            if attempt == max_attempts - 1:

                raise

            # Exponential backoff
            wait_time = 2 ** attempt

            time.sleep(
                wait_time
            )

    raise RuntimeError(
        "Gemini request failed after retries."
    )


# =========================================================
# TERMINAL TEST
# =========================================================

if __name__ == "__main__":

    print(
        "\n======================================"
    )

    print(
        "DHANYA AI INTERACTIONS API TEST"
    )

    print(
        "======================================"
    )

    try:

        chat = create_chat()

        answer = ask_assistant(
            chat=chat,
            question=(
                "How can I include this food "
                "in my diet?"
            ),
            detected_food="Chick Peas",
            confidence=0.96,
            category="Pulse"
        )

        print(
            "\nDhanya AI:"
        )

        print(
            answer
        )

        print(
            "\n======================================"
        )

        print(
            "ASSISTANT TEST COMPLETED"
        )

        print(
            "======================================"
        )

    except Exception as error:

        print(
            "\nAssistant Error:"
        )

        print(
            str(error)
        )