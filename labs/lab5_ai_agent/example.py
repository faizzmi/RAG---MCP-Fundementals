from google import genai
from google.genai import types
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from config import gemini_key

def call_llm(prompt: str, system_instruction: str = None, model: str = "gemini-3.5-flash") -> str:
    # Call Gemini with a single prompt string and an optional system instruction.
    client = genai.Client(api_key=gemini_key)
    config = types.GenerateContentConfig(system_instruction=system_instruction) if system_instruction else None

    response = client.models.generate_content(
        model=model,
        contents=prompt,
        config=config,
    )
    return response.text


def fetch_user_preferences(user_id: str = None) -> str:
    return "Prefers concise answers, budget-friendly options, and Malaysian context."

# Step 1: ask the LLM to pull structured detail out of the raw question.
def extract_question_details(question: str) -> str:
    system_instruction = "You are a helpful assistant that extracts detailed information from user questions."
    prompt = f"Extract detailed information from the following question: '{question}'."
    return call_llm(prompt, system_instruction=system_instruction)

# Step 2: ask the LLM to choose the best answer given details + preferences
def choose_best_answer(question_details: str, preferences: str) -> str:
    system_instruction = "You are a helpful assistant that selects the best answer based on user preferences."
    prompt = (
        f"Based on the extracted information: '{question_details}' "
        f"and user preferences: '{preferences}', select the best answer."
    )
    return call_llm(prompt, system_instruction=system_instruction)

# Formats the final response shown to the user.
def user_answer(question: str, decision: str) -> str:
    return f"The best answer to your question '{question}' is: '{decision}'"

# Orchestrates the full extraction -> preference-matching -> decision pipeline.
def run_agent(question: str, user_id: str = None) -> str:
    question_details = extract_question_details(question)
    preferences = fetch_user_preferences(user_id)
    decision = choose_best_answer(question_details, preferences)
    return user_answer(question, decision)


if __name__ == "__main__":
    question = input("What is your question?\n")
    print(run_agent(question))


# $ py -m example
# What is your question?
# how to organize thoug and make a clear way of thinking
# The best answer to your question 'how to organize thoug and make a clear way of thinking' is: 'Here is a concise, budget-friendly guide tailored to a Malaysian context to help you organize your thoughts and build a clear way of thinking.

# ---

# ### 1. Tactical Tools (Get it out of your head)
# *   **The RM2 "Brain Dump":** Buy a cheap notebook from **Mr. DIY** or **Eco-Shop**. Whenever your mind feels cluttered, write down *everything* you are thinking about without editing yourself. Once it is on paper, your brain can relax.
# *   **Free Mind Mapping:** Use a blank A4 paper or download a free app like **XMind** or **Google Keep** (fully free) to visually connect your thoughts.      
# *   **Bullet Outlining:** Structure your day using simple bullet points. Group them into three simple "buckets": *Kerja* (Work), *Peribadi* (Personal), and *Lain-lain* (Others).

# ### 2. Cognitive Frameworks (How to structure your thinking)
# *   **The Feynman Technique:** Try explaining your complex thoughts or problems in simple, everyday English or Bahasa Melayu—as if you are explaining it to a child. If you struggle to simplify it, you need to clarify your thoughts.      
# *   **The "Urgent vs. Important" Matrix:** Divide your tasks into four boxes. Focus only on what is truly urgent and important today, and push the rest to next week.

# ### 3. Lifestyle & Habits (Keep your mind clear)
# *   **Digital Detox:** Limit late-night scrolling on TikTok, Shopee, or Instagram. High digital consumption directly contributes to mental clutter and "brain fog."
# *   **Free Mindfulness:** Take 10 minutes to walk at your local *Taman Rekreasi* (recreational park) without looking at your phone, or use free breathing apps like **Insight Timer** to calm a racing mind.
# *   **Sleep:** Never underestimate a good night's sleep. It is the cheapest and most effective way to restore cognitive clarity.'