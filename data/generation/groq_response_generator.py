import os
import time
from groq import Groq, RateLimitError, APIError

CANARY = "[ODSB-CANARY-7F3A2B] "
PLACEHOLDER = "<TO_BE_GENERATED>"  # used in paraphrase_eval scaffolding

SYSTEM_PROMPT_TECH = (
    "You are a helpful, friendly assistant. Answer the user's technical "
    "question concretely with specifics. Keep your answer under 100 words. "
    "Do not include meta-commentary, system tags, labels, or formatting "
    "markers — just the answer text itself."
)


SYSTEM_PROMPT_EMO = (
    "You are a warm, friendly assistant in a casual conversation. The user "
    "has shared how they're feeling. Acknowledge what they shared in a "
    "natural, human way and ask one open-ended follow-up question. Keep your "
    "response to 1-2 sentences (under 40 words). Respond conversationally — "
    "no clinical language, no suggestions to seek professional help, no "
    "disclaimers about being an AI, no formatting markers, just a warm "
    "natural reply."
)

SYSTEM_PROMPT_EMO_AFTER_TECH = (
    "You are a warm, friendly assistant in a casual conversation. The user "
    "previously asked a technical question and you answered it. Now they've "
    "shared how they're feeling. Acknowledge what they shared in a natural, "
    "human way and ask one open-ended follow-up. Keep it to 1-2 sentences "
    "(under 40 words). No clinical language, no suggestions to seek "
    "professional help, no disclaimers about being an AI, no formatting "
    "markers — just a warm natural reply."
)

SYSTEM_PROMPT_TECH_AFTER_EMO = (
    "You are a helpful, friendly assistant in a casual conversation. "
    "The user previously shared how they were feeling emotionally. "
    "Now they are asking a technical question. Answer the technical "
    "question concretely and clearly with useful specifics while "
    "maintaining a warm conversational tone. Keep your response under "
    "100 words. No meta-commentary, no disclaimers about being an AI, "
    "no formatting markers, no labels — just the response text."
)

SYSTEM_PROMPT_BENIGN = (
    "You are a helpful, friendly assistant. Respond naturally to the user's "
    "request in 1-3 sentences (under 60 words). Do not include any tags or "
    "labels — just the response text."
)

PRIMARY_MODELS = [
    "llama-3.3-70b-versatile",
    "openai/gpt-oss-120b",
    "meta-llama/llama-4-scout-17b-16e-instruct",
]
FALLBACK_MODEL = "llama-3.1-8b-instant"


class MultiModelGroqGenerator:
    def __init__(self, temperature=0.85):
        self.client = Groq()
        self.temperature = temperature
        self.usage = {m: 0 for m in PRIMARY_MODELS + [FALLBACK_MODEL]}

    def select_system_prompt(self, role_position, intent_sequence):
        """Pick the right system prompt for this assistant turn.

        role_position: which assistant turn (0 = first, 1 = second)
        intent_sequence: list like ['emotional_state', 'technical_request']
                         or empty for condition D
        """
        if not intent_sequence:
            return SYSTEM_PROMPT_BENIGN  # condition D

        if role_position >= len(intent_sequence):
            return SYSTEM_PROMPT_BENIGN

        intent = intent_sequence[role_position]

        if intent == "emotional_state":
            if role_position == 0:
                return SYSTEM_PROMPT_EMO
            return SYSTEM_PROMPT_EMO_AFTER_TECH
        elif intent == "technical_request":
            if role_position == 0:
                return SYSTEM_PROMPT_TECH
            return SYSTEM_PROMPT_TECH_AFTER_EMO
        return SYSTEM_PROMPT_BENIGN

    def generate(self, conversation_so_far, system_prompt, row_index):
        """Generate one assistant response.
        Retries on empty/short responses by falling through to the next model
        in the chain. If all models fail, raises RuntimeError.
        """
        messages = [{"role": "system", "content": system_prompt}] + conversation_so_far

        primary = PRIMARY_MODELS[row_index % len(PRIMARY_MODELS)]
        # Build full chain: primary, then the OTHER primaries, then fallback.
        # This way if one model refuses (safety filter), we try a different
        # primary before giving up to the small fallback model.
        other_primaries = [m for m in PRIMARY_MODELS if m != primary]
        model_chain = [primary] + other_primaries + [FALLBACK_MODEL]

        last_err = None
        for model in model_chain:
            for attempt in range(3):
                try:
                    resp = self.client.chat.completions.create(
                        model=model,
                        messages=messages,
                        temperature=self.temperature,
                        max_tokens=150,
                        top_p=0.95,
                    )
                    text = resp.choices[0].message.content
                    text = text.strip() if text else ""

                    # Reject empty or refusal-shaped responses
                    if len(text) < 15:
                        last_err = ValueError(f"Response too short: {text!r}")
                        print(f"  [{model}] short response ({len(text)} chars), retrying")
                        time.sleep(1)
                        continue

                    # Reject obvious refusal patterns
                    lower = text.lower()
                    refusal_markers = [
                        "i can't help", "i cannot help", "i'm not able to",
                        "as an ai", "i'm an ai", "i am an ai",
                        "please reach out to", "please contact a",
                        "if you're in crisis", "if you are in crisis",
                    ]
                    if any(marker in lower for marker in refusal_markers):
                        last_err = ValueError(f"Refusal detected: {text[:80]!r}")
                        print(f"  [{model}] refusal pattern, falling through")
                        break  # break attempt loop, try next model

                    self.usage[model] = self.usage.get(model, 0) + 1
                    return text, model

                except RateLimitError as e:
                    last_err = e
                    wait = 2 ** attempt
                    print(f"  [{model}] rate limited, waiting {wait}s...")
                    time.sleep(wait)
                except APIError as e:
                    last_err = e
                    print(f"  [{model}] API error: {e}, retrying...")
                    time.sleep(1)
            print(f"  [{model}] exhausted, trying next model")

        raise RuntimeError(f"All models failed for row {row_index}. Last: {last_err}")

    def print_usage(self):
        print("\n=== Model usage ===")
        total = sum(self.usage.values())
        for m, n in self.usage.items():
            pct = (n / total * 100) if total else 0
            print(f"  {m}: {n} ({pct:.1f}%)")
        print(f"  TOTAL: {total}")