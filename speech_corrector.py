from nlp import GPTClient

class SpeechCorrector:
    """Fixes STT errors using GPT contextual understanding."""
    def __init__(self, gpt: GPTClient):
        self.gpt = gpt

    def correct(self, raw_text: str, memory_summary: str = "") -> str:
        if not raw_text.strip():
            return raw_text
        prompt = (
            "You are a speech-to-text corrector for an emergency assistant (RescueHub).\n"
            "Fix grammar and recognition errors in the user's voice transcript without changing meaning.\n"
            "Prefer emergency-related words (fire, burn, injury, ambulance, address).\n"
            "Return only the corrected sentence.\n"
            f"Context:\n{memory_summary}\nUser said (possibly wrong): {raw_text}"
        )
        response = self.gpt.chat([
            {"role": "system", "content": prompt}
        ])
        return response.strip()
