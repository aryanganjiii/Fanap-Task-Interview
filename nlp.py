import os, requests, json
from dotenv import load_dotenv
load_dotenv()

class GPTClient:
    def __init__(self, model="gpt-4o-mini"):
        self.model = model
        self.api_key = os.getenv("GAPGPT_API_KEY")
        self.base_url = os.getenv("GAPGPT_BASE_URL", "https://api.gapgpt.com/v1")
        if not self.api_key:
            raise ValueError("Missing GAPGPT_API_KEY in .env")

    def chat(self, messages):
        url = f"{self.base_url}/chat/completions"
        headers = {"Authorization": f"Bearer {self.api_key}"}
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": 0.3,
            "max_tokens": 200
        }
        r = requests.post(url, json=payload, headers=headers, timeout=30)
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"].strip()

    def parse_user_turn(self, memory_text: str, user_input: str) -> dict:
        """Interpret intent & slots dynamically."""
        url = f"{self.base_url}/chat/completions"
        headers = {"Authorization": f"Bearer {self.api_key}"}
        system = {
            "role": "system",
            "content": (
                "You are the NLP parser for RescueHub.\n"
                "Analyze the user's input (with possible STT errors) in context of previous dialogue.\n"
                "Return only a compact JSON object:\n"
                "{"
                "\"intent\": \"fire|medical|other\","
                "\"address\": \"string or null\","
                "\"injury\": true/false/null,"
                "\"severity\": \"high|low|null\","
                "\"escalate_to_medical\": true/false"
                "}"
            )
        }
        user = {"role": "user", "content": f"Context:\n{memory_text}\n\nCaller: {user_input}"}
        data = {"model": self.model, "messages": [system, user], "temperature": 0.1}
        r = requests.post(url, json=data, headers=headers, timeout=40)
        try:
            content = r.json()["choices"][0]["message"]["content"]
            return json.loads(content)
        except Exception as e:
            print("[Parse Error]", e)
            return {"intent": None, "address": None, "injury": None, "severity": None, "escalate_to_medical": False}
