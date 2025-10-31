from collections import deque

class ConversationMemory:
    def __init__(self, max_turns=8):
        self.history = deque(maxlen=max_turns)

    def add(self, role: str, content: str):
        self.history.append({"role": role, "content": content})

    def to_messages(self):
        return list(self.history)

    def get_summary(self):
        return "\n".join([f"{m['role']}: {m['content']}" for m in self.history])
