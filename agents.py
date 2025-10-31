from dataclasses import dataclass
from typing import Optional, Tuple
import json
from tools import dispatch_resources
from memory import ConversationMemory
from nlp import GPTClient
from vector_memory import VectorMemory
import re

# ====== Context ======
@dataclass
class Ctx:
    address: Optional[str] = None
    severity: Optional[str] = None
    injuries: Optional[bool] = None
    injury_desc: Optional[str] = None
    active_agent: Optional[str] = None
    done: bool = False
    escalation_done: bool = False
    incident_type: Optional[str] = None
    medical_probe_done: bool = False
    asked_address_once: bool = False
    had_fire: bool = False
    had_medical: bool = False


# ====== Dynamic Dispatcher ======
class DynamicDispatcher:
    def __init__(self, gpt: GPTClient):
        self.gpt = gpt

    def infer_incident_type(self, memory_text: str, latest_user: str) -> str:
        reply = self.gpt.chat([
            {"role": "system", "content": (
                "You are RescueHub's incident classifier. "
                "Decide the emergency type. Return only: fire, medical, or both."
            )},
            {"role": "user", "content": f"Context:\n{memory_text}\nCaller: {latest_user}"}
        ])
        r = reply.lower().strip()
        if "both" in r:
            return "both"
        if "fire" in r:
            return "fire"
        if "medical" in r:
            return "medical"
        return "unknown"


# ====== Fire Agent ======
class FireAgent:
    name = "Fire Agent"

    def __init__(self, gpt: GPTClient, dispatcher: DynamicDispatcher):
        self.gpt = gpt
        self.dispatcher = dispatcher

    def _analyze_injury(self, user_text: str, memory_text: str) -> Optional[bool]:
        msg = [
            {"role": "system", "content": (
                "Check if the user clearly mentions any person being injured, burned, or hurt. "
                "Reply only with one word: yes, no, or unknown.")},
            {"role": "user", "content": f"Conversation:\n{memory_text}\nUser said: {user_text}"}
        ]
        try:
            result = self.gpt.chat(msg).lower().strip()
            if "yes" in result:
                return True
            if "no" in result:
                return False
            return None
        except Exception:
            return None

    def handle(self, user: str, ctx: Ctx, memory: ConversationMemory) -> Tuple[str, Ctx]:
        parsed = self.gpt.parse_user_turn(memory.get_summary(), user)
        ctx.address = ctx.address or parsed.get("address")

        if not ctx.address:
            return "I’m sorry to hear that. Can you tell me your address?", ctx

        ctx.had_fire = True
        if ctx.injuries is None and not ctx.escalation_done:
            ctx.escalation_done = True
            return "Thank you. Are there any injuries?", ctx

        if ctx.injuries is None and ctx.escalation_done:
            guess = self._analyze_injury(user, memory.get_summary())
            if guess is True:
                ctx.injuries = True
                ctx.active_agent = "medical"
                return "I'm escalating this to our medical team. Please hold.", ctx
            elif guess is False:
                ctx.injuries = False
                ctx.incident_type = "fire"
                res = dispatch_resources(ctx.incident_type, ctx.address, injuries=False)
                ctx.done = True
                return (
                    f"We’re dispatching {', '.join(res.resources)} to {ctx.address}. "
                    "Please stay safe until firefighters arrive."
                ), ctx
            else:
                return "Could you please confirm — are there any injuries?", ctx

        if ctx.injuries is True:
            ctx.active_agent = "medical"
            return "I'm escalating this to our medical team. Please hold.", ctx

        if ctx.injuries is False:
            ctx.incident_type = "fire"
            res = dispatch_resources(ctx.incident_type, ctx.address, injuries=False)
            ctx.done = True
            return (
                f"We’re dispatching {', '.join(res.resources)} to {ctx.address}. "
                "Please stay safe until firefighters arrive."
            ), ctx

        return "Understood. Please hold while I confirm the details.", ctx



# ====== Medical Agent ======
class MedicalAgent:
    name = "Medical Agent"

    def __init__(self, gpt: GPTClient, dispatcher: DynamicDispatcher):
        self.gpt = gpt
        self.dispatcher = dispatcher

    def _heuristic_check(self, text: str) -> bool:
        keywords = [
            r"second[- ]?degree", r"third[- ]?degree", r"burn",
            r"fracture", r"broken",
            r"bleeding", r"blood", r"cut", r"wound", r"laceration",
            r"bruise", r"swollen",
            r"head", r"concussion", r"unconscious", r"fainted"
        ]
        return any(re.search(k, text.lower()) for k in keywords)

    def _heuristic_type(self, text: str) -> str:
        t = text.lower()
        if re.search(r"second[- ]?degree|third[- ]?degree|burn", t):
            return "burn"
        if re.search(r"fracture|broken", t):
            return "fracture"
        if re.search(r"bleeding|blood|cut|wound|laceration", t):
            return "bleeding"
        if re.search(r"head|concussion|unconscious|fainted", t):
            return "head"
        return "other"

    def _analyze_medical_context(self, memory_text: str, user_text: str) -> dict:
        system_prompt = (
            "You are RescueHub's medical triage AI.\n"
            "Analyze the user's latest message about an injury.\n"
            "Return structured JSON only like:\n"
            "{"
            "  \"injury_type\": \"burn | fracture | bleeding | head | other\","
            "  \"has_enough_info\": true/false,"
            "  \"next_question\": \"short medical question or null\""
            "}\n\n"
            "Rules:\n"
            "- If user clearly describes something like 'second-degree burn', 'broken leg', or 'heavy bleeding', set has_enough_info=true.\n"
            "- If vague ('he’s hurt', 'he’s in pain'), set has_enough_info=false and propose a follow-up.\n"
            "- If has_enough_info=true, next_question must be null.\n"
            "- NEVER repeat identical questions."
        )
        msg = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Conversation so far:\n{memory_text}\nUser said: {user_text}"}
        ]
        try:
            raw = self.gpt.chat(msg)
            data = json.loads(raw)
        except Exception:
            data = {
                "injury_type": "unknown",
                "has_enough_info": False,
                "next_question": "Can you describe the injury in more detail?"
            }

        # Heuristic override: if we detect concrete info, force enough_info
        if not data.get("has_enough_info") and self._heuristic_check(user_text):
            data["has_enough_info"] = True
            data["injury_type"] = self._heuristic_type(user_text)
            data["next_question"] = None

        # Sanity: map unknown to heuristic guess if available
        if data.get("injury_type") in [None, "", "unknown"]:
            data["injury_type"] = self._heuristic_type(user_text)

        return data

    def _extract_address_regex(self, text: str) -> Optional[str]:
        t = text.strip()
        street_words = r"(street|st\.?|ave(nue)?\.?|road|rd\.?|blvd\.?|lane|ln\.?|way|platz|straße|strasse)"
        pat = rf"([A-Za-z0-9\-]+(?:\s+[A-Za-z0-9\-]+){{0,5}}\s+{street_words})\b"
        m = re.search(pat, t, flags=re.IGNORECASE)
        return m.group(1) if m else None

    def _extract_address_llm_then_regex(self, memory_text: str, user_text: str) -> Optional[str]:
        addr = None
        try:
            parsed = self.gpt.parse_user_turn(memory_text, user_text)
            addr = (parsed or {}).get("address")
        except Exception:
            pass
        if not addr:
            addr = self._extract_address_regex(user_text)
        return addr

    def handle(self, user: str, ctx: Ctx, memory: ConversationMemory) -> Tuple[str, Ctx]:
        if isinstance(user, str) and user.strip().lower().startswith("system: follow up"):
            return "I understand there’s an injury. Can you describe what happened?", ctx

        if not ctx.injury_desc and isinstance(user, str):
            ctx.injury_desc = user

        if not ctx.address:
            extracted = self._extract_address_llm_then_regex(memory.get_summary(), user)
            if extracted:
                ctx.address = extracted

        if not ctx.address:
            if not ctx.asked_address_once:
                ctx.asked_address_once = True
                return "What is your exact address?", ctx
            else:
                return "Please provide the full address (number + street).", ctx

        analysis = self._analyze_medical_context(memory.get_summary(), user)
        enough = analysis.get("has_enough_info", False)
        injury_type = analysis.get("injury_type", "other")
        next_q = analysis.get("next_question") or "Can you describe the injury in more detail?"

        if not ctx.medical_probe_done:
            ctx.medical_probe_done = True
            return next_q, ctx

        if enough:
            ctx.had_medical = True
            if ctx.had_fire and ctx.had_medical:
                ctx.incident_type = "both"
                res = dispatch_resources(ctx.incident_type, ctx.address, injuries=True)
                ctx.done = True
                return (
                    f"Thank you. We’re dispatching {', '.join(res.resources)} to {ctx.address}. "
                    "Please stay calm and keep the patient safe until help arrives."
                ), ctx

            else:
                ctx.incident_type = "medical"

        if ctx.severity != "asked_followup":
            ctx.severity = "asked_followup"
            return next_q, ctx

        ctx.incident_type = "medical" if ctx.active_agent == "medical" else "both"
        ctx.done = True
        res = dispatch_resources(ctx.incident_type, ctx.address, injuries=True)
        return (
            f"Thank you. We’re dispatching {', '.join(res.resources)} to {ctx.address}. "
            "Please stay calm and keep the patient safe until help arrives."
        ), ctx


# ====== Orchestrator ======
class Orchestrator:
    def __init__(self, gpt: GPTClient):
        self.dispatcher = DynamicDispatcher(gpt)
        self.fire = FireAgent(gpt, self.dispatcher)
        self.medical = MedicalAgent(gpt, self.dispatcher)
        self.memory = ConversationMemory()
        self.memory_vec = VectorMemory()
        self.gpt = gpt

    def detect_initial_agent(self, first_input: str) -> str:
        try:
            result = self.gpt.chat([
                {"role": "system", "content": (
                    "Classify this sentence as 'fire' or 'medical'. If both, choose 'fire'."
                )},
                {"role": "user", "content": first_input}
            ])
            r = result.lower()
            if "medical" in r:
                return "medical"
            return "fire"
        except Exception:
            return "fire"

    def _is_explicit_recall_query(self, text: str) -> bool:
        t = text.lower()
        return any(k in t for k in ["remember", "previous", "last time", "earlier report"])

    def step(self, user_text: str, ctx: Ctx) -> Tuple[str, Ctx]:
        current_type = self.detect_initial_agent(user_text)

        if self._is_explicit_recall_query(user_text):
            results, sims = self.memory_vec.search(user_text, top_k=3, return_distance=True)
            threshold = 0.80
            relevant = [
                r for r, s in zip(results, sims)
                if s > threshold and r.get("incident") == current_type
            ]
            if relevant:
                reply = (
                    f"Yes, I remember your previous {current_type} report. "
                    f"Do you want to update it or add new information?"
                )
                self.memory.add("assistant", reply)
                return f"RescueHub: {reply}", ctx

        self.memory.add("user", user_text)
        self.memory_vec.add_memory(user_text, current_type)

        if ctx.active_agent is None:
            ctx.active_agent = current_type

        if ctx.active_agent == "fire":
            reply, ctx = self.fire.handle(user_text, ctx, self.memory)
            self.memory.add("assistant", f"Fire Agent: {reply}")

            if ctx.escalation_done and ctx.active_agent == "medical":
                med_reply, ctx = self.medical.handle("system: follow up", ctx, self.memory)
                full = f"Fire Agent: {reply}\nMedical Agent: {med_reply}"
                self.memory.add("assistant", full)
                return full, ctx

        elif ctx.active_agent == "medical":
            reply, ctx = self.medical.handle(user_text, ctx, self.memory)
            self.memory.add("assistant", f"Medical Agent: {reply}")

        return f"{ctx.active_agent.title()} Agent: {reply}", ctx
