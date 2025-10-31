from typing import List, Tuple, Optional, Dict
from vector_memory import VectorMemory
import json, re
from datetime import datetime
from pathlib import Path

class MemoryManager:

    def __init__(self, gpt_client, persist_dir: str = "memory_store"):
        self.vector = VectorMemory(persist_dir=persist_dir)
        self.gpt = gpt_client

        self.persist_dir = Path(persist_dir)
        self.persist_dir.mkdir(exist_ok=True)
        self.incidents_path = self.persist_dir / "incidents.json"

        if self.incidents_path.exists():
            self.incidents: List[Dict] = json.loads(self.incidents_path.read_text(encoding="utf-8"))
        else:
            self.incidents = []

    # ---------- vector snapshots ----------
    def add_entry(self, user_text: str, assistant_reply: str = "", incident: str = "unknown"):
        chunk = f"User: {user_text}\nAssistant: {assistant_reply}"
        self.vector.add_memory(chunk, incident=incident)

    # ---------- persistence ----------
    def _save_incidents(self):
        self.incidents_path.write_text(json.dumps(self.incidents, ensure_ascii=False, indent=2), encoding="utf-8")

    def _now(self) -> str:
        return datetime.utcnow().isoformat()

    # ---------- address helpers ----------
    def _normalize_address(self, text: Optional[str]) -> Optional[str]:
        if not text:
            return None
        t = text.strip().lower()
        t = t.replace("straße", "strasse")
        t = re.sub(r"\bst\.?\b", "street", t)
        t = re.sub(r"\brd\.?\b", "road", t)
        t = re.sub(r"\save\.?\b", "avenue", t)
        t = re.sub(r"\bln\.?\b", "lane", t)
        t = re.sub(r"(\w)-(\w)", r"\1 \2", t)
        t = re.sub(r"\s+", " ", t)
        return t

    def _similar_enough(self, a: str, b: str) -> bool:
        return a and b and (a == b or a in b or b in a)

    # ---------- incidents API ----------
    def find_by_address(self, raw_address: Optional[str]) -> Optional[Dict]:
        if not raw_address:
            return None
        naddr = self._normalize_address(raw_address)
        if not naddr:
            return None
        for r in reversed(self.incidents):
            if self._similar_enough(naddr, r.get("address", "")):
                return r
        return None

    def upsert_from_ctx(self, ctx, source: str = "agent") -> Dict:
        addr = self._normalize_address(ctx.address) or ""
        inc_type = (ctx.incident_type or ctx.active_agent or "unknown").lower()

        target = None
        for r in reversed(self.incidents):
            if self._similar_enough(addr, r.get("address","")):
                target = r
                break

        if target:
            if ctx.injuries is not None:
                target["injuries"] = bool(ctx.injuries)
            if ctx.injury_desc:
                target["injury_desc"] = ctx.injury_desc
            if inc_type and inc_type != "unknown":
                target["incident_type"] = inc_type
            if getattr(ctx, "done", False) or getattr(ctx, "dispatched", False):
                target["dispatched"] = True
            target.setdefault("history", []).append({
                "ts": self._now(),
                "source": source,
                "ctx": {
                    "address": addr,
                    "incident_type": inc_type,
                    "injuries": ctx.injuries,
                    "injury_desc": ctx.injury_desc
                }
            })
            self._save_incidents()
            return target

        rec = {
            "id": f"inc_{len(self.incidents)+1}",
            "ts": self._now(),
            "address": addr,
            "incident_type": inc_type,
            "injuries": bool(ctx.injuries) if ctx.injuries is not None else None,
            "injury_desc": ctx.injury_desc or "",
            "dispatched": bool(getattr(ctx, "done", False) or getattr(ctx, "dispatched", False)),
            "history": [{
                "ts": self._now(),
                "source": source,
                "ctx": {
                    "address": addr,
                    "incident_type": inc_type,
                    "injuries": ctx.injuries,
                    "injury_desc": ctx.injury_desc
                }
            }]
        }
        self.incidents.append(rec)
        self._save_incidents()
        return rec

    def recall_context(self, user_text: str, current_incident: Optional[str] = None,
                       top_k: int = 3, min_similarity: float = 0.80,
                       require_same_incident: bool = False, return_summary: bool = True) -> str:
        try:
            results, sims = self.vector.search(user_text, top_k=top_k, return_distance=True)
        except TypeError:
            results = self.vector.search(user_text, top_k=top_k)
            sims = [1.0] * len(results)

        if not results:
            return ""

        norm: List[dict] = []
        for r in results:
            if isinstance(r, dict):
                text = r.get("text", "")
                incident = r.get("incident", "unknown")
            else:
                text = str(r)
                incident = "unknown"
            norm.append({"text": text, "incident": incident})

        filtered: List[str] = []
        for item, score in zip(norm, sims):
            if score < min_similarity:
                continue
            if require_same_incident and current_incident:
                inc = (item.get("incident") or "unknown").lower()
                cur = current_incident.lower()
                same = (inc == cur or inc == "both" or cur == "both")
                if not same:
                    continue
            text = (item.get("text") or "").strip()
            if text:
                filtered.append(text)

        if not filtered:
            return ""

        joined = "\n---\n".join(filtered)
        prompt = (
            "You are RescueHub's memory summarizer.\n"
            "Summarize in 1–2 sentences:\n\n" + joined
        )
        try:
            summary = self.gpt.chat([
                {"role":"system","content":prompt},
                {"role":"user","content":f"User's current query: {user_text}"}
            ])
            return summary.strip()
        except Exception:
            return ""
