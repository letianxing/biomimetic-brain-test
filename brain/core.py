from __future__ import annotations

from dataclasses import dataclass
import time
import re


@dataclass(frozen=True)
class BrainDecision:
    path: str
    eligible: bool
    text: str = ""
    action_intent: str = "none"
    reason: str = ""
    person_id: str = ""


class EligibilityGate:
    def allow(self, attention: dict, transcript: dict) -> tuple[bool, str]:
        if not transcript.get("is_final") or not str(transcript.get("text") or "").strip():
            return False, "no_final_transcript"
        if not attention.get("listen"):
            return False, "attention_not_listening"
        if not attention.get("addressed_to_robot"):
            return False, "not_addressed_to_robot"
        if float(attention.get("confidence", 0)) < 0.52:
            return False, "attention_confidence_low"
        return True, "eligible"


class FastBrain:
    def decide(self, text: str) -> BrainDecision | None:
        rules = {
            "你好": "你好，我在听。",
            "停止": "好的，我停下了。",
            "闭嘴": "好的，我先安静。",
            "你是谁": "我是正在进行感知和记忆联调的机器人。",
        }
        compact = re.sub(r"^(?:reachy|机器人|小圆|小园|小元|小袁)[，,。！!\s]*", "", text.strip(), flags=re.I).strip("，,。！!？? ")
        if not compact and re.match(r"^(?:小圆|小园|小元|机器人|reachy)",text.strip(),re.I):
            return BrainDecision("fast",True,"我在，你说。",reason="name_call_acknowledgement")
        for token, reply in rules.items():
            if token == compact:
                return BrainDecision("fast", True, reply, "none", f"explicit:{token}")
        return None


class SlowBrain:
    def __init__(self, llm=None):
        self.llm = llm

    def respond(self, text: str, memories: list[dict]) -> str:
        if self.llm is not None:
            try:
                return self.llm.respond(text, memories)
            except Exception:
                pass
        remembered = [str(item.get("text") or "") for item in memories if item.get("text")]
        if "记得" in text or "之前" in text:
            return f"我记得我们之前提到过：{remembered[0]}" if remembered else "我还没有找到相关记忆，但我会记住这次对话。"
        if remembered:
            return f"我听到了：{text}。我还记得你之前说过：{remembered[0]}"
        return f"我听到了：{text}。"


class Arbitrator:
    def choose(self, fast: BrainDecision | None, slow_text: str, person_id: str) -> BrainDecision:
        if fast is not None:
            return BrainDecision(**{**fast.__dict__, "person_id": person_id})
        return BrainDecision("slow", True, slow_text, "none", "slow_dialogue", person_id)


class Brain:
    def __init__(self, llm=None):
        self.gate, self.fast, self.slow, self.arbitrator = EligibilityGate(), FastBrain(), SlowBrain(llm), Arbitrator()
        self.last_owner_seen_ms = 0

    def dialogue(self, attention: dict, transcript: dict, memories: list[dict]) -> BrainDecision:
        allowed, reason = self.gate.allow(attention, transcript)
        person_id = str(attention.get("person_id") or "")
        if not allowed:
            return BrainDecision("gate", False, reason=reason, person_id=person_id)
        text = str(transcript.get("text") or "").strip()
        fast = self.fast.decide(text)
        return self.arbitrator.choose(fast, "" if fast else self.slow.respond(text, memories), person_id)

    def owner_return(self, people: list[dict], now_ms: int | None = None) -> BrainDecision | None:
        stamp = int(now_ms or time.time() * 1000)
        owner = next((p for p in people if p.get("active") and p.get("role") == "owner"), None)
        if owner is None:
            return None
        if self.last_owner_seen_ms and stamp - self.last_owner_seen_ms < 30_000:
            return None
        self.last_owner_seen_ms = stamp
        return BrainDecision("proactive", True, "哇哦，你又回来了，我想你了。", "greet_owner", "owner_return", str(owner.get("resolved_person_id") or "owner"))
