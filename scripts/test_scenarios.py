#!/usr/bin/env python3
"""Deterministic checks for the three physical test contracts."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from brain.core import Brain

b = Brain()
assert not b.dialogue({"listen": True, "addressed_to_robot": False, "confidence": .9}, {"is_final": True, "text": "你好"}, []).eligible
reply = b.dialogue({"listen": True, "addressed_to_robot": True, "confidence": .9}, {"is_final": True, "text": "你好"}, [])
assert reply.path == "fast" and reply.text
memory = b.dialogue({"listen": True, "addressed_to_robot": True, "confidence": .9}, {"is_final": True, "text": "你还记得吗"}, [{"text": "主人喜欢喝咖啡"}])
assert "主人喜欢喝咖啡" in memory.text
greeting = b.owner_return([{"active": True, "role": "owner", "resolved_person_id": "owner"}], 100000)
assert greeting and greeting.action_intent == "greet_owner"
print("scenario_1_attention: PASS")
print("scenario_2_memory_dialogue: PASS")
print("scenario_3_owner_return_proactive: PASS")
