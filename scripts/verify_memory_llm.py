#!/usr/bin/env python3
"""Isolated real HRI + Qwen integration check, without microphone/camera capture."""
import json
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import time
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from brain.io import get_json
from brain.llm import OpenAICompatibleClient
from brain.memory import ConversationMemory

root = Path(__file__).resolve().parents[2]
with socket.socket() as sock:
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
with tempfile.TemporaryDirectory(prefix="brain-memory-check-") as directory:
    service = subprocess.Popen([str(root / "hri-memory-service/target/debug/hri-memory-service"),
                                "--host", "127.0.0.1", "--port", str(port), "--data-dir", directory],
                               stdout=subprocess.DEVNULL)
    memory = None
    try:
        url = f"http://127.0.0.1:{port}"
        deadline = time.monotonic() + 10
        while True:
            try:
                get_json(url + "/v1/health")
                break
            except Exception:
                if time.monotonic() > deadline:
                    raise
                time.sleep(.1)
        scope = {"subject_id": "local-user", "robot_id": "reachy-mini", "session_id": "isolated-verification"}
        memory = ConversationMemory(url, scope, directory + "/outbox.db")
        for index, (person, text) in enumerate([("owner", "我想周六去紫竹院公园"), ("person2", "那就下午三点在东门见")]):
            event = {"scope": scope, "source_system": "biomimetic-brain-test", "source_topic": "/brain/heard_utterance",
                     "kind": "heard_utterance", "trace_id": str(index), "entity": {"person_id": person},
                     "text": text, "structured": {"attention": {"addressed_to_robot": False}, "identity_source": "voice_enrollment"},
                     "observed_at_ms": 1000 + index}
            memory.enqueue(event)
        memory.flush()
        assert memory.status()["pending"] == 0, memory.status()
        # Remove local copies from retrieval to verify the real HRI hits protocol.
        with memory.lock:
            memory.db.execute("DELETE FROM outbox")
            memory.db.commit()
        records = memory.search("他们约在哪里几点见面")
        assert len(records) == 2 and {r["entity"]["person_id"] for r in records} == {"owner", "person2"}
        llm = OpenAICompatibleClient("http://127.0.0.1:18050/v1/chat/completions", "qwen3.5-9b")
        reply = llm.respond("刚才他们约在哪里、几点见面？请区分是谁说的。", records)
        assert ("三" in reply or "3" in reply) and "东门" in reply and "公园" in reply, reply
        assert not any(negative in reply for negative in ("并没有", "并未约定", "没有约定")), reply
        owner_records = memory.search("", "owner")
        greeting = llm.respond("主人刚刚出现，请自然搭话。", owner_records,
                               proactive={"resolved_person_id": "owner", "role": "owner", "emotion_valid": True, "emotion_label": "happy"})
        assert greeting, "empty proactive reply"
        print(json.dumps({"memory_records": len(records), "recall_reply": reply, "owner_greeting": greeting}, ensure_ascii=False, indent=2))
    finally:
        if memory:
            memory.close()
        service.terminate()
        service.wait(timeout=5)
