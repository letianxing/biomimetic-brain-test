#!/usr/bin/env python3
"""Read-only readiness and live evidence summary; does not start recording."""
import json
from pathlib import Path
import socket
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from brain.io import get_json
from llm_endpoint import health_url

failed = False
for name, url in [
    ("Qwen", health_url()),
    ("Memory", "http://127.0.0.1:8788/v1/health"),
    ("Vision", "http://127.0.0.1:8080/api/state"),
    ("Voice", "http://127.0.0.1:8090/api/state"),
    ("Attention", "http://127.0.0.1:8092/api/state"),
    ("Brain", "http://127.0.0.1:8094/api/state"),
]:
    try:
        data = get_json(url)
        print(name, "HTTP OK")
        if name == "Voice":
            print(json.dumps({key: data.get(key) for key in ("running", "asr_backend", "asr_error", "error", "input_device_info", "playback", "last_speaker") if key != "last_speaker"}, ensure_ascii=False))
            scene = data.get("audio_scene") or {}
            print("  audio_scene:", json.dumps({key: scene.get(key) for key in ("classifier_ready", "error", "dropped_frames", "music")}, ensure_ascii=False))
            print("  ros:", json.dumps(data.get("ros"), ensure_ascii=False))
            if scene.get("error"):
                failed = True
            if not data.get("running"):
                print("  尚未开始录音：8090 选择设备并点击 Start")
            elif data.get("asr_backend") == "none" or data.get("error") or data.get("asr_error"):
                failed = True
        if name == "Attention":
            print(json.dumps(data.get("diagnostics"), ensure_ascii=False))
            print("  utterances:", len(data.get("utterances", [])), "interruption_candidate:", bool(data.get("interruption_candidate")))
        if name == "Brain":
            print(json.dumps({key: data.get(key) for key in ("llm", "s2s", "memory", "error")}, ensure_ascii=False))
            if not (data.get("llm") or {}).get("enabled"):
                print("  ERROR: Brain 未配置 Qwen")
                failed = True
    except Exception as exc:
        failed = True
        print(name, "FAILED", str(exc))
try:
    with socket.create_connection(("127.0.0.1", 9090), timeout=2):
        print("ROS bridge TCP OK")
except OSError as exc:
    failed = True
    print("ROS bridge FAILED", exc)
raise SystemExit(1 if failed else 0)
