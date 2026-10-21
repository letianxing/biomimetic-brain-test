from __future__ import annotations
import base64
import json
import os
import subprocess
import tempfile
import threading
import time
import urllib.request
from urllib.parse import urlparse
import uuid
import wave


def opener(url):
    return urllib.request.build_opener(urllib.request.ProxyHandler({})) if urlparse(url).hostname in {"localhost", "127.0.0.1", "::1"} else urllib.request.build_opener()


def get_json(url):
    with opener(url).open(url, timeout=1.5) as response:
        return json.load(response)


def post_json(url, payload, timeout=3):
    request = urllib.request.Request(url, json.dumps(payload, ensure_ascii=False).encode(), {"content-type": "application/json"})
    with opener(url).open(request, timeout=timeout) as response:
        return json.load(response)


class MacTTS:
    """Brain-owned synthesis; Voice plays PCM and supplies its exact echo reference."""
    def __init__(self, enabled=True, voice="Tingting", on_playback=None, voice_url=None):
        self.enabled, self.voice = enabled, voice
        self.rate = max(120, min(300, int(os.environ.get("BRAIN_TTS_RATE", "210"))))
        self.on_playback = on_playback or (lambda speaking: None)
        self.voice_url = (voice_url or os.environ.get("BRAIN_VOICE_URL", "http://127.0.0.1:8090")).rstrip("/")
        self.process = None
        self.closed = False
        self.lock = threading.RLock()
        self.play_lock = threading.Lock()
        self.playback_id = ""
        self.cancel = threading.Event()

    def speak(self, text, cancel=None, on_playback=None, turn_id=""):
        notify = on_playback or self.on_playback
        cancel = cancel or threading.Event()
        with self.play_lock:
            if not self.enabled or not text or self.closed or cancel.is_set():
                return False
            self.cancel = cancel
            key = uuid.uuid4().hex
            with tempfile.TemporaryDirectory(prefix="brain-tts-") as directory:
                path = directory + "/reply.wav"
                try:
                    with self.lock:
                        if cancel.is_set() or self.closed:
                            return False
                        self.process = subprocess.Popen(["/usr/bin/say", "-v", self.voice, "-r", str(self.rate), "-o", path,
                                                         "--file-format=WAVE", "--data-format=LEI16@16000", text],
                                                        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                        process = self.process
                    code = process.wait()
                    if cancel.is_set() or self.closed:
                        return False
                    if code:
                        raise RuntimeError(f"macOS speech synthesis exited {code}")
                    with wave.open(path, "rb") as wav:
                        if wav.getnchannels() != 1 or wav.getsampwidth() != 2:
                            raise RuntimeError("expected mono PCM16 from macOS say")
                        payload = {"id": key, "turn_id": turn_id, "text": text, "sample_rate_hz": wav.getframerate(),
                                   "pcm_s16le": base64.b64encode(wav.readframes(wav.getnframes())).decode()}
                    with self.lock:
                        if cancel.is_set() or self.closed:
                            return False
                        self.playback_id = key
                        result = post_json(self.voice_url + "/api/playback/start", payload)
                    if result.get("error"):
                        raise RuntimeError(result["error"])
                    notify(True)
                    while not cancel.wait(.04) and not self.closed:
                        state = get_json(self.voice_url + "/api/playback")
                        if state.get("id") != key:
                            return False
                        if state.get("error"):
                            raise RuntimeError(state["error"])
                        if not state.get("speaking"):
                            return True
                    return False
                finally:
                    try:
                        if self.playback_id == key:
                            self._stop_playback(key,20)
                    finally:
                        self.playback_id = ""
                        self.process = None
                        notify(False)

    def _stop_playback(self,key,fade_ms):
        result=post_json(self.voice_url+"/api/playback/stop",{"id":key,"fade_ms":fade_ms})
        until=time.monotonic()+.25+fade_ms/1000
        while result.get("id")==key and result.get("speaking") and time.monotonic()<until:
            time.sleep(.005)
            result=get_json(self.voice_url+"/api/playback")
        return result

    def interrupt(self,fade_ms=20):
        self.cancel.set()
        with self.lock:
            if self.process is not None and self.process.poll() is None:
                self.process.terminate()
            key = self.playback_id
        if key:
            self._stop_playback(key,fade_ms)

    def close(self):
        self.closed = True
        self.interrupt()
