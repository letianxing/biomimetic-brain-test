"""ROS topic input and trace output, with bounded queues and HTTP fallback visibility."""
from collections import deque
import json
import queue
import threading
import time


class BrainRosStream:
    def __init__(self, url, session, snapshot):
        self.url, self.session, self.snapshot = url, session, snapshot
        self.stop_event = threading.Event()
        self.outputs = queue.Queue(maxsize=256)
        self.turns = deque(maxlen=512)
        self.lock = threading.Lock()
        self.latest = None
        self.received = 0.
        # The attention workspace is the unified subscription point. Brain still
        # acts on /attention/brain_input; this is kept for tracing and for the
        # consumers that will read the workspace instead.
        self.workspace = None
        self.workspace_received = 0.
        self.connected = False
        self.error = ""
        self.dropped = 0
        self.thread = threading.Thread(target=self._run, daemon=True, name="brain-ros-stream")
        self.thread.start()

    def publish_trace(self, event):
        try: self.outputs.put_nowait(event)
        except queue.Full: self.dropped += 1

    def workspace_contents(self):
        """Latest broadcast cycle, or why there is none."""
        with self.lock:
            if self.workspace is None:
                return {"available": False, "reason": "no_workspace_broadcast_received"}
            age = time.monotonic() - self.workspace_received
            return dict(self.workspace, available=age < 1.5, age_ms=int(age * 1000))

    def input(self):
        with self.lock:
            if self.latest is None or time.monotonic() - self.received > .5:
                return None
            data = dict(self.latest)
            data["utterances"] = list(self.turns) + list(data.get("utterances", []))
            self.turns.clear()
            return data

    def status(self):
        return {"connected": self.connected, "error": self.error, "dropped": self.dropped,
                "input_age_ms": round((time.monotonic()-self.received)*1000) if self.received else None}

    def close(self):
        self.stop_event.set()
        self.thread.join(timeout=2)

    def _run(self):
        try: import websocket
        except ImportError as exc:
            self.error = str(exc)
            return
        while not self.stop_event.is_set():
            connection = None
            try:
                connection = websocket.create_connection(self.url, timeout=.3, http_no_proxy=["127.0.0.1", "localhost"])
                for topic in ("/brain/trace", "/brain/state"):
                    connection.send(json.dumps({"op": "advertise", "topic": topic, "type": "std_msgs/msg/String"}))
                for topic in ("/attention/brain_input", "/attention/utterance", "/attention/workspace"):
                    connection.send(json.dumps({"op": "subscribe", "topic": topic, "type": "std_msgs/msg/String", "queue_length": 32}))
                self.connected, self.error = True, ""
                last_state = 0.
                while not self.stop_event.is_set():
                    for _ in range(32):
                        try: event = self.outputs.get_nowait()
                        except queue.Empty: break
                        connection.send(json.dumps({"op": "publish", "topic": "/brain/trace", "msg": {"data": json.dumps(event, ensure_ascii=False)}}))
                    if time.monotonic()-last_state >= .5:
                        connection.send(json.dumps({"op": "publish", "topic": "/brain/state", "msg": {"data": json.dumps(self.snapshot(), ensure_ascii=False)}}))
                        last_state = time.monotonic()
                    connection.settimeout(.02)
                    try: message = json.loads(connection.recv())
                    except websocket.WebSocketTimeoutException: continue
                    if message.get("op") != "publish": continue
                    payload = json.loads((message.get("msg") or {}).get("data") or "{}")
                    if payload.get("session_id") != self.session: continue
                    with self.lock:
                        if message.get("topic") == "/attention/brain_input":
                            self.latest, self.received = payload, time.monotonic()
                        elif message.get("topic") == "/attention/workspace":
                            self.workspace, self.workspace_received = payload, time.monotonic()
                        elif message.get("topic") == "/attention/utterance":
                            if len(self.turns) == self.turns.maxlen: self.dropped += 1
                            self.turns.append(payload["utterance"])
            except Exception as exc:
                self.error = str(exc)
            finally:
                self.connected = False
                if connection: connection.close()
            self.stop_event.wait(.3)
