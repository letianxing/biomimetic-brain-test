"""Durable local outbox plus the HRI event/search protocol."""
import json
import os
import sqlite3
import threading
from pathlib import Path
from .io import post_json
from .memory_echo import is_playback_echo


def record_key(item):
    return json.dumps([item.get("scope",{}),item.get("trace_id") or item.get("event_id")],sort_keys=True)


class ConversationMemory:
    def __init__(self, url, scope, path=None):
        self.url, self.scope = url.rstrip("/"), scope
        path = path or os.environ.get("BRAIN_MEMORY_OUTBOX", str(Path(__file__).resolve().parents[1] / "data" / "outbox.sqlite3"))
        if path != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(path, check_same_thread=False)
        self.db.execute("CREATE TABLE IF NOT EXISTS outbox (key TEXT PRIMARY KEY, payload TEXT NOT NULL, sent INTEGER DEFAULT 0)")
        self.db.commit()
        self.lock = threading.RLock()
        self.error = ""
        self.stop_event = threading.Event()
        self.thread = None
        self.closed = False

    def start(self):
        self.thread = threading.Thread(target=self._run, daemon=True, name="brain-memory")
        self.thread.start()

    def enqueue(self, event):
        key = json.dumps([event["scope"], event["kind"], event["trace_id"]], sort_keys=True)
        with self.lock:
            self.db.execute("INSERT OR IGNORE INTO outbox(key,payload) VALUES (?,?)", (key, json.dumps(event, ensure_ascii=False)))
            self.db.commit()

    def flush(self):
        with self.lock:
            rows = self.db.execute("SELECT key,payload FROM outbox WHERE sent=0 LIMIT 32").fetchall()
        for key, payload in rows:
            if self.stop_event.is_set():
                break
            try:
                post_json(self.url + "/v1/events", json.loads(payload))
                with self.lock:
                    self.db.execute("UPDATE outbox SET sent=1 WHERE key=?", (key,))
                    self.db.commit()
                self.error = ""
            except Exception as exc:
                self.error = str(exc)
                break

    def search(self, query="", person_id=""):
        records = {}
        relevant = []
        try:
            for kind in ("heard_utterance", "dialogue_turn"):
                payload = {"scope": self.scope, "kind": kind, "limit": 12, "all_sessions": True}
                if person_id and person_id != "unknown":
                    payload["entity_id"] = person_id
                # Query hits plus recent turns: recency supports follow-ups with no lexical overlap.
                for query_text in (query, "") if query else ("",):
                    result = post_json(self.url + "/v1/search", dict(payload, query_text=query_text))
                    for hit in result.get("hits", []):
                        item = hit["record"]
                        records[item["event_id"]] = item
                        if query_text and float(hit.get("text_score") or 0) > 0:
                            relevant.append((float(hit.get("score") or 0), item))
            self.error = ""
        except Exception as exc:
            self.error = str(exc)
        # Include not-yet-delivered turns, so a temporary memory outage loses no context.
        with self.lock:
            local = self.db.execute("SELECT payload FROM outbox ORDER BY rowid DESC LIMIT 100").fetchall()
        for (raw,) in local:
            item = json.loads(raw)
            if item.get("kind") not in {"heard_utterance", "dialogue_turn"}:
                continue
            if any(item["scope"].get(k) != self.scope.get(k) for k in ("subject_id","robot_id")) or (person_id and person_id != "unknown" and item["entity"].get("person_id") != person_id):
                continue
            records[record_key(item)] = item
        unique = {record_key(item): item for item in records.values()}
        all_records = list(unique.values())
        unique = {key: item for key, item in unique.items() if not is_playback_echo(item, all_records)}
        relevant = [(score, item) for score, item in relevant if not is_playback_echo(item, all_records)]
        recent = sorted(unique.values(), key=lambda x: x.get("observed_at_ms", 0), reverse=True)[:8]
        ranked = [item for _, item in sorted(relevant, key=lambda pair: pair[0], reverse=True)[:6]]
        selected = {}
        for item in ranked + recent:
            selected.setdefault(record_key(item), item)
        return list(selected.values())[:12]

    def forget_person(self, person_id, all_sessions=True):
        """Ask the store to drop everything it holds about this person.

        Also drops anything still queued in the outbox for them: a sentence
        that has not been sent yet is still a sentence about to be remembered.
        """
        person = str(person_id or "").strip()
        if not person:
            return {"error": "person_id is required"}
        with self.lock:
            rows = self.db.execute("SELECT key, payload FROM outbox WHERE sent=0").fetchall()
            pending_dropped = 0
            for row_key, payload in rows:
                try:
                    record = json.loads(payload)
                except ValueError:
                    continue
                entity = record.get("entity") or {}
                structured = record.get("structured") or {}
                if person in {entity.get("person_id"), entity.get("voice_id"), entity.get("face_id"),
                              structured.get("person_id")}:
                    self.db.execute("DELETE FROM outbox WHERE key=?", (row_key,))
                    pending_dropped += 1
            self.db.commit()
        try:
            result = post_json(self.url + "/v1/forget",
                               {"scope": self.scope, "person_id": person,
                                "all_sessions": bool(all_sessions)})
        except Exception as exc:
            return {"error": str(exc)[:200], "pending_dropped": pending_dropped}
        return dict(result, pending_dropped=pending_dropped)

    def status(self):
        with self.lock:
            pending = self.db.execute("SELECT count(*) FROM outbox WHERE sent=0").fetchone()[0]
        return {"pending": pending, "error": self.error}

    def _run(self):
        while not self.stop_event.is_set():
            self.flush()
            self.stop_event.wait(.5)

    def close(self):
        self.stop_event.set()
        if self.thread:
            self.thread.join(timeout=4)
        with self.lock:
            if not self.closed:
                self.db.close()
                self.closed = True
