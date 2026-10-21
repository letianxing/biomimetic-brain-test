import tempfile
import threading
import time
import unittest
from unittest.mock import Mock, patch
from brain.runtime import BrainRuntime
from brain.llm import ReplyCancelled
from brain.memory import ConversationMemory


def wait_until(predicate, timeout=2):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(.01)
    raise AssertionError("condition timed out")


def turn(key, text, addressed=True, person="person1"):
    stamp = int(time.time() * 1000)
    return {"utterance_id": key, "stamp_ms": stamp, "person_id": person,
            "transcript": {"text": text, "is_final": True, "emitted_ms": stamp},
            "attention": {"listen": addressed, "addressed_to_robot": addressed, "confidence": .9}}


class SpeechTests(unittest.TestCase):
    def setUp(self):
        self.runtime = BrainRuntime(memory_path=":memory:")
        self.runtime.memory.search = Mock(return_value=[])
        self.runtime.tts = Mock()
        self.runtime.tts.speak.return_value = True
        self.addCleanup(self.runtime.stop)

    def test_bystanders_recorded_with_identity_without_reply(self):
        a, b = turn("a", "周六去公园", False), turn("b", "下午三点", False, "person2")
        self.runtime.step({"utterances": [a, b]})
        self.runtime.step({"utterances": [a, b]})
        import json
        records = [json.loads(row[0]) for row in self.runtime.memory.db.execute("SELECT payload FROM outbox").fetchall()]
        self.assertEqual(len([record for record in records if record["kind"] == "heard_utterance"]), 2)
        self.runtime.tts.speak.assert_not_called()
        self.assertEqual({h["person_id"] for h in self.runtime.state["heard"]}, {"person1", "person2"})

    def test_only_attention_confirmed_partial_cancels_blocked_llm(self):
        started = threading.Event()
        def respond(text, memories, cancel=None, **kwargs):
            started.set()
            cancel.wait(2)
            raise ReplyCancelled()
        self.runtime.llm = Mock()
        self.runtime.llm.respond.side_effect = respond
        self.runtime.step({"utterances": [turn("a", "讲一个故事")]})
        self.assertTrue(started.wait(1))
        def candidate(addressed):
            return {"interruption_candidate": {"key": "b", "stamp_ms": int(time.time() * 1000),
                    "attention": {"listen": True, "addressed_to_robot": addressed, "confidence": .9}}}
        self.runtime.step(candidate(False))
        self.assertFalse(self.runtime.cancel.is_set())
        self.runtime.step(candidate(True))
        self.runtime.candidate_since -= 150
        self.runtime.step(candidate(True))
        self.assertTrue(self.runtime.cancel.is_set())
        self.runtime.tts.interrupt.assert_called_once()
        self.runtime.tts.speak.assert_not_called()

    def test_cancelled_late_reply_never_speaks_and_new_turn_does(self):
        started, release = threading.Event(), threading.Event()
        def respond(text, memories, **kwargs):
            started.set()
            release.wait(2)
            return "过期回复"
        self.runtime.llm = Mock()
        self.runtime.llm.respond.side_effect = respond
        self.runtime.step({"utterances": [turn("old", "讲故事")]})
        self.assertTrue(started.wait(1))
        self.runtime.step({"utterances": [turn("new", "你好")]})
        release.set()
        wait_until(lambda: not self.runtime.jobs)
        self.runtime.tts.speak.assert_called_once()
        self.assertEqual(self.runtime.tts.speak.call_args.args[0], "你好，我在听。")

    def test_owner_presence_no_longer_opens_its_own_turn(self):
        """Somebody familiar coming back is raised by attention as a candidate
        (arrival_event) and arbitrated with everything else, rather than being a
        second place that can start the robot talking."""
        owner={'resolved_person_id':'owner','role':'owner','identity_confidence':.9,'active':True}
        for stamp in (100000, 106000, 112000):
            with patch('brain.runtime.now_ms', return_value=stamp):
                self.runtime.step({'person_hypotheses':[owner],
                                   'attention':{'person_id':'owner','listen':True,'confidence':.9}})
        import time as _t; _t.sleep(.2)
        self.runtime.tts.speak.assert_not_called()


class MemoryTests(unittest.TestCase):
    def test_outbox_survives_restart_and_parses_hri_hits(self):
        scope = {"subject_id": "local-user", "robot_id": "reachy-mini", "session_id": "test"}
        with tempfile.TemporaryDirectory() as directory:
            memory = ConversationMemory("http://localhost:1", scope, directory + "/memory.db")
            event = {"scope": scope, "trace_id": "one", "kind": "heard_utterance", "text": "公园", "entity": {"person_id": "person1"}, "observed_at_ms": 1}
            memory.enqueue(event)
            with patch("brain.memory.post_json", side_effect=OSError("offline")):
                memory.flush()
            self.assertEqual(memory.status()["pending"], 1)
            restored = ConversationMemory("http://localhost:1", scope, directory + "/memory.db")
            with patch("brain.memory.post_json", return_value={"hits": [{"record": {**event, "event_id": "id"}}]}):
                results = restored.search("公园")
                self.assertEqual(len(results), 1)
                self.assertEqual(results[0]["text"], "公园")
                restored.flush()
            self.assertEqual(restored.status()["pending"], 0)
            restored.close()
            memory.close()

class AudioObservationTests(unittest.TestCase):
    def test_reflex_is_observed_once_without_speaking_or_actuating(self):
        runtime = BrainRuntime(memory_path=":memory:")
        runtime.tts = Mock()
        stamp = int(time.time() * 1000)
        event = {"id": "audio:1", "kind": "orient", "stamp_ms": stamp, "direction_deg": 130}
        state = {"voice": {"audio_scene": {"music": {"music_state": True, "genre": "jazz"}, "events": [event]}}}
        try:
            runtime.step(state)
            runtime.step(state)
            self.assertEqual(len(runtime.state['audio_actions']), 1)
            self.assertFalse(runtime.state['audio_actions'][0]['executed'])
            runtime.tts.speak.assert_not_called()
            self.assertEqual(runtime.memory.status()['pending'], 1)
        finally:
            runtime.stop()
