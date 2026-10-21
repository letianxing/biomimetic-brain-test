"""清空某个人的对话记忆：库里、未发送队列里、以及进程里的缓存。"""
import json
import unittest
import unittest.mock
from collections import deque


class ForgetPersonTests(unittest.TestCase):
    def runtime(self):
        from brain.runtime import BrainRuntime
        runtime = BrainRuntime(memory_path=":memory:")
        self.addCleanup(runtime.stop)
        runtime.memory.forget_person = unittest.mock.Mock(return_value={"removed": 7, "pending_dropped": 0})
        return runtime

    def heard(self, person, text, stamp):
        return {"utterance_id": f"{person}-{stamp}", "person_id": person, "stamp_ms": stamp,
                "transcript": {"text": text, "emitted_ms": stamp, "ended_ms": stamp},
                "attention": {"addressed_to_robot": True}}

    def test_the_live_caches_forget_too(self):
        runtime = self.runtime()
        runtime.state["heard"] = [self.heard("天行", "上海迪士尼挺好玩的", 1000),
                                  self.heard("guest_2", "冰淇淋很贵", 1100)]
        runtime.state["events"] = [{"person_id": "天行", "text": "我也想去", "stamp_ms": 1050},
                                   {"person_id": "guest_2", "text": "是挺贵", "stamp_ms": 1150}]
        runtime.held = deque([{"person_id": "天行", "text": "你觉得呢", "stamp_ms": 1200},
                              {"person_id": "guest_2", "text": "能便宜点吗", "stamp_ms": 1250}], maxlen=8)
        runtime.answered = deque([("天行", "上海迪士尼挺好玩的", 1000)], maxlen=16)
        runtime.owners["天行"] = {"since": 1000}
        runtime.voice_bindings["天行"] = ["stranger_dd"]
        runtime.state["current_turn"] = {"person_id": "天行", "turn_id": "t1"}

        result = runtime.forget_person("天行")

        self.assertEqual([turn["person_id"] for turn in runtime.state["heard"]], ["guest_2"])
        self.assertEqual([event["person_id"] for event in runtime.state["events"]], ["guest_2"])
        self.assertEqual([item["person_id"] for item in runtime.held], ["guest_2"])
        self.assertEqual(list(runtime.answered), [])
        self.assertNotIn("天行", runtime.owners)
        self.assertNotIn("天行", runtime.voice_bindings)
        self.assertIsNone(runtime.state["current_turn"])
        self.assertEqual(result["live"], {"heard": 1, "events": 1, "held": 1, "answered": 1})
        self.assertEqual(result["stored"]["removed"], 7)
        self.assertIn("声纹与人脸模板没有删除", result["note"])
        runtime.memory.forget_person.assert_called_once_with("天行", all_sessions=True)

    def test_somebody_else_is_left_alone(self):
        runtime = self.runtime()
        runtime.state["heard"] = [self.heard("guest_2", "冰淇淋很贵", 1100)]
        runtime.forget_person("天行")
        self.assertEqual(len(runtime.state["heard"]), 1)

    def test_no_name_is_refused(self):
        runtime = self.runtime()
        self.assertIn("error", runtime.forget_person("  "))
        runtime.memory.forget_person.assert_not_called()

    def test_the_outbox_drops_anything_still_queued_for_them(self):
        from brain.memory import ConversationMemory
        memory = ConversationMemory("http://127.0.0.1:1", {"subject_id": "s", "robot_id": "r",
                                                           "session_id": "lab"}, ":memory:")
        self.addCleanup(memory.close)
        for person, text in (("天行", "上海迪士尼挺好玩的"), ("guest_2", "冰淇淋很贵")):
            memory.enqueue({"scope": memory.scope, "source_system": "brain", "source_topic": "/t",
                            "kind": "heard_utterance", "entity": {"person_id": person},
                            "text": text, "structured": {}, "observed_at_ms": 1000,
                            "trace_id": person})
        with unittest.mock.patch("brain.memory.post_json", return_value={"removed": 0}):
            result = memory.forget_person("天行")
        self.assertEqual(result["pending_dropped"], 1)
        remaining = memory.db.execute("SELECT payload FROM outbox").fetchall()
        self.assertEqual(len(remaining), 1)
        self.assertEqual(json.loads(remaining[0][0])["entity"]["person_id"], "guest_2")
