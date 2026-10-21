"""机器人自己去认人，以及记忆按人召回。

Two halves of the same thing. Without a name the robot cannot scope anything:
the panel prints 说话人13996, the model is told a stranger said something, and
memory is searched across everybody because there is nobody to search for. So
it asks — once, for somebody it has both seen and heard — and once it has an
answer, everything downstream has an identity to key on.
"""
import time
import unittest
from unittest.mock import Mock, patch

from brain.runtime import BrainRuntime


def ask_event(person='participant_007'):
    return {"candidate_id": "enrolment:" + person, "kind": "enrolment_request",
            "person_id": person, "salience": .4, "urgency": .25,
            "summary": "还不知道这个人是谁，可以问一次名字",
            "detail": {"participant_id": person, "label": "说话人7"}}


def said(text, stamp, person='participant_007'):
    return {"utterance_id": "u" + str(stamp), "person_id": person, "participant_id": person,
            "stamp_ms": stamp,
            "transcript": {"text": text, "started_ms": stamp - 1000, "ended_ms": stamp,
                           "is_final": True, "emitted_ms": stamp,
                           "speaker": {"speaker_id": "spk_2", "similarity": .3}},
            "attention": {"listen": True, "addressed_to_robot": True, "confidence": .9,
                          "reasons": ["addressed_form"]}}


class AskingTests(unittest.TestCase):
    def runtime(self):
        runtime = BrainRuntime(memory_path=':memory:')
        runtime.tts = Mock(enabled=True)
        runtime.tts.speak.return_value = True
        runtime.llm = Mock()
        runtime.llm.respond.return_value = '好的。'
        runtime.memory.search = Mock(return_value=[])
        self.addCleanup(runtime.stop)
        return runtime

    def settle(self, runtime):
        deadline = time.monotonic() + 3
        while runtime.jobs and time.monotonic() < deadline:
            time.sleep(.01)

    def test_it_asks_the_question_itself_without_the_model(self):
        runtime = self.runtime()
        with patch('brain.runtime.post_json', return_value={}) as posted:
            runtime.step({"utterances": [], "attention": {"target_id": "none", "listen": False},
                          "workspace_events": [ask_event()]})
            self.settle(runtime)
        self.assertEqual(runtime.tts.speak.call_args.args[0], '我还没记住你，怎么称呼？')
        runtime.llm.respond.assert_not_called()
        # …and perception is told it was actually asked, so it stops proposing it.
        self.assertTrue(any('asked-for-a-name' in str(call.args[0]) for call in posted.call_args_list))

    def test_the_answer_binds_the_name_and_says_so(self):
        runtime = self.runtime()
        runtime.enrolment = {"awaiting": "participant_007"}
        stamp = int(time.time() * 1000)
        with patch('brain.runtime.post_json',
                   return_value={"success": True, "person_id": "小李", "message": "小李，记住了。"}) as posted:
            runtime.step({"utterances": [said('我叫小李', stamp)],
                          "attention": {"target_id": "participant_007", "listen": True},
                          "enrolment": {"awaiting": "participant_007"}})
            self.settle(runtime)
        self.assertEqual(runtime.tts.speak.call_args.args[0], '小李，记住了。')
        request = next(call for call in posted.call_args_list if 'enroll-participant' in str(call.args[0]))
        self.assertEqual(request.args[1]['participant_id'], 'participant_007')
        self.assertEqual(request.args[1]['name'], '我叫小李')

    def test_an_answer_that_is_not_a_name_is_just_a_sentence(self):
        runtime = self.runtime()
        stamp = int(time.time() * 1000)
        with patch('brain.runtime.post_json',
                   return_value={"success": False, "reason": "no_name_in_answer"}):
            runtime.step({"utterances": [said('你先说你叫什么', stamp)],
                          "attention": {"target_id": "participant_007", "listen": True},
                          "enrolment": {"awaiting": "participant_007"}})
            self.settle(runtime)
        runtime.llm.respond.assert_called_once()

    def test_a_name_is_bound_to_whoever_said_it_not_to_whoever_was_asked(self):
        """别人插话报自己的名字，也是他自己的名字，不是被问的那个人的。"""
        runtime = self.runtime()
        stamp = int(time.time() * 1000)
        with patch('brain.runtime.post_json',
                   return_value={"success": True, "person_id": "小李", "message": "小李，记住了。"}) as posted:
            runtime.step({"utterances": [said('我叫小李', stamp, person='天行')],
                          "attention": {"target_id": "天行", "listen": True},
                          "enrolment": {"awaiting": "participant_007"}})
            self.settle(runtime)
        request = next(call for call in posted.call_args_list if 'enroll-participant' in str(call.args[0]))
        self.assertEqual(request.args[1]['participant_id'], '天行')

    def test_an_unprompted_introduction_registers_without_being_asked(self):
        """没问也会说「我的名字叫天行」——那一刻就该真的记下来，而不是嘴上说记住了。"""
        runtime = self.runtime()
        stamp = int(time.time() * 1000)
        with patch('brain.runtime.post_json',
                   return_value={"success": True, "person_id": "天行", "message": "天行，记住了。"}) as posted:
            runtime.step({"utterances": [said('我的名字叫天行。', stamp)],
                          "attention": {"target_id": "participant_007", "listen": True}})
            self.settle(runtime)
        self.assertTrue(any('enroll-participant' in str(call.args[0]) for call in posted.call_args_list))
        self.assertEqual(runtime.tts.speak.call_args.args[0], '天行，记住了。')
        runtime.llm.respond.assert_not_called()


class ScopedRecallTests(unittest.TestCase):
    """记忆按人召回：不知道是谁才全局找。"""

    def runtime(self):
        runtime = BrainRuntime(memory_path=':memory:')
        runtime.tts = Mock(enabled=True)
        runtime.tts.speak.return_value = True
        runtime.llm = Mock()
        runtime.llm.respond.return_value = '好的。'
        self.addCleanup(runtime.stop)
        return runtime

    def settle(self, runtime):
        deadline = time.monotonic() + 3
        while runtime.jobs and time.monotonic() < deadline:
            time.sleep(.01)

    def record(self, person, text):
        return {"event_id": person + text, "trace_id": person + text, "kind": "heard_utterance",
                "text": text, "entity": {"person_id": person}, "observed_at_ms": 1789551200000,
                "scope": {"session_id": "s"}, "structured": {}}

    def test_a_known_speaker_is_answered_from_their_own_history(self):
        runtime = self.runtime()
        runtime.memory.search = Mock(side_effect=lambda text, identity=None: (
            [self.record('天行', '天行上周去了迪士尼'), self.record('天行', '天行喜欢公园'),
             self.record('天行', '天行怕晒')] if identity == '天行' else
            [self.record('陌生人2', '陌生人2要去外滩')]))
        stamp = int(time.time() * 1000)
        runtime.step({"utterances": [said('我们上次说的那个地方', stamp, person='天行')],
                      "attention": {"target_id": "天行", "listen": True}})
        self.settle(runtime)

        turn = runtime.state['current_turn']
        self.assertEqual(turn['memory_scope'], 'this_person')
        self.assertTrue(all('天行' in item['text'] for item in turn['memories']))

    def test_an_unknown_speaker_falls_back_to_the_room(self):
        runtime = self.runtime()
        runtime.memory.search = Mock(side_effect=lambda text, identity=None:
                                     [] if identity else [self.record('陌生人2', '陌生人2要去外滩')])
        stamp = int(time.time() * 1000)
        runtime.step({"utterances": [said('那有什么好玩的', stamp, person='unknown')],
                      "attention": {"target_id": "none", "listen": True}})
        self.settle(runtime)

        self.assertEqual(runtime.state['current_turn']['memory_scope'], 'room_only_identity_unknown')


if __name__ == '__main__':
    unittest.main()
