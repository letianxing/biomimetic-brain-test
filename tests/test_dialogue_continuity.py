import time
import unittest
from unittest.mock import Mock

from brain.dialogue_continuity import ExpectedAnswer
from brain.runtime import BrainRuntime


class ExpectedAnswerTests(unittest.TestCase):
    """Brain records what it asked; the attention system decides who answered."""

    def test_only_a_finished_question_becomes_an_open_expectation(self):
        pending = ExpectedAnswer()
        pending.offered('q', 'a', '你想听听建议吗？', 1500, True)
        self.assertEqual(pending.summary(2000)['question'], '你想听听建议吗？')

        # An interrupted question is not offered at all; a statement clears it.
        self.assertIsNone(ExpectedAnswer().offered('q', 'a', '你想听听建议吗？', 1500, False))
        self.assertIsNone(ExpectedAnswer().summary(2000))

        pending.offered('q', 'a', '好的。', 1500, True)
        self.assertIsNone(pending.summary(2000))

        pending.offered('q', 'unknown', '要我回答吗？', 1500, True)
        self.assertIsNone(pending.summary(2000))

    def test_the_expectation_expires_and_can_be_closed(self):
        pending = ExpectedAnswer()
        pending.offered('q', 'a', '要我回答吗？', 1500, True)
        self.assertIsNone(pending.summary(20000))

        pending.offered('q', 'a', '要我回答吗？', 1500, True)
        pending.resolved('b')
        self.assertIsNotNone(pending.summary(2000))
        pending.resolved('a')
        self.assertIsNone(pending.summary(2000))

    def test_the_snapshot_carries_the_question_text_to_attention(self):
        runtime = BrainRuntime(memory_path=':memory:')
        self.addCleanup(runtime.stop)
        runtime.expected_answer.offered('prior', 'a', '需要我给建议吗？', int(time.time() * 1000) - 200, True)
        summary = runtime.snapshot()['attention_memory']['expected_answer']
        self.assertEqual(summary['person_id'], 'a')
        self.assertEqual(summary['question'], '需要我给建议吗？')


class RuntimeTests(unittest.TestCase):
    def test_answer_gated_by_attention_reaches_the_model_with_its_question(self):
        """No look-away rule here: attention already resolved the addressee."""
        runtime = BrainRuntime(memory_path=':memory:')
        runtime.tts = Mock(enabled=True)
        runtime.tts.speak.return_value = True
        runtime.llm = Mock()
        runtime.llm.respond.return_value = '那就周六去公园吧。'
        runtime.memory.search = Mock(return_value=[])
        self.addCleanup(runtime.stop)

        stamp = int(time.time() * 1000)
        runtime.expected_answer.offered('prior', 'a', '需要我给建议吗？', stamp - 500, True,
                                        {'recent_conversation': [{'person_id': 'a', 'text': '我还没想好周末去哪'}]})
        turn = {'utterance_id': 'answer', 'person_id': 'a', 'stamp_ms': stamp, 'speech_act': 'backchannel',
                'transcript': {'text': '是的', 'started_ms': stamp - 100, 'ended_ms': stamp, 'is_final': True,
                               'emitted_ms': stamp, 'speaker': {'speaker_id': 'a', 'similarity': .9}},
                'attention': {'listen': True, 'addressed_to_robot': True, 'confidence': .77,
                              'reasons': ['expected_answer_to_robot_question', 'expected_answer']}}
        runtime.step({'utterances': [turn], 'attention': {'target_id': 'none', 'listen': False}})

        deadline = time.monotonic() + 2
        while runtime.jobs and time.monotonic() < deadline:
            time.sleep(.01)
        runtime.llm.respond.assert_called_once()
        context = runtime.state['current_turn']['model_context']
        self.assertEqual(context['answering_robot_question']['question'], '需要我给建议吗？')
        self.assertEqual(context['current_user_text'], '是的')
        self.assertEqual(context['answering_robot_question']['preceding_conversation'][0]['text'], '我还没想好周末去哪')
        self.assertIsNone(runtime.expected_answer.summary(stamp))

    def test_an_utterance_attention_did_not_gate_stays_a_backchannel(self):
        runtime = BrainRuntime(memory_path=':memory:')
        runtime.tts = Mock(enabled=True)
        runtime.llm = Mock()
        self.addCleanup(runtime.stop)
        stamp = int(time.time() * 1000)
        runtime.expected_answer.offered('prior', 'a', '需要我给建议吗？', stamp - 500, True)
        turn = {'utterance_id': 'stray', 'person_id': 'a', 'stamp_ms': stamp, 'speech_act': 'backchannel',
                'transcript': {'text': '是的', 'started_ms': stamp - 100, 'is_final': True, 'emitted_ms': stamp,
                               'speaker': {'speaker_id': 'a', 'similarity': .9}},
                'attention': {'listen': False, 'addressed_to_robot': False, 'reasons': []}}
        runtime.step({'utterances': [turn], 'attention': {'target_id': 'none', 'listen': False}})
        self.assertFalse(runtime.jobs)
        runtime.llm.respond.assert_not_called()


if __name__ == '__main__':
    unittest.main()
