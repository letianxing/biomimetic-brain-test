"""慢脑忙的时候，别把问题吞掉。

现场 2026-09-20 16:48：面板选的是 DeepSeek，接口回 503「Service is too busy」，
控制台上一行红字，机器人一声不吭——站在人的角度，它就是没理你。配了两个模型，
选中的那个此刻答不了，就让另一个把这一轮答完；两个都答不了，也要开口说一句
「连不上大脑」，而不是沉默。
"""
import time
import unittest
from unittest.mock import Mock, patch

from brain.llm import TransientLlmError
from brain.runtime import BrainRuntime


def asked(text='你能列举一下吗？', person='天行'):
    stamp = int(time.time() * 1000)
    return {'utterance_id': 'q' + str(stamp), 'person_id': person, 'participant_id': person,
            'stamp_ms': stamp,
            'transcript': {'text': text, 'started_ms': stamp - 1200, 'ended_ms': stamp,
                           'is_final': True, 'emitted_ms': stamp,
                           'speaker': {'speaker_id': person, 'similarity': .8}},
            'attention': {'listen': True, 'addressed_to_robot': True, 'confidence': .9,
                          'reasons': ['sentence_is_a_question']}}


class BusyModelTests(unittest.TestCase):
    def runtime(self, backend='deepseek'):
        runtime = BrainRuntime(memory_path=':memory:')
        runtime.tts = Mock(enabled=True)
        runtime.tts.speak.return_value = True
        runtime.memory.search = Mock(return_value=[])
        runtime.llm_backend = backend
        self.addCleanup(runtime.stop)
        return runtime

    def settle(self, runtime):
        deadline = time.monotonic() + 4
        while runtime.jobs and time.monotonic() < deadline:
            time.sleep(.01)

    def test_the_other_model_answers_this_turn(self):
        runtime = self.runtime()
        runtime.llm = Mock()
        runtime.llm.respond.side_effect = TransientLlmError("LLM HTTP 503: service is too busy")
        spare = Mock()
        spare.respond.return_value = '可以，三样：豆浆、油条、粥。'

        with patch('brain.llm.configured_llm', return_value=spare):
            runtime.step({'utterances': [asked()], 'attention': {'target_id': '天行', 'listen': True}})
            self.settle(runtime)

        self.assertEqual(runtime.tts.speak.call_args.args[0], '可以，三样：豆浆、油条、粥。')
        fallback = runtime.state['current_turn']['llm_fallback']
        self.assertEqual(fallback['from'], 'deepseek')
        self.assertIn('503', fallback['reason'])
        # 面板选的那个不变：这一轮换人答，不是把开关拨了
        self.assertEqual(runtime.llm_backend, 'deepseek')

    def test_a_real_mistake_is_not_retried_on_the_other_model(self):
        """400 这种是这次请求本身有问题，换个模型也一样，别白跑一趟。"""
        runtime = self.runtime()
        runtime.llm = Mock()
        runtime.llm.respond.side_effect = RuntimeError("LLM HTTP 400: bad request")
        spare = Mock()

        with patch('brain.llm.configured_llm', return_value=spare):
            runtime.step({'utterances': [asked()], 'attention': {'target_id': '天行', 'listen': True}})
            self.settle(runtime)

        spare.respond.assert_not_called()

    def test_when_nothing_answers_it_still_says_something(self):
        runtime = self.runtime()
        runtime.llm = Mock()
        runtime.llm.respond.side_effect = TransientLlmError("LLM HTTP 503: service is too busy")

        with patch('brain.llm.configured_llm', return_value=None):
            runtime.step({'utterances': [asked()], 'attention': {'target_id': '天行', 'listen': True}})
            self.settle(runtime)

        self.assertEqual(runtime.tts.speak.call_args.args[0], runtime.FALLBACK_LINE)
        self.assertEqual(runtime.state['current_turn']['status'], 'answered_without_a_model')

    def test_and_does_not_keep_saying_it(self):
        runtime = self.runtime()
        runtime.llm = Mock()
        runtime.llm.respond.side_effect = TransientLlmError("LLM HTTP 503: service is too busy")

        with patch('brain.llm.configured_llm', return_value=None):
            for _ in range(3):
                runtime.step({'utterances': [asked()], 'attention': {'target_id': '天行', 'listen': True}})
                self.settle(runtime)

        spoken = [call.args[0] for call in runtime.tts.speak.call_args_list]
        self.assertEqual(spoken.count(runtime.FALLBACK_LINE), 1)


class RetryTests(unittest.TestCase):
    def test_a_busy_answer_is_asked_again_before_giving_up(self):
        from brain.llm import OpenAICompatibleClient
        client = OpenAICompatibleClient('http://127.0.0.1:1', 'test')
        client.RETRY_BACKOFF_S = (0., 0.)
        calls = []

        def flaky(messages, cancel, on_delta, lookup_likely=False, speaker=None):
            calls.append(1)
            if len(calls) < 3:
                raise TransientLlmError('LLM HTTP 503: too busy')
            return '好的。'

        client._converse = flaky
        self.assertEqual(client.respond('在吗', []), '好的。')
        self.assertEqual(len(calls), 3)


if __name__ == '__main__':
    unittest.main()
