"""慢的那几轮，慢在跑了三趟模型，不是网络。

实测同一时刻的单趟往返：本地 Qwen 首块 143–353 ms，DeepSeek 首块 206–234 ms、
全程 497–864 ms。而现场「上海晴，挺热。」从请求到首字 7515 ms。差的不是网络，
是趟数：模型先凭空答了天气 → 被打回「你没查就说了天气」 → 调工具 → 再答一遍。

天气和时间本来就是查得到、而且已经预取好的事实（预取命中是 0 ms）。问题里
一看就是问这个，就把结果连同问题一起给它，一趟就能答完。
"""
import unittest
from unittest.mock import patch

from brain.llm import OpenAICompatibleClient


class OneTripForFactsTests(unittest.TestCase):
    def client(self):
        return OpenAICompatibleClient('http://127.0.0.1:1', 'test')

    def test_a_weather_question_arrives_with_the_weather_already_in_it(self):
        client = self.client()
        seen = {}

        def capture(messages, cancel, on_delta, lookup_likely=False, speaker=None):
            seen['context'] = messages[1]['content']
            return '上海晴，挺热。'

        client._converse = capture
        with patch('brain.tools.run_tool',
                   return_value={'city': '上海', 'condition': '晴', 'max_c': 29, 'min_c': 22}):
            client.respond('今天上海天气怎么样', [])

        self.assertIn('tool_results', seen['context'])
        self.assertIn('get_weather', seen['context'])
        self.assertIn('29', seen['context'])

    def test_and_a_date_question_with_the_date(self):
        client = self.client()
        seen = {}
        client._converse = lambda messages, *a, **k: seen.setdefault('c', messages[1]['content']) or '周六。'
        with patch('brain.tools.run_tool', return_value={'weekday': '星期六', 'date': '2026-09-20'}):
            client.respond('今天星期几', [])
        self.assertIn('get_datetime', seen['c'])

    def test_an_ordinary_sentence_carries_nothing_extra(self):
        client = self.client()
        seen = {}
        client._converse = lambda messages, *a, **k: seen.setdefault('c', messages[1]['content']) or '好呀。'
        client.respond('周末去公园吧', [])
        self.assertNotIn('tool_results', seen['c'])

    def test_a_lookup_that_fails_does_not_poison_the_context(self):
        client = self.client()
        seen = {}
        client._converse = lambda messages, *a, **k: seen.setdefault('c', messages[1]['content']) or '没查到。'
        with patch('brain.tools.run_tool', return_value={'error': 'unreachable'}):
            client.respond('今天天气怎么样', [])
        self.assertNotIn('tool_results', seen['c'])


class CountingTheTripsTests(unittest.TestCase):
    def test_every_pass_is_counted(self):
        client = OpenAICompatibleClient('http://127.0.0.1:1', 'test')
        replies = [('我这就帮你查。', []), ('上海晴，二十九度，挺热的，记得防晒。', [])]

        def scripted(messages, cancel, on_delta):
            return replies.pop(0) if replies else ('好的。', [])

        client._send = scripted
        client._shorten_if_needed = lambda text, messages, cancel: text
        client.respond('周末去哪玩', [])
        # 第一趟答了却没查 → 被打回 → 第二趟
        self.assertEqual(client.round_trips, 2)
        self.assertEqual(client.nudges, 1)


if __name__ == '__main__':
    unittest.main()
