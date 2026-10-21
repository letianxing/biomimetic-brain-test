import json
import threading
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer

from brain.llm import OpenAICompatibleClient


class TestLLM(unittest.TestCase):
    def test_openai_compatible_client(self):
        requests=[]
        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                requests.append(json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0)))))
                body = json.dumps({"choices": [{"message": {"content": "本地模型回复"}}]}).encode()
                self.send_response(200); self.send_header("content-type", "application/json"); self.send_header("content-length", str(len(body))); self.end_headers(); self.wfile.write(body)
            def log_message(self, *args): pass
        server = HTTPServer(("127.0.0.1", 0), Handler); threading.Thread(target=server.handle_request, daemon=True).start()
        self.addCleanup(server.server_close)
        self.assertEqual(OpenAICompatibleClient(f"http://127.0.0.1:{server.server_port}", "test").respond("我们几点见面", [{"text":"下午三点在公园见","kind":"heard_utterance","entity":{"person_id":"p1"}}]), "本地模型回复")
        content=json.loads(requests[0]["messages"][1]["content"])
        self.assertEqual(content["current_user_text"],"我们几点见面")
        self.assertEqual(content["history"][0]["text"],"下午三点在公园见")
        self.assertIn("先直接回答当前这句话",requests[0]["messages"][0]["content"])
        # The brain is an agent now: the lookups it is allowed to make travel
        # with every request, so the model can answer "今天天气怎么样" instead of
        # asking the person which city they meant.
        self.assertEqual([tool["function"]["name"] for tool in requests[0]["tools"]],
                         ["get_weather","get_datetime","remember_person"])


class ToolTests(unittest.TestCase):
    def test_a_tool_call_is_executed_and_answered_from_its_result(self):
        import brain.llm as llm
        requests=[]
        replies=[{"choices":[{"message":{"content":"","tool_calls":[{"id":"c1","type":"function",
                   "function":{"name":"get_weather","arguments":"{\"city\":\"深圳\"}"}}]}}]},
                 {"choices":[{"message":{"content":"深圳今天多云，26到32度。"}}]}]
        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                requests.append(json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0)))))
                body=json.dumps(replies[len(requests)-1]).encode()
                self.send_response(200); self.send_header("content-type","application/json")
                self.send_header("content-length",str(len(body))); self.end_headers(); self.wfile.write(body)
            def log_message(self,*args): pass
        from http.server import ThreadingHTTPServer
        server=ThreadingHTTPServer(("127.0.0.1",0),Handler)
        threading.Thread(target=server.serve_forever,daemon=True).start()
        self.addCleanup(server.shutdown)
        calls=[]
        def fake_tool(name,arguments,context=None):
            calls.append((name,arguments)); return {"city":"深圳","condition":"多云","min_c":26,"max_c":32}
        original=llm.run_tool; llm.run_tool=fake_tool; self.addCleanup(lambda: setattr(llm,"run_tool",original))

        answer=llm.OpenAICompatibleClient(f"http://127.0.0.1:{server.server_port}","test").respond("今天天气怎么样",[])

        self.assertEqual(answer,"深圳今天多云，26到32度。")
        self.assertEqual(calls,[("get_weather",'{"city":"深圳"}')])
        # the tool result has to be in the second request, or the answer is invented
        self.assertEqual(requests[1]["messages"][-1]["role"],"tool")
        self.assertIn("多云",requests[1]["messages"][-1]["content"])

    def test_a_tool_that_fails_does_not_kill_the_turn(self):
        import brain.llm as llm
        from brain.tools import run_tool
        self.assertEqual(run_tool("no_such_tool",{})["error"],"unknown_tool")
        self.assertEqual(run_tool("get_weather","not json")["error"],"unknown_location")


if __name__ == "__main__": unittest.main()

class CancellationTests(unittest.TestCase):
    def test_cancellation_closes_waiting_http_request(self):
        from http.server import ThreadingHTTPServer
        from brain.llm import ReplyCancelled
        import time
        entered, release, cancelled = threading.Event(), threading.Event(), threading.Event()
        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                self.rfile.read(int(self.headers.get("Content-Length", 0)))
                entered.set()
                release.wait(2)
            def log_message(self, *args): pass
        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        serving = threading.Thread(target=server.serve_forever, daemon=True)
        serving.start()
        errors = []
        def request():
            try:
                OpenAICompatibleClient(f"http://127.0.0.1:{server.server_port}", "test").respond("test", [], cancel=cancelled)
            except Exception as exc:
                errors.append(exc)
        worker = threading.Thread(target=request)
        try:
            worker.start()
            self.assertTrue(entered.wait(1))
            cancelled.set()
            worker.join(.5)
            self.assertFalse(worker.is_alive())
            self.assertIsInstance(errors[0], ReplyCancelled)
        finally:
            release.set()
            server.shutdown()
            server.server_close()
            worker.join(2)


class ReplyLengthTests(unittest.TestCase):
    """问天气就答"多云"，不要念气象报告。

    From the panel: 「上海多云，最高29度，最低22度。」 — fourteen characters with the
    punctuation stripped, so it sat exactly on the hard limit and was let
    through, and it is precisely the data recital the prompt asks it not to
    produce. A reply over the normal length that recites digits now gets one
    pass to say it plainly.
    """

    def client(self, shorter="多云，挺舒服"):
        client = OpenAICompatibleClient("http://127.0.0.1:1/v1", "test")
        asked = []

        def send(messages, cancel, on_delta):
            asked.append(messages[-1]["content"])
            return shorter, []

        client._send = send
        return client, asked

    def test_a_weather_report_inside_the_hard_limit_is_still_said_again(self):
        client, asked = self.client()
        self.assertEqual(client._shorten_if_needed("上海多云，最高29度，最低22度。", [], None), "多云，挺舒服")
        self.assertIn("不要念数据", asked[0])

    def test_a_short_answer_with_a_number_is_left_alone(self):
        client, asked = self.client()
        self.assertEqual(client._shorten_if_needed("下午两点", [], None), "下午两点")
        self.assertEqual(asked, [])

    def test_a_long_answer_without_numbers_is_still_shortened(self):
        client, asked = self.client("嗯，我在")
        self.assertEqual(client._shorten_if_needed("我明白你的意思，咱们可以慢慢聊这件事情", [], None), "嗯，我在")
        self.assertEqual(len(asked), 1)

    def test_a_reply_that_is_already_short_is_left_alone(self):
        client, asked = self.client()
        self.assertEqual(client._shorten_if_needed("多云", [], None), "多云")
        self.assertEqual(asked, [])
