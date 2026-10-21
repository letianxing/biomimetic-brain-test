"""在面板上换慢脑：本地 Qwen ↔ DeepSeek，key 不进代码也不进状态。"""
import json
import os
import threading
import unittest
import unittest.mock
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from tempfile import TemporaryDirectory

from brain import llm_backends
from brain.llm import OpenAICompatibleClient


class KeyHandlingTests(unittest.TestCase):
    def test_the_key_comes_from_the_environment(self):
        with unittest.mock.patch.dict(os.environ, {"DEEPSEEK_API_KEY": "sk-from-env"}):
            self.assertEqual(llm_backends.resolve("deepseek")["api_key"], "sk-from-env")

    def test_or_from_a_file_named_by_the_environment(self):
        with TemporaryDirectory() as folder:
            path = Path(folder) / "key.txt"
            path.write_text("sk-from-file\n", encoding="utf-8")
            with unittest.mock.patch.dict(os.environ, {"DEEPSEEK_API_KEY": "",
                                                       "BRAIN_DEEPSEEK_KEY_FILE": str(path)}):
                self.assertEqual(llm_backends.resolve("deepseek")["api_key"], "sk-from-file")

    def test_a_missing_key_is_reported_not_guessed(self):
        with unittest.mock.patch.dict(os.environ, {"DEEPSEEK_API_KEY": "",
                                                   "BRAIN_DEEPSEEK_KEY_FILE": "/nope/does-not-exist"}):
            row = next(item for item in llm_backends.describe() if item["id"] == "deepseek")
        self.assertFalse(row["key_present"])
        self.assertFalse(row["available"])
        self.assertIn("API key", row["reason"])

    def test_what_the_console_sees_never_contains_the_key(self):
        with unittest.mock.patch.dict(os.environ, {"DEEPSEEK_API_KEY": "sk-secret-value"}):
            rendered = json.dumps(llm_backends.describe(), ensure_ascii=False)
        self.assertNotIn("sk-secret-value", rendered)
        self.assertIn('"key_present": true', rendered)

    def test_the_local_model_needs_no_key_and_says_where_the_words_go(self):
        with unittest.mock.patch.dict(os.environ, {"BRAIN_LLM_BASE_URL": "http://172.16.60.162:18050/v1"}):
            row = next(item for item in llm_backends.describe() if item["id"] == "qwen_162")
        self.assertTrue(row["available"])
        self.assertIsNone(row["key_present"])
        self.assertIn("不出本地网络", row["where"])
        cloud = next(item for item in llm_backends.describe() if item["id"] == "deepseek")
        self.assertIn("离开这台机器", cloud["where"])


class RequestShapeTests(unittest.TestCase):
    """llama.cpp 要 enable_thinking=false，云端 API 不认这个字段。"""

    def capture(self, **kwargs):
        seen = []

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                seen.append(json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0)))))
                body = json.dumps({"choices": [{"message": {"content": "好"}}]}).encode()
                self.send_response(200); self.send_header("content-type", "application/json")
                self.send_header("content-length", str(len(body))); self.end_headers(); self.wfile.write(body)

            def log_message(self, *args): pass

        server = HTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=server.handle_request, daemon=True).start()
        self.addCleanup(server.server_close)
        client = OpenAICompatibleClient(f"http://127.0.0.1:{server.server_port}", "test", **kwargs)
        client.respond("你好", [])
        return seen[0]

    def test_llama_cpp_gets_the_thinking_switch(self):
        self.assertEqual(self.capture(template_kwargs=True)["chat_template_kwargs"],
                         {"enable_thinking": False})

    def test_a_cloud_api_does_not(self):
        self.assertNotIn("chat_template_kwargs", self.capture(template_kwargs=False))


class SwitchTests(unittest.TestCase):
    def runtime(self):
        from brain.runtime import BrainRuntime
        runtime = BrainRuntime(memory_path=":memory:")
        self.addCleanup(runtime.stop)
        return runtime

    def test_an_unknown_backend_changes_nothing(self):
        runtime = self.runtime()
        before = runtime.llm
        result = runtime.set_llm("not-a-backend")
        self.assertFalse(result["switched"])
        self.assertIs(runtime.llm, before)
        self.assertIn("unknown backend", result["error"])

    def test_a_backend_that_will_not_answer_is_not_switched_to(self):
        runtime = self.runtime()
        before, broken = runtime.llm, unittest.mock.Mock()
        broken.respond.side_effect = RuntimeError("connection refused")
        with unittest.mock.patch("brain.llm.configured_llm", return_value=broken):
            result = runtime.set_llm("deepseek")
        self.assertFalse(result["switched"])
        self.assertIs(runtime.llm, before)
        self.assertIn("connection refused", result["error"])

    def test_a_working_backend_is_switched_to_and_reported(self):
        runtime = self.runtime()
        working = unittest.mock.Mock()
        working.respond.return_value = "好"
        working.model, working.url = "deepseek-chat", "https://api.deepseek.com/v1/chat/completions"
        with unittest.mock.patch("brain.llm.configured_llm", return_value=working):
            result = runtime.set_llm("deepseek")
        self.assertTrue(result["switched"])
        self.assertIs(runtime.llm, working)
        self.assertIs(runtime.brain.slow.llm, working)
        self.assertEqual(runtime.llm_backend, "deepseek")
        self.assertEqual(runtime.state["llm"]["model"], "deepseek-chat")
        self.assertTrue(any(item["kind"] == "slow_brain_switched" for item in runtime.state["timeline"]))

    def test_and_can_be_switched_back(self):
        runtime = self.runtime()
        working = unittest.mock.Mock()
        working.respond.return_value = "好"
        working.model, working.url = "qwen3.5-9b", "http://172.16.60.162:18050/v1/chat/completions"
        with unittest.mock.patch("brain.llm.configured_llm", return_value=working):
            runtime.set_llm("deepseek")
            result = runtime.set_llm("qwen_162")
        self.assertTrue(result["switched"])
        self.assertEqual(runtime.llm_backend, "qwen_162")
