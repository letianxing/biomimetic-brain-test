from __future__ import annotations
import http.client
import json
import os
import socket
import threading
from urllib.parse import urlparse
from .prompt_context import build_context, context_budget, fit_to_budget
from .tools import TOOLS, run_tool
import re

# Saying it is 25 degrees is a claim about the world that the model has no way to
# know. In the field it answered "上海今天晴，最高温25度" without calling anything,
# and "我这就帮你查查" without ever querying. Prompt wording did not stop either;
# this does.
LOOKUP_CLAIM = re.compile(r"(度|℃|气温|天气|晴|阴|多云|下雨|下雪|带伞|湿度|风速"
                          r"|几点|现在.*点|星期[一二三四五六日天]|今天.*号|日期)")
LOOKUP_PROMISE = re.compile(r"(我这就(查|看)|我查一下|帮你查|马上查|稍等.*查)")
# Replies are meant to be one short spoken line. The model drifts longer however
# the prompt is worded, so the length is checked and one compression pass is
# allowed. Counting Chinese characters, ignoring punctuation.
REPLY_LIMIT = int(os.environ.get("BRAIN_REPLY_CHARS", "6"))
REPLY_HARD_LIMIT = int(os.environ.get("BRAIN_REPLY_CHARS_MAX", "14"))
PUNCTUATION = re.compile(r"[\s，,。.!！?？、：:；;\"'“”‘’（）()~～]")


class TransientLlmError(RuntimeError):
    """现在忙，不是答不了：重试一次可能就好了。"""


class ReplyCancelled(Exception):
    pass


class OpenAICompatibleClient:
    def __init__(self, base_url, model, api_key="", timeout=60., template_kwargs=True, backend=""):
        self.url = base_url.rstrip("/")
        if not self.url.endswith("/chat/completions"):
            self.url += "/chat/completions" if self.url.endswith("/v1") else "/v1/chat/completions"
        self.model, self.api_key, self.timeout = model, api_key, timeout
        # llama.cpp needs enable_thinking=false or Qwen3.5 answers with an
        # empty content and a full reasoning block; a cloud API rejects the
        # unknown field, so it travels with the backend.
        self.template_kwargs, self.backend = bool(template_kwargs), str(backend or "")
        # What was actually looked up for the last turn, so the trace can show
        # whether an answer came from a tool or from the model's own belief.
        self.last_tool_results = []
        self.round_trips = self.nudges = self.shortenings = 0
        self.last_error = ""

    def respond(self, user_text, memories, cancel=None, proactive=None, on_delta=None, perception=None):
        cancel = cancel or threading.Event()
        prompt, context = build_context(user_text, memories, proactive, perception)
        self.last_tool_results = []
        # 这一轮到底跑了几趟模型。A turn that took seven seconds and one that took
        # one look identical in the log unless this is written down; the field
        # slow turns were three round trips (answer → 「你没查就说了天气」 →
        # tool call → answer), not a slow network.
        self.round_trips = self.nudges = self.shortenings = 0
        messages = [{"role": "system", "content": prompt},
                    {"role": "user", "content": json.dumps(context, ensure_ascii=False, separators=(",", ":"))}]
        # Only this sentence decides whether a lookup is likely. Reading the
        # recent conversation for it meant that once anybody mentioned the
        # weather, everything said afterwards was treated as a weather question:
        # "好累呀" came back as a Shanghai forecast.
        lookup_likely = self._needs_a_lookup(user_text)
        # 他问的就是天气或时间，而这些都是查得到的、已经缓存好的事实。
        # `_facts_for` 自己判断问的是不是这两件事——`_needs_a_lookup` 尺度不一样，
        # 它是用来看模型的回答有没有凭空说数字的，「今天星期几」并不触发它。
        # Handing the answer over with the question costs nothing (the forecast
        # is prefetched on a schedule, so the lookup is 0 ms) and saves the
        # round trips that made these turns the slowest of all: the model
        # answered from thin air, was sent back to use the tool, called it, and
        # answered again — three passes for one sentence.
        facts = self._facts_for(user_text)
        if facts:
            context = dict(context, tool_results=facts)
            messages[1]["content"] = json.dumps(context, ensure_ascii=False, separators=(",", ":"))
        try:
            return self._retrying(messages, cancel, on_delta, lookup_likely,
                                  (perception or {}).get("current_speaker") or {})
        except RuntimeError as exc:
            # The estimate is conservative but it is still an estimate, and the
            # server rejects the whole request rather than truncating it. One
            # retry on half the budget is a shorter memory, not a dead turn.
            if "exceed_context_size" not in str(exc) or cancel.is_set():
                raise
            prompt, context = fit_to_budget(prompt, context, context_budget() // 2)
            messages[0]["content"], messages[1]["content"] = prompt, json.dumps(context, ensure_ascii=False, separators=(",", ":"))
            return self._retrying(messages, cancel, on_delta, lookup_likely,
                                  (perception or {}).get("current_speaker") or {})

    def _retrying(self, messages, cancel, on_delta, lookup_likely, speaker):
        """Ask again when the answer was 「现在忙」 rather than an answer.

        A cloud API says 503 and means it for a second or two. The field turn
        that produced this went straight to a red line on the console and the
        robot said nothing at all — from the person's side it simply did not
        answer. One quick retry, then a slower one, then it is the caller's
        problem to fall back to another model.
        """
        attempts = len(self.RETRY_BACKOFF_S) + 1
        for attempt in range(attempts):
            try:
                return self._converse(messages, cancel, on_delta, lookup_likely, speaker=speaker)
            except TransientLlmError:
                self.last_error = "transient"
                if attempt == attempts - 1 or cancel.is_set():
                    raise
                if cancel.wait(self.RETRY_BACKOFF_S[attempt]):
                    raise ReplyCancelled()

    def _converse(self, messages, cancel, on_delta, lookup_likely=False, speaker=None):
        """One turn, with at most one round of looking things up.

        A tool call produces no spoken text, so nothing reaches the speaker until
        the answer is grounded. One round only: two lookups for one spoken
        sentence is a conversation the person is waiting through, and anything
        that needs more than that is not something to answer out loud.
        """
        # Nothing is streamed. A six-character reply gains nothing from being
        # spoken as it is generated, and streaming is how the robot ended up
        # saying two things at once when a turn was retried or superseded.
        spoken = None
        for attempt in range(3):
            self.round_trips += 1
            text, tool_calls = self._send(messages, cancel, spoken if attempt == 0 else None)
            if not tool_calls:
                # Nothing has been spoken yet (replies are not streamed), so
                # this can run on any answer: if it states the weather or the
                # time, or promises to look something up, without a tool having
                # been used, it goes back. "明天呢" after a weather question is a
                # weather question, and the model knew it — it just answered
                # "还没查到" instead of reaching for the tool.
                if attempt == 0 and spoken is None and self._needs_a_lookup(text):
                    # It stated the weather, or promised to look it up, without
                    # looking anything up. Make it use the tool, then speak.
                    messages = messages + [
                        {"role": "assistant", "content": text},
                        {"role": "user", "content": "你刚才的回答里有天气或时间这类你无法凭空知道的内容，"
                                                    "或者你说了要去查却没查。现在先调用工具，再用一句话回答。"}]
                    self.nudges += 1
                    continue
                if not text:
                    raise RuntimeError("LLM returned empty content")
                text = self._shorten_if_needed(text, messages, cancel)
                if on_delta is not None and text:
                    on_delta(text)
                return text
            messages = messages + [{"role": "assistant", "content": text or "", "tool_calls": tool_calls}]
            if attempt >= 2:
                break
            for call in tool_calls:
                function = call.get("function") or {}
                # 谁在说话，由感知层给，不由模型给：remember_person 只从模型那里拿名字。
                result = run_tool(function.get("name"), function.get("arguments"),
                                  context=speaker or {})
                self.last_tool_results.append({"name": function.get("name"), "result": result})
                messages.append({"role": "tool", "tool_call_id": call.get("id") or "0",
                                 "name": function.get("name"),
                                 "content": json.dumps(result, ensure_ascii=False)})
        return text

    def _shorten_if_needed(self, text, messages, cancel):
        """One pass to say it shorter. Never truncates: half a sentence is worse."""
        body = PUNCTUATION.sub("", str(text or ""))
        # Numbers are where it stops sounding like a person. "上海多云，最高29度，
        # 最低22度。" is fourteen characters — inside the hard limit, and exactly
        # the weather report the user asked it not to read out. Any reply over
        # the normal length that recites digits gets one pass to say it plainly;
        # the retry already asks for no data.
        recites_numbers = bool(re.search(r"\d", body)) and len(body) > REPLY_LIMIT
        if len(body) <= REPLY_HARD_LIMIT and not recites_numbers:
            return text
        self.shortenings += 1
        self.round_trips += 1
        try:
            shorter, _ = self._send(messages + [
                {"role": "assistant", "content": text},
                {"role": "user", "content": f"太长了。用不超过 {REPLY_LIMIT} 个字重说一遍，"
                                            "只保留最要紧的意思，说人话，不要念数据。"}], cancel, None)
        except Exception:
            return text
        return shorter or text

    @staticmethod
    def _facts_for(user_text):
        """Look up what the question is about, before the model asks for it."""
        from .tools import run_tool
        said = str(user_text or "")
        facts = {}
        if re.search(r"(天气|气温|度|℃|冷|热|下雨|下雪|带伞|湿度|风)", said):
            weather = run_tool("get_weather", {})
            if weather and not weather.get("error"):
                facts["get_weather"] = weather
        if re.search(r"(几点|现在.*点|星期|周几|几号|日期|今天.*号)", said):
            when = run_tool("get_datetime", {})
            if when and not when.get("error"):
                facts["get_datetime"] = when
        return facts

    @staticmethod
    def _needs_a_lookup(text):
        text = str(text or "")
        return bool(LOOKUP_CLAIM.search(text) or LOOKUP_PROMISE.search(text))

    # 云端忙、网关抖一下、本地模型正在换模型——这些不是"这句话回答不了"，
    # 是"现在别问"。DeepSeek 在现场就直接回过 503 Service is too busy。
    RETRY_STATUSES = {408, 425, 429, 500, 502, 503, 504}
    RETRY_BACKOFF_S = (.4, 1.2)

    def _send(self, messages, cancel, on_delta):
        payload = {"model": self.model, "messages": messages, "tools": TOOLS,
                   "temperature": .3, "max_tokens": int(os.environ.get("BRAIN_MAX_TOKENS", "128")), "stream": on_delta is not None}
        if self.template_kwargs:
            payload["chat_template_kwargs"] = {"enable_thinking": False}
        url = urlparse(self.url)
        cls = http.client.HTTPSConnection if url.scheme == "https" else http.client.HTTPConnection
        connection = cls(url.hostname, url.port, timeout=self.timeout)
        finished = threading.Event()
        transport = {"socket": None}
        def watch():
            while not finished.wait(.03):
                if cancel.is_set():
                    sock = connection.sock or transport["socket"]
                    if sock:
                        try: sock.shutdown(socket.SHUT_RDWR)
                        except OSError: pass
                    connection.close()
                    if sock is not None:
                        return
        watcher = threading.Thread(target=watch, daemon=True)
        watcher.start()
        try:
            if cancel.is_set():
                raise ReplyCancelled()
            headers = {"content-type": "application/json"}
            if self.api_key:
                headers["authorization"] = "Bearer " + self.api_key
            connection.request("POST", url.path + ("?" + url.query if url.query else ""),
                               json.dumps(payload, ensure_ascii=False).encode(), headers)
            transport["socket"] = connection.sock
            response = connection.getresponse()
            if response.status != 200:
                body = response.read(200)
                if response.status in self.RETRY_STATUSES:
                    raise TransientLlmError(f"LLM HTTP {response.status}: {body!r}")
                raise RuntimeError(f"LLM HTTP {response.status}: {body!r}")
            calls = {}
            if on_delta is not None and "text/event-stream" in response.getheader("content-type", ""):
                fragments = []
                while not cancel.is_set():
                    line = response.readline()
                    if not line:
                        break
                    if not line.startswith(b"data:"):
                        continue
                    payload = line[5:].strip()
                    if payload == b"[DONE]":
                        break
                    message = json.loads(payload)
                    choices = message.get("choices") or []
                    delta = (choices[0].get("delta") or {}) if choices else {}
                    content = delta.get("content")
                    if isinstance(content, str) and content:
                        fragments.append(content)
                        on_delta(content)
                    # Tool calls arrive in pieces like text does, and the
                    # arguments are a JSON string built up across chunks.
                    for part in delta.get("tool_calls") or []:
                        slot = calls.setdefault(part.get("index", 0),
                                                {"id": "", "type": "function", "function": {"name": "", "arguments": ""}})
                        if part.get("id"):
                            slot["id"] = part["id"]
                        function = part.get("function") or {}
                        if function.get("name"):
                            slot["function"]["name"] = function["name"]
                        if function.get("arguments"):
                            slot["function"]["arguments"] += function["arguments"]
                text = "".join(fragments).strip()
            else:
                message = json.loads(response.read())["choices"][0]["message"]
                text = str(message.get("content") or "").strip()
                for index, call in enumerate(message.get("tool_calls") or []):
                    calls[index] = call
                if on_delta is not None and text:
                    on_delta(text)
            if cancel.is_set():
                raise ReplyCancelled()
            tool_calls = [calls[key] for key in sorted(calls)]
            if not text and not tool_calls:
                raise RuntimeError("LLM returned empty content")
            return text, tool_calls
        except Exception:
            if cancel.is_set():
                raise ReplyCancelled()
            raise
        finally:
            finished.set()
            connection.close()


def configured_llm(backend=None):
    """The slow brain named by the console, or the one in the environment."""
    from .llm_backends import DEFAULT_BACKEND, resolve
    try:
        spec = resolve(backend or DEFAULT_BACKEND)
    except KeyError:
        spec = None
    if spec and spec["base_url"]:
        return OpenAICompatibleClient(spec["base_url"], spec["model"], spec["api_key"], spec["timeout"],
                                      template_kwargs=spec["template_kwargs"], backend=spec["name"])
    url = os.environ.get("BRAIN_LLM_BASE_URL", "").strip()
    return OpenAICompatibleClient(url, os.environ.get("BRAIN_LLM_MODEL", "local-model"),
                                  os.environ.get("BRAIN_LLM_API_KEY", ""), float(os.environ.get("BRAIN_LLM_TIMEOUT", "60"))) if url else None
