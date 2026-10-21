from __future__ import annotations
from collections import deque
from dataclasses import asdict
import os
import re
import threading
import time
import uuid
from .core import Brain, BrainDecision
from .io import MacTTS, get_json, post_json
from .registration import parse_registration
from .llm import configured_llm, ReplyCancelled
from .memory import ConversationMemory, record_key
from .streaming_speech import StreamingSpeech
from .attention_trigger import AttentionTrigger
from .dialogue_continuity import ExpectedAnswer


# Phases the attention loop passes through constantly on its way somewhere:
# they are visible live on the 8092 panel and say nothing after the fact.
TRANSIENT_PHASES = {"ACQUIRING", "ORIENTING"}


def now_ms():
    return int(time.time() * 1000)


class BrainRuntime:
    def __init__(self, attention_url="http://127.0.0.1:8092", memory_url="http://127.0.0.1:8788", memory_path=None):
        self.attention_url = attention_url.rstrip("/")
        from .llm_backends import DEFAULT_BACKEND
        self.llm_backend = DEFAULT_BACKEND
        self.llm = configured_llm(self.llm_backend)
        self.scope = {"subject_id": "local-user", "robot_id": "reachy-mini", "session_id": os.environ.get("SESSION_ID", "mac-first-test")}
        self.memory = ConversationMemory(memory_url, self.scope, memory_path)
        self.brain = Brain(self.llm)
        self.tts = MacTTS(on_playback=self._on_playback)
        self.state = {"decision": None, "events": [], "heard": [],
                      "llm": self._llm_status(),
                      "tts": {"enabled": True, "voice": "Tingting", "output": "Voice PCM / macOS default"},
                      "s2s": {"owner": "brain", "speaking": False, "thinking": False, "error": "", "interruptions": 0},
                      "memory": {"pending": 0, "error": ""}}
        self.running = False
        self.stop_event = threading.Event()
        self.cancel = threading.Event()
        self.generation = 0
        self.lock = threading.RLock()
        self.seen = set()
        self.last_candidate = ""
        self.candidate_since = 0
        self.interrupted_keys = set()
        self.owners = {}
        self.interaction_transition_seen=set()
        self.last_phase_traced=None
        self.lock_lost_since=0
        self.state["initiative"]={}
        self.crossmodal_seen=set()
        self.reflex_seen = set()
        self.last_startle_ms = 0
        self.pending_cough=None
        self.last_cough_care_ms=0
        self.attention_trigger=AttentionTrigger()
        self.expected_answer=ExpectedAnswer()
        self.pending_group_turn=None
        self.pending_user_turn = None
        # What we have just answered, to catch the same sentence arriving twice.
        # In the field "你是谁呀" was answered twice two seconds apart: one piece
        # of speech, two finals, two identical replies.
        self.answered = deque(maxlen=16)
        # Sentences attention thought were probably for us but was not sure
        # about. Not answered when heard; answered if attention then settles on
        # the person who said them, which is what "等注意力到了再结合上下文回"
        # actually means in code.
        self.held = deque(maxlen=8)
        # Which voices belong to which person, as reported by perception. Brain
        # does not work this out; it only needs to know that a sentence held
        # under a voice is this person's sentence.
        self.voice_bindings = {}
        self.session_context = {}
        self.enrolment = {}
        self.participant_index = {}
        self._last_client = None
        self.active_stream=None
        self.yield_timer=None
        self.yield_generation=None
        self.yield_complete=None
        self.state["reflex_attention"] = None
        self.last_human_ms = 0
        self.proactive_wait_ms = max(1000, int(os.environ.get("BRAIN_PROACTIVE_WAIT_MS", "5000")))
        self.jobs = set()
        self.audio_event_ids = set()
        self.state["audio_scene"] = {}
        self.state["audio_actions"] = []
        self.state["tts"]["rate"] = self.tts.rate
        self.state["current_turn"] = None
        self.state["timeline"] = []
        self.trace_sequence = 0
        self.run_id = uuid.uuid4().hex[:10]
        self.ros_stream = None
        self.state["input_transport"] = "http"
        self.state["s2s"]["generating"] = False
        self.slots = threading.Semaphore(2)

    def start(self):
        if self.running:
            return
        self.running = True
        self.memory.start()
        if os.environ.get("BRAIN_ROSBRIDGE_URL"):
            from .ros_stream import BrainRosStream
            self.ros_stream = BrainRosStream(os.environ["BRAIN_ROSBRIDGE_URL"], self.scope["session_id"], lambda: self.snapshot(compact=True))
        self.thread = threading.Thread(target=self._loop, daemon=True, name="brain-perception")
        self.thread.start()

    def stop(self):
        self.stop_event.set()
        if self.yield_timer:self.yield_timer.cancel()
        self.running = False
        self.cancel.set()
        if self.ros_stream:
            self.ros_stream.close()
        self.tts.close()
        if hasattr(self, "thread"):
            self.thread.join(timeout=3)
        with self.lock:
            jobs = list(self.jobs)
        for job in jobs:
            job.join(timeout=4)
        if not any(job.is_alive() for job in jobs):
            self.memory.close()

    def _llm_status(self, error=""):
        """What the console shows about the slow brain. Never the key itself."""
        from .llm_backends import describe
        return {"enabled": self.llm is not None,
                "backend": self.llm_backend,
                "model": getattr(self.llm, "model", "rule-fallback"),
                "endpoint": getattr(self.llm, "url", ""),
                "backends": describe(),
                "error": error}

    @staticmethod
    def _passes(client):
        """这一轮跑了几趟模型。

        A seven-second turn and a one-second turn look identical in the log
        without it — and the slow ones in the field were three passes (answer
        → 「你没查就说了天气」 → tool call → answer), not a slow network.
        """
        def counted(name):
            value = getattr(client, name, None)
            return value if isinstance(value, int) else None
        tools = getattr(client, "last_tool_results", None)
        return {"round_trips": counted("round_trips"),
                "sent_back_to_use_a_tool": counted("nudges"),
                "shortened": counted("shortenings"),
                "tools": [str(item.get("name")) for item in tools
                          if isinstance(item, dict)] if isinstance(tools, list) else []}

    def _answer_with_a_fallback(self, call, details):
        """慢脑忙不过来时，换另一个模型把这句话答完。

        The field turn that prompted this: DeepSeek answered 503 「Service is
        too busy」, the console showed a red line, and the robot said nothing —
        from where the person was standing it simply ignored the question. Two
        models are configured; when the chosen one cannot answer *right now*,
        the other one answers this turn. The console selection does not change,
        and the trace records that the answer came from somewhere else.
        """
        from .llm import TransientLlmError, configured_llm
        from .llm_backends import BACKENDS
        self._last_client = self.llm
        try:
            return call(self.llm)
        except ReplyCancelled:
            raise
        except Exception as exc:                                  # noqa: BLE001
            transient = isinstance(exc, TransientLlmError) or "HTTP 5" in str(exc) or "HTTP 429" in str(exc)
            others = [name for name in BACKENDS if name != self.llm_backend]
            if not transient or not others:
                raise
            for name in others:
                spare = configured_llm(name)
                if spare is None:
                    continue
                self._trace("llm_fallback", details.get("turn_id", ""), was=self.llm_backend,
                            now=name, error=str(exc)[:160])
                details["llm_fallback"] = {"from": self.llm_backend, "to": name,
                                           "reason": str(exc)[:160]}
                with self.lock:
                    self.state["s2s"]["error"] = f"{self.llm_backend} 忙，这一轮用 {name} 回答"
                self._last_client = spare
                return call(spare)
            raise

    # 兜底那句话。Short, honest, and not an apology loop: it says the model is
    # unreachable, not that the robot did not understand.
    FALLBACK_LINE = "我这会儿连不上大脑，等一下再问我一次。"
    FALLBACK_QUIET_MS = 30000          # 别每句话都说一遍

    def _say_we_could_not_answer(self, details, turn, exc):
        """The model could not be reached. Say so once, rather than nothing."""
        if not (self.tts and getattr(self.tts, "enabled", False)):
            return
        text = str((turn or {}).get("transcript", {}).get("text") or "").strip()
        if not text:
            return                      # 没有人问问题，就没有必要报告故障
        stamp = now_ms()
        if stamp - int(getattr(self, "_said_we_failed_ms", 0)) < self.FALLBACK_QUIET_MS:
            return
        self._said_we_failed_ms = stamp
        try:
            self.tts.speak(self.FALLBACK_LINE)
        except Exception:                                    # noqa: BLE001
            return
        self._trace("fallback_spoken", details.get("turn_id", ""), error=str(exc)[:160])
        details.update(assistant_text=self.FALLBACK_LINE, status="answered_without_a_model")

    def set_llm(self, backend):
        """Switch the slow brain, keeping the old one if the new one will not answer.

        A cloud backend means the conversation leaves this machine, so the
        switch is explicit and the console says where the words go. The probe
        is one token: an endpoint that cannot answer that will not answer a
        turn either, and finding out during a turn means a silence.
        """
        from .llm import configured_llm
        from .llm_backends import BACKENDS
        name = str(backend or "")
        if name not in BACKENDS:
            return dict(self._llm_status(error=f"unknown backend {name}"), switched=False)
        candidate = configured_llm(name)
        if candidate is None:
            self.state["llm"] = self._llm_status(error=f"{name} 没有配置好（缺少地址或 API key）")
            return dict(self.state["llm"], switched=False)
        try:
            probe = candidate.respond("说一个字", [], cancel=threading.Event(),
                                      perception={"current_speaker": {"person_id": "diagnostic"}})
        except Exception as exc:
            self.state["llm"] = self._llm_status(error=f"{name} 不可用：{str(exc)[:160]}")
            return dict(self.state["llm"], switched=False)
        with self.lock:
            self.llm, self.llm_backend = candidate, name
            self.brain.slow.llm = candidate
            self.state["llm"] = self._llm_status()
        self._trace("slow_brain_switched", "", reason=name, text=str(probe)[:40])
        return dict(self.state["llm"], switched=True, probe=str(probe)[:40])

    def forget_person(self, person_id, all_sessions=True):
        """Forget everything said by or to this person, here and in the store.

        Asked for from the console. The stored records go, and so do the live
        caches — a person is not forgotten while the last two minutes of them
        are still sitting in the context that goes to the model, in the queue
        of sentences waiting for attention, or in the list of who has spoken.

        Identity is deliberately untouched: the voiceprint and the face
        template belong to the perception services, and deleting those is a
        different and larger act than forgetting a conversation.
        """
        person = str(person_id or "").strip()
        if not person:
            return {"error": "person_id is required"}
        removed = {"heard": 0, "events": 0, "held": 0, "answered": 0}
        with self.lock:
            heard = [turn for turn in self.state.get("heard", []) if turn.get("person_id") != person]
            removed["heard"] = len(self.state.get("heard", [])) - len(heard)
            self.state["heard"] = heard
            events = [event for event in (self.state.get("events") or [])
                      if event.get("person_id") != person]
            removed["events"] = len(self.state.get("events") or []) - len(events)
            self.state["events"] = events
            kept = deque((item for item in self.held if item.get("person_id") != person), maxlen=8)
            removed["held"] = len(self.held) - len(kept)
            self.held = kept
            answered = deque(((who, text, when) for who, text, when in self.answered if who != person),
                             maxlen=16)
            removed["answered"] = len(self.answered) - len(answered)
            self.answered = answered
            self.owners.pop(person, None)
            self.voice_bindings.pop(person, None)
            current = self.state.get("current_turn") or {}
            if current.get("person_id") == person:
                self.state["current_turn"] = None
        self.expected_answer.resolved(person)
        stored = self.memory.forget_person(person, all_sessions=all_sessions)
        self._trace("person_forgotten", "", person_id=person,
                    reason=f"stored={stored.get('removed')} live={sum(removed.values())}")
        return {"person_id": person, "live": removed, "stored": stored,
                "note": "只清对话记忆；声纹与人脸模板没有删除。"}

    def configure_tts(self, enabled, voice, rate=None):
        self.tts.enabled, self.tts.voice = enabled, voice or "Tingting"
        if rate is not None:
            self.tts.rate = max(120, min(300, int(rate)))
        self.state["tts"].update(enabled=enabled, voice=self.tts.voice, rate=self.tts.rate)
        if not enabled:
            self.tts.interrupt()

    def _on_playback(self, speaking):
        self.state["s2s"]["speaking"] = speaking
        if speaking:
            self.state["s2s"]["thinking"] = False

    def _last_heard_human_ms(self):
        """Newest transcribed human utterance. Raw VAD is not usable for this:
        it goes active on the robot's own playback, which would let an echo
        re-arm anything keyed on "somebody spoke"."""
        return max((int(turn.get("stamp_ms") or 0) for turn in self.state.get("heard", [])
                    if turn.get("person_id") not in {"robot", None}), default=0)

    def _recent_participants(self):
        """Who has actually spoken in this session, newest timestamp per person.

        A voice counts for the person it belongs to. Somebody who talked while
        nobody could see them is filed under a voice cluster; once perception
        has bound that voice to a face, they have been part of this conversation
        all along, and turning to us is joining it rather than starting one.
        """
        participants={}
        for turn in self.state.get("heard",[]):
            person=turn.get("person_id")
            if person and person not in {"unknown","robot"}:
                participants[person]=max(participants.get(person,0),int(turn.get("stamp_ms") or 0))
        for person,voices in (self.voice_bindings or {}).items():
            for voice in voices:
                if voice in participants:
                    participants[person]=max(participants.get(person,0),participants[voice])
        return participants

    # Our own words come back to us through the microphone. The voice service
    # catches most of that by matching text against what it played; this is the
    # backstop for the ones it mishears.
    OWN_REPLY_WINDOW_MS = 20000
    OWN_REPLY_SIMILARITY = .7

    def _our_recent_replies(self, stamp):
        texts = []
        current = self.state.get("current_turn") or {}
        if current.get("assistant_text"):
            texts.append(current["assistant_text"])
        for event in (self.state.get("events") or [])[:4]:
            if event.get("text") and 0 <= stamp - int(event.get("stamp_ms") or 0) <= self.OWN_REPLY_WINDOW_MS:
                texts.append(event["text"])
        return texts

    def _sounds_like_our_own_reply(self, spoken_text, stamp):
        from difflib import SequenceMatcher
        for said in self._our_recent_replies(stamp):
            ours = re.sub(r"[\s，,。.!！?？、]", "", str(said))
            if not ours or len(spoken_text) < 2:
                continue
            if spoken_text in ours or ours in spoken_text:
                return True
            if SequenceMatcher(None, spoken_text, ours).ratio() >= self.OWN_REPLY_SIMILARITY:
                return True
        return False

    # How recently they have to have spoken for what they said to outrank
    # "somebody came back". Two minutes is about as long as a person will
    # accept being answered about something they said before turning round.
    CONTEXT_PRIORITY_MS = 120000

    def _something_to_say(self, trigger, held, stamp):
        """Why this attention arousal should become speech, and about what.

        Attention settling on somebody says where to look. Whether there is
        anything to say is a separate question, and what to say it about is a
        third one. Order matters and the field showed why: the man talked to
        himself — 我今天身体不太舒服。想出去玩。— then turned to the robot, and
        it greeted him as somebody who had just walked in. Both were true; the
        thing he had just said is the better answer.

        So: a sentence we deliberately kept, then anything he said in the last
        couple of minutes that we never answered, then an exchange of ours
        left open, and only then a candidate that arrived with its own
        content. Failing all of that, if the loop with this person is closed,
        the robot looks and says nothing.
        """
        person = trigger["person_id"]
        if held:
            return "held_sentence", held["text"]
        recent = self._recent_unanswered(person, stamp, self._is_engaging_us(trigger))
        if recent:
            return "their_recent_words", recent
        pending = self.expected_answer.summary(stamp)
        if pending and pending.get("person_id") == person:
            return "we_are_waiting_for_their_answer", ""
        if any(item.get("person_id") == person for item in self.expected_answer.pending_unfinished(stamp)):
            return "unfinished_with_this_person", ""
        if trigger.get("summary") or trigger.get("detail"):
            return "workspace_candidate", ""       # arrival / presence / affect / memory event
        if not self._conversation_closed(person, stamp):
            return "their_last_sentence_has_no_reply", ""
        return "", ""

    def _ask_for_a_name(self, state, stamp):
        """「我还没记住你，怎么称呼？」——问一次，问完就等。

        The robot cannot scope a memory, a session or a reply to a person it
        has no name for. Perception raises this only for somebody it has both
        seen and heard, at most twice, and only when the room is quiet; the
        sentence is fixed rather than generated, because a model asked to
        improvise here produces a different question every time and people
        answer the question, not the mood.
        """
        if self.pending_user_turn or self.pending_group_turn:
            return False
        events = [event for event in (state.get("workspace_events") or [])
                  if event.get("kind") == "enrolment_request"]
        if not events:
            return False
        request = events[0]
        person = str(request.get("person_id") or "")
        detail = request.get("detail") or {}
        if not person:
            return False
        self._trace("asked_for_a_name", "", person_id=person, reason="seen_and_heard_but_unnamed")
        try:
            post_json(self.attention_url + "/api/asked-for-a-name",
                      {"participant_id": person, "stamp_ms": stamp}, timeout=1.)
        except Exception:                                  # noqa: BLE001 - asking is best effort
            pass
        self._submit({"utterance_id": "ask_name:" + str(stamp), "person_id": person,
                      "stamp_ms": stamp, "speak_because": "ask_for_a_name",
                      "transcript": {"text": ""},
                      "ask_name": {"participant_id": person, "label": detail.get("label") or person},
                      "attention": (state.get("attention") or {})})
        return True

    # 自我介绍的说法。People introduce themselves without being asked, and the
    # registration must not depend on the robot having got its question in first
    # — nor on the model deciding to call a tool.
    SELF_INTRODUCTION = re.compile(r"(我叫|我的名字(是|叫)|你可以叫我|可以叫我|就叫我|记住我)")

    def _name_answer(self, turn):
        """Bind a name to this face and this voice, from what they just said.

        Two ways in: the robot asked 「怎么称呼」 and this is the answer, or they
        introduced themselves unprompted. Either way it goes through the
        services that own those profiles, and only for the person perception
        says is speaking. If there is no name in the sentence — they said
        something else entirely — this returns nothing and the sentence is
        handled like any other.
        """
        awaiting = str((self.enrolment or {}).get("awaiting") or "")
        who = str(turn.get("participant_id") or turn.get("person_id") or "")
        text = str((turn.get("transcript") or {}).get("text") or "")
        introduced = bool(self.SELF_INTRODUCTION.search(text))
        if not who or who in {"unknown", "robot"}:
            return None
        if not (introduced or (awaiting and who == awaiting)):
            return None
        if awaiting and who != awaiting and not introduced:
            return None
        try:
            result = post_json(self.attention_url + "/api/enroll-participant",
                               {"name": text, "participant_id": who,
                                "utterance_id": turn.get("utterance_id")}, timeout=10.)
        except Exception as exc:                            # noqa: BLE001
            self._trace("name_registration_failed", turn.get("utterance_id", ""), error=str(exc))
            return None
        if not result.get("success") and result.get("reason") == "no_name_in_answer":
            return None                                     # 他说的不是名字，照常回答
        if result.get("already_known"):
            # 第一次是注册，之后就是普通对话。Repeating 「我已经认识你啦」 every
            # time somebody says their own name is not a conversation; the
            # identity is on file, so let the reply be an ordinary one.
            return None
        self._trace("name_registration", turn.get("utterance_id", ""), result=result)
        name = str(result.get("person_id") or "")
        if result.get("success") and name and name != who:
            # 这一轮之前他说的话，也是他说的。Brain keeps its own short window of
            # what it heard; leaving it under the old id would make the next
            # reply talk about 「说话人7」 and 「小李」 as two people.
            with self.lock:
                for item in self.state.get("heard", []):
                    if item.get("person_id") == who or item.get("participant_id") == who:
                        item["person_id"] = name
                        item["participant_id"] = name
                for item in self.held:
                    if item.get("person_id") == who:
                        item["person_id"] = name
        return BrainDecision("registration", True, result.get("message") or "好的。",
                             reason="enrolled_by_asking" if result.get("success") else "enrolment_retry",
                             person_id=result.get("person_id") or "")

    # 别人之间的话，永远不接。Ambient talk the gate refused can still be picked
    # up when the person turns to the robot — that is the 自言自语→回头 case —
    # but only while it is this fresh, and only if it was not aimed at somebody
    # else. Two minutes later it is not an invitation, it is eavesdropping.
    AMBIENT_PICKUP_MS = 30000

    @staticmethod
    def _aimed_elsewhere(turn):
        """Did the addressee judgement say this was for another person?"""
        attention = turn.get("attention") or {}
        detail = attention.get("addressee_detail") or {}
        terms = set(detail.get("terms") or {})
        if terms & {"addresses_another_person", "third_person_about_robot"}:
            return True
        if "another_speaker_has_the_floor" in set(attention.get("reasons") or ()):
            return True
        addressee = str(turn.get("addressee") or "")
        return addressee.startswith("person:")

    def _for_us(self, turn, stamp=0, engaged=False):
        """Was this sentence ours to answer?

        Three outcomes come out of the addressee gate: answer it, keep it in
        case attention arrives on whoever said it, or let it go. The first two
        are ours whenever attention does arrive. The third was being answered
        as well — in the field 「我感觉肚子有点饿」 scored 0, was refused as not
        addressed to the robot, and was replied to five seconds later because
        attention had settled on the man who said it.

        So the third case now needs the person to be *engaging* the robot right
        now — a sustained look, which is what a silent invitation is — and the
        sentence to be recent and not aimed at anybody else. Attention arriving
        says where to look; it does not make everything said nearby ours.
        """
        attention = turn.get("attention") or {}
        if attention.get("addressed_to_robot") or turn.get("held_for_attention") \
                or attention.get("held_for_attention"):
            return not self._aimed_elsewhere(turn)
        if not engaged or self._aimed_elsewhere(turn):
            return False
        return 0 <= int(stamp) - int(turn.get("stamp_ms") or 0) <= self.AMBIENT_PICKUP_MS

    @staticmethod
    def _is_engaging_us(trigger):
        """Are they turning to the robot, or is it only the robot noticing them?

        Both happen and they are not the same. A sustained look, or somebody
        walking back in after being away, is an opening — picking up what they
        just said is the point of it. Quiet company, an expression, a memory
        the robot wants to raise: those are the *robot* deciding to speak, and
        digging a sentence that was never addressed to it out of the last
        minute is how it ends up answering a conversation it was not in.
        """
        trigger = trigger or {}
        state = str(trigger.get("engagement_state") or "")
        if state in {"INVITED", "ENGAGED", "arrival_event"}:
            return True
        return bool({"gaze_engaged", "silent_invitation", "mutual_gaze", "returned_after_absence"} &
                    set(trigger.get("reasons") or ()))

    def _names_for(self, person_id):
        """Every id that means this person: the face, the voice, the participant.

        The room is talked about in three vocabularies — the camera's person,
        the voiceprint cluster, and the participant perception binds them into —
        and 「转过头来」 only works if a sentence filed under one of them can be
        found from another. This is the translation table.
        """
        person = str(person_id or "")
        names = {person} | set((self.voice_bindings or {}).get(person) or ())
        index = self.participant_index or {}
        for table in ("visual", "speaker"):
            mapped = (index.get(table) or {}).get(person)
            if mapped:
                names.add(mapped)
            for observation, participant in (index.get(table) or {}).items():
                if participant == person or participant in names:
                    names.add(observation)
                    names.add(participant)
        return {name for name in names if name and name not in {"unknown", "robot"}}

    def _said_by(self, turn, names):
        return bool(names & {str(turn.get("person_id") or ""), str(turn.get("participant_id") or "")})

    def _recent_unanswered(self, person_id, stamp, engaged=False):
        """The newest thing they said that we have not replied to, if it is fresh.

        Ambient talk counts. It was not addressed to the robot, which is why it
        was only remembered; attention arriving on the person who said it is
        what makes it answerable, and answering it is what the room expects.
        """
        names = self._names_for(person_id)
        replied = max((int(event.get("stamp_ms") or 0) for event in (self.state.get("events") or [])
                       if str(event.get("person_id") or "") in names and event.get("text")), default=0)
        for turn in self.state.get("heard", []):
            if not self._said_by(turn, names) or not self._for_us(turn, stamp, engaged):
                continue
            when = int(turn.get("stamp_ms") or 0)
            if when <= replied or not 0 <= stamp - when <= self.CONTEXT_PRIORITY_MS:
                continue
            text = str((turn.get("transcript") or {}).get("text") or "").strip()
            if text:
                return text
        return ""

    def _conversation_closed(self, person_id, stamp):
        """Did their last sentence already get a reply from us?"""
        names = self._names_for(person_id)
        heard = max((int(turn.get("stamp_ms") or 0) for turn in self.state.get("heard", [])
                     if self._said_by(turn, names) and self._for_us(turn, stamp, True)), default=0)
        if not heard:
            return True                            # they have not said anything to us at all
        replied = max((int(event.get("stamp_ms") or 0) for event in (self.state.get("events") or [])
                       if str(event.get("person_id") or "") in names and event.get("text")), default=0)
        return replied >= heard

    # How long after taking a sentence the same person is still finishing the
    # same thought rather than asking something new.
    CONTINUATION_MS = 2500

    def _continues_the_turn_in_flight(self, turn, stamp):
        """Same person, still talking, while we are working on their last line.

        One thought said in three breaths is one question. Answering each
        breath produces a reply nobody asked for, and cutting our own sentence
        in half to start the next one is worse: the field heard a lot of
        half-finished replies. Nothing here decides what to say — it decides
        whether this is a new question at all.
        """
        current = self.state.get("current_turn") or {}
        if not current or current.get("proactive"):
            return None
        if str(current.get("person_id") or "") != str(turn.get("person_id") or ""):
            return None      # somebody else speaking is a real interruption
        s2s = self.state["s2s"]
        if not (s2s.get("thinking") or s2s.get("generating") or s2s.get("speaking")):
            return None
        if 0 <= stamp - int(current.get("received_ms") or 0) > self.CONTINUATION_MS:
            return None
        return "already_speaking" if current.get("playback_started_ms") else "not_yet_spoken"

    def _recent_addressees(self):
        """Who was last addressing the robot, which is not the same as who spoke.

        An exchange that is already running does not end because the camera lost
        them; attention uses this so a follow-up off camera is still part of the
        same conversation.
        """
        addressees={}
        for turn in self.state.get("heard",[]):
            person=turn.get("person_id")
            if not person or person in {"unknown","robot"}:continue
            if not (turn.get("attention") or {}).get("addressed_to_robot"):continue
            addressees[person]=max(addressees.get(person,0),int(turn.get("stamp_ms") or 0))
        return addressees

    def _attention_goals(self, stamp):
        """What Brain currently wants, as top-down bias for attention.

        These are intentions Brain already acts on; publishing them lets
        attention weight the people they concern instead of each side keeping a
        private copy. A goal biases competition, it never grants a turn.
        """
        goals=[]
        pending=self.expected_answer.summary(stamp)
        if pending:
            goals.append({"kind":"awaiting_answer","person_id":pending["person_id"],
                          "strength":.65,"expires_ms":pending["expires_ms"]})
        turn=self.state.get("current_turn") or {}
        if turn.get("person_id") and turn.get("status") in {"thinking","synthesizing","speaking"} and not turn.get("reflex"):
            goals.append({"kind":"holding_floor","person_id":turn["person_id"],"strength":.85,
                          "expires_ms":stamp+2000})
        for waiting in (self.pending_group_turn,self.pending_user_turn):
            if waiting and waiting.get("person_id"):
                goals.append({"kind":"deferred_turn","person_id":waiting["person_id"],"strength":.5,
                              "expires_ms":stamp+5000})
        for person,presence in self.owners.items():
            if presence.get("greeted"):continue
            goals.append({"kind":"greet_owner","person_id":person,"strength":.4,"expires_ms":stamp+10000})
        return goals[:8]

    def snapshot(self, compact=False):
        import copy
        with self.lock:
            data = {key: self.state.get(key) for key in ("current_turn", "timeline", "tts", "s2s", "memory", "llm", "error", "audio_scene", "input_transport", "ros", "reflex_attention", "initiative")} if compact else self.state
            participants=self._recent_participants();addressees=self._recent_addressees()
            summary={"recent_participants":[{"person_id":p,"last_heard_ms":t,"last_addressed_ms":addressees.get(p)}
                                            for p,t in sorted(participants.items(),key=lambda item:item[1],reverse=True)[:16]],
                     "expected_answer":self.expected_answer.summary(now_ms()),
                     "unfinished":self.expected_answer.pending_unfinished(now_ms()),
                     "goals":self._attention_goals(now_ms())}
            return copy.deepcopy(dict(data,session_id=self.scope["session_id"],stamp_ms=now_ms(),attention_memory=summary))

    def _trace(self, kind, turn_id, **fields):
        with self.lock:
            self.trace_sequence += 1
            event = dict(id=self.trace_sequence, trace_id=f"{self.run_id}:{self.trace_sequence}", stamp_ms=now_ms(), kind=kind, turn_id=turn_id, **fields)
            self.state["timeline"] = ([event] + self.state["timeline"])[:120]
        record = self._event("brain_trace", event["trace_id"], str(event.get("text") or event.get("reason") or kind), fields.get("person_id", "unknown"), event, event["stamp_ms"])
        record["entity"]["track_id"] = turn_id
        self.memory.enqueue(record)
        if self.ros_stream:
            self.ros_stream.publish_trace(event)

    def _turn_playback(self, generation, details, speaking):
        stamp = now_ms()
        field = "playback_started_ms" if speaking else "playback_ended_ms"
        if not speaking and not details.get("playback_started_ms"):
            return
        if not details.get(field) or not speaking:
            details[field] = stamp
        details["status"] = "speaking" if speaking else "synthesizing"
        self._trace("playback_started" if speaking else "playback_stopped", details["turn_id"],
                    text=details.get("current_chunk") or details["assistant_text"], person_id=details["person_id"])
        with self.lock:
            if generation == self.generation:
                self._on_playback(speaking)
            if generation == self.generation:
                self.state["current_turn"] = dict(details)

        if not speaking and self.yield_generation==generation and self.yield_complete:
            self.yield_complete()

    # Weather is fetched on a schedule, not inside a turn: the cold lookup is
    # two sequential internet requests and measured 3.86 seconds, which is
    # where the field's 6.5 seconds between question and first word went.
    WEATHER_REFRESH_S = 240

    def _keep_the_weather_warm(self):
        from .tools import prefetch
        while not self.stop_event.wait(5 if not self.state.get("weather") else self.WEATHER_REFRESH_S):
            try:
                self.state["weather"] = prefetch()
            except Exception as exc:
                self.state["weather"] = {"warm": False, "reason": str(exc)[:120]}

    def _loop(self):
        threading.Thread(target=self._keep_the_weather_warm, daemon=True, name="brain-weather").start()
        while not self.stop_event.is_set():
            try:
                streamed = self.ros_stream.input() if self.ros_stream else None
                self.state["input_transport"] = "ros2" if streamed else "http_fallback"
                if self.ros_stream: self.state["ros"] = self.ros_stream.status()
                self.step(streamed if streamed is not None else get_json(self.attention_url + "/api/state"))
                self.state["error"] = ""
            except Exception as exc:
                self.state["error"] = str(exc)
            self.state["memory"] = self.memory.status()
            self.stop_event.wait(.05)

    def _interrupt(self, reason, evidence=None, record_interruption=True, fade_ms=20):
        with self.lock:
            if self.yield_timer:self.yield_timer.cancel()
            self.yield_generation=None
            self.state["s2s"]["yield_pending"]=False
            self.cancel.set()
            self.generation += 1
            interrupted_generation=self.generation
            self.state["s2s"]["thinking"] = False
            self.state["s2s"]["generating"] = False
        try:
            self.tts.interrupt(fade_ms=fade_ms)
            with self.lock:
                if self.generation==interrupted_generation:self.state["s2s"]["speaking"]=False
        except Exception as exc:
            self.state["s2s"]["error"] = str(exc)
        field="last_interruption" if record_interruption else "last_deferral"
        self.state["s2s"][field] = {"reason": reason, "stamp_ms": now_ms()}
        if record_interruption:self.state["s2s"]["interruptions"] += 1
        else:self.state["s2s"]["proactive_deferrals"]=self.state["s2s"].get("proactive_deferrals",0)+1
        turn = self.state.get("current_turn")
        if turn:
            if turn.get("proactive") and not turn.get("playback_started_ms"):
                owner=self.owners.get(turn.get("person_id"))
                if owner:owner.update(greeted=False,last_greeting=0,focus_since=None)
            self.state["current_turn"] = dict(turn, status="interrupted")
        self._trace("interrupted" if record_interruption else "proactive_deferred", turn["turn_id"] if turn else "", reason=reason, evidence=evidence)

    def step(self, state):
        stamp = now_ms()
        for transition in reversed(state.get("interaction_transitions",[])):
            identity=f"{transition.get('stamp_ms')}:{transition.get('id')}"
            if identity not in self.interaction_transition_seen:
                if len(self.interaction_transition_seen)>1000:self.interaction_transition_seen.clear()
                self.interaction_transition_seen.add(identity)
                phase=(transition.get("to") or ["unknown"])[0]
                # Acquiring a face and orienting to a sound happen every couple
                # of seconds by design. Recording each one buried every actual
                # decision — twenty-two of the last twenty-two timeline entries
                # were phase churn — and wrote one memory record per bounce.
                if phase in TRANSIENT_PHASES or phase==self.last_phase_traced:
                    continue
                self.last_phase_traced=phase
                self._trace("attention_changed","",reason=phase,evidence=transition)
        self.voice_bindings = state.get("voice_bindings") or {}
        # 还不知道名字的人，和此刻在等谁回答名字。Perception decides when it is
        # worth asking; Brain decides whether it is free to ask, and says it.
        self.enrolment = state.get("enrolment") or {}
        # 同一个人的三种叫法：镜头里的、声音的、绑定后的参与者。
        self.participant_index = state.get("participant_index") or {}
        # Which conversation the current speaker is in, with its own roster and
        # its own ordered transcript. Two people discussing lunch across the
        # room must not end up in the prompt for a question about the project.
        self.session_context = state.get("session_context") or {}
        self.state["perception_context"] = {"people": state.get("person_hypotheses") or state.get("people") or [],
                                             "reflex_attention": state.get("reflex_attention"), "attention_distribution": (state.get("attention") or {}).get("attention_distribution")}
        current=self.state.get("current_turn") or {}
        self._drop_a_greeting_nobody_is_there_for(state, current, stamp)
        self.state["reflex_attention"] = state.get("reflex_attention")
        focus = state.get("reflex_attention") or {}
        if (focus.get("active") and focus.get("kind") == "startle" and focus.get("id")
                and 0 <= stamp - int(focus.get("stamp_ms") or 0) < 2000 and focus.get("id") not in self.reflex_seen):
            if len(self.reflex_seen) >= 512:
                self.reflex_seen.clear()
            self.reflex_seen.add(focus["id"])
            self._trace("startle_attention", focus["id"], evidence=focus)
            if stamp-self.last_startle_ms >= 8000:
                self.last_startle_ms = stamp
                self._submit({"person_id":focus.get("target_id","environment"), "utterance_id":"reflex:"+focus["id"],
                              "stamp_ms":stamp,"transcript":{"text":""}}, reflex_event=focus)
        self.state["crossmodal_events"]=state.get("crossmodal_events") or []
        for event in self.state["crossmodal_events"]:
            if event.get("id") and event["id"] not in self.crossmodal_seen:
                if len(self.crossmodal_seen)>2000:self.crossmodal_seen.clear()
                self.crossmodal_seen.add(event["id"])
                self.memory.enqueue(self._event("crossmodal_event",event["id"],event["kind"],event.get("person_id","environment"),event,event["stamp_ms"]))
        voice = state.get("voice") or {}
        self._observe_audio_scene(voice.get("audio_scene") or {}, stamp)
        vad = voice.get("last_vad") or {}
        partial=voice.get("last_streaming_transcript") or {}
        fresh_partial=bool(partial.get("text") and partial.get("source")!="robot_echo" and not partial.get("is_final") and 0<=stamp-int(partial.get("emitted_ms") or 0)<1000)
        # Raw VAD goes active on our own playback, so it is not evidence that a
        # person is talking. The partial is already filtered for echo; the level
        # only counts while we are quiet.
        playing=bool((voice.get("playback") or {}).get("speaking") or voice.get("robot_speaking")
                     or self.state["s2s"].get("speaking"))
        human_activity=bool((vad.get("active") and not playing and 0<=stamp-int(vad.get("stamp_ms") or 0)<1000)
                            or fresh_partial)
        if human_activity:
            self.last_human_ms = stamp
            current=self.state.get("current_turn") or {}
            # Only a turn the robot started by itself gives way to speech. A
            # turn that is answering what this person just said is not a
            # greeting that can wait: in the field 「我想出去玩了。」 was answered
            # with 「好的，你想去哪玩?」 and the reply was cancelled as a
            # postponed greeting while he stood there looking at it.
            self_initiated=bool(current.get("proactive") or current.get("speak_because")=="workspace_candidate")
            if self_initiated and self.state["s2s"].get("thinking") and not current.get("playback_started_ms"):
                self._interrupt("user_speech_preempts_proactive", record_interruption=False)
        candidate = state.get("interruption_candidate") or {}
        gate = candidate.get("attention") or {}
        key = candidate.get("key", "")
        valid = (key and 0 <= stamp - int(candidate.get("stamp_ms") or 0) < 500 and
                 gate.get("listen") and gate.get("addressed_to_robot") and float(gate.get("confidence") or 0) >= .62 and (candidate.get("plan") or {}).get("mode")!="continue")
        if valid and not self._startle_in_progress(stamp):
            if key != self.last_candidate:
                self.last_candidate, self.candidate_since = key, stamp
            if stamp - self.candidate_since >= int((candidate.get("plan") or {}).get("confirmation_ms",120)) and key not in self.interrupted_keys and (self.state["s2s"]["thinking"] or self.state["s2s"]["speaking"] or self.state["s2s"].get("generating")):
                self.interrupted_keys.add(key)
                if len(self.interrupted_keys) > 512:
                    self.interrupted_keys = {key}
                plan=candidate.get("plan") or {}
                if plan.get("mode")=="yield":self._request_yield("attention_confirmed_user_speech:"+key,candidate,plan)
                else:self._interrupt("attention_confirmed_user_speech:"+key,candidate,fade_ms=int(plan.get("fade_ms",20)))
        else:
            self.last_candidate = ""
        for turn in state.get("utterances", []):
            key = turn["utterance_id"]
            if key in self.seen:
                continue
            self.seen.add(key)
            if len(self.seen) > 4096:
                self.seen = {item["utterance_id"] for item in state.get("utterances", [])}
            transcript = turn["transcript"]
            if turn.get("source") == "robot_echo" or float(transcript.get("self_echo_probability") or 0) >= .75:
                self.memory.enqueue(self._event("robot_echo", key, transcript.get("text", ""), "robot", turn, turn["stamp_ms"]))
                self._trace("echo_rejected", key, text=transcript.get("text", ""), evidence=transcript.get("echo_match"))
                continue
            turn=dict(turn)
            # Attention decided this answers our open question; Brain only adds
            # the question text, because only Brain knows what it asked. A short
            # "是的" is then an answer rather than a backchannel.
            if "expected_answer_to_robot_question" in set((turn.get("attention") or {}).get("reasons") or []):
                pending=self.expected_answer.pending
                if pending and pending["person_id"]==turn.get("person_id"):
                    turn["dialogue_link"]=dict(pending)
                    turn["speech_act"]="answer_to_robot";turn["addressee"]="robot"
            self._remember_heard(turn)
            if self.pending_group_turn and turn["utterance_id"]!=self.pending_group_turn["utterance_id"]:
                self._trace("group_turn_withdrawn",self.pending_group_turn["utterance_id"],reason="another_human_continued")
                self.pending_group_turn=None
            if turn.get("speech_act")=="backchannel":
                self._trace("backchannel_observed",key,text=transcript.get("text",""),reason="acknowledgement_not_floor_request")
                continue
            gate = turn.get("attention") or {}
            if turn.get("held_for_attention") and not gate.get("addressed_to_robot"):
                self.held.append({"person_id": turn.get("person_id"), "text": transcript.get("text", ""),
                                  "utterance_id": key, "stamp_ms": stamp,
                                  "detail": turn.get("addressee_detail")})
                self._trace("held_pending_attention", key, text=transcript.get("text", ""),
                            person_id=turn.get("person_id", "unknown"),
                            reason="probably_for_us_but_not_certain")
                continue
            allowed, reason = self.brain.gate.allow(gate, transcript)
            if not allowed:
                self._trace("gate_rejected", key, text=transcript.get("text", ""), reason=reason, person_id=turn.get("person_id", "unknown"))
                if not self.state["s2s"]["thinking"] and not self.state["s2s"]["speaking"]:
                    self.state["decision"] = asdict(BrainDecision("gate", False, reason=reason, person_id=turn.get("person_id", "unknown")))
                continue
            # Replay is useful for memory recovery, but must not replay old audible replies.
            if not 0 <= stamp - int(transcript.get("emitted_ms") or turn.get("stamp_ms") or 0) < 8000:
                continue
            spoken_text = re.sub(r"[\s，,。.!！?？、]", "", str(transcript.get("text") or ""))
            if spoken_text and self._sounds_like_our_own_reply(spoken_text, stamp):
                # The voice service marks our playback as an echo by matching
                # the text it heard against the text it played; one misheard
                # character gets under that bar. In the field "我是小圆呀" came
                # back as "我是小圆样", was answered, and the answer came back
                # too: one question, three replies. Whatever we just said is
                # not a question for us, however it was transcribed.
                self._trace("own_reply_heard_again", key, text=transcript.get("text", ""),
                            person_id=turn.get("person_id", "unknown"),
                            reason="transcript_is_almost_exactly_what_we_just_said")
                continue
            if spoken_text and any(person == turn.get("person_id") and text == spoken_text
                                   and 0 <= stamp - when < 6000
                                   for person, text, when in self.answered):
                self._trace("duplicate_transcript_skipped", key, text=transcript.get("text", ""),
                            reason="same_sentence_answered_moments_ago")
                continue
            self.answered.append((turn.get("person_id"), spoken_text, stamp))
            # The question has been taken up, so stop telling attention we wait.
            self.expected_answer.resolved(turn.get("person_id"))
            if turn.get("addressee")=="group":
                self.pending_group_turn=turn
                self._trace("group_turn_waiting",key,reason="leave_floor_for_human_response")
                continue
            owner_id=turn.get("person_id")
            owners=[p for p in state.get("person_hypotheses",[]) if p.get("role")=="owner" and p.get("resolved_person_id")==owner_id]
            if owners or (transcript.get("speaker") or {}).get("speaker_role")=="owner":
                presence=self.owners.setdefault(owner_id,{"since":stamp,"last_seen":stamp,"last_greeting":0})
                presence.update(greeted=True,last_seen=stamp)
                if owners:presence["tracking_key"]=owners[0].get("temporary_id")
            continuation=self._continues_the_turn_in_flight(turn,stamp)
            if self._startle_in_progress(stamp) or self.state["s2s"].get("yield_pending"):
                self.pending_user_turn=turn
                self._trace("user_turn_deferred",turn["utterance_id"],reason="startle_has_priority" if self._startle_in_progress(stamp) else "waiting_for_phrase_boundary")
            elif continuation=="already_speaking":
                # We are mid-sentence on their previous line. Finish it, then
                # take this one: 说了就说完. The pending slot already drops it
                # if it goes stale.
                self.pending_user_turn=turn
                self._trace("user_turn_deferred",turn["utterance_id"],
                            reason="same_person_still_talking_let_our_sentence_finish")
            else:
                if continuation=="not_yet_spoken":
                    # Nothing of the earlier answer has been said, so there is
                    # nothing to finish and no reason for two replies: this
                    # sentence replaces it, with the earlier one kept as
                    # context for what to answer.
                    current=self.state.get("current_turn") or {}
                    turn["supersedes"]=current.get("turn_id")
                    turn["superseded_text"]=current.get("user_text")
                self.pending_user_turn=None
                self._submit(turn)
        speaking_now=any(self.state["s2s"].get(key) for key in ("thinking","generating","speaking"))
        if (self.pending_user_turn and not speaking_now and not self._startle_in_progress(stamp)
                and not self.state["s2s"].get("yield_pending")):
            pending,self.pending_user_turn=self.pending_user_turn,None
            if 0<=stamp-int(pending["transcript"].get("emitted_ms") or pending.get("stamp_ms") or 0)<8000:
                self._submit(pending)
        group=self.pending_group_turn
        if group:
            age=stamp-int(group["transcript"].get("emitted_ms") or group.get("stamp_ms") or 0)
            if age>=8000:self.pending_group_turn=None
            elif age>=700 and stamp-self.last_human_ms>=700 and not any(self.state["s2s"].get(k) for k in ("speaking","thinking","generating")) and not self._startle_in_progress(stamp):
                self.pending_group_turn=None
                self._submit(group)
        cough=self.pending_cough
        if cough and stamp-cough["stamp_ms"]>6000:self.pending_cough=None
        elif cough and stamp-cough["stamp_ms"]>=1000 and stamp-self.last_human_ms>=1000:
            attention=state.get("attention") or {}
            if (attention.get("target_kind")=="person" and attention.get("listen")
                    and not any(self.state["s2s"].get(k) for k in ("thinking","speaking","generating"))
                    and stamp-self.last_cough_care_ms>=30000):
                self.pending_cough=None;self.last_cough_care_ms=stamp
                self._submit({"utterance_id":"care:"+cough["id"],"person_id":attention.get("person_id","unknown"),"stamp_ms":stamp,"transcript":{"text":""},"care_event":cough})
        busy=any(self.state["s2s"].get(k) for k in ("thinking","speaking","generating"))
        if not busy and self._ask_for_a_name(state, stamp):
            return
        # Attention decides that somebody is engaging us; Brain only decides
        # whether it is free to answer. No interaction case is matched here.
        trigger=self.attention_trigger.update(dict(state,workspace_events=state.get("workspace_events") or []),
                                              stamp,busy,
                                              self._recent_participants(),self._last_heard_human_ms(),
                                              turn_in_flight=bool(self.pending_user_turn or self.pending_group_turn))
        if trigger:
            presence=self.owners.get(trigger["person_id"])
            if presence:presence["greeted"]=True
            # If this person said something we held, answer that rather than
            # opening on nothing. Attention arriving is what makes the held
            # sentence answerable.
            # The sentence may be filed under a voice rather than a person: the
            # crowd was talking off camera, so all attention had was a cluster.
            # Perception says which voices belong to this face; without that
            # the robot turns to the person who just addressed it and opens on
            # nothing.
            owners=self._names_for(trigger["person_id"])
            held=None
            for item in list(self.held):
                if item["person_id"] in owners and 0<=stamp-item["stamp_ms"]<=30000:
                    held=item
            reason,about=self._something_to_say(trigger,held,stamp)
            if not reason:
                # Attention arriving is not a reason to talk. Looking at
                # somebody the robot has already finished a conversation with
                # and saying something is how it ended up answering a sentence
                # from minutes ago: the loop with that person was closed and
                # there was nothing new, so the model was handed the old
                # context and found something in it to reply to.
                self._trace("attention_without_anything_to_say","",person_id=trigger["person_id"],
                            reason="conversation_with_this_person_is_closed")
                self._owner_presence(state,stamp)
                return
            if held:
                self.held=deque((item for item in self.held if item is not held),maxlen=8)
                self._trace("held_answered_on_attention",held["utterance_id"],text=held["text"],
                            person_id=held["person_id"],reason="attention_arrived")
            self._submit({"utterance_id":"engagement:"+str(stamp),"person_id":trigger["person_id"],"stamp_ms":stamp,
                          "speak_because":reason,
                          "transcript":{"text":about},
                          "attention":(state.get("attention") or {}),"attention_trigger":trigger,
                          "held_utterance":held})
        self._owner_presence(state, stamp)

    def _request_yield(self,reason,evidence,plan):
        if not self.state["s2s"].get("speaking"):
            self._interrupt(reason,evidence,fade_ms=int(plan.get("fade_ms",40)))
            return
        with self.lock:
            generation=self.generation
            if self.yield_generation==generation:return
            self.yield_generation=generation
            self.state["s2s"]["yield_pending"]=True
            stream=self.active_stream[1] if self.active_stream and self.active_stream[0]==generation else None
        def complete():
            with self.lock:
                if self.generation!=generation or self.yield_generation!=generation or self.stop_event.is_set():return
                self.yield_generation=None
                self._interrupt(reason,evidence,fade_ms=int(plan.get("fade_ms",40)))
        self._trace("yield_requested",(self.state.get("current_turn") or {}).get("turn_id",""),reason=reason,evidence=plan)
        self.yield_complete=complete
        if stream:stream.request_yield(complete)
        self.yield_timer=threading.Timer(max(0,min(500,int(plan.get("max_boundary_wait_ms",350))))/1000,complete)
        self.yield_timer.daemon=True;self.yield_timer.start()

    def _startle_in_progress(self,stamp):
        turn=self.state.get("current_turn") or {}
        return bool(turn.get("reflex") and turn.get("status") in {"thinking","synthesizing","speaking"}
                    and 0<=stamp-int(turn.get("received_ms") or 0)<2000)

    def _observe_audio_scene(self, scene, stamp):
        self.state["audio_scene"] = {key: scene.get(key) for key in ("acoustic", "music", "classifier_ready", "error")}
        for event in scene.get("events", []):
            key = event.get("id")
            if not key or key in self.audio_event_ids or not 0 <= stamp - event.get("stamp_ms", 0) < 1500:
                continue
            self.audio_event_ids.add(key)
            if len(self.audio_event_ids) > 1024:
                self.audio_event_ids = {item["id"] for item in scene.get("events", [])}
            kind = event.get("kind")
            if kind=="cough":
                self.pending_cough=event
                self.memory.enqueue(self._event("audio_event",key,"cough_observed","environment",event,event["stamp_ms"]))
                continue
            intent = {"startle": "startle_reflex", "orient": "orient_to_sound", "music_beat": "beat_pulse"}.get(kind)
            if not intent:
                continue
            action = dict(event, action_intent=intent, executed=False)
            self.state["audio_actions"] = ([action] + self.state["audio_actions"])[:50]
            self.memory.enqueue(self._event("audio_event", key, intent, "environment", action, event["stamp_ms"]))

    def _event(self, kind, trace_id, text, person_id, structured, stamp):
        return {"scope": self.scope, "source_system": "biomimetic-brain-test", "source_topic": "/brain/" + kind,
                "kind": kind, "trace_id": trace_id, "text": text, "entity": {"person_id": person_id},
                "structured": structured, "observed_at_ms": stamp}

    def _remember_heard(self, turn):
        text = str(turn["transcript"].get("text") or "")
        self.last_human_ms = max(self.last_human_ms, int(turn["transcript"].get("emitted_ms") or turn.get("stamp_ms") or 0))
        self.memory.enqueue(self._event("heard_utterance", turn["utterance_id"], text,
                                       turn.get("person_id", "unknown"), turn, turn["stamp_ms"]))
        self.state["heard"] = ([turn] + self.state["heard"])[:50]
        self._trace("heard", turn["utterance_id"], text=text, person_id=turn.get("person_id", "unknown"), attention=turn.get("attention"))

    # A lock lost for one cycle is attention passing through a reflex, not the
    # person leaving: ORIENTING and ACQUIRING happen every couple of seconds by
    # design, and each one moves target_id off the person for a frame.
    LOCK_LOST_MS = 600

    def _drop_a_greeting_nobody_is_there_for(self, state, current, stamp):
        """Abandon a self-started turn if the person it was for is gone.

        Only a self-started turn, and only before it has said anything. The
        rule used to apply to every attention-driven turn and all the way
        through playback, so a reply was cut mid-sentence whenever the camera
        blinked or a sound pulled attention aside for one frame — which is most
        seconds. Once it has started a sentence it finishes it; if the person
        really has left, the next turn simply will not happen.
        """
        self_initiated = bool(current.get("proactive") or current.get("speak_because") == "workspace_candidate")
        pending = any(self.state["s2s"].get(key) for key in ("thinking", "generating"))
        if not (self_initiated and pending) or current.get("playback_started_ms"):
            self.lock_lost_since = 0
            return
        if self._owner_focused(state, current.get("person_id"), stamp, allow_held=True):
            self.lock_lost_since = 0
            return
        self.lock_lost_since = self.lock_lost_since or stamp
        if stamp - self.lock_lost_since >= self.LOCK_LOST_MS:
            self.lock_lost_since = 0
            self._interrupt("owner_attention_lock_lost")

    def _owner_focused(self,state,key,stamp,allow_held=False):
        attention=state.get("attention") or {}
        target=attention.get("target_id")
        observed=int(attention.get("stamp_ms") or state.get("stamp_ms") or stamp)
        return bool(attention.get("person_id")==key and attention.get("target_kind","person")=="person"
                    and (not target or target=="person:"+str(key)) and (attention.get("listen") or (allow_held and attention.get("focus_status")=="held_briefly" and attention.get("focus_age_ms",9999)<=650))
                    and float(attention.get("confidence") or 0)>=.62 and 0<=stamp-observed<750)

    def _owner_presence(self, state, stamp):
        if self.pending_group_turn:return
        self.state["initiative"]={}
        people = state.get("person_hypotheses") or state.get("people") or []
        present = {str(p.get("resolved_person_id") or p.get("person_id")): p for p in people
                   if p.get("active", True) and p.get("role") == "owner" and float(p.get("identity_confidence") or p.get("best_identity_probability") or 0) >= .6}
        active = [p for p in people if p.get("active", True)]
        for key, previous in self.owners.items():
            if key not in present or not self._owner_focused(state,key,stamp,allow_held=True):
                previous["focus_since"] = None
            still_visible = any(str(p.get("resolved_person_id") or p.get("person_id")) == key or (previous.get("tracking_key") and p.get("temporary_id") == previous["tracking_key"]) for p in active)
            if still_visible:
                previous["last_seen"] = stamp
            elif stamp - previous["last_seen"] >= 5000:
                previous["greeted"] = False
                previous["since"] = 0
        for key, owner in present.items():
            previous = self.owners.setdefault(key, {"since": stamp, "last_seen": stamp, "greeted": False, "last_greeting": 0})
            previous["tracking_key"] = owner.get("temporary_id")
            if not previous["since"]:
                previous["since"] = stamp
            previous["last_seen"] = stamp
            attention = state.get("attention") or {}
            focused = self._owner_focused(state,key,stamp)
            self.state["initiative"][key]={"locked":focused,"already_interacted":bool(previous["greeted"]),"wait_remaining_ms":self.proactive_wait_ms}
            if not focused:
                held=self._owner_focused(state,key,stamp,allow_held=True)
                self.state["initiative"][key]["holding"]=held
                if not held:previous["focus_since"] = None
                continue
            if previous.get("focus_since") is None: previous["focus_since"] = stamp
            quiet_since = max(previous["focus_since"], self.last_human_ms)
            self.state["initiative"][key]["wait_remaining_ms"]=max(0,self.proactive_wait_ms-(stamp-quiet_since))
            if (stamp-quiet_since < self.proactive_wait_ms or previous["greeted"] or stamp - previous["since"] < 800
                    or (previous.get("last_greeting") and stamp-previous["last_greeting"] < 30000)
                    or self.state["s2s"]["thinking"] or self.state["s2s"]["speaking"] or self.state["s2s"].get("generating")):
                continue
            # Deliberately does not open a turn. Somebody familiar coming back
            # is a candidate like any other and attention raises it
            # (arrival_event), where it can be weighed against what else is
            # going on. This rule used to submit its own turn with a written-out
            # instruction as the user text — which the model then read aloud
            # ("主人刚回来，咱们聊点轻松的吧"). Tracking stays; the turn does not.
            previous["greeted"] = True
            previous["last_greeting"] = stamp

    def _perception_for_llm(self):
        scene = self.state.get("audio_scene") or {}
        acoustic = scene.get("acoustic") or {}
        music = scene.get("music") or {}
        now = now_ms()
        people = (self.state.get("perception_context") or {}).get("people") or []
        return {"audio": {key:acoustic.get(key) for key in ("stamp_ms","rms_dbfs","direction_deg","direction_valid")} if now-acoustic.get("stamp_ms",0)<1500 else {"available":False},
                "music": {key:music.get(key) for key in ("valid","music_state","genre","genre_valid","bpm","bpm_valid","top_events")},
                "recent_reflexes": [e for e in self.state.get("audio_actions",[]) if now-e.get("stamp_ms",0)<5000][:4],
                "people": [{key:p.get(key) for key in ("resolved_person_id","role","emotion_label","emotion_valid","gesture","gaze_score")} for p in people[:4]],
                "crossmodal_events": [e for e in self.state.get("crossmodal_events",[]) if now-e.get("stamp_ms",0)<5000][:4],
                "attention_focus": ((self.state.get("perception_context") or {}).get("attention_distribution") or {}).get("focus"),
                "physical_actions_executed":False}

    def _stream_reply(self, text, memories, owner, generation, cancel, details):
        details["path"] = "proactive" if owner else "slow"
        def refresh():
            with self.lock:
                if generation == self.generation:
                    self.state["current_turn"] = dict(details)
        def chunk_ready(chunk):
            if not details.get("first_chunk_ready_ms"):
                details["first_chunk_ready_ms"] = now_ms()
                details["status"] = "synthesizing"
            self._trace("phrase_ready", details["turn_id"], text=chunk)
            refresh()
        def playback(chunk, speaking):
            details["current_chunk"] = chunk
            self._turn_playback(generation, details, speaking)
        speaker = StreamingSpeech(self.tts, cancel, chunk_ready, playback, details["turn_id"])
        with self.lock:
            if generation==self.generation:self.active_stream=(generation,speaker)
        def delta(fragment):
            if cancel.is_set() or generation != self.generation:
                return
            if not details.get("first_token_ms"):
                details["first_token_ms"] = now_ms()
                self._trace("first_token", details["turn_id"])
            details["assistant_text"] += fragment
            refresh()
            speaker.feed(fragment)
        with self.lock:
            if generation == self.generation:
                self.state["s2s"]["generating"] = True
        try:
            try:
                details["llm_request_ms"]=now_ms()
                refresh()
                model_perception=dict(self._perception_for_llm(),current_speaker=details.get("speaker_context"))
                from .prompt_context import build_context
                _,details["model_context"]=build_context(text,memories,owner,model_perception)
                refresh()
                reply = self._answer_with_a_fallback(
                    lambda client: client.respond(text, memories, cancel=cancel, proactive=owner,
                                                  on_delta=delta, perception=model_perception),
                    details)
            except ReplyCancelled:
                if speaker.error:
                    raise speaker.error
                speaker.abort()
                return speaker.text, False
            if cancel.is_set():
                speaker.abort()
                return speaker.text, False
            if not speaker.text and reply:
                delta(reply)
            details["decision_ready_ms"] = now_ms()
            self._trace("reply_ready", details["turn_id"], text=reply, path=details["path"])
            refresh()
            return reply, speaker.finish()
        except Exception as exc:
            details.update(status="error", error=str(exc))
            speaker.abort()
            raise
        finally:
            with self.lock:
                if generation == self.generation:
                    self.state["s2s"]["generating"] = False

    def _registration_reply(self, turn):
        import re
        transcript=turn.get("transcript") or {}
        text=str(transcript.get("text") or "")
        intent=parse_registration(text) is not None or transcript.get("repeats_enrollment") or re.search(r"^(?:(?:小圆|小园|小元|小袁)[，, ]*)?(?:(?:请|帮我|我要|我想|再|重新)[，, ]*)*(?:注册|登记|记住我|认识一下|我是你的主人|我是主人)", text)
        if not intent:
            return None
        speaker=transcript.get("speaker") or {}
        role=speaker.get("speaker_role") if speaker.get("speaker_id") not in {None,"unknown","robot"} and float(speaker.get("similarity") or 0)>=.6 else None
        if role not in {"owner","known"}:
            people=(self.state.get("perception_context") or {}).get("people") or []
            recognized=next((p for p in people if p.get("active",True) and p.get("resolved_person_id")==turn.get("person_id") and float(p.get("identity_confidence") or p.get("best_identity_probability") or 0)>=.75),None)
            role=recognized.get("role") if recognized else None
        if role not in {"owner","known"}:
            return None
        reply="你已经是主人啦，不要来挑逗我了。" if role=="owner" else "我已经认识你啦，不用再注册了。"
        return BrainDecision("fast",True,reply,reason="already_registered",person_id=turn.get("person_id",""))

    def _voice_registration(self, turn, cancel):
        request=parse_registration(turn.get("transcript",{}).get("text",""))
        if request is None:
            return None
        existing=self._registration_reply(turn)
        if existing:
            return existing
        if not request["name"]:
            return BrainDecision("registration",True,"请面对摄像头告诉我：你好小圆，我叫你的名字，请记住我。",reason="registration_name_missing")
        if cancel.is_set():
            raise ReplyCancelled()
        self._trace("registration_started",turn["utterance_id"],name=request["name"],role=request["role"])
        try:
            result=post_json(self.attention_url+"/api/guided-registration",dict(request,utterance_id=turn["utterance_id"]),timeout=12)
            success=bool(result.get("success"))
            reply=result.get("message") or "这次没有采集清楚，请面对摄像头再说一次。"
            if success and not result.get("already_registered"):
                reply=f"记住啦，{result.get('person_id') or request['name']}，你是我的{'主人' if request['role']=='owner' else '朋友'}了。"
            self._trace("registration_finished",turn["utterance_id"],success=success,result=result)
            return BrainDecision("registration",True,reply,reason="registration_complete" if success else "registration_retry")
        except Exception as exc:
            self._trace("registration_finished",turn["utterance_id"],success=False,error=str(exc))
            return BrainDecision("registration",True,"这次注册没有完成。请确认摄像头和麦克风在采集，再面对我说一次。",reason="registration_error")

    def _submit(self, turn, owner=None, reflex_event=None):
        if self.state["s2s"]["thinking"] or self.state["s2s"]["speaking"] or self.state["s2s"].get("generating"):
            self._interrupt("startle_reflex" if reflex_event else
                            "same_person_kept_talking" if turn.get("supersedes") else "new_addressed_final",
                            fade_ms=5 if reflex_event else 20,
                            record_interruption=not turn.get("supersedes"))
        with self.lock:
            self.generation += 1
            generation = self.generation
            cancel = self.cancel = threading.Event()
            self.state["s2s"].update(thinking=True, error="")
            details = {"turn_id": turn["utterance_id"], "person_id": turn.get("person_id", "unknown"),
                       "user_text": "" if owner else turn["transcript"]["text"], "assistant_text": "",
                       "attention": turn.get("attention", {}), "attention_trigger":turn.get("attention_trigger"), "proactive": bool(owner), "reflex": reflex_event, "status": "thinking",
                       # The sentence this one replaced, when the same person
                       # kept talking before we said anything. It is already in
                       # the recent context the model reads; this is so the
                       # console can show which line is being answered.
                       "supersedes": turn.get("supersedes"), "superseded_text": turn.get("superseded_text"),
                       # Why it opened its mouth at all, in the console and the
                       # trace: a kept sentence, something they just said, an
                       # exchange left open, or a candidate with its own content.
                       "speak_because": turn.get("speak_because"),
                       "received_ms": now_ms(), "asr_ended_ms": turn["transcript"].get("ended_ms"),
                       "asr_emitted_ms": turn["transcript"].get("emitted_ms"), "decision_ready_ms": None,
                       "playback_started_ms": None, "playback_ended_ms": None, "error": ""}
            self.state["current_turn"] = dict(details)
        self._trace("attention_trigger_started" if turn.get("attention_trigger") else "reflex_started" if reflex_event else "proactive_started" if owner else "gate_accepted", turn["utterance_id"],
                    text=details["user_text"], person_id=details["person_id"], attention=details["attention"])
        def job():
            acquired = False
            try:
                while not cancel.is_set() and not self.stop_event.is_set():
                    if self.slots.acquire(timeout=.05):
                        acquired = True
                        break
                if not acquired or cancel.is_set():
                    return
                text = turn["transcript"]["text"]
                person = turn.get("person_id", "unknown")
                details["memory_started_ms"]=now_ms()
                registration = (BrainDecision("care",True,"慢慢来，别着急。",reason="cough_care") if turn.get("care_event")
                                else BrainDecision("registration",True,"我还没记住你，怎么称呼？",reason="ask_for_name") if turn.get("ask_name")
                                else (self._name_answer(turn) or self._voice_registration(turn, cancel)) if not reflex_event and not owner else None)
                # 召回要有判定依据：先按人，不知道是谁才全局找。
                # Searching everybody's history for every sentence is how the
                # robot answered one person with another person's afternoon.
                # The identity the perception layer settled on is what scopes
                # it; without one there is nothing to scope by and the room's
                # own history is all there is.
                memories, scope = [], "none"
                if not reflex_event and not registration:
                    if person in {"", "unknown"}:
                        memories = self.memory.search(text)[:12]
                        scope = "room_only_identity_unknown"
                    else:
                        personal, combined = [], {}
                        aliases=(turn["transcript"].get("speaker") or {}).get("aliases") or []
                        for identity in [person]+aliases[:4]:
                            personal.extend(self.memory.search(text, identity))
                        for item in personal:
                            combined.setdefault(record_key(item),item)
                        scope = "this_person"
                        if len(combined) < 3:
                            # Thin on their own history: the room's is better
                            # than nothing, and every line says whose it is.
                            for item in self.memory.search(text):
                                combined.setdefault(record_key(item),item)
                            scope = "this_person_then_room"
                        memories=list(combined.values())[:12]
                details["memory_scope"]=scope
                details["memory_finished_ms"]=now_ms()
                recent=[]
                if not owner and not reflex_event:
                    end=int(turn["transcript"].get("ended_ms") or turn.get("stamp_ms") or now_ms())
                    for heard in reversed(self.state.get("heard",[])):
                        if heard["utterance_id"]!=turn["utterance_id"] and 0<=end-int(heard.get("stamp_ms") or 0)<=120000:
                            recent.append({"utterance_id":heard["utterance_id"],"session_id":self.scope["session_id"],"person_id":heard.get("person_id"),"text":heard["transcript"].get("text"),"to_robot":(heard.get("attention") or {}).get("addressed_to_robot"),"stamp_ms":heard.get("stamp_ms"),
                                "started_ms":heard["transcript"].get("started_ms"),"ended_ms":heard["transcript"].get("ended_ms"),
                                "identity":{k:heard[k] for k in ("identity_source","identity_conflict","visual_person_id","identity_confidence") if k in heard},
                                "background":{"addressee":heard.get("addressee"),"group_context":heard.get("group_context"),"visible_participants":heard.get("visible_participants",[]),"speech_act":heard.get("speech_act"),"overlap_probability":heard.get("overlap_probability"),"visual_observations":(heard.get("evidence") or {}).get("vision",[])}})
                    # Our own turns belong in the conversation we hand the
                    # model. Without them it saw only his side and answered
                    # 「我觉得上迪士尼乐园挺好玩的。」 with 「上海迪士尼挺好玩的。」
                    # — a parrot, because as far as it could tell nothing had
                    # been said back yet. The prompt already tells it that a
                    # line from 小圆 is its own; there were none to see.
                    for event in list(self.state.get("events") or [])[:16]:
                        when=int(event.get("stamp_ms") or 0)
                        # Not `text`: that name holds the sentence being answered
                        # a few lines up, and shadowing it sent our own last
                        # reply to the model as the user's question.
                        spoken=str(event.get("text") or "").strip()
                        if spoken and 0<=end-when<=120000:
                            recent.append({"utterance_id":"reply:"+str(when),"session_id":self.scope["session_id"],
                                           "person_id":"小圆","text":spoken,"stamp_ms":when,"said_by_robot":True})
                    recent.sort(key=lambda item:int(item.get("stamp_ms") or 0))
                    session=dict(self.session_context or {})
                    if session.get("transcript"):
                        # The session's own transcript replaces the room-wide
                        # window: complete for this conversation, ordered, and
                        # numbered so the order cannot be lost in serialisation.
                        # Every member appears in `participants` even if they
                        # have been silent or are off camera.
                        # person_id is the participant the perception layer
                        # settled on, and the label is what to call them. A line
                        # whose speaker was inferred says so: 「unknown」 read as
                        # 天行 was the model guessing, not the system deciding.
                        recent=[{"utterance_id":row.get("utterance_id") or f"s{row['i']}",
                                 "session_id":session.get("session_id"),
                                 "person_id":row.get("speaker_label") or row.get("speaker"),
                                 "participant_id":row.get("speaker"),
                                 "identity_status":row.get("identity_status") or "confirmed",
                                 "text":row.get("text"),
                                 "to_robot":bool(row.get("to_robot")),"stamp_ms":row.get("stamp_ms"),
                                 "order":row.get("i")} for row in session["transcript"]]
                        for event in list(self.state.get("events") or [])[:16]:
                            when=int(event.get("stamp_ms") or 0)
                            spoken=str(event.get("text") or "").strip()
                            if spoken and 0<=end-when<=120000:
                                recent.append({"utterance_id":"reply:"+str(when),
                                               "session_id":session.get("session_id"),
                                               "person_id":"小圆","text":spoken,"stamp_ms":when,
                                               "said_by_robot":True})
                        recent.sort(key=lambda item:int(item.get("stamp_ms") or 0))
                        for order,row in enumerate(recent,start=1):
                            row["order"]=order
                details["recent_conversation"]=recent[-16:]
                details["session"]={k:v for k,v in (self.session_context or {}).items() if k!="transcript"}
                details["speaker_context"]={"held_utterance":turn.get("held_utterance"),
                                    "session":details["session"],
                                    # 谁在说话，以及这个「谁」确不确定——工具要用它，
                                    # 提示词也要用它，免得模型自己认人。
                                    "participant_id":turn.get("participant_id") or person,
                                    "participant_label":turn.get("participant_label") or "",
                                    "identity_status":turn.get("identity_status") or "",
                                    "identity_ladder":turn.get("identity_ladder") or {},
                                    "utterance_id":turn["utterance_id"],"stamp_ms":turn.get("stamp_ms"),"session_id":self.scope["session_id"],"dialogue_link":turn.get("dialogue_link"),"attention_trigger":turn.get("attention_trigger"),"group_context":turn.get("group_context"),"recent_conversation":recent[-16:],"person_id":person,"voice":turn["transcript"].get("speaker"),"attention":turn.get("attention"),"visual_evidence":(turn.get("evidence") or {}).get("vision",[])}
                memories = [m for m in memories if m.get("trace_id") != turn["utterance_id"]]
                if cancel.is_set():
                    return
                details["memories"] = [{"event_id": m.get("event_id"), "trace_id": m.get("trace_id"),
                                        "person_id": (m.get("entity") or {}).get("person_id"),
                                        "text": str(m.get("text") or "")[:200], "kind": m.get("kind")} for m in memories[:12]]
                streamed = False
                fast = BrainDecision("reflex", True, "哎呀！", "startle", "startle_exclamation", person) if reflex_event else None if owner else (registration or self._registration_reply(turn) or self.brain.fast.decide(text))
                if fast:
                    decision = BrainDecision(**{**asdict(fast), "person_id": person})
                elif self.llm and os.environ.get("BRAIN_STREAM_REPLY", "0") == "1":
                    # Off by default. Replies are one short line now, so playing
                    # them phrase by phrase buys nothing and is how a retried or
                    # superseded turn got half said before being replaced.
                    streamed = True
                    reply, completed = self._stream_reply(text, memories, owner, generation, cancel, details)
                    if not reply and cancel.is_set():
                        return
                    decision = BrainDecision("proactive" if owner else "slow", True, reply,
                                             reason="owner_return_llm" if owner else "llm", person_id=person)
                elif self.llm:
                    # Same context the streaming path builds; only the speaking
                    # is different — the whole line, once, after it is decided.
                    details["path"] = "proactive" if owner else "slow"
                    details["llm_request_ms"] = now_ms()
                    model_perception = dict(self._perception_for_llm(), current_speaker=details.get("speaker_context"))
                    from .prompt_context import build_context
                    _, details["model_context"] = build_context(text, memories, owner, model_perception)
                    with self.lock:
                        if generation == self.generation:
                            self.state["s2s"]["generating"] = True
                    def spoken_once(fragment):
                        if not details.get("first_token_ms"):
                            details["first_token_ms"] = now_ms()
                            self._trace("first_token", details["turn_id"])
                        details["assistant_text"] += fragment
                    reply = self._answer_with_a_fallback(
                        lambda client: client.respond(text, memories, cancel=cancel, proactive=owner,
                                                      on_delta=spoken_once, perception=model_perception),
                        details)
                    details["llm_passes"] = self._passes(self._last_client or self.llm)
                    with self.lock:
                        if generation == self.generation:
                            self.state["s2s"]["generating"] = False
                    decision = BrainDecision("proactive" if owner else "slow", True, reply,
                                             reason="owner_return_llm" if owner else "llm", person_id=person)
                else:
                    reply = "你回来啦，想聊聊吗？" if owner else self.brain.slow.respond(text, memories)
                    decision = BrainDecision("proactive" if owner else "slow", True, reply, reason="rule_fallback", person_id=person)
                if not streamed:
                    with self.lock:
                        if cancel.is_set() or generation != self.generation or self.stop_event.is_set():
                            return
                        details.update(path=decision.path, assistant_text=decision.text, decision_ready_ms=now_ms(),
                                       first_chunk_ready_ms=now_ms(), status="synthesizing" if self.tts.enabled else "muted")
                        self.state["current_turn"] = dict(details)
                    self._trace("reply_ready", details["turn_id"], text=decision.text, path=decision.path)
                    completed = self.tts.speak(decision.text, cancel=cancel, turn_id=details["turn_id"],
                                               on_playback=lambda speaking: self._turn_playback(generation, details, speaking))
                with self.lock:
                    if generation == self.generation:
                        self.state["decision"] = asdict(decision)
                details["status"] = "interrupted" if cancel.is_set() else "completed" if completed else "muted" if not self.tts.enabled else "stopped"
                with self.lock:
                    if generation == self.generation:
                        self.state["current_turn"] = dict(details)
                if generation==self.generation and not cancel.is_set():
                    self.expected_answer.offered(details["turn_id"],person,decision.text,details.get("playback_ended_ms") or now_ms(),bool(completed),
                        {"recent_conversation":details.get("recent_conversation",[]),"attention_trigger":turn.get("attention_trigger")})
                self._trace("turn_finished", details["turn_id"], status=details["status"])
                # Whatever prompted this turn, the room has just heard us. An
                # internally generated candidate must not follow it immediately.
                self.attention_trigger.note_reply(int(time.time()*1000))
                data = {**asdict(decision), "stamp_ms": now_ms(), "playback_completed": bool(completed),
                        "interrupted": cancel.is_set(), "user_text": "" if owner else text,
                        "trigger": owner if owner else turn, "memory_count": len(memories), "timing": dict(details)}
                self.state["events"] = ([data] + self.state["events"])[:50]
                self.memory.enqueue(self._event("dialogue_turn", turn["utterance_id"] + ":reply", decision.text,
                                               person, data, data["stamp_ms"]))
            except ReplyCancelled:
                pass
            except Exception as exc:
                if generation == self.generation and (not cancel.is_set() or details.get("status") == "error"):
                    self.state["s2s"]["error"] = str(exc)
                    details.update(status="error", error=str(exc))
                    self._trace("error", details["turn_id"], error=str(exc))
                    # 兜底：被问到了就别装作没听见。Both models failing is rare
                    # and it is still not a reason to stand there silently —
                    # from the person's side an unanswered question is the
                    # robot ignoring them, and they repeat themselves at it.
                    self._say_we_could_not_answer(details, turn, exc)
                    self.state["current_turn"] = dict(details)
            finally:
                if acquired:
                    self.slots.release()
                if generation == self.generation:
                    self.state["s2s"]["thinking"] = False
                with self.lock:
                    self.jobs.discard(threading.current_thread())
        thread = threading.Thread(target=job, daemon=True, name="brain-reply")
        with self.lock:
            self.jobs.add(thread)
        thread.start()
