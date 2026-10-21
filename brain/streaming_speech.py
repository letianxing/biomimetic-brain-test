"""Overlap LLM generation with cancellable phrase-by-phrase TTS playback."""
import queue
import re
import threading


class StreamingSpeech:
    def __init__(self, tts, cancel, on_chunk, on_playback, turn_id=""):
        self.tts, self.cancel = tts, cancel
        self.turn_id = turn_id
        self.on_chunk, self.on_playback = on_chunk, on_playback
        self.items = queue.Queue(maxsize=16)
        self.buffer = ""
        self.text = ""
        self.error = None
        self.completed = True
        self.yield_callback = None
        self.yield_lock = threading.Lock()
        self.thread = threading.Thread(target=self._play, daemon=True, name="streaming-tts")
        self.thread.start()

    def feed(self, delta):
        if self.cancel.is_set():
            return
        self.text += delta
        self.buffer += delta
        while self.buffer:
            match = re.search(r"[。！？!?；;\n]", self.buffer)
            cut = match.end() if match else 0
            if not cut:
                comma = re.search(r"[，,：:]", self.buffer)
                if comma and comma.end() >= 12:
                    cut = comma.end()
            if not cut and len(self.buffer) >= 40:
                cut = self.buffer.rfind(" ", 15, 40) or 40
                if cut < 0:
                    cut = 40
            if not cut:
                break
            chunk, self.buffer = self.buffer[:cut], self.buffer[cut:]
            self._queue(chunk)

    def _queue(self, chunk):
        chunk = chunk.strip()
        if not chunk or self.cancel.is_set():
            return
        self.on_chunk(chunk)
        while not self.cancel.is_set():
            try:
                self.items.put(chunk, timeout=.05)
                return
            except queue.Full:
                continue

    def request_yield(self,callback):
        with self.yield_lock:self.yield_callback=callback

    def _play(self):
        while not self.cancel.is_set():
            with self.yield_lock:callback=self.yield_callback
            if callback:
                callback();return
            try:
                chunk = self.items.get(timeout=.05)
            except queue.Empty:
                continue
            with self.yield_lock:callback=self.yield_callback
            if callback:
                callback();return
            if chunk is None:
                return
            try:
                played = self.tts.speak(chunk, cancel=self.cancel, turn_id=self.turn_id,
                                        on_playback=lambda speaking: self.on_playback(chunk, speaking))
                self.completed = self.completed and bool(played)
            except Exception as exc:
                self.error = exc
                self.cancel.set()
                return

    def finish(self):
        self._queue(self.buffer)
        self.buffer = ""
        while not self.cancel.is_set():
            try:
                self.items.put(None, timeout=.05)
                break
            except queue.Full:
                continue
        while self.thread.is_alive():
            self.thread.join(timeout=.05)
        if self.error:
            raise self.error
        return self.completed and not self.cancel.is_set()

    def abort(self):
        was_cancelled = self.cancel.is_set()
        self.cancel.set()
        try:
            if not was_cancelled:
                self.tts.interrupt()
        finally:
            self.thread.join(timeout=4)
