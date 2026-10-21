"""Record that the robot asked someone something and is still owed an answer.

This is a memory fact, not a gate. Brain produces it because only Brain knows
what it said; the attention system consumes it through the session snapshot and
decides there whether a later utterance answers it. Nothing here inspects an
incoming turn.
"""
import re

QUESTION = re.compile(r'[?？]|(?:吗|么|呢)[。！!\s]*$')


class ExpectedAnswer:
    def __init__(self, window_ms=10000, unfinished_limit=8):
        self.pending = None
        self.window_ms = window_ms
        # Questions that ran out without an answer. Attention may raise one of
        # these again later; Brain only records that it happened.
        self.unfinished = []
        self.unfinished_limit = unfinished_limit

    def offered(self, turn_id, person, text, ended_ms, completed, context=None):
        """Called once the robot's question actually finished playing."""
        if not completed or person in {None, '', 'unknown', 'robot'}:
            return
        if not QUESTION.search(text):
            self.pending = None
            return
        self.pending = {'turn_id': turn_id, 'person_id': person, 'question': text, 'ended_ms': ended_ms,
                        'expires_ms': ended_ms + self.window_ms, 'context': context or {}}

    def resolved(self, person_id):
        """Drop the fact once that person's answer has been taken up."""
        if self.pending and self.pending['person_id'] == person_id:
            self.pending = None
        self.unfinished = [item for item in self.unfinished if item['person_id'] != person_id]

    def _expire(self, pending):
        """An unanswered question is unfinished business, not a failure to log."""
        self.unfinished = [item for item in self.unfinished if item['id'] != pending['turn_id']]
        self.unfinished.append({'id': pending['turn_id'], 'kind': 'unanswered_question',
                                'person_id': pending['person_id'], 'question': pending['question'],
                                'turn_id': pending['turn_id'], 'stamp_ms': pending['ended_ms'],
                                'summary': pending['question']})
        del self.unfinished[:-self.unfinished_limit]

    def pending_unfinished(self, stamp):
        """Expire anything still open before reporting, so the list is current."""
        self.summary(stamp)
        return list(self.unfinished)

    def summary(self, stamp):
        """What the attention system is told; expired questions disappear."""
        pending = self.pending
        if not pending:
            return None
        if stamp > pending['expires_ms']:
            self._expire(pending)
            self.pending = None
            return None
        return {key: pending[key] for key in ('person_id', 'expires_ms', 'turn_id', 'question')}
