"""Take a turn when the attention system reports somebody is engaging us.

Attention already fused gaze, working memory and what was said into one
engagement belief, so no interaction case is recognised here. Brain only decides
whether it is free to take the floor.

Re-arming is the whole difficulty. An invitation means "there is recent context,
they are looking at me, and I have not answered it yet". Once we have answered,
the same context must not invite again — and it will look brand new to any key
derived from attention's own state, because the robot speaking pushes attention
out of the invited phase and straight back into it when the reply ends. So
re-arming is tied to something neither attention nor our own voice can churn: a
real human utterance has to have been transcribed since we last took a turn.
Raw voice activity is not good enough — it goes active on the robot's own
playback, so an echo would re-arm the very loop this is meant to stop.
"""

MIN_INTERVAL_MS = 8000   # backstop against churn even when speech is continuous
ARRIVAL_QUIET_MS = 5000  # somebody who has not spoken yet is not interrupted into
# What somebody just said outranks anything the robot thought of by itself. In
# the field both fired at once: it answered "喂" and, in the same breath, asked
# about something the person had said a minute earlier. Two replies to two
# different things is not a conversation. The order is fixed:
#   1 live speech addressed to us          (handled by Brain's own turn path)
#   2 continuation of the current exchange (attention's INVITED / expected answer)
#   3 audio-visual invitation              (sustained gaze)
#   4 anything internally generated        (unfinished questions, affect, presence)
# Levels 2-4 all pass through here; a level-4 candidate stands down whenever
# there is live speech in flight or the exchange is still warm.
INTERNAL_COOLDOWN_MS = 8000   # an internal candidate keeps quiet this long after we last spoke
INVITED_COOLDOWN_MS = 5000    # and so does an invitation: one reply per thing said, not two
INTERNAL_KINDS = ('memory_event', 'affect_event', 'presence_event', 'arrival_event')
# Somebody spoke in the room this recently and the opening has to join what is
# going on, not greet. Whether they were talking *to us* is beside the point: a
# person who turns from a conversation to the robot is asking about the
# conversation, not announcing their arrival.
ROOM_ACTIVE_MS = 40000


class AttentionTrigger:
    def __init__(self, states=('INVITED',), min_interval_ms=MIN_INTERVAL_MS,
                 event_kinds=('memory_event', 'affect_event', 'presence_event', 'arrival_event'), arrival_quiet_ms=ARRIVAL_QUIET_MS):
        self.states = set(states)
        self.event_kinds = set(event_kinds)
        self.arrival_quiet_ms = arrival_quiet_ms
        self.min_interval_ms = min_interval_ms
        self.last_fired_ms = 0
        self.last_spoke_ms = 0
        self.last_person = None
        self.last_reason = "no_cycle_yet"

    def status(self):
        return {"last_fired_ms": self.last_fired_ms or None, "last_person": self.last_person,
                "reason": self.last_reason, "min_interval_ms": self.min_interval_ms}

    def note_reply(self, stamp):
        """Brain tells us when it finished saying something, whatever prompted it."""
        self.last_spoke_ms = max(int(self.last_spoke_ms or 0), int(stamp or 0))

    def update(self, state, stamp, busy=False, participants=(), last_heard_ms=0, turn_in_flight=False):
        attention = state.get('attention') or {}
        engagement = attention.get('engagement') or {}
        name = str(engagement.get('state') or '')
        person = engagement.get('person_id')
        fresh = 0 <= stamp - int(attention.get('stamp_ms') or state.get('stamp_ms') or stamp) < 750

        if busy:
            self.last_reason = "busy"
            return None
        if turn_in_flight:
            # Somebody is being answered right now. Whatever we thought of can wait.
            self.last_reason = "live_turn_outranks_internal_candidate"
            return None
        if not fresh:
            self.last_reason = "attention_stale"
            return None
        from_workspace = False
        events = [item for item in (state.get('workspace_events') or [])
                  if item.get('kind') in self.event_kinds and item.get('person_id')]
        if name not in self.states or not person:
            # Attention may also have raised something nobody is doing right now:
            # unfinished business, somebody present and silent who looks low, or
            # simply somebody who has been sitting here a long time in silence.
            # Those are candidates, not permission; the same guards apply.
            if not events:
                self.last_reason = "not_invited"
                return None
            event = events[0]
            if event.get('kind') in INTERNAL_KINDS and self.last_spoke_ms and \
                    stamp - self.last_spoke_ms < INTERNAL_COOLDOWN_MS:
                # Level 4 against a conversation that is still warm. Each
                # producer owns its own silence rule; this one only stops the
                # robot from answering somebody and then, in the same breath,
                # raising something it thought of by itself.
                self.last_reason = "just_spoke_internal_candidate_waits"
                return None
            from_workspace = True
            name, person = event['kind'], event['person_id']
            engagement = {'state': name, 'person_id': person,
                          'confidence': float(event.get('share') or 0),
                          'reasons': list(event.get('reasons') or [])}
            attention = dict(attention, workspace_event=event)
        if self.last_fired_ms:
            if stamp - self.last_fired_ms < self.min_interval_ms:
                self.last_reason = "min_interval"
                return None
            # Nobody has said anything since we last spoke, so the context we
            # already answered is the only context there is.
            if int(last_heard_ms or 0) <= self.last_fired_ms:
                self.last_reason = "nothing_new_since_our_last_turn"
                return None

        if self.last_spoke_ms and stamp - self.last_spoke_ms < INVITED_COOLDOWN_MS:
            # We have only just finished a sentence. In the field the reply to
            # "我感觉今天很热" was followed six seconds later by a second opening
            # about the same thing, because an invitation is re-armed the moment
            # the robot stops talking. One reply per thing said.
            self.last_reason = "just_spoke_nothing_new_to_add"
            return None
        # Opening on somebody who has not spoken yet means starting a
        # conversation rather than joining one, so the room has to have settled
        # first. This used to live in Brain's owner rule; it belongs with every
        # other guard on taking the floor.
        # Workspace candidates carry their own silence requirements, so this
        # applies only to opening on somebody attention says is inviting us.
        if not from_workspace and person not in set(participants) and last_heard_ms:
            if stamp - int(last_heard_ms) < self.arrival_quiet_ms:
                self.last_reason = "arrival_needs_a_quiet_room"
                return None

        self.last_fired_ms = stamp
        self.last_spoke_ms = stamp
        self.last_person = person
        self.last_reason = "fired"
        return {'kind': 'attention_engagement', 'engagement_state': name, 'person_id': person, 'stamp_ms': stamp,
                'summary': (attention.get('workspace_event') or {}).get('summary'),
                'detail': (attention.get('workspace_event') or {}).get('detail'),
                'confidence': float(engagement.get('confidence') or 0),
                'reasons': list(engagement.get('reasons') or []),
                'participants': sorted(participants),
                'spoke_this_session': person in set(participants),
                'room_recently_active': bool(last_heard_ms and stamp - int(last_heard_ms) < ROOM_ACTIVE_MS),
                'evidence': {'attention': attention},
                'interpretation': 'attention_reports_engagement_not_a_spoken_question'}
