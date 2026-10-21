"""Exclude strongly evidenced playback echoes from recall without rewriting history."""
import re


def is_playback_echo(item, records):
    if item.get('kind') != 'heard_utterance':
        return False
    data = item.get('structured') or {}
    transcript = data.get('transcript') or {}
    if data.get('source') == 'robot_echo' or transcript.get('source') == 'robot_echo':
        return True
    if float(transcript.get('self_echo_probability') or 0) >= .75:
        return True
    # Infer only a long exact fragment during recorded playback, never from text alone.
    candidate = re.sub(r'[^\w]', '', str(item.get('text') or '').lower())
    start, end = transcript.get('started_ms'), transcript.get('ended_ms')
    if len(candidate) < 8 or not start or not end or end <= start:
        return False
    if float((transcript.get('speaker') or {}).get('similarity') or 0) >= .75:
        return False
    for other in records:
        if other.get('kind') != 'dialogue_turn':
            continue
        reply = other.get('structured') or {}
        timing = reply.get('timing') or {}
        a, b = timing.get('playback_started_ms'), timing.get('playback_ended_ms')
        reference = re.sub(r'[^\w]', '', str(other.get('text') or '').lower())
        if a and b and candidate in reference and max(0,min(end,b+600)-max(start,a)) >= .8*(end-start):
            return True
    return False
