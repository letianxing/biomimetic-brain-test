"""Facts the robot cannot know from perception or memory, looked up instead of guessed.

Without these the model had two options for "今天天气怎么样": invent a number, or
ask the user what they meant. It kept choosing the second, which is why every
reply sounded like a question. A lookup that fails says so and the robot says it
could not check — that is still better than a plausible invented temperature.
"""
from __future__ import annotations
import json
import os
import time
import urllib.parse
import urllib.request

TIMEOUT_S = float(os.environ.get('BRAIN_TOOL_TIMEOUT', '6'))
GEOCODE_URL = 'https://geocoding-api.open-meteo.com/v1/search'
FORECAST_URL = 'https://api.open-meteo.com/v1/forecast'

# WMO weather interpretation codes, in the words a person would use.
CONDITIONS = {0: '晴', 1: '大致晴朗', 2: '多云', 3: '阴', 45: '有雾', 48: '有雾凇',
              51: '毛毛雨', 53: '小雨', 55: '中雨', 56: '冻雨', 57: '冻雨',
              61: '小雨', 63: '中雨', 65: '大雨', 66: '冻雨', 67: '冻雨',
              71: '小雪', 73: '中雪', 75: '大雪', 77: '雪粒',
              80: '阵雨', 81: '阵雨', 82: '强阵雨', 85: '阵雪', 86: '强阵雪',
              95: '雷阵雨', 96: '雷阵雨伴冰雹', 99: '雷阵雨伴冰雹'}

TOOLS = [
    {'type': 'function', 'function': {
        'name': 'get_weather',
        'description': '查询天气实况与今天的最高最低气温。用户问天气、冷不冷、要不要带伞时调用；不要凭印象回答。',
        'parameters': {'type': 'object', 'properties': {
            'city': {'type': 'string', 'description': '城市名。用户没说城市就不要填，默认用机器人所在城市。'}},
            'required': []}}},
    {'type': 'function', 'function': {
        'name': 'get_datetime',
        'description': '查询当前日期、时间和星期。用户问今天几号、星期几、现在几点时调用。',
        'parameters': {'type': 'object', 'properties': {}, 'required': []}}},
    {'type': 'function', 'function': {
        'name': 'remember_person',
        'description': ('把当前说话人的名字、人脸和声纹真正记进档案。只有当他自己说出自己叫什么'
                        '（「我叫X」「我的名字叫X」「你可以叫我X」「记住我，我是X」）时才调用，'
                        '而且只填他说的那个名字。别人提到的第三个人不算，你自己猜的不算。'
                        '不调用这个工具，你就没有真的认识他——不要只在嘴上说「记住了」。'),
        'parameters': {'type': 'object', 'properties': {
            'name': {'type': 'string', 'description': '他自己说出来的名字，原样填写。'}},
            'required': ['name']}}},
]


# A city does not move and the weather does not change between two sentences.
# The lookup cost the person three seconds of silence, most of it geocoding the
# same city again.
_PLACES = {}
_FORECASTS = {}
FORECAST_TTL_S = float(os.environ.get('BRAIN_WEATHER_TTL', '600'))


def prefetch(city=None):
    """Warm the forecast before anybody asks, and say whether it worked.

    Measured on this machine: a cold get_weather takes 3.86 seconds — two
    sequential requests to open-meteo, geocoding then forecast — and a warm one
    takes 0 ms. That cold call sat inside a turn, which is where the field saw
    6.5 seconds between the question and the first word. Weather does not
    change in the time it takes to ask about it, so it is fetched on a
    schedule instead and the question is answered from the cache.
    """
    city = (city or os.environ.get('BRAIN_LOCATION', '')).strip()
    if not city:
        return {'warm': False, 'reason': 'no BRAIN_LOCATION configured'}
    cached = _FORECASTS.get(city)
    age = time.time() - cached[0] if cached else None
    if cached and age < FORECAST_TTL_S / 2:
        return {'warm': True, 'age_s': round(age, 1), 'city': city}
    report = get_weather(city)
    return {'warm': not report.get('error'), 'city': city,
            'reason': report.get('error') or '', 'observed_at': report.get('observed_at')}


def _fetch(url, params):
    query = urllib.parse.urlencode(params)
    with urllib.request.urlopen(f'{url}?{query}', timeout=TIMEOUT_S) as response:
        return json.loads(response.read().decode('utf-8'))


def get_datetime(**_ignored):
    now = time.localtime()
    weekday = '一二三四五六日'[now.tm_wday]
    hour, minute = now.tm_hour, now.tm_min
    part = '凌晨' if hour < 6 else '早上' if hour < 11 else '中午' if hour < 13 else '下午' if hour < 18 else '晚上'
    twelve = hour % 12 or 12
    half = '整' if minute == 0 else '半' if minute == 30 else f'{minute}分'
    return {'date': time.strftime('%Y-%m-%d', now), 'time': time.strftime('%H:%M', now),
            'weekday': f'星期{weekday}',
            # How a person would say it, so the robot does not read out 14:29.
            'spoken': f'{part}{twelve}点{half}'}


def get_weather(city=None, **_ignored):
    city = (city or os.environ.get('BRAIN_LOCATION', '')).strip()
    if not city:
        # Not a failure to hide: nobody told the robot where it is, and guessing
        # a city would produce a confident wrong answer.
        return {'error': 'unknown_location', 'detail': '没有配置机器人所在城市（BRAIN_LOCATION）'}
    cached = _FORECASTS.get(city)
    if cached and time.time() - cached[0] < FORECAST_TTL_S:
        return cached[1]
    try:
        place = _PLACES.get(city)
        if place is None:
            found = (_fetch(GEOCODE_URL, {'name': city, 'count': 1, 'language': 'zh'}) or {}).get('results') or []
            if not found:
                return {'error': 'city_not_found', 'city': city}
            place = _PLACES[city] = found[0]
        data = _fetch(FORECAST_URL, {
            'latitude': place['latitude'], 'longitude': place['longitude'],
            'current': 'temperature_2m,relative_humidity_2m,weather_code,wind_speed_10m',
            'daily': 'temperature_2m_max,temperature_2m_min,weather_code',
            'timezone': place.get('timezone') or 'auto', 'forecast_days': 2})
    except Exception as exc:
        return {'error': 'lookup_failed', 'detail': str(exc)[:120]}
    current, daily = data.get('current') or {}, data.get('daily') or {}
    def day(key, index):
        values = daily.get(key) or []
        return values[index] if len(values) > index else None
    def first(key):
        return day(key, 0)
    report = {'city': place.get('name') or city,
              'condition': CONDITIONS.get(int(current.get('weather_code', -1)), '未知'),
              'temp_c': current.get('temperature_2m'), 'max_c': first('temperature_2m_max'),
              'min_c': first('temperature_2m_min'), 'humidity_percent': current.get('relative_humidity_2m'),
              'wind_kmh': current.get('wind_speed_10m'), 'observed_at': current.get('time'),
            # "明天呢" is a weather question and the tool has to be able to answer
            # it; asking for one day meant the model could only say it did not know.
            'tomorrow': {'condition': CONDITIONS.get(int(day('weather_code', 1) or -1), '未知'),
                         'max_c': day('temperature_2m_max', 1), 'min_c': day('temperature_2m_min', 1)}}
    _FORECASTS[city] = (time.time(), report)
    return report


ATTENTION_URL = os.environ.get('BRAIN_ATTENTION_URL', 'http://127.0.0.1:8092').rstrip('/')


def remember_person(name, context=None):
    """真的把人记住：名字 + 这张脸 + 这个声音，写进两个服务各自的档案。

    Saying 「记住啦」 is not remembering. In the field the robot answered
    「你是天行呀」 to somebody whose identity store was empty — it had read the
    name off the transcript two lines earlier and was role-playing recognition.
    This is the difference: perception binds the name to the face template and
    the voiceprint of the person who is speaking right now, or it reports that
    it could not and the robot says so.

    Which person that is comes from perception's own attribution, never from
    the model: the model supplies the name it just heard, nothing else.
    """
    context = context or {}
    person = str(context.get('participant_id') or context.get('person_id') or '')
    if not person or person in {'unknown', 'robot'}:
        return {'success': False, 'reason': 'no_speaker', 'message': '现在还分不清是谁在说话'}
    payload = json.dumps({'name': str(name or ''), 'participant_id': person,
                          'utterance_id': str(context.get('utterance_id') or '')}).encode()
    request = urllib.request.Request(ATTENTION_URL + '/api/enroll-participant', data=payload,
                                     headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(request, timeout=TIMEOUT_S) as response:
        return json.loads(response.read().decode() or '{}')


HANDLERS = {'get_weather': get_weather, 'get_datetime': get_datetime,
            'remember_person': remember_person}
# Tools that need to know who is speaking, as opposed to ones that only need
# their arguments. The context is perception's, not the model's.
NEEDS_CONTEXT = {'remember_person'}


def run_tool(name, arguments, context=None):
    """Never raises: a broken tool has to come back as something the robot can say."""
    handler = HANDLERS.get(name)
    if handler is None:
        return {'error': 'unknown_tool', 'name': name}
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments or '{}')
        except ValueError:
            arguments = {}
    if not isinstance(arguments, dict):
        arguments = {}
    if name in NEEDS_CONTEXT:
        arguments = dict(arguments, context=context or {})
    try:
        return handler(**arguments)
    except Exception as exc:
        return {'error': 'tool_failed', 'name': name, 'detail': str(exc)[:120]}
