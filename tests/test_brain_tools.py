"""天气在轮次之外取，不然问一句要等四秒。

Measured on this machine: a cold get_weather is 3.86 s — geocoding then
forecast, two sequential internet requests — and a warm one is 0 ms. The field
saw 6.5 s between the question and the first word, and the model was not the
problem: the same reply path with no tool call measured 506 ms end to end.
"""
import os
import time
import unittest
import unittest.mock

from brain import tools


class PrefetchTests(unittest.TestCase):
    def setUp(self):
        self.forecasts = dict(tools._FORECASTS)
        self.places = dict(tools._PLACES)
        self.addCleanup(lambda: (tools._FORECASTS.clear(), tools._FORECASTS.update(self.forecasts),
                                 tools._PLACES.clear(), tools._PLACES.update(self.places)))

    def forecast_response(self):
        return {'current': {'temperature_2m': 30.8, 'relative_humidity_2m': 48, 'weather_code': 2,
                            'wind_speed_10m': 7.2, 'time': '2026-09-18T11:45'},
                'daily': {'temperature_2m_max': [31.3, 30.2], 'temperature_2m_min': [23.9, 25.2],
                          'weather_code': [2, 80]}}

    def test_a_warm_cache_is_not_fetched_again(self):
        tools._FORECASTS['上海'] = (time.time(), {'city': '上海', 'condition': '多云'})
        with unittest.mock.patch.object(tools, '_fetch', side_effect=AssertionError('should not fetch')):
            status = tools.prefetch('上海')
        self.assertTrue(status['warm'])
        self.assertLess(status['age_s'], 5)

    def test_a_stale_cache_is_refreshed_in_the_background(self):
        tools._FORECASTS['上海'] = (time.time() - tools.FORECAST_TTL_S, {'city': '上海'})
        tools._PLACES['上海'] = {'latitude': 31.2, 'longitude': 121.5, 'timezone': 'Asia/Shanghai', 'name': '上海'}
        with unittest.mock.patch.object(tools, '_fetch', return_value=self.forecast_response()):
            status = tools.prefetch('上海')
        self.assertTrue(status['warm'])
        self.assertEqual(tools._FORECASTS['上海'][1]['condition'], '多云')

    def test_and_the_question_then_costs_nothing(self):
        tools._PLACES['上海'] = {'latitude': 31.2, 'longitude': 121.5, 'timezone': 'Asia/Shanghai', 'name': '上海'}
        tools._FORECASTS.pop('上海', None)
        with unittest.mock.patch.object(tools, '_fetch', return_value=self.forecast_response()):
            tools.prefetch('上海')
        with unittest.mock.patch.object(tools, '_fetch', side_effect=AssertionError('should not fetch')):
            report = tools.run_tool('get_weather', {'city': '上海'})
        self.assertEqual(report['condition'], '多云')

    def test_no_location_is_reported_rather_than_guessed(self):
        with unittest.mock.patch.dict(os.environ, {'BRAIN_LOCATION': ''}):
            status = tools.prefetch('')
        self.assertFalse(status['warm'])
        self.assertIn('BRAIN_LOCATION', status['reason'])

    def test_a_failed_lookup_says_so_and_leaves_the_cache_alone(self):
        tools._FORECASTS.pop('上海', None)
        tools._PLACES['上海'] = {'latitude': 31.2, 'longitude': 121.5, 'timezone': 'Asia/Shanghai', 'name': '上海'}
        with unittest.mock.patch.object(tools, '_fetch', side_effect=OSError('network down')):
            status = tools.prefetch('上海')
        self.assertFalse(status['warm'])
        self.assertEqual(status['reason'], 'lookup_failed')
        self.assertNotIn('上海', tools._FORECASTS)
