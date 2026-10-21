from contextlib import nullcontext
from pathlib import Path
import os
import subprocess
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import mac_stack
import mac_stack_config as config
import mac_stack_support as support


class LauncherTests(unittest.TestCase):
    def test_stale_environment_cannot_change_selected_model_or_ports(self):
        with patch.dict(os.environ, {'VOICE_ASR_COMMAND': 'stale', 'VOICE_ASR_SHERPA_MODEL': '/old', 'ROSBRIDGE_PORT': '19090', 'BRAIN_LLM_BASE_URL': 'http://old'}):
            env = config.environment('same-session', 'vosk')
        self.assertEqual(env['VOICE_ASR_SHERPA_MODEL'], '')
        self.assertEqual(env['VOICE_ASR_COMMAND'], '')
        self.assertEqual(env['ROSBRIDGE_PORT'], '9090')
        self.assertEqual(env['BRAIN_LLM_BASE_URL'], 'http://old')
        self.assertEqual(env['SESSION_ID'], 'same-session')

    def test_asr_alternatives_include_their_real_model_files(self):
        required = config.required_files(['voice'], 'whisper')
        self.assertIn('Whisper base', required)
        self.assertIn('Vosk Chinese', required)
        self.assertNotIn('Sherpa Zipformer Chinese', required)
        self.assertIn('AST AudioSet music', required)
        self.assertIn('Wav2Vec2 ten genres', required)

    def test_attention_and_brain_share_session_without_double_s2s(self):
        specs = config.specs('test-session')
        self.assertNotIn('--s2s', specs['attention'].command)
        self.assertEqual(specs['attention'].command[-1], specs['brain'].env['SESSION_ID'])

    def test_full_launcher_rejects_existing_component_instance(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(support, 'RUN', Path(directory)), patch.object(support, 'fingerprint', return_value='date python ' + support.SCRIPT + ' qwen'):
            with support.instance('qwen'):
                with self.assertRaises(RuntimeError):
                    with support.instance('all'):
                        self.fail('duplicate launcher accepted')
            self.assertFalse((Path(directory) / 'qwen.json').exists())

    def test_unrelated_process_is_not_owned(self):
        with patch.object(support, 'fingerprint', return_value='python unrelated.py'):
            self.assertFalse(support.owned({'pid': 123, 'fingerprint': 'python unrelated.py'}))

    def test_cleanup_terminates_owned_process_group(self):
        child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)'], start_new_session=True)
        try:
            support.stop_processes([('test', child)])
            self.assertIsNotNone(child.poll())
        finally:
            if child.poll() is None:
                child.kill()
                child.wait()

    def test_failed_readiness_still_cleans_started_children(self):
        args = SimpleNamespace(component='voice', session='test', asr='sherpa', tse='baseline', camera=0,
                               dry_run=False, no_capture=True, no_browser=True)
        child = Mock()
        with tempfile.TemporaryDirectory() as directory, patch.object(mac_stack, 'RUN', Path(directory)), \
             patch.object(mac_stack, 'preflight'), patch.object(mac_stack, 'instance', return_value=nullcontext()), \
             patch.object(mac_stack, 'port_open', return_value=False), patch.object(mac_stack.subprocess, 'Popen', return_value=child), \
             patch.object(mac_stack, 'wait_ready', side_effect=RuntimeError('failed')), patch.object(mac_stack, 'stop_processes') as cleanup:
            with self.assertRaises(RuntimeError):
                mac_stack.run(args)
            cleanup.assert_called_once_with([('voice', child)])

    def test_capture_waits_for_models_and_uses_selected_device(self):
        args = SimpleNamespace(profile='mac_builtin', device='MacBook Pro麦克风', tse='baseline')
        with patch.object(mac_stack, 'request', side_effect=[{'running': True}, {'asr_model_ready': True, 'speaker_model_ready': True, 'audio_scene': {'classifier_ready': True}}]) as request:
            mac_stack.start_capture(args)
            self.assertIn('profile=mac_builtin', request.call_args_list[0].args[0])
            self.assertTrue(request.call_args_list[0].kwargs['post'])
