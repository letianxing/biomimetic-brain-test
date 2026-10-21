import unittest,os,sys
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from llm_endpoint import endpoint,remote,health_url
from mac_stack_config import environment
class EndpointTests(unittest.TestCase):
    def test_default_remote_and_local_override(self):
        with patch.dict(os.environ,{},clear=True):
            self.assertEqual(endpoint(),'http://172.16.60.162:18050/v1/chat/completions')
            self.assertTrue(remote());self.assertEqual(health_url(),'http://172.16.60.162:18050/health')
            self.assertEqual(environment('test')['BRAIN_LLM_BASE_URL'],endpoint())
        with patch.dict(os.environ,{'BRAIN_LLM_BASE_URL':'http://127.0.0.1:18050/v1/chat/completions'}):self.assertFalse(remote())
