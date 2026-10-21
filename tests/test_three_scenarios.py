import unittest
from brain.core import Brain

class Scenarios(unittest.TestCase):
    def test_attention_gate_rejects_bystander_and_answers_addressed_user(self):
        b=Brain(); t={"is_final":True,"text":"你好"}
        self.assertFalse(b.dialogue({"listen":True,"addressed_to_robot":False,"confidence":.9},t,[]).eligible)
        self.assertEqual(b.dialogue({"listen":True,"addressed_to_robot":True,"confidence":.9},t,[]).text,"你好，我在听。")
    def test_memory_is_used_in_slow_reply(self):
        d=Brain().dialogue({"listen":True,"addressed_to_robot":True,"confidence":.8},{"is_final":True,"text":"你还记得吗"},[{"text":"主人喜欢喝咖啡"}])
        self.assertIn("主人喜欢喝咖啡",d.text)
    def test_owner_return_is_proactive_once(self):
        b=Brain(); people=[{"active":True,"role":"owner","resolved_person_id":"owner"}]
        self.assertIn("又回来了",b.owner_return(people,100000).text)
        self.assertIsNone(b.owner_return(people,100500))
if __name__=='__main__': unittest.main()

class FastPathTests(unittest.TestCase):
    def test_greeting_inside_question_does_not_swallow_question(self):
        from unittest.mock import Mock
        model = Mock()
        model.respond.return_value = "我们约下午三点。"
        decision = Brain(model).dialogue({"listen": True, "addressed_to_robot": True, "confidence": .9},
                                        {"is_final": True, "text": "你好，刚才约几点？"}, [])
        self.assertEqual(decision.text, "我们约下午三点。")
        model.respond.assert_called_once()
