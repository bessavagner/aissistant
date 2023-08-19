import unittest
from aissistant.core import Conversation

class TestBaseConversation(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.conversation = Conversation()

    def test_answer_returns_response_content(self):
        prompt = "Hello, how are you?"
        response = self.conversation.answer(prompt)
        self.assertIsInstance(response, str)
        self.assertNotEqual(response, "")

if __name__ == '__main__':
    unittest.main()
