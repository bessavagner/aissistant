"""
    Package "aissistant"

    This module provides package's api
"""

import json
import logging

from pathlib import Path

from genaikit.constants import ROLES

from .base import BaseConversation
from .base import BaseContext

logger = logging.getLogger('client')


class Conversation(BaseConversation):

    def __init__(self,
                 set_up: str = None,
                 gpt4=False,
                 temperature=0,
                 open_ai_key=None,
                 organization=None
    ):

        super().__init__(
            gpt4=gpt4,
            temperature=temperature,
            open_ai_key=open_ai_key,
            organization=organization
        )

        if set_up:

            self.message = {
                'role': ROLES[0],
                'content': set_up
            }


class QuestionContext:
    
    def __init__(self,
                 *args,
                 text: str = None,
                 set_up: str = None,
                 max_tokens=500,
                 **kwargs):
        self.chat = Conversation(*args, set_up=set_up, **kwargs)
        self.context_text = """Answer the question based on the context below,
        and if the question can't be answered based on the context, say
        \"I don't know\"\n\nContext: {}\n\n---\n\nQuestion: {}\nAnswer:"""
        self.context = BaseContext(
            text,
            model=self.chat.model,
            max_tokens=max_tokens
        )
        self.history = []

    def answer(self, question: str, max_length=1800):
        context = self.context.create(question, max_length=max_length)
        prompt = self.context_text.format(context, question)
        answer = self.chat.answer(prompt)
        self.history.append({
            'question': question,
            'answer': answer
        })
        return answer

    def save_history(self, path: str | Path):
        json.dump({'history': self.history}, path)
