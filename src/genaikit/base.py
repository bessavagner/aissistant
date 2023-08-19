"""
    Package "aissistant"

    This module provides package's base logic, factories and abstractions
"""
import os
import logging
from pathlib import Path

import openai
import numpy as np
import pandas as pd

from openai import Embedding
from openai.embeddings_utils import distances_from_embeddings
from dotenv import load_dotenv

from genaikit.constants import ROLES
from genaikit.constants import MODELS
from genaikit.constants import MODELS_EMBEDDING
from genaikit.constants import MAX_TOKENS
from genaikit.constants import DEBUG

from genaikit.utils import number_of_tokens
from genaikit.utils import text_to_embeddings


logger = logging.getLogger('client')
debugger = logging.getLogger('standard')


class BaseConversation:

    def __init__(
            self,
            gpt4=False,
            temperature=0,
            open_ai_key=None,
            organization=None
    ):
        """
        Base class for managing conversations using OpenAI Chat API.
        
        Parameters:
            gpt4 (bool, optional): Specifies whether to use GPT-4 model.
                Default is False.
            temperature (float, optional): The temperature parameter for
                generating responses. Default is 0.
            open_ai_key (str, optional): The API key for accessing OpenAI
                services. If not provided, it will be retrieved from the
                environment variable OPENAI_API_KEY.
            organization (str, optional): The organization ID for OpenAI API.
                If not provided, it will be retrieved from the environment
                variable OPENAI_API_ORGANIZATION.
        
        Attributes:
            model (str): The model to be used for generating responses.
            temperature (float): The temperature parameter for generating
                responses.
            messages (list): List of conversation messages.
            max_tokens (int): Maximum number of tokens allowed for the
                conversation.
            last_response (object): The last response received from the
                ChatCompletion API.
        """
        if open_ai_key is None:

            load_dotenv()
            open_ai_key = os.getenv("OPENAI_API_KEY")

            if open_ai_key is None:
                raise RuntimeError("`open_ai_key` not set")
            if organization is None:
                organization = os.getenv("OPENAI_API_ORGANIZATION")

        openai.api_key = open_ai_key
        openai.organization = organization

        self.model = MODELS[1]  # gpt-3.5-turbo-0613
        self.temperature = temperature

        if gpt4:
            # self.model = MODELS[3]  # gpt-4-0613
            logger.warning(
                'Using gpt-4 is not supported yet. Using gpt-3.5-turbo-0613'
            )

        self.messages = []
        self.messages_backup = []
        self.max_tokens = dict(MAX_TOKENS)[self.model]
        self.last_response = None

    def answer(self, prompt):
        """
        Generate a response to the given prompt.
        
        Parameters:
            prompt (str): The prompt message for generating a response.
        
        Returns:
            str: The generated response content.
        """
        self._update(ROLES[1], prompt)

        self.last_response = openai.ChatCompletion.create(
            model=self.model,
            messages=self.messages,
            temperature=self.temperature
        )

        response_content = self.last_response.choices[0].message.content
        if DEBUG:
            debugger.info('Response: %s', response_content)
        self._update(ROLES[2], response_content)

        return response_content

    def current_number_of_tokens(self,):
        return number_of_tokens(self.messages, self.model)

    def _update(self, role: str, content: str):

        """
        Update the conversation with a new message.
        
        Parameters:
            role (str): The role of the message (e.g.,
                'system', 'user', 'assistant').
            content (str): The content of the message.
        """
        if role not in ROLES:
            raise KeyError(f"`role` must be one of: {ROLES}")

        if not isinstance(content, str):
            raise TypeError('`content` must be a string')

        message = {
            'role': role,
            'content': content
        }
        self.messages.append(message)
        self.messages_backup.append(message)
        self._reduce_number_of_tokens_if_needed()

    def _reduce_number_of_tokens_if_needed(self,):
        
        n_tokens = number_of_tokens(self.messages, self.model)
        
        if n_tokens > self.max_tokens:
            
            reduced = False
            
            messages = self.messages
            if len(messages) > 1:
                messages = messages[:-1]
            for idx, message in enumerate(messages):
                content = (
                    "Make a summary of the following text"
                    f": {message['content']}"
                )
                
                if DEBUG:
                    debugger.info(
                        (
                            'Maximum number of tokens exceeded. '
                            'Summarizing message: %s'
                        ),
                        message['content']
                    )
                
                message_ = {
                    'role': ROLES[1],
                    'content': content
                }
                response = openai.ChatCompletion.create(
                            model=self.model,
                            messages=[message_],
                            temperature=self.temperature
                        )
                content = response.choices[0].message.content
                self.messages[idx]['content'] = content
                
                if DEBUG:
                    debugger.info('Message summarized: %s', message['content'])
            
                n_tokens = number_of_tokens(self.messages, self.model)
                if n_tokens < self.max_tokens:
                    reduced = True
                    break
            
            if not reduced:
                if len(self.messages) == 1:
                    raise NotImplementedError(
                        'Conversation exceeds maximum number of tokens '
                        'and cannot be reduced'
                    )
                if n_tokens > self.max_tokens:
                    if DEBUG:
                        debugger.warning(
                            (
                                'Maximum number of tokens exceeded: '
                                '%s/%s. Removing head message: %s'

                            ),
                            n_tokens,
                            self.max_tokens,
                            self.messages[0]['content']
                        )
                    for idx, message in self.messages:
                        if message['role'] == ROLES[1]:
                            self.messages.pop(idx)  # TODO improve history retention
                            break

class BaseContext:
    def __init__(self,
                 text: str = None,
                 model: str = MODELS[1],
                 max_tokens: int = 500):
        
        self.text = text
        self.embeddings = None
        self.max_tokens = max_tokens
        
        if text is not None:
            if not isinstance(text, str):
                raise TypeError('`text` must a string')
            self.embeddings = text_to_embeddings(  # TODO edit embeddings
                text,
                model=model,
                max_tokens=max_tokens
            )
    
    def generate_embeddings(
            self,
            source: str | pd.DataFrame | Path,
            model: str = MODELS[1],
            max_tokens: int = None
    ) -> pd.DataFrame:
        if max_tokens is None:
            max_tokens = self.max_tokens
        if isinstance(source, pd.DataFrame):
            self.embeddings = source

        elif isinstance(source, Path):
            self.embeddings = pd.read_parquet(source, engine='pyarrow')
        else:
            if not isinstance(source, str):
                raise TypeError(
                    '`source` must either be a DataFrame, '
                    'Path object or a string'
                )
            self.embeddings = text_to_embeddings(  # TODO edit embeddings
                source,
                model=model,
                max_tokens=max_tokens
            )
        return self.embeddings

    def save_embeddings(
            self,
            path: str | Path,
    ):
        self.embeddings.to_parquet(path, engine='pyarrow')

    def create(self, question: str, max_length=1800):
        result = []
        current_length = 0
        data = self.embeddings
        
        question_embedding = Embedding.create(
            input=question, engine=MODELS_EMBEDDING[0]
        )['data'][0]['embedding']
        
        data['distances'] = distances_from_embeddings(
            question_embedding,
            data['embeddings'].values,
            distance_metric='cosine'
        )
        
        for _, row in data.sort_values('distances', ascending=True).iterrows():

            current_length = number_of_tokens("\n-".join(result))
            # current_length += row['tokens'] + 4

            if current_length > max_length:
                break

            # Else add it to the text that is being returned
            result.append(row["chunks"])
        if DEBUG:
            debugger.info(
                'Context created. Length: %s', current_length
            )
        # Return the context
        return "\n-".join(result)


class BaseHandleData:
    def __init__(self,):
        pass


class BasePdfHandler:
    def __init__(self, path: str | Path) -> None:
        pass