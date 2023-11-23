import os
import logging
from abc import ABC
from abc import abstractmethod
from pathlib import Path
from typing import Callable

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

from .constants import MODELS
from .constants import DEBUG

logger = logging.getLogger('standard')


class BaseContext(ABC):
    def __init__(self,
                 text: str = None,
                 max_tokens: int = 500,
                 openai_key=None,
                 openai_organization=None,
                 **kwargs):
        
        self.text = text
        self.embeddings = None
        self.json = "{}"
        self.max_tokens = max_tokens
        self.openai_key = openai_key
        self.openai_organization = openai_organization

    def save_embeddings(
            self,
            path: str | Path,
    ):
        if self.embeddings is not None:
            self.embeddings.to_parquet(path, engine='pyarrow')
            return
        if DEBUG:
            logger.warning(
                'Embeddings weren\'t generated yet. Nothing to save.'
            )

    def to_json(self,):
        if self.embeddings is not None:
            return self.embeddings.to_json(force_ascii=False)
        if DEBUG:
            logger.warning(
                'Embeddings weren\'t generated yet. Returning {}.'
            )
        return {}

    @abstractmethod
    def generate_embeddings(
            self,
            source: str | pd.DataFrame | Path | dict,
            model: str = MODELS[1],
            max_tokens: int = None
    ) -> pd.DataFrame:
        pass

    @abstractmethod
    def generate_context(self, question: str, max_length=1800) -> str:
        pass


class BaseChatter(ABC):
    def __init__(self,
                 set_up: str = None,
                 open_ai_key=None,
                 organization=None,
                 **kwargs):
        """
        Initialize a BaseChatter instance.

        Parameters
        ----------
        open_ai_key : str, optional
            OpenAI API key.
        organization : str, optional
            OpenAI organization key.
        **kwargs : dict
            Additional keyword arguments.

        Returns
        -------
        None
        """
        # Load credentials
        self.open_ai_key = open_ai_key or os.getenv("OPENAI_API_KEY")
        self.organization = organization or os.getenv(
            "OPENAI_API_ORGANIZATION"
        )

        # API setup
        self.client = OpenAI(
            api_key=self.open_ai_key,
            organization=self.organization,
            **kwargs
        )

        # setting
        self.set_up = set_up

    @abstractmethod
    def answer(self, prompt, **kwargs):
        """
        Generate a response to the given prompt.

        Parameters:
            prompt (str): The prompt message for generating a response.

        Returns:
            str: The generated response content.
        """

    @abstractmethod
    def _update(self, role: str, content: str):

        """
        Update the conversation with a new message.

        Parameters:
            role (str): The role of the message (e.g.,
                'system', 'user', 'assistant').
            content (str): The content of the message.
        """

    @abstractmethod
    def _reduce_number_of_tokens_if_needed(self, **kwargs):
        """
        Reduce messages using self.client.chat.completions.create.

        Parameters:
        ----------
            **kwargs: arguments for self.client.chat.completions.create
        """


class BaseQuestionContext(ABC):
    
    def __init__(self,
                 *args,  # args for agent
                 text: str = None,
                 **kwargs):  # kwargs for agent
        self.text = text
        self.chatter = None  # ChatGPT agent (chat completion)
        self.instruction = ''  # context instruction for agent
        self.history = []

    @abstractmethod
    def answer(self,
               question: str,
               context: str,
               use_agent=True,
               conversation=True) -> str:
        pass
