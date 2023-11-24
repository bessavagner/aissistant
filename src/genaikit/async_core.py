import json
import logging
from pathlib import Path
from typing import Callable

import pandas as pd

from openai import InternalServerError
from openai import AsyncOpenAI
from openai import APIError

from .constants import MODELS, MAX_TOKENS, ROLES
from .constants import EMBEDDINGS_COLUMNS
from .constants import MODELS_EMBEDDING
from .constants import DEBUG
from .prompts import CONTEXT_SETUP
from .errors import APIContextError
from .errors import ContextError
from .errors import MessageError
from .utils import distances_from_embeddings
from .utils import number_of_tokens

from .nlp.processors import TextProcessor

from .async_base import AsyncBaseQuestionContext
from .async_base import AsyncBaseContext
from .async_base import AsyncBaseChatter

logger = logging.getLogger('standard')


class AsyncContext(AsyncBaseContext):
    """
    A class representing a context for contextual chat applications.

    Parameters:
    - text (str, optional): The initial text for the context.
    - model (str, optional): The model to use for generating embeddings.
    - max_tokens (int, optional): The maximum number of tokens per chunk.
    - openai_key (str, optional): The OpenAI API key.
    - openai_organization (str, optional): The OpenAI organization.
    - **kwargs: Additional keyword arguments.

    Methods:
    - generate_embeddings(source, model=MODELS[1], max_tokens=None) ->
        pd.DataFrame:
        Generate embeddings from a source.

        Parameters:
        - source (str | pd.DataFrame | Path | dict): The source data for
            generating embeddings.
        - model (str, optional): The model to use for generating embeddings.
        - max_tokens (int, optional): The maximum number of tokens per chunk.

        Returns:
        pd.DataFrame: The generated embeddings.

    - generate_Asynccontext(question, max_length=1800) -> str:
        Generate a context based on a question.

        Parameters:
        - question (str): The question for generating the context.
        - max_length (int, optional): The maximum length of the context.

        Returns:
        str: The generated context.

    """
    def __init__(self,
                 text: str = None,
                 model: str = MODELS[1],
                 max_tokens: int = 500,
                 openai_key=None,
                 openai_organization=None,
                 **kwargs):
        super().__init__(
            text,
            model,
            max_tokens,
            openai_key=openai_key,
            openai_organization=openai_organization,
            **kwargs
        )

    async def generate_embeddings(
            self,
            source: str | pd.DataFrame | Path | dict = None,
            model: str = MODELS[1],
            max_tokens: int = None  # tokens per chunk
    ) -> pd.DataFrame:
        if source is None:
            source = self.text
        if max_tokens is None:
            max_tokens = self.max_tokens
        if isinstance(source, pd.DataFrame):
            self.embeddings = source

        elif isinstance(source, Path):
            self.embeddings = pd.read_parquet(source, engine='pyarrow')
        elif isinstance(source, dict):
            if EMBEDDINGS_COLUMNS != tuple(source.keys()):
                raise ContextError(
                    f"Keys of `source` must be: {','.join(EMBEDDINGS_COLUMNS)}"
                )
            lengths = list(map(len, source.values()))
            if not all(x == lengths[0] for x in lengths[1:]):
                raise ContextError(
                    "Items of `source`must have the same length"
                )
            self.json = source
            self.embeddings = pd.DataFrame(source)
            return self.embeddings
        else:
            if not isinstance(source, str):
                raise TypeError(
                    '`source` must either be a DataFrame, '
                    'Path object, a string or a dict'
                )
            try:
                self.embeddings = await TextProcessor().async_embeddings(
                    source,
                    model=model,
                    max_tokens=max_tokens,
                    openai_key=self.openai_key,
                    openai_organization=self.openai_organization
                )
            except APIError as err:
                message = (
                    f"OpenAI's error: {err.message} "
                    f"(code {err.code}) "
                    "Try again in a few minutes."
                )
                raise APIContextError(message) from err
        data = json.loads(
            self.embeddings.to_json(
                orient='table',
                index=False,
                force_ascii=False
            )
        )['data']
        self.json = {column: [] for column in EMBEDDINGS_COLUMNS}
        for row in data:
            for key, value in row.items():
                self.json[key].append(value)
        return self.embeddings

    async def generate_context(self,
                               question: str,
                               max_length=1800) -> str:
        result = []
        current_length = 0
        data = self.embeddings
        client = AsyncOpenAI(
            api_key=self.openai_key, organization=self.openai_organization
        )

        question_embedding = await client.embeddings.create(
            input=question, model=MODELS_EMBEDDING[0]
        )

        data['distances'] = distances_from_embeddings(
            question_embedding.data[0].embedding,
            data['embeddings'].values,
            distance_metric='cosine'
        )

        for _, row in data.sort_values('distances', ascending=True).iterrows():

            current_length = number_of_tokens("\n-".join(result))

            if current_length > max_length:
                break

            # Else add it to the text that is being returned
            result.append(row["chunks"])
        if DEBUG:
            logger.info(
                'Context created. Length: %s', current_length
            )
        # Return the context
        return "\n-".join(result)

    def save_embeddings(
            self,
            path: str | Path,
    ):
        self.embeddings.to_parquet(path, engine='pyarrow')


class AsyncChatter(AsyncBaseChatter):
    def __init__(self,
                 use_gpt4=False,
                 temperature=0,
                 open_ai_key=None,
                 organization=None,
                 set_up: str = None,
                 **kwargs):
        """
        Initialize a Chatter instance.

        Parameters
        ----------
        use_gpt4 : bool, optional
            Flag indicating whether to use GPT-4.
        temperature : float, optional
            Temperature parameter for model output.
        open_ai_key : str, optional
            OpenAI API key.
        organization : str, optional
            OpenAI organization key.
        set_up : str, optional
            Additional setup information.
        **kwargs : dict
            Additional keyword arguments.

        Returns
        -------
        None
        """
        super().__init__(set_up, open_ai_key, organization, **kwargs)

        # Model setup
        ## gpt-4 or gpt-3.5-turbo-1106
        self.model = MODELS[3] if use_gpt4 else MODELS[0]
        self.max_tokens = dict(MAX_TOKENS)[self.model]
        self.temperature = temperature

        # Data
        self.messages = []
        self.messages_backup = []
        self.messages_summarizer = []
        self.last_response = None

        if self.set_up:

            self.messages.append(
                {
                    'role': ROLES[0],
                    'content': self.set_up
                }
            )

    async def answer(self, prompt, use_agent=True, conversation=True, **kwargs):
        """
        Generate a response to the given prompt.

        Parameters:
        -----------
            prompt (str): The prompt message for generating a response.
            use_agent (bool): whether use a separate GPT chat to summarize
                large messages
            conversation (bool): whether stream messages in a single
                conversation

        Returns:
            str: The generated response content.
        """
        await self._update(ROLES[1], prompt, use_agent=use_agent)
        if self.set_up:
            messages = self.messages if conversation else self.messages[-2:]
        else:
            messages = self.messages if conversation else [
                self.messages[0], self.messages[-1]
            ]

        self.last_response = await self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            **kwargs
        )

        response_content = self.last_response.choices[0].message.content
        if DEBUG:
            logger.info('Response: %s', response_content)
        await self._update(ROLES[2], response_content, use_agent=use_agent)

        return response_content

    async def _update(self, role: str, content: str, use_agent=True):

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
        await self._reduce_number_of_tokens_if_needed(use_agent)

    async def _reduce_number_of_tokens_if_needed(
            self,
            use_agent=True,
            model=None,
            **kwargs
    ):
        """
        Reduces the number of tokens in the conversation if it exceeds the
        maximum allowed. It tries to summarize the conversation first and
        removes the oldest messages if necessary.
        """
        if model is None:
            model = self.model
        n_tokens = number_of_tokens(self.messages, model)
        while n_tokens > self.max_tokens:
            if use_agent and len(self.messages) > 2:
                # Attempt to summarize the conversation
                summary_prompt = (
                    "Please summarize the following conversation:\n"
                )
                if self.set_up:
                    summary_prompt += "\n".join(
                        message['content'] for message in self.messages[1:-1]
                    )
                else:
                    summary_prompt += "\n".join(
                        message['content'] for message in self.messages[:-1]
                    )

                summary_response = await self.client.chat.completions.create(
                    model=model,
                    messages=[{'role': ROLES[0], 'content': summary_prompt}],
                    temperature=self.temperature,
                    max_tokens=self.max_tokens // 4,  # Limit summary length
                    **kwargs
                )
                summary_content = summary_response.choices[0].\
                    message.content.strip()

                # Replace the conversation with the summary and the
                # last message
                if self.set_up:
                    self.messages = [
                        self.messages[0],
                        {'role': ROLES[1], 'content': summary_content},
                        self.messages[-1]
                    ]
                else:
                    self.messages = [
                        {'role': ROLES[1], 'content': summary_content},
                        self.messages[-1]
                    ]

                if DEBUG:
                    logger.info(
                        'Conversation summarized: %s', summary_content
                    )
            else:
                # Remove the oldest message if summarization is not possible
                # or did not help
                if self.set_up:
                    removed_message = self.messages.pop(1)
                else:
                    removed_message = self.messages.pop(0)
                if DEBUG:
                    logger.warning(
                        'Removed oldest message: %s',
                        removed_message['content']
                    )

            # Recalculate the number of tokens after modification
            n_tokens = number_of_tokens(self.messages, model)

            if n_tokens <= self.max_tokens:
                break  # Exit the loop if we are within the token limit

            if len(self.messages) <= 2:
                # If we cannot reduce further, raise an error
                if self.set_up or len(self.messages) == 1:
                    raise MessageError(
                        'Conversation exceeds maximum number of '
                        'tokens and cannot be reduced further.'
                    )


class AsyncMiniChatter(AsyncChatter):
    def __init__(self, open_ai_key=None, organization=None, **kwargs):
        """
        Initialize a MiniChatter instance.

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
        super().__init__(open_ai_key, organization, **kwargs)
        self.model = MODELS[1]  # gpt-3.5-turbo
        self.max_tokens = dict(MAX_TOKENS)[self.model]


class AsyncAdvancedChatter(AsyncChatter):
    def __init__(self, open_ai_key=None, organization=None, **kwargs):
        """
        Initialize an AdvancedChatter instance.

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
        super().__init__(
            use_gpt4=True,
            open_ai_key=open_ai_key,
            organization=organization,
            **kwargs
        )

    def change_model(self, model: str | int):
        """
        Change the model used by the chatter.

        Parameters
        ----------
        model : Union[str, int]
            The model to switch to. Can be a string representing the model name
            or an integer index.

        Returns
        -------
        None
        """
        if isinstance(model, str):
            if model not in MODELS:
                raise ValueError(f"`model` must be one of {','.join(MODELS)}")
            self.model = model
            self.max_tokens = dict(MAX_TOKENS)[model]
        elif isinstance(model, int):
            if model >= len(MODELS) or model < 0:
                models = '\n'.join(
                    [f"{idx+1}. {model}" for idx, model in enumerate(MODELS)]
                )
                raise ValueError(
                    f"`model` must be: {models}"
                )
            self.model = MODELS[model]
            self.max_tokens = MAX_TOKENS[model][1]


class AsyncQuestionContext(AsyncBaseQuestionContext):
    def __init__(self,
                 *args,
                 text: str = None,
                 set_up: str = None,
                 model=MODELS[1],  # context and agent's model
                 **kwargs):  # agent's **kwargs
        """
        Initialize a QuestionContext instance.

        Parameters
        ----------
        *args : list
            Variable length argument list.
        text : str, optional
            The text for the context.
        set_up : str, optional
            Additional setup information for the context.
        max_tokens : int, optional
            Maximum tokens per context's chunks.
        model : str, optional
            The context and agent's model.
        **kwargs : dict
            Additional keyword arguments for the agent.

        About set up:
        ------------
        The default setup does the following:

        1. **Defines the Role**: It clearly states that the AI is an assistant
            with broad knowledge, setting the expectation for the type of
            responses it should give.

        2. **Specifies the Task**: It instructs the AI to provide accurate and
            detailed responses, emphasizing the quality of the information.

        3. **Guides on Context Use**: It tells the AI to use the provided
            context to inform its responses, which is crucial for a contextual
            bot.

        4. **Handles Insufficient Context**: It gives a clear protocol for what
        to do if the context is not enough to answer the question.

        5. **Emphasizes Communication Goals**: It stresses the importance of
            clarity, relevance, and conciseness, which are key for user
            satisfaction.

        Remember that the effectiveness of the setup text can vary depending
            on the specific use case and the nature of the questions being
            asked. It may require further refinement through testing and
            iteration.

        Returns
        -------
        None
        """
        super().__init__(*args, text=text, set_up=set_up, **kwargs)
        self.instruction = (
            "Context: ###\n{}\n###\n"
            "User's question: ###\n{}\n###\n"
        )
        self.chatter = AsyncAdvancedChatter(
            *args, set_up=CONTEXT_SETUP, **kwargs
        )
        self.chatter.change_model(model)

    async def answer(self,
                     question: str,
                     context: str,
                     use_agent=True,
                     conversation=True):
        """
        Generate an answer for the given question and context.

        Parameters
        ----------
        question : str
            The question to answer.
        context : str
            The context for generating the answer.
        max_length : int, optional
            Maximum length of the answer.
        use_agent : bool, optional
            Flag indicating whether to use the agent for summarizing messages.
        conversation : bool, optional
            Flag indicating whether the answer should stream conversation.

        Returns
        -------
        str
            The generated answer.
        """

        prompt = self.instruction.format(context, question)
        try:
            answer = await self.chatter.answer(
                prompt, use_agent=use_agent, conversation=conversation
            )
        except InternalServerError as err:
            return (
                f"Openai server error: code: {err.code} - "
                f"message: {err.message}"
            )
        self.history.append({
            'question': question,
            'answer': answer
        })
        return answer
