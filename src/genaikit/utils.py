"""
    Package "aissistant"

    This module provides helpful objects
"""
import re
import logging
import subprocess

from pathlib import Path

import tiktoken
import pandas as pd
import numpy as np

from PyPDF2 import PdfReader

from openai import Embedding

from .constants import MODELS
from .constants import MODELS_EMBEDDING
from .constants import MAX_TOKENS

logger = logging.getLogger("client")
debugger = logging.getLogger("standard")


def important_action(arg=False):
    if not arg:
        return None
    return "Action!"


def split_into_sentences(text: str, minimal_length: int = 50) -> list[str]:
    sentences = []
    for sentence in text.split(". "):
        if len(sentence) > minimal_length:
            sentences.append(sentence)
    return sentences


def token_splitter(
    text: str,
    model: str = MODELS[1],
    max_tokens: int = 500,
    minimal_length: int = 50
):
    encoding = None

    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")

    sentences = split_into_sentences(text, minimal_length=minimal_length)
    n_tokens = [
        len(encoding.encode(" " + sentence)) for sentence in sentences
    ]
    
    # total_tokens = sum(n_tokens)  # TODO manage max_tokens
    # if total_tokens >= dict(MAX_TOKENS)[model]:
    #     new_max_tokens = dict(MAX_TOKENS)[model] // len(sentences)
    #     logger.warning(
    #         (
    #             "`max_tokens=%s` produces higher number of "
    #             "tokens (%s) than the model %s allows. "
    #             ", Reducing to %s."
    #         ),
    #         max_tokens,
    #         total_tokens,
    #         model,
    #         new_max_tokens,
    #     )
    #     max_tokens = new_max_tokens

    total_tokens = 0
    chunks = []
    tokens = []
    chunk = []

    if model == MODELS[1]:  # note: future models may require this to change
        for sentence, n_token in zip(sentences, n_tokens):
            if total_tokens + n_token > max_tokens and chunk:
                chunks.append(". ".join(chunk) + ".")
                tokens.append(total_tokens)
                chunk = []
                total_tokens = 0

            if n_token > max_tokens:
                continue

            chunk.append(sentence)
            total_tokens += n_token + 1
        # shortened = []

        # # Loop through the dataframe
        # for sentence in sentences:

        #     # If the text is None, go to the next row
        #     if sentence is None:
        #         continue

        #     # If the number of tokens is greater than the max number of tokens, split the text into chunks
        #     if len(encoding.encode(sentence)) > max_tokens:
        #         shortened += split_into_many(sentence, max_tokens)

        #     # Otherwise, add the text to the list of shortened texts
        #     else:
        #         shortened.append(sentence)
        # data = pd.DataFrame(shortened, columns=['chunks'])
        # data['n_tokens'] = data.chunks.apply(lambda x: len(encoding.encode(x)))
        array = np.array([chunks, tokens]).T
        data = pd.DataFrame(array, columns=('chunks', 'n_tokens',))
        data['n_tokens'] = data['n_tokens'].astype('int')
        return data
    
    raise NotImplementedError(  # TODO choose another error
        f"number_of_tokens() is not presently implemented for model {model}. "
        "See https://github.com/openai/openai-python/blob/main/chatml.md for "
        "information on how messages are converted to tokens."
        ""
    )

# Function to split the text into chunks of a maximum number of tokens
def split_into_many(text, max_tokens, model: str = MODELS[1]):
    tokenizer = None

    try:
        tokenizer = tiktoken.encoding_for_model(model)
    except KeyError:
        tokenizer = tiktoken.get_encoding("cl100k_base")

    # Split the text into sentences
    sentences = text.split('. ')

    # Get the number of tokens for each sentence
    n_tokens = [len(tokenizer.encode(" " + sentence)) for sentence in sentences]

    chunks = []
    tokens_so_far = 0
    chunk = []

    # Loop through the sentences and tokens joined together in a tuple
    for sentence, token in zip(sentences, n_tokens):

        # If the number of tokens so far plus the number of tokens in the current sentence is greater
        # than the max number of tokens, then add the chunk to the list of chunks and reset
        # the chunk and tokens so far
        if tokens_so_far + token > max_tokens:
            chunks.append(". ".join(chunk) + ".")
            chunk = []
            tokens_so_far = 0

        # If the number of tokens in the current sentence is greater than the max number of
        # tokens, go to the next sentence
        if token > max_tokens:
            continue

        # Otherwise, add the sentence to the chunk and add the number of tokens to the total
        chunk.append(sentence)
        tokens_so_far += token + 1

    return chunks

def number_of_tokens(messages: str|list[str], model: str = MODELS[1]):
    """
    Returns the number of tokens used by a list of messages.

    Parameters
    ----------
    messages : str or list of str
        A single message or a list of messages to be processed. Each message
        can be a string.
    model : str, optional
        The name of the model used for token encoding (default is MODELS[1]).

    Returns
    -------
    int
        The total number of tokens used by the provided messages.

    Raises
    ------
    NotImplementedError
        If the function is not presently implemented for the given model.

    Notes
    -----
    The function calculates the number of tokens used by messages. The number
    of tokens
    is derived from the encoding of the messages according to the specified
    model.
    If the model is not found in the pre-defined MODELS list, the function will
    fall back
    to using the "cl100k_base" model for token encoding.

    Each message is expected to be in the form of a dictionary with 'role' and
    'content' keys,
    representing the sender role and the content of the message, respectively.
    The function
    calculates the token count considering the special tokens used for message
    encoding,
    such as <im_start> and <im_end>. For future models, token counts may vary,
    so this
    behavior is subject to change.

    The function raises a NotImplementedError if the provided model is not
    supported. Users can refer to the provided link for information on how
    messages are converted to tokens for each specific model.

    Examples
    --------
    >>> messages = [
    ...     {
    ...         'role': 'user',
    ...         'content': "Hello, how are you?"
    ...     },
    ...     {
    ...         'role': 'assistant',
    ...         'content': "I'm doing great! How can I assist you?"
    ...     }
    ... ]
    >>> num_tokens = number_of_tokens(messages)
    >>> print(num_tokens)
    23

    >>> single_message = "This is a test message."
    >>> num_tokens = number_of_tokens(single_message, model="my_custom_model")
    >>> print(num_tokens)
    8
    """
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")
    if isinstance(messages, str):
        messages = [
            {
                'role': 'user',
                'content': messages
            }
        ]
    if model == MODELS[1]:  # note: future models may
        num_tokens = 0  # deviate from this
        for message in messages:
            # every message follows
            # <im_start>{role/name}\n{content}<im_end>\n
            num_tokens += 4
            for key, value in message.items():
                num_tokens += len(encoding.encode(value))
                if key == "name":  # if True, the role is omitted
                    num_tokens += -1  # role is always required and 1 token
        num_tokens += 2  # every reply is primed with <im_start>assistant
        return num_tokens
    raise NotImplementedError(  # TODO choose another error
        f"number_of_tokens() is not presently implemented for model {model}. "
        "See https://github.com/openai/openai-python/blob/main/chatml.md for "
        "information on how messages are converted to tokens."
        ""
    )


def text_to_embeddings(
        text: str, model: str = MODELS[1], max_tokens: int = 500
):
    data = token_splitter(text, model, max_tokens)
    data['embeddings'] = data.chunks.apply(lambda x: Embedding.create(
            input=x, engine=MODELS_EMBEDDING[0]
        )['data'][0]['embedding']
    )
    # data['embeddings'] = data['embeddings'].apply(eval).apply(np.array)
    return data


def test_with_flake8(file_path):
    # Run Flake8 on the file
    try:
        result = subprocess.run(
            ["flake8", file_path], capture_output=True, text=True, check=True
        )

        # Check the return code
        if result.returncode == 0:
            logger.info("No Flake8 errors found!")
            return
        # Print the Flake8 output
        logger.info(result.stdout)
        return result.stdout
    except subprocess.CalledProcessError as err:
        return err.output


def clean_lines_and_spaces(text):
    text = text.replace("\n", " ")
    text = text.replace("\\n", " ")
    text = text.replace("  ", " ")
    text = text.replace("  ", " ")
    return text

def clean_text(text):
    """
    Sanitizes the input text by removing special characters (excluding spaces,
    digits, and alphabets),
    bullet points (•), and extra spaces. Periods are retained in the sanitized
    text.

    Parameters
    ----------
    text : str
        The text to be sanitized.

    Returns
    -------
    str
        The sanitized text without special characters and extra spaces,
        but with periods retained.

    Examples
    --------
    >>> text_to_sanitize = \"\"\"
    ...     Hello! This is a sample text with special characters: @#$%^&*(),
    ...     bullet points •, extra spaces, and new lines.
    ...
    ...     The text will be sanitized to remove all these elements.
    ... \"\"\"
    >>> sanitized_text = sanitize_text(text_to_sanitize)
    >>> print(sanitized_text)
    Hello This is a sample text with special characters bullet points extra spaces and new lines. The text will be sanitized to remove all these elements.
    """
    text = re.sub(r'[^\w\s.]', '', text)
    text = clean_lines_and_spaces(text)
    text = text.replace('•', '')
    text = text.strip()
    return text

def pdf2string(path: str | Path):
    reader = PdfReader(path)
    entire_text = ''
    for page in reader.pages:
        entire_text += page.extract_text()
    return entire_text
