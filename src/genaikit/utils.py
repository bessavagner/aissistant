import re
import asyncio
from typing import List
from pathlib import Path

import tiktoken

import pandas as pd
import numpy as np
from scipy import spatial
from pypdf import PdfReader

from openai import OpenAI
from openai import AsyncOpenAI


from .constants import MODELS
from .constants import EMBEDDINGS_COLUMNS
from .constants import MODELS_EMBEDDING
from .constants import TOKENIZER


def split_into_sentences(text: str, minimal_length: int = 50) -> list[str]:
    """
    Split a text into sentences.

    Parameters:
    - text (str): The input text.
    - minimal_length (int, optional): The minimum length of a sentence.

    Returns:
    list[str]: A list of sentences.
    """
    sentences = []
    for sentence in text.split(". "):
        if len(sentence) > minimal_length:
            sentences.append(sentence)
    return sentences

def number_of_tokens(messages: str | list[str], model: str = TOKENIZER[0]):
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
    encoding = get_encoding(model)
    if isinstance(messages, str):
        messages = [
            {
                'role': 'user',
                'content': messages
            }
        ]
    # if model == MODELS[1]:  # note: future models may
    if True:  # note: future models may
        num_tokens = 0      # deviate from this
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

def get_encoding(model: str = None):
    try:
        if model:
            return tiktoken.encoding_for_model(model)
        return tiktoken.get_encoding(TOKENIZER[0])
    except KeyError:
        return tiktoken.get_encoding(TOKENIZER[0])

def token_splitter(
    text: str,
    model: str = MODELS[1],
    max_tokens: int = 500,
    minimal_length: int = 50
):
    """
    Split a text into tokens.

    Parameters:
    - text (str): The input text.
    - model (str, optional): The model to use for tokenization.
    - max_tokens (int, optional): The maximum number of tokens per chunk.
    - minimal_length (int, optional): The minimum length of a sentence.

    Returns:
    pd.DataFrame: The tokenized data.
    """
    encoding = None

    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        encoding = tiktoken.get_encoding(TOKENIZER[0])

    sentences = split_into_sentences(text, minimal_length=minimal_length)
    n_tokens = [
        len(encoding.encode(" " + sentence)) for sentence in sentences
    ]

    total_tokens = 0
    chunks = []
    tokens = []
    chunk = []

    # if model == MODELS[1]:  # note: future models may require this to change
    if True:  # note: future models may require this to change
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

        array = np.array([chunks, tokens]).T
        data = pd.DataFrame(array, columns=(
            EMBEDDINGS_COLUMNS[0], EMBEDDINGS_COLUMNS[1],)
        )
        data[EMBEDDINGS_COLUMNS[1]] = data[EMBEDDINGS_COLUMNS[1]].astype('int')
        return data
    
    raise NotImplementedError(  # TODO choose another error
        f"number_of_tokens() is not presently implemented for model {model}. "
        "See https://github.com/openai/openai-python/blob/main/chatml.md for "
        "information on how messages are converted to tokens."
        ""
    )

async def async_text_to_embeddings(
        text: str,
        model: str = MODELS[1],
        max_tokens: int = 500,
        openai_key=None,
        openai_organization=None
):
    """
    Convert text to embeddings.

    Parameters:
    - text (str): The input text.
    - model (str, optional): The model to use for generating embeddings.
    - max_tokens (int, optional): The maximum number of tokens per chunk.
    - openai_key (str, optional): The OpenAI API key.
    - openai_organization (str, optional): The OpenAI organization.

    Returns:
    pd.DataFrame: The data with embeddings.
    """
    data = token_splitter(text, model, max_tokens)
    client = AsyncOpenAI(api_key=openai_key, organization=openai_organization)

    # Create a list to store the tasks
    tasks = []

    # Create a coroutine to apply to each row
    async def create_embedding(row):
        embedding = await client.embeddings.create(
            input=row[EMBEDDINGS_COLUMNS[0]],
            model=MODELS_EMBEDDING[0]
        )
        return embedding.data[0].embedding

    # Schedule the coroutines as tasks
    for _, row in data.iterrows():
        tasks.append(create_embedding(row))

    # Gather the results
    embeddings = await asyncio.gather(*tasks)

    # Assign the results to the new column
    data[EMBEDDINGS_COLUMNS[2]] = embeddings

    return data

def text_to_embeddings(
        text: str,
        model: str = MODELS[1],
        max_tokens: int = 500,
        openai_key=None,
        openai_organization=None
):
    """
    Convert text to embeddings.

    Parameters:
    - text (str): The input text.
    - model (str, optional): The model to use for generating embeddings.
    - max_tokens (int, optional): The maximum number of tokens per chunk.
    - openai_key (str, optional): The OpenAI API key.
    - openai_organization (str, optional): The OpenAI organization.

    Returns:
    pd.DataFrame: The data with embeddings.
    """
    data = token_splitter(text, model, max_tokens)
    client = OpenAI(api_key=openai_key, organization=openai_organization)

    # Create a list to store the tasks
    embeddings = []

    # Create a coroutine to apply to each row
    def create_embedding(row):
        embedding = client.embeddings.create(
            input=row[EMBEDDINGS_COLUMNS[0]],
            model=MODELS_EMBEDDING[0]
        )
        return embedding.data[0].embedding

    # Schedule the coroutines as tasks
    for _, row in data.iterrows():
        embeddings.append(create_embedding(row))

    # Assign the results to the new column
    data[EMBEDDINGS_COLUMNS[2]] = embeddings

    return data

def distances_from_embeddings(
    query_embedding: List[float],
    embeddings: List[List[float]],
    distance_metric="cosine",
) -> List[List]:
    """
    Calculate distances between a query embedding and a list of embeddings.

    Parameters:
    - query_embedding (List[float]): The embedding of the query.
    - embeddings (List[List[float]]): A list of embeddings.
    - distance_metric (str, optional): The distance metric to use.

    Returns:
    List[List]: A list of distances.
    """
    distance_metrics = {
        "cosine": spatial.distance.cosine,
        "L1": spatial.distance.cityblock,
        "L2": spatial.distance.euclidean,
        "Linf": spatial.distance.chebyshev,
    }
    distances = [
        distance_metrics[distance_metric](query_embedding, embedding)
        for embedding in embeddings
    ]
    return distances


def code_to_str(path: str | Path, comments: str = '#{}'):
    filepath = Path(path)
    code = comments.format(filepath.name) + '\n'
    with open(path, 'r', encoding='utf-8') as file_:
        code += file_.read()
    return code

def codes_to_str(path: str | Path,
                 suffix: str = '.py',
                 comments: str = '#',
                 exclude: list[str] = None) -> str:
    code = ''
    
    for item in Path(path).glob(f"*{suffix}"):
        if exclude and item.name in exclude:
            continue
        name = item.name
        with open(item, 'r', encoding='utf-8') as file_:
            code += f"{comments} {name}\n{file_.read()}\n"
    
    return code

def python_codes_to_str(path: str | Path, exclude: list[str] = None) -> str:
    return code_to_str(path, exclude=exclude)

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
        but with periods, colons and semi-colons retained.

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
    text = re.sub(r'[^\w\s.;,\'\"]', '', text)
    text = clean_lines_and_spaces(text)
    text = text.replace('•', '')
    text = text.strip()
    return text

def pdf2text(filename, wdir, clean=True, return_reader=False):
    """
    Extract text from a PDF file and optionally clean it.

    Parameters
    ----------
    filename : str
        The name of the PDF file.
    wdir : str
        The working directory where the PDF file is located.
    clean : bool, optional
        Whether to clean the extracted text. Defaults to True.
    return_reader : bool, optional
        Whether to return the PdfReader object. Defaults to False.

    Returns
    -------
    str or tuple
        The extracted text and, optionally, the PdfReader object.
    """
    path = Path(wdir) / Path(filename)
    if path.suffix == "":
        path = Path(wdir) / Path(f"{filename}.pdf")
    reader = PdfReader(path)
    entire_text = ""
    for page in reader.pages:
        entire_text += page.extract_text()
    if clean:
        entire_text = clean_text(entire_text)
    if return_reader:
        return entire_text, reader
    return entire_text

