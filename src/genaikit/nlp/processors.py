import asyncio
from typing import List
from pathlib import Path

import tiktoken
import numpy as np
import pandas as pd

from openai import OpenAI, AsyncOpenAI

from genaikit.constants import MODELS
from genaikit.constants import EMBEDDINGS_COLUMNS
from genaikit.constants import MODELS_EMBEDDING
from genaikit.utils import get_encoding

from .base import BaseTextProcessor


class TextProcessor(BaseTextProcessor):
    def __init__(self, *args, pipeline: str = 'en_core_web_lg', **kwargs):
        super().__init__(*args, pipeline=pipeline, **kwargs)

    def split(self, text) -> np.ndarray:
        self.text = text
        doc = self.nlp(self.text)
        self.sequences = [sent.text for sent in doc.sents]
        return self.sequences

    def to_chunks(self,
                  text: str = None,
                  model: str = MODELS[1],
                  max_tokens: int = 100) -> List[str]:
        if text is not None:
            self.text = text
            self.split(text)
        # group sentences semantically with a maximum number of tokens
        # using tiktoken to compute tokens
        # example maximum number of tokens
        chunks = []
        tokens = []
        current_chunk = []
        current_tokens = 0
        encoding = get_encoding(model)

        for sentence in self.sequences:

            encoded_text = encoding.encode(sentence + ' ')
            if current_tokens + len(encoded_text) <= max_tokens:
                current_chunk.append(sentence)
                current_tokens += len(encoded_text)
            else:
                chunks.append(' '.join(current_chunk))
                tokens.append(current_tokens)
                current_chunk = [sentence]
                current_tokens = len(encoded_text)

        if current_chunk:
            chunks.append(' '.join(current_chunk))
            tokens.append(current_tokens)
        self.chunks = chunks
        self.n_tokens = tokens
        return chunks

    def group_by_semantics(self,
                           data: str | List[str] = None,
                           model: str = MODELS[1],
                           max_tokens: int = 100,
                           threshold: float = 0.8) -> List[str]:
        
        if data is not None:
            if isinstance(data, list):
                if not self.n_tokens:
                    encoding = get_encoding(model)
                    self.n_tokens = [
                        len(encoding.encode(chunk)) for chunk in data
                    ]
                self.chunks = data
            else:
                self.to_chunks(
                    text=data,
                    model=model,
                    max_tokens=max_tokens,
                )
        
        docs = [self.nlp(sentence) for sentence in self.chunks]
        segments = []
        start_idx = 0
        end_idx = 1
        segment = [self.chunks[start_idx]]
        while end_idx < len(docs):
            if docs[start_idx].similarity(docs[end_idx]) >= threshold:
                segment.append(docs[end_idx].text)
            else:
                segments.append(" ".join(segment))
                start_idx = end_idx
                segment = [self.chunks[start_idx]]
            end_idx += 1
        if segment:
            segments.append(" ".join(segment))
        self.segments = segments
        return segments

    def to_dataframe(self,
                     data: str | List[str] = None,
                     model: str = MODELS[1],
                     max_tokens: int = 120,
                     threshold: float = 0.8) -> pd.DataFrame:
        if data is not None:
            self.group_by_semantics(
                data=data,
                model=model,
                max_tokens=max_tokens,
                threshold=threshold
            )

        # encoding = get_encoding(model)

        chunks = self.chunks
        n_tokens = self.n_tokens
        # n_tokens = [
        #     len(encoding.encode(" " + chunk))
        #     for chunk in chunks
        # ]
        self.dataframe = pd.DataFrame({'chunks': chunks, 'n_tokens': n_tokens})
        return self.dataframe

    def embeddings(self,
                   data: str | List[str] | pd.DataFrame = None,
                   model: str = MODELS[1],
                   max_tokens: int = 120,
                   threshold: float = .8,
                   openai_key=None,
                   openai_organization=None) -> pd.DataFrame:
        if data is not None and not isinstance(data, pd.DataFrame):
            self.to_dataframe(
                data=data,
                model=model,
                max_tokens=max_tokens,
                threshold=threshold
            )
        client = OpenAI(api_key=openai_key, organization=openai_organization)

        embeddings = []
        def create_embedding(row):
            embedding = client.embeddings.create(
                input=row[EMBEDDINGS_COLUMNS[0]],
                model=MODELS_EMBEDDING[0]
            )
            return embedding.data[0].embedding

        for _, row in self.dataframe.iterrows():
            embeddings.append(create_embedding(row))

        self.dataframe[EMBEDDINGS_COLUMNS[2]] = embeddings

        return self.dataframe

    async def async_embeddings(self,
                               data: str | List[str] | pd.DataFrame = None,
                               model: str = MODELS[1],
                               max_tokens: int = 120,
                               threshold: float = .8,
                               openai_key=None,
                               openai_organization=None,
                               **kwargs) -> pd.DataFrame:
        if data is not None:
            self.to_dataframe(
                data=data,
                model=model,
                max_tokens=max_tokens,
                threshold=threshold
            )
        client = AsyncOpenAI(
            api_key=openai_key,
            organization=openai_organization,
            max_retries=1,
            **kwargs
        )

        tasks = []

        async def create_embedding(row):
            embedding = await client.embeddings.create(
                input=row[EMBEDDINGS_COLUMNS[0]],
                model=MODELS_EMBEDDING[0]
            )
            return embedding.data[0].embedding

        for _, row in self.dataframe.iterrows():
            tasks.append(create_embedding(row))
        
        self.dataframe[EMBEDDINGS_COLUMNS[2]] = await asyncio.gather(*tasks)

        return self.dataframe


def naive_split(
    text: str, minimal_length: int = 50
) -> list[str]:
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

def naive_token_splitter(
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
    encoding = get_encoding(model)

    sentences = naive_split(text, minimal_length=minimal_length)
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

def naive_text_to_embeddings(
        text: str,
        model: str = MODELS[1],
        max_tokens: int = 500,
        openai_key=None,
        openai_organization=None
):
    processor = TextProcessor()
    processor.dataframe = naive_token_splitter(text, model, max_tokens)
    return processor.embeddings(
        model=model,
        openai_key=openai_key,
        openai_organization=openai_organization
    )
