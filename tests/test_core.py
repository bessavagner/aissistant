import pytest
import shutil
import tempfile
from pathlib import Path
import pandas as pd
import numpy as np
from pandas.testing import assert_frame_equal
from unittest.mock import patch, Mock
from genaikit.core import Chatter, Context, QuestionContext

from genaikit.utils import get_encoding
from genaikit.constants import EMBEDDINGS_COLUMNS
from genaikit.nlp.processors import TextProcessor


encoding = get_encoding()

foo_chunks = [
    "Test sentence.", "another test sentence.", "one more test sentence."
]
foo_tokens = [
    len(encoding.encode(sent)) for sent in foo_chunks
]
foo_embdd = [len(sent.split(' '))*[0.1] for sent in foo_chunks]
embeddings_json = {
    k:v for k,v in zip(EMBEDDINGS_COLUMNS, (foo_chunks, foo_tokens, foo_embdd))
}
foo_df = pd.DataFrame(embeddings_json)

# Mock response to mimic the behavior of the actual OpenAI client.
# object: client = genaikit.core.OpenAI()
# method mocked: client.chat.completions.create
# object mocked (return_value): client.choices[0].message.content
class MockResponse:
    def __init__(self):
        self.choices = [MockChoice()]

class MockChoice:
    def __init__(self):
        self.message = MockMessage()

class MockMessage:
    def __init__(self):
        self.content = "Mocked response content"

# Mock response to mimic the behavior of OpenAI
# objects: client = genaikit.nlp.processors.OpenAI() or
# methods mocked: client.embeddings.create
# object mocked (return_value): client.data[0].embedding
class MockEmbedding:
    def __init__(self, chunks):
        self.data = [MockData(chunks)]

class MockData:
    def __init__(self, chunks):
        self.embedding = [len(chunk.split(' '))*[0.1] for chunk in chunks]

@pytest.fixture
def mock_openai_client():
    with patch('genaikit.base.OpenAI') as mock:
        mock.return_value.chat.completions.create = Mock(return_value=MockResponse())
        mock.return_value.embeddings.create = Mock(return_value=MockEmbedding(foo_chunks))
        yield mock

def test_chatter_initialization():
    chatter = Chatter(open_ai_key="test_key", organization="test_org")
    assert chatter.open_ai_key == "test_key"
    assert chatter.organization == "test_org"
    assert chatter.model == "gpt-3.5-turbo-1106"
    assert chatter.max_tokens == 16385
    assert chatter.temperature == 0

def test_chatter_answer(mock_openai_client):
    chatter = Chatter(open_ai_key="test_key", organization="test_org")
    with patch("genaikit.core.OpenAI") as mock_openai:
        mock_openai.return_value.chat.completions.create.return_value = MockResponse()
        response = chatter.answer("test prompt")
        assert response == "Mocked response content"

def test_context_initialization():
    context = Context(text="test text", openai_key="test_key", openai_organization="test_org")
    assert context.text == "test text"
    assert context.openai_key == "test_key"
    assert context.openai_organization == "test_org"
    assert context.max_tokens == 90

def test_context_generate_embeddings_from_dataframe():
    context = Context(openai_key="test_key", organization="test_org")
    with patch("genaikit.core.OpenAI") as mock_openai:
        mock_openai.return_value.chat.completions.create.return_value = MockEmbedding(foo_chunks)
        embeddings = context.generate_embeddings(foo_df)
    assert_frame_equal(embeddings, foo_df)

def test_context_generate_embeddings_from_path():
    context = Context(openai_key="test_key", organization="test_org")
    with patch("genaikit.core.pd.read_parquet") as mock_read_parquet:
        mock_read_parquet.return_value = foo_df
        embeddings = context.generate_embeddings(Path('path'))
        assert_frame_equal(embeddings, foo_df)

def test_context_generate_embeddings_from_dict():
    context = Context(openai_key="test_key", organization="test_org")
    embeddings = context.generate_embeddings(embeddings_json)
    assert_frame_equal(embeddings, foo_df)

def test_context_generate_embeddings_from_text():
    context = Context(openai_key="test_key", organization="test_org")
    with patch.object(TextProcessor, "embeddings") as mock_embeddings:
        mock_embeddings.return_value = foo_df
        embeddings = context.generate_embeddings("test text")
        assert_frame_equal(embeddings, foo_df)

# def test_context_generate_context():
#     context = Context(openai_key="test_key", organization="test_org")
#     context.embeddings = foo_df
#     with patch("genaikit.core.OpenAI") as mock_openai:
#         mock_openai.return_value.embeddings.create.return_value = MockEmbedding(foo_df["chunks"])
#         generated_context = context.generate_context("test question")
#         assert generated_context == "One sentence\n-Another sentence"

# def test_question_context_initialization():
#     question_context = QuestionContext(openai_key="test_key", organization="test_org")
#     assert question_context.instruction == "Context: ###\n{}\n###\nUser's question: ###\n{}\n###\n"
#     assert question_context.chatter.open_ai_key == "test_key"
#     assert question_context.chatter.organization == "test_org"
#     assert question_context.chatter.model == "gpt-3.5-turbo-1106"
#     assert question_context.chatter.max_tokens == 16385
#     assert question_context.chatter.temperature == 0

# def test_question_context_answer():
#     question_context = QuestionContext(openai_key="test_key", organization="test_org")
#     with patch("genaikit.core.OpenAI") as mock_openai:
#         mock_openai.return_value.chat.completions.create.return_value = MockResponse()
#         response = question_context.answer("test question", "test context")
#         assert response == "Mocked response content"