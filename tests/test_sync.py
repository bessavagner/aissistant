import pytest
from unittest.mock import patch, Mock

import pandas as pd
from pandas.testing import assert_frame_equal


from genaikit.core import Chatter, Context, QuestionContext

# Mock response to mimic the behavior of the actual OpenAI client
embeddings_json = {
    "chunks": ['blah', 'bleh', 'phew'],
    "n_tokens": [1, 1, 1],
    "embeddings": [[0.1], [0.1], [0.1]]
}
df_foo = pd.DataFrame(embeddings_json)

class MockResponse:
    def __init__(self):
        self.choices = [MockChoice()]

class MockChoice:
    def __init__(self):
        self.message = MockMessage()

class MockMessage:
    def __init__(self):
        self.content = "Mocked response content"

class MockEmbedding:
    def __init__(self):
        self.data = [MockData()]

class MockData:
    def __init__(self):
        self.embedding = df_foo

@pytest.fixture
def mock_openai_client():
    with patch('genaikit.base.OpenAI') as mock:
        mock.return_value.chat.completions.create = Mock(return_value=MockResponse())
        mock.return_value.embeddings.create = Mock(return_value=MockEmbedding())
        yield mock

# Test Chatter class
def test_chatter_answer(mock_openai_client):
    with patch('genaikit.core.OpenAI') as mock_openai:
        mock_openai.return_value.chat.completions.create.return_value = MockResponse()
        chatter = Chatter(open_ai_key='test_key')
        response = chatter.answer("Hello, how are you?")
        assert response == "Mocked response content"

def test_embeddings_creation_df(mock_openai_client):
    with patch('genaikit.core.OpenAI') as mock_openai:
        mock_openai.return_value.embeddings.create.return_value = MockEmbedding()
        context = Context(open_ai_key='test_key')
        embeddings = context.generate_embeddings(df_foo)
        assert_frame_equal(embeddings, df_foo)
        assert context.json == context.json

# def test_embeddings_creation(mock_openai_client):
#     with patch('genaikit.core.OpenAI') as mock_openai:
#         with patch('genaikit.nlp.processors.OpenAI') as mock_openai_utils:
#             mock_openai.return_value.embeddings.create.return_value = MockEmbedding()
#             mock_openai_utils.return_value.embeddings.create.return_value = MockEmbedding()
#             context = Context(open_ai_key='test_key')
#             text = "Test text"
#             embeddings = context.generate_embeddings(df_foo)
#             print(embeddings)
#             print(df_foo)
#             assert_frame_equal(embeddings, df_foo)
#             assert context.json == embeddings

# Test QuestionContext class integration with Chatter
def test_question_context_integration(mock_openai_client):
    with patch('genaikit.core.OpenAI') as mock_openai:
        mock_openai.return_value.chat.completions.create.return_value = MockResponse()
        question_context = QuestionContext(text="This is a test context")
        response = question_context.answer("What is the context?", "Test context")
        assert question_context.chatter.last_response.choices[0].message.content == "Mocked response content"