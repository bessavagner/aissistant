import re
import logging
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import pandas as pd
from pandas.testing import assert_frame_equal

from genaikit.utils import get_encoding
from genaikit.constants import MODELS

from genaikit.nlp.processors import (
    TextProcessor,
    naive_split,
    naive_token_splitter,
    naive_text_to_embeddings
)

logger = logging.getLogger('standard')

path = Path(__file__).resolve().parent / 'sample/king.txt'
sampĺe_text_king = ''
with open(path, 'r', encoding='utf-8') as file_:
    sampĺe_text_king = file_.read()
SAMPLE_TEXT = re.split(r'(?<=:\s)(.*?)(?=\n)', sampĺe_text_king)[-1]


class MockEmbedding:
    def __init__(self, chunks):
        self.data = [MockData(chunks)]

class MockData:
    def __init__(self, chunks):
        self.embedding = [len(chunk)*[0.1] for chunk in chunks]



class TestTextProcessor(unittest.TestCase):
    def setUp(self):
        self.processor = TextProcessor()

    def test_split(self):
        # Test functionality
        text = "This is a sample text. It has multiple sentences."
        self.assertGreater(len(self.processor.split(text)), 1)

        # Test edge case: empty input
        self.assertEqual(self.processor.split(''), [])

        # Test error handling: invalid input type
        with self.assertRaises(ValueError):
            self.processor.split(123)

    def test_to_chunks(self):
        # Test functionality
        text = "This is a sample text. It has multiple sentences."
        self.processor.to_chunks(text)
        self.assertEqual(self.processor.text, text)
        self.assertGreater(len(self.processor.sequences), 1)

        # Test error handling: invalid input type
        with self.assertRaises(ValueError):
            self.processor.to_chunks(123)

    def test_group_by_semantics(self):
        # Test functionality
        data = "This is a sample text. It has multiple sentences."
        self.processor.group_by_semantics(data)
        self.assertEqual(self.processor.text, data)
        self.assertGreater(len(self.processor.sequences), 1)

        # Test error handling: invalid input type
        with self.assertRaises(ValueError):
            self.processor.group_by_semantics(123)

    def test_to_dataframe(self):
        # Test functionality
        model = MODELS[0]
        encoding = get_encoding(model)
        data = ["This is a sample text.", "It has multiple sentences."]
        n_tokens = [len(encoding.encode(chunk)) for chunk in data]
        expected_output = pd.DataFrame({'chunks': data, 'n_tokens': n_tokens})
        dataframe = self.processor.to_dataframe(data, model=model)
        self.assertEqual(self.processor.chunks, data)
        self.assertIsNotNone(self.processor.segments)
        assert_frame_equal(dataframe, expected_output)

        # Test error handling: invalid input type
        with self.assertRaises(ValueError):
            self.processor.to_dataframe(123)

    def test_max_tokens(self,):
        max_tokens = 90
        self.processor.to_chunks(SAMPLE_TEXT, max_tokens=max_tokens)
        self.assertEqual(
            len(self.processor.chunks), len(self.processor.n_tokens)
        )
        self.assertTrue(
            all(n < max_tokens for n in self.processor.n_tokens)
        )

    def test_embeddings(self,):
        data = ["This is a sample text.", "It has multiple sentences."]
        with patch('genaikit.nlp.processors.OpenAI') as mock:
            mock.return_value.embeddings.create.return_value = MockEmbedding(data)
            self.processor.embeddings(data, openai_key='foo')
            self.assertGreater(len(self.processor.chunks), 1)
            self.assertEqual(
                len(self.processor.chunks), len(self.processor.n_tokens)
            )
            self.assertIn('embeddings', self.processor.dataframe.columns)

# TODO test naive processors
