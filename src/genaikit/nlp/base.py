
from abc import ABC, abstractmethod
from typing import List

import spacy

import pandas as pd


class BaseTextProcessor(ABC):
    def __init__(self, pipeline: str = 'en_core_web_sm'):
        super().__init__()
        self.nlp = spacy.load(pipeline)
        self.text = ''
        self.sequences = []
        self.chunks = []
        self.n_tokens = []
        self.segments = []
        self.dataframe = pd.DataFrame([])

    @abstractmethod
    def split(self, text: str) -> List[str]:
        pass

    @abstractmethod
    def to_chunks(self, text: str, model: str) -> List[str]:
        pass

    @abstractmethod
    def group_by_semantics(self, text: str, model: str) -> pd.DataFrame:
        pass