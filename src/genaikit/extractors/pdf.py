from genaikit.utils import (
    clean_text,
    pdf2text
)

from .base import BasePdfReader

class PdfToText(BasePdfReader):
    def __init__(self, filename=None, wdir='') -> None:
        self.__raw = None
        self.__text = None
        self.__reader = None
        self.__n_pages = None
        if filename is not None:
            self.read(filename, wdir)

    @property
    def raw(self,):
        return self.__raw

    @property
    def text(self,):
        return self.__text

    @property
    def reader(self,):
        return self.__reader

    @property
    def n_pages(self):
        return self.__n_pages

    def read(self, filename: str, wdir: str = ''):
        self.__raw, self.__reader = pdf2text(
            filename, wdir, clean=False, return_reader=True
        )
        self.__text = clean_text(self.__raw)
        self.__n_pages = len(self.reader.pages)
        self.__raw = pdf2text(filename, wdir, clean=False)
        self.__text = clean_text(self.__raw)
        return self.__text
