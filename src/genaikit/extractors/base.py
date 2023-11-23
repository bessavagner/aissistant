class BasePdfReader(ABC):
    @property
    @abstractmethod
    def raw(
        self,
    ) -> str:
        pass

    @property
    @abstractmethod
    def text(
        self,
    ) -> str:
        pass

    @abstractmethod
    def read(self, filename: str, wdir: str = "") -> str:
        pass
