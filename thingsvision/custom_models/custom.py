import abc


class Custom(metaclass=abc.ABCMeta):
    def __init__(self, device) -> None:
        self.device = device
        self.backend = "pt"

    @abc.abstractmethod
    def create_model(self):
        """Create the model"""

    def get_backend(self):
        return self.backend
