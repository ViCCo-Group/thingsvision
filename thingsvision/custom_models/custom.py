import abc 

class Custom(metaclass=abc.ABCMeta):
    def __init__(self, device, backend) -> None:
        self.device = device 
        self.backend = backend

    @abc.abstractmethod
    def create_model(self, backend):
        """ Create the model """