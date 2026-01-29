import abc

class BasePolicy(abc.ABC):
    def __init__(self):
        super(BasePolicy, self).__init__()

    @abc.abstractmethod
    def act(self):
        raise NotImplementedError("'act' needs to be implemented")

    @abc.abstractmethod
    def save(self, path: str):
        raise NotImplementedError("'save' needs to be implemented")
