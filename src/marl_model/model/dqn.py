from .network_generator import PlainNetworkGenerator
from .model_param import SimpleModelParam, ClassicControlModelParam
from .base_model import BaseModel


class DQNModel(SimpleModelParam, PlainNetworkGenerator, BaseModel):

    def __repr__(self) -> str:
        cls = f'dqn-{self.input_size[0]}x{self.input_size[1]}-{self.output_size}'
        return cls

class ClassicControlDQNModel(ClassicControlModelParam, PlainNetworkGenerator, BaseModel):

    def __repr__(self) -> str:
        cls = f'dqn-{self.input_size[0]}x{self.input_size[1]}-{self.output_size}'
        return cls
