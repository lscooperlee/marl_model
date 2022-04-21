from .network_generator import ConvolutionNetworkGenerator, ConvolutionAtariNetworkGenerator
from .model_param import SimpleModelParam, AtariModelParam
from .base_model import BaseModel


class CDQNModel(SimpleModelParam, ConvolutionNetworkGenerator, BaseModel):

    def __repr__(self) -> str:
        cls = f'cdqn-{self.input_size[0]}x{self.input_size[1]}-{self.output_size}'
        return cls


class CDQNAtariModel(AtariModelParam, ConvolutionAtariNetworkGenerator, BaseModel):

    def __repr__(self) -> str:
        cls = f'cdqn-{self.input_size[0]}x{self.input_size[1]}-{self.output_size}'
        return cls