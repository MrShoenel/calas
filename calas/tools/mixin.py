import torch
from torch import nn
from typing import Generic, TypeVar, Optional, Self
from types import TracebackType


T_Mod = TypeVar(name='T_Mod', bound=nn.Module)


class NoGradNoTrainMixin(Generic[T_Mod]):
    def __init__(self, module: T_Mod):
        self._wrapped_module = module
    

    def __enter__(self) -> Self:
        self.was_training = self._wrapped_module.training
        self._wrapped_module.eval()
        self.ng = torch.no_grad()
        self.ng.__enter__()
        return self
    
    
    def __exit__(self, exc_type: Optional[type[BaseException]]=None, exc_value: Optional[BaseException]=None, traceback: Optional[TracebackType]=None) -> None:
        self.ng.__exit__(exc_type=exc_type, exc_value=exc_value, traceback=traceback)
        if self.was_training:
            self._wrapped_module.train()
