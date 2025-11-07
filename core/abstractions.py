from typing import Protocol, TypeVar, Generic, runtime_checkable
from abc import ABC, abstractmethod
import torch
import numpy as np
from dataclasses import dataclass
from enum import IntEnum, auto


T = TypeVar('T')
T_co = TypeVar('T_co', covariant=True)
ModelInput = TypeVar('ModelInput', bound=torch.Tensor)
ModelOutput = TypeVar('ModelOutput', bound=torch.Tensor)


class Phase(IntEnum):
    TRAIN = auto()
    VALIDATE = auto()
    TEST = auto()
    INFERENCE = auto()


@runtime_checkable
class Differentiable(Protocol):
    def forward(self, x: torch.Tensor) -> torch.Tensor: ...
    def parameters(self): ...


@runtime_checkable
class Stateful(Protocol[T_co]):
    def state_dict(self) -> dict: ...
    def load_state_dict(self, state: dict) -> None: ...


@runtime_checkable
class Composable(Protocol[T]):
    def __call__(self, x: T) -> T: ...


class Architecture(ABC, Generic[ModelInput, ModelOutput]):
    
    @abstractmethod
    def encode(self, x: ModelInput) -> torch.Tensor:
        raise NotImplementedError
    
    @abstractmethod
    def decode(self, z: torch.Tensor) -> ModelOutput:
        raise NotImplementedError
    
    def __call__(self, x: ModelInput) -> ModelOutput:
        return self.decode(self.encode(x))


class Layer(ABC):
    
    @abstractmethod
    def compute(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError
    
    @abstractmethod
    def flops(self) -> int:
        raise NotImplementedError
    
    @abstractmethod
    def receptive_field(self) -> int:
        raise NotImplementedError


@dataclass(frozen=True)
class Hyperparameters:
    depth: int
    width: int
    dropout_rate: float
    
    def validate(self) -> bool:
        return (
            self.depth > 0 and
            self.width > 0 and
            0.0 <= self.dropout_rate <= 1.0
        )


@dataclass(frozen=True)
class TrainingConfig:
    batch_size: int
    learning_rate: float
    weight_decay: float
    gradient_clip_norm: float
    
    def validate(self) -> bool:
        return all([
            self.batch_size > 0,
            self.learning_rate > 0,
            self.weight_decay >= 0,
            self.gradient_clip_norm > 0
        ])


@runtime_checkable
class Optimizer(Protocol):
    def step(self) -> None: ...
    def zero_grad(self) -> None: ...


@runtime_checkable
class LossFunction(Protocol):
    def __call__(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor: ...


@runtime_checkable
class Metric(Protocol):
    def update(self, pred: np.ndarray, target: np.ndarray) -> None: ...
    def compute(self) -> float: ...
    def reset(self) -> None: ...


class Preprocessor(ABC, Generic[T]):
    
    @abstractmethod
    def fit(self, data: T) -> 'Preprocessor':
        raise NotImplementedError
    
    @abstractmethod
    def transform(self, data: T) -> T:
        raise NotImplementedError
    
    def fit_transform(self, data: T) -> T:
        return self.fit(data).transform(data)


class FeatureExtractor(ABC):
    
    @abstractmethod
    def extract(self, raw_data: np.ndarray) -> np.ndarray:
        raise NotImplementedError
    
    @property
    @abstractmethod
    def output_dim(self) -> int:
        raise NotImplementedError


class InferenceEngine(ABC, Generic[ModelInput, ModelOutput]):
    
    @abstractmethod
    def predict(self, x: ModelInput) -> ModelOutput:
        raise NotImplementedError
    
    @abstractmethod
    def predict_batch(self, x: ModelInput, batch_size: int) -> ModelOutput:
        raise NotImplementedError


@runtime_checkable
class Serializable(Protocol):
    def serialize(self) -> bytes: ...
    def deserialize(self, data: bytes) -> None: ...


class Registry(Generic[T]):
    
    def __init__(self):
        self._registry: dict[str, type[T]] = {}
    
    def register(self, name: str):
        def decorator(cls: type[T]) -> type[T]:
            self._registry[name] = cls
            return cls
        return decorator
    
    def create(self, name: str, *args, **kwargs) -> T:
        if name not in self._registry:
            raise ValueError(f"Unknown type: {name}")
        return self._registry[name](*args, **kwargs)
    
    def list_available(self) -> list[str]:
        return list(self._registry.keys())


class LazyEvaluator(Generic[T]):
    
    def __init__(self, fn: Composable[T]):
        self._fn = fn
        self._cache: dict[int, T] = {}
    
    def __call__(self, x: T) -> T:
        key = hash(x.tobytes() if isinstance(x, np.ndarray) else str(x))
        if key not in self._cache:
            self._cache[key] = self._fn(x)
        return self._cache[key]
    
    def clear_cache(self) -> None:
        self._cache.clear()


class Pipeline(Generic[T]):
    
    def __init__(self, *stages: Composable[T]):
        self._stages = stages
    
    def __call__(self, x: T) -> T:
        for stage in self._stages:
            x = stage(x)
        return x
    
    def __len__(self) -> int:
        return len(self._stages)
