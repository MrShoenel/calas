from typing import TypeVar, Callable, Generic


T = TypeVar('T')


class _Empty:
    """Placeholder class for non-existent values."""
    pass


class Lazy(Generic[T]):
    """
    Simple (and definitely not perfect) implementation of Lazy<T>.
    Produces a synchronized value.
    """
    def __init__(self, factory: Callable[[], T]) -> None:
        self.factory = factory
        self._value = _Empty()

    @property
    def value(self) -> T:
        if isinstance(self._value, _Empty):
            # Try to make it:
            self._value = self.factory()
        return self._value
