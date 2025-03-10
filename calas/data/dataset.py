import sys
import numpy as np
from abc import ABC, abstractmethod
from dataclasses import dataclass
from itertools import batched, groupby
from typing import Sequence, Callable, Generic, Generator, Iterable, final, override, Optional, TypeVar, Self, Protocol, TypeAlias, Any



T = TypeVar(name='T')
U = TypeVar(name='U')
T_contra = TypeVar(name='T_contra', contravariant=True)

class SupportsDunderLT(Protocol[T_contra]):
    def __lt__(self, other: T_contra, /) -> bool: ...

class SupportsDunderGT(Protocol[T_contra]):
    def __gt__(self, other: T_contra, /) -> bool: ...

SupportsRichComparison: TypeAlias = SupportsDunderLT[Any] | SupportsDunderGT[Any]
SupportsRichComparisonT = TypeVar(name='SupportsRichComparisonT', bound=SupportsRichComparison)


@dataclass
class Batch(Generic[T]):
    index: int
    items: 'list[T]'
    
    def __len__(self) -> int:
        return len(self.items)
    
    def __getitem__(self, idx: int|slice) -> T|list[T]:
        return self.items[idx]
    
    def __iter__(self) -> Generator[T, None, None]:
        for val in self.items:
            yield val


T_Dataset_co = TypeVar(name='T_Dataset_co', bound='DatasetBase[T]', covariant=True)
U_Dataset_co = TypeVar(name='U_Dataset_co', bound='DatasetBase[U]', covariant=True)


class DatasetBase(Generic[T, T_Dataset_co], ABC):
    """
    TODO: Decompose groupby and traintestsplit into mixins.
    """

    @property
    @abstractmethod
    def all_data(self) -> Sequence[T]: # pragma: no cover
        """
        Should be overridden in subclasses to return a `Sequence[T]` of all of the
        dataset's items. This property is used by many other methods. This type
        guarantees __iter__, __len__, __contains__, __getitem__, and __reversed__.
        """
        ...
    
    @final
    def __iter__(self) -> Generator[T, None, None]:
        for item in self.all_data:
            yield item
    
    @final
    def __contains__(self, item: T) -> bool:
        return item in self.all_data
    
    @final
    def __getitem__(self, index: int) -> T:
        return self.all_data[index]
    
    @final
    def __getitem__(self, index: slice) -> list[T]:
        return self.all_data[index]
    
    @final
    def __len__(self) -> int:
        return len(self.all_data)
    
    @property
    def size(self) -> int:
        return len(self.all_data)
    
    @property
    def is_empty(self) -> bool:
        return self.size == 0
    
    def shuffle(self, ds_fac: Callable[[Iterable[T]], T_Dataset_co], seed: Optional[int]=None) -> T_Dataset_co:
        if seed is None:
            seed = np.random.randint(low=0, high=sys.maxsize)
        gen = np.random.default_rng(seed=seed)
        items = list(self.all_data).copy()
        gen.shuffle(items)
        return ds_fac(items)
    
    def select(self, ds_fac: Callable[[Iterable[U]], U_Dataset_co], selector: Callable[[T], U]) -> U_Dataset_co:
        return ds_fac(map(selector, self.all_data))
    
    def select_items(self, selector: Callable[[T], U]) -> Iterable[U]:
        return map(selector, self.all_data)
    
    def order_by(self, ds_fac: Callable[[Iterable[T]], T_Dataset_co], sort_key: Callable[[T], SupportsRichComparisonT], reverse: bool=False) -> T_Dataset_co:
        return ds_fac(sorted(self.all_data, key=sort_key, reverse=reverse))
    
    def _group_by_order_by(self, ds_fac: Callable[[Iterable[T]], T_Dataset_co], grp_key: Callable[[T], U], sort_key: Optional[Callable[[T], SupportsRichComparisonT]]=None) -> Iterable[tuple[U, T_Dataset_co]]:
        for key, group in groupby(iterable=self.all_data, key=grp_key):
            group = sorted(group, key=sort_key) if isinstance(sort_key, Callable) else list(group)
            yield key, ds_fac(group)
    
    def group_by(self, ds_fac: Callable[[Iterable[T]], T_Dataset_co], key: Callable[[T], U]) -> Iterable[tuple[U, T_Dataset_co]]:
        return self._group_by_order_by(ds_fac=ds_fac, grp_key=key, sort_key=None)
    
    def group_by_order_by(self, ds_fac: Callable[[Iterable[T]], T_Dataset_co], grp_key: Callable[[T], U], sort_key: Optional[Callable[[T], SupportsRichComparisonT]]=None, sort_groups_by_key: bool=False) -> Iterable[tuple[U, T_Dataset_co]]:
        """
        Performs grouping and orders items within each group.
        """
        res = self._group_by_order_by(ds_fac=ds_fac, grp_key=grp_key, sort_key=sort_key)
        if not sort_groups_by_key:
            return res
        return sorted(res, key=lambda tpl: tpl[0])
    
    def where(self, ds_fac: Callable[[Iterable[T]], T_Dataset_co], predicate: Callable[[T], bool]) -> T_Dataset_co:
        return ds_fac([item for item in self.all_data if predicate(item)])
    
    def take(self, ds_fac: Callable[[Iterable[T]], T_Dataset_co], num: int) -> T_Dataset_co:
        assert isinstance(num, int)
        return ds_fac(self.all_data[0:num])
    
    def take_while(self, ds_fac: Callable[[Iterable[T]], T_Dataset_co], predicate: Callable[[T], bool]) -> T_Dataset_co:
        assert isinstance(predicate, Callable)
        return ds_fac([item for item in self.all_data if predicate(item)])
    
    def skip(self, ds_fac: Callable[[Iterable[T]], T_Dataset_co], num: int) -> T_Dataset_co:
        assert isinstance(num, int)
        return ds_fac(self.all_data[num:])
    
    def skip_while(self, ds_fac: Callable[[Iterable[T]], T_Dataset_co], predicate: Callable[[T], bool]) -> T_Dataset_co:
        return self.take_while(ds_fac=ds_fac, predicate=lambda item: not predicate(item))
    
    
    def train_test_split(self, ds_fac: Callable[[Iterable[T]], T_Dataset_co], seed: Optional[int]=None, test_ratio: float=0.1) -> tuple[T_Dataset_co, T_Dataset_co]:
        all_data = list(self.all_data).copy()
        n_test = max(1, round(test_ratio * len(all_data)))
        
        if isinstance(seed, int):
            gen = np.random.default_rng(seed=seed)
            gen.shuffle(all_data)

        train = ds_fac(all_data[n_test:])
        test = ds_fac(all_data[0:n_test])

        return train, test
    
    def iter_batched(self, batch_size: int) -> Iterable[Batch[T]]:
        assert isinstance(batch_size, int) and batch_size > 0

        for idx_batch, batch in enumerate(batched(iterable=self.all_data, n=batch_size)):
            yield Batch(index=idx_batch, items=list(batch))



class ListDataset(DatasetBase[T, Self]):
    """
    A default implementation for list-based datasets.
    """
    def __init__(self, items: Iterable[T]=[]):
        super().__init__()
        self._all_data: list[T] = []
        self._all_data.extend(items)
    
    def _factory(self, items: Iterable[T]) -> Self:
        return ListDataset(items=items)
    
    def append(self, item: T) -> Self:
        self._all_data.append(item)
        return self
    
    def extend(self, items: Iterable[T]) -> Self:
        self._all_data.extend(items)
        return self
    
    def sort(self, key: Callable[[T], SupportsRichComparisonT], reverse: bool=False) -> Self:
        self._all_data.sort(key=key, reverse=reverse)
        return self
    
    def clear(self) -> Self:
        self._all_data.clear()
        return self
    
    @property
    def all_data_(self) -> Sequence[T]:
        """
        An "in-place" version of all data, i.e., returns the actual underlying list.
        """
        return self._all_data
    
    @property
    @override
    def all_data(self) -> Sequence[T]:
        """
        Returns a shallow copy of the underlying list that holds all data.
        """
        return self._all_data.copy()
    
    @override
    def shuffle(self, seed: Optional[int]=None) -> Self:
        return super().shuffle(ds_fac=self._factory, seed=seed)
    
    @override
    def select(self, selector: Callable[[T], U]) -> 'ListDataset[U]':
        return super().select(ds_fac=self._factory, selector=selector)
    
    @override
    def order_by(self, sort_key: Callable[[T], SupportsRichComparisonT], reverse = False) -> Self:
        return super().order_by(ds_fac=self._factory, sort_key=sort_key, reverse=reverse)
    
    @override
    def group_by(self, key: Callable[[T], U]):
        return super().group_by(ds_fac=self._factory, key=key)
    
    @override
    def group_by_order_by(self, grp_key: Callable[[T], U], sort_key: Optional[Callable[[T], SupportsRichComparisonT]]=None, sort_groups_by_key: bool=False):
        return super().group_by_order_by(ds_fac=self._factory, grp_key=grp_key, sort_key=sort_key, sort_groups_by_key=sort_groups_by_key)
    
    @override
    def where(self, predicate: Callable[[T], bool]) -> Self:
        return super().where(ds_fac=self._factory, predicate=predicate)
    
    @override
    def skip(self, num: int) -> Self:
        return super().skip(ds_fac=self._factory, num=num)
    
    @override
    def skip_while(self, predicate: Callable[[T], bool]) -> Self:
        return super().skip_while(ds_fac=self._factory, predicate=predicate)
    
    @override
    def take(self, num: int) -> Self:
        return super().take(ds_fac=self._factory, num=num)
    
    @override
    def take_while(self, predicate: Callable[[T], bool]) -> Self:
        return super().take_while(ds_fac=self._factory, predicate=predicate)
    
    @override
    def train_test_split(self, seed: Optional[int]=None, test_ratio: float=0.1) -> tuple[Self, Self]:
        return super().train_test_split(ds_fac=self._factory, seed=seed, test_ratio=test_ratio)
