import contextlib
import numpy as np

__all__ = [
    "Tensor",
    "tensor",
    "manual_seed",
    "argmax",
    "no_grad",
    "device",
    "cuda",
    "version",
    "long",
]

long = np.int64

_rng = np.random.default_rng()


def manual_seed(seed: int) -> None:
    global _rng
    _rng = np.random.default_rng(seed)


class Tensor:
    def __init__(self, data):
        if isinstance(data, Tensor):
            self._array = data._array.copy()
        else:
            self._array = np.array(data)

    def to(self, device):
        return self

    def numpy(self):
        return self._array

    def item(self):
        return self._array.item()

    def squeeze(self, axis=None):
        return Tensor(np.squeeze(self._array, axis=axis))

    def detach(self):
        return Tensor(self._array)

    def __iter__(self):
        return iter(self._array)

    def __len__(self):
        return len(self._array)

    def __array__(self):
        return self._array

    def __repr__(self):
        return f"Tensor({self._array!r})"


def tensor(data, dtype=None):
    array = np.array(data, dtype=dtype)
    return Tensor(array)


def argmax(data, dim=None):
    array = data._array if isinstance(data, Tensor) else np.array(data)
    if dim is None:
        return Tensor(array.argmax())
    return Tensor(array.argmax(axis=dim))


class _NoGrad:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return False


def no_grad():
    return _NoGrad()


class Device:
    def __init__(self, kind: str):
        self.type = kind

    def __repr__(self):
        return f"device(type='{self.type}')"


def device(kind: str) -> Device:
    return Device(kind)


class _Cuda:
    @staticmethod
    def is_available() -> bool:
        return False

    @staticmethod
    def device_count() -> int:
        return 0


cuda = _Cuda()


class _Version:
    cuda = None


version = _Version()


class Generator:
    pass


class _Module:
    pass


class _NN:
    Module = _Module


nn = _NN()
