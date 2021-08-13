"""
WARNING: This file contains advanced Python black magic, it's essentially meta-programming.
If you're not too familiar with Python's inner workings, don't touch anything.
"""
from typing import Tuple
from torch import Tensor as PT_Tensor


class MetaTensor(type):
    """Ok, so bare with me, it's a wee bit weird:
    According to https://stackoverflow.com/questions/12447036/why-getitem-cannot-be-classmethod
    When I call Tensor[*args], it doesn't resolve to Tensor.__getitem__ but instead to type(Tensor).__getitem__
    By default, type(AnyClass) == type, where AnyCall is just a class declared like class AnyClass: ...
    (aka inheriting from object).
    So the only way to overwrite __getitem__ in Tensor[*args] is to make a metaclass, and set Tensor's metaclass to it.
    Once __getitem__ gets called, its argument
    """

    @classmethod
    def __getitem__(mcs, shape: Tuple):
        # Makes a "fake type" that the type annotation system won't mark as an error
        # Also, make it inherit from Pytorch's Tensor type so that it doesn't think it's the wrong type
        # [:] == [slice(None, None, None)]

        if not isinstance(shape, tuple):
            shape = (shape,)
        shape = tuple(d if d is not None else slice(None, None, None) for d in shape)
        dynamicAnnotationType = type(
            f"Tensor{len(shape)}d", (_Tensor, PT_Tensor, object), dict(type_shape=shape)
        )

        return dynamicAnnotationType


# Public
class Tensor(metaclass=MetaTensor):
    pass


# Private
class _Tensor:
    """A type annotation that can be used to keep trace of the size of each dimension.

    It can be used like so:
    thingy: Tensor[batch, channels, iterations + 1, height, width] = weird_function_that_does_things(x)
    Here we know that thingy has size `iterations + 1` along the third dimension.

    This type doesn't do anything or check anything at runtime, it's just for clarity.

    When you don't know the size of an axis ahead of time, you can use ":" instead like [a, b, :, c],
    or None like [a, b, None, c].
    """

    type_shape: Tuple

    def __new__(cls, *args, **kwargs) -> bool:
        """Hacky way to make the Tensor[1,2,3](my_tensor) syntax work"""

        return all(map(cls._check, args)) and all(map(cls._check, kwargs.values()))

    @classmethod
    def check(cls, *args, **kwargs):
        for i, arg in enumerate(args):
            if not cls._check(arg):
                raise TypeError(
                    f"Bad shape for args {i}. Should be {cls.type_shape} but was {arg.shape}"
                )

        for k, v in kwargs.items():
            if not cls._check(v):
                raise TypeError(
                    f"Bad shape for {k}. Should be {cls.type_shape} but was {v.shape}"
                )

    @classmethod
    def _check(cls, x) -> bool:
        """Checks that the number and size of dimensions are respected. You can pass this to assert"""
        valid = x.ndim == len(cls.type_shape)

        if not valid:
            return False

        for real_dim, type_dim in zip(x.shape, cls.type_shape):
            if isinstance(type_dim, slice):
                continue
            valid &= real_dim == type_dim

        return valid


# It's just to make it clear in the code

Tensor5d = Tensor[:, :, :, :, :]  # Yes, that one get used.... ugh
Tensor4d = Tensor[:, :, :, :]
Tensor3d = Tensor[:, :, :]
Tensor2d = Tensor[:, :]
Tensor1d = Tensor[:]
