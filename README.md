# Tensor Type
[![PyPI version](https://badge.fury.io/py/tensor_type.svg)](https://badge.fury.io/py/tensor_type)

Annotates shapes of PyTorch Tensors using type annotation in Python3, and provides optional runtime shape validation.

This comes in very handy when debugging complex programs that manipulate huge `torch.Tensor`s where shape (dimensions) vary widely and are hard to track down.

I got tired of writing tons of `assert my_tensor.shape == (batch, channels, height, width)` over and over, so I made that utility, but then
I got tired of copy/pasting it into every new projects from my Gist of it, so here I finally made it a library that I can pip install everywhere.

## Getting started

```sh
pip3 install tensor_type
```

`tensor_type` only works with PyTorch, but that's only because I make the annotation type inherit from `torch.Tensor` to satisfy static annotations.


## Usage
```python3
from tensor_type import Tensor, Tensor3d, Tensor4d
import torch

# You can use the type in function's signatures

def my_obscure_function(x: Tensor4d) -> Tensor3d:
    return x.sum(dim=3)/x.mean()

t = my_obscure_function(x=torch.rand(3,2,4,2))

# You can check the shape with an explicit assert
assert Tensor3d(t)

# Or you can check it with the .check() method which will produce a nicer error message
Tensor3d.check(t)

# Check specific shape
assert Tensor[3, 2, 4](t)

# This will match no matter the size of the second axis
assert Tensor[3, :, 4](t)

batch = 64
channels = 3
h, w = 256, 512

# You can statically annotate the shape like so
# This WILL NOT be checked at run time, it's just for clarity

my_tensor: Tensor[batch, channels, h, w] = load_images(...)

# You can assert it later like so:
assert Tensor[batch, channels, h, w](my_tensor)

# You can define new "types" like so:
ImageBatch = Tensor[64, 3, :, :]

# And then use the new type
assert ImageBatch(torch.rand(64, 3, 256, 256))
assert ImageBatch(torch.rand(64, 3, 512, 512))
assert not ImageBatch(torch.rand(64, 1, 256, 256))
```

## Development

To install the latest version from Github, run:

```
git clone git@github.com:sam1902/tensor_type.git tensor_type
cd tensor_type
pip3 install -U .
```
