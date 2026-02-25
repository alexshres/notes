# PyTorch Internals

Notes from [ezyang](https://blog.ezyang.com/2019/05/pytorch-internals/).


## Tensor
Tensors are an n-dimensional data structure containing some sort of scalar
type. Generally a Tensor class contains the data, and the metadata: the size of
the tensor, shape of tensor, type of elements (`dtype`), and device the tensor lives on. Another additional metadata included is the stride. 

__Strides__ are an important part of how tensors can be represented. A `2x2`
matrix can be stored contiguously in memory in row major order:

```
MATRIX M
1   2 
3   4


M IN MEMORY:
0x10: 1
0x14: 2
0x18: 3
0x1C: 4 

MATRIX A
1   2   3
4   5   6
```

Here, the size of our matrix is `size=[2,2]` and our stride is `stride=[2,1]`.
To get `M[i,j]` from memory, we do the following operation: `i*stride[0]
+ j*stride[1]`.


### Tensor Representation
When we look at a subset of a tensor, we don't create a new tensor, we just
return a tensor which is a different view on the underlying data. For example,
if we do `M[1:]` we are "viewing" the first row of our matrix but the view is
of the original matrix. If we update something in the view, that update will be
reflected back in M. If we did `M[:1,2].T` we are first getting the matrix:
```
M[:1,2]
2   3
5   6

M[:1, 2].T
2   5
3   6
```

Where the final view just rearranges the data using the stride feature but the
elements are still stored in the same exact memory location.

To implement this functionality of having multiple views on a tensor, we need
to decouple the notion of the tensor (user-visible concept) and the actual
memory location of the data of the tensor - also known as __storage__.

Here's a picture from the blog:
![Storage and Tensor relationship](./imgs/decoupled_tensor.png)



