# NumPy and Torch `as_strided` and `sum` 



### Outer Product

Outer product between two vectors $a \in \mathbb{R}^{n}$ and $b \in
\mathbb{R}^{m}$ will result in a matrix $M \in \mathbb{R}^{n \times m}$.

The code as a for loop looks like this:

```python

for i in range(n):
    for j in range(m):
        M[i, j] = a[i + 0*j] + b[0*i + j]
```

Our index equations are thus:
$$
    \text{id}_{x_{a}} = i + 0 * j           \\
    \text{id}_{x_{b}} = 0 * i + j
$$

If we differentiat, we get $\partial id_{a}$ wrt $i$ being $1$ and $\partial
id_{a}$ wrt $j$ being $0$. Similarly for $\text{idx}_{b}$ we get $\partial id_{b}$ wrt $i$ being $0$ and $\partial id_{b}$ wrt $j$ being $1$.

So our code looks like this:

```python
n = 3
m = 4

a = torch.randn(n)
b = torch.randn(m)

outer_act = torch.outer(a, b)

# one size for both
size = (n, m)

stride_a = (1, 0)
stride_b = (0, 1)

outer_our = torch.as_strided(u, size, stride_a) * torch.as_strided(b, size,
stride_b)

torch. testing.assert_allclose(outer_act, outer_our)
```


### Matrix Multiplication

Say we have matrices $A \in \mathbb{R}^{n \times m}$ and $B \in \mathbb{R}^{n
\times p}$, then $C=AB \in \mathbb{R}^{m \times p}$. The pseudocode for matmul
looks something like:

```python

for i in n:         # loop through A's rows
    for j in p:     # loop through B's columns
        for k in m: # shared dimension (A's columns in ith row and B's rows in
        jth column)
            C[i, j] = A[i, k] + B[k, j]
```



