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

For the matrix $C$ when it's stored in data it's stored in row major format so
using the same variables as above:

```python
# shape = [width, height]
A.shape = [n, m]
B.shape = [m, p]
C.shape = [n, p]

# (row * width) + col
A[i, k] = A_arr[i*m + 1*k] = A_arr[i*m + j*0 + k*1]
B[k, j] = B_arr[k*m + 1*j] = B_arr[i*0 + j*1 + k*m]
C[i, j] = C_arr[i*p + 1*j] = C_arr[i*p + j*1 + k*0]

dAdi = m
dAdj = 0
dAdk = 1

A_stride = (m, 0, 1)

dBdi = 0
dBdj = 1
dBdk = m

B_stride = (0, 1, m)
```

To align $A$ for matmul, we need to expand it do include the $p$ dimension of
$B$:

$A_ext \in \mathbb{R}^{n \times p \times m}$
$B_ext \in \mathbb{R}^{n \times p \times m}$



```python
Example:

# shape = [n, m]
A.shape = [2, 3]

# shape = [m, p]
B.shape = [3, 3]

A = [[a0, a1, a2],
     [a3, a4, a5]]

B = [[b0, b1, b2],
     [b3, b4, b5],
     [b6, b7, b8]]

A_ext.shape = [2, 3, 3]

```




