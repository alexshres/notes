# Classification and Regression Trees

## Binary Decision Trees- [Classification and Regression Trees](#classification-and-regression-trees)
  - [Binary Decision Tres](#binary-decision-tres)

### Binary Tree:
* Every tree has 2 or 0 nodes
* Leaf node creates a partition of the input space
* decisions at each node involve a single feature
* for cont. vars, splits always of the form $x_{i} \leq t$
* anything in the leaf node gets the same prediction


### Regression Trees

* decision trees give the partition of $\mathcal{X}$ into regions $\{region R_1, \ldots, R_M\}$
* if $c_m$ is the value of a tree partition then we have that $f(x) = \sum_{m=1}^{M}c_{m}\mathbb{1}(x \in R_{m})$
* how to choose $c_1, \ldots, c_M$
* loss function is $(\hat{y} - y)^{2}
* $\hat{c}_{m} = \text{ave}(y_i | x_i \in R_{m})$
* we don't want to over split where each input has its own leaf node
* can limit by letting the number of terminal nodes $T$ to measure complexity (or depth)
* need a splitting variable and a split point
* to find a splitting point (say we only have one feature):
  * we sort the data points (n total)
  * there are $n-1$ total splitting locations, we just go halfway in between
  * we try each one and see which one minimizes the loss function
* for multiple features:
  * brute force through each feature
  * for each feature find the best split point
  * compare best split points across all features and see which one minimizes the loss
* after this first split we have two regions
* for each region we recursively do this again
* computation cost:
  * sort points: nlogn
  * sort features: dlogd 
* it is a __very exhaustive__ search
* greedy since we are finding the best at each step, not minimizing the "global tree"
* controlling complexity 
  * limit max depth
  * require all leaf nodes to contain min number of points
  * node must have at least a certain number of data points to split
  * __Backward Pruning__ __CART__:
    * build a really big tree: (e.g. every regions have $\leq 5$ points)
    * "__Prune__" the tree back greedily all the way to the root, assessing performance on validation


### Classification Trees

* first we denote the proportion of obseravation in some region $R_m$ with class $k$ by the total number of points in $R_m$ as $N_m$ and using an indicator function, get the count of points in $R_m$ that have the classification of $k$
* we then predict the classification for region $R_m$ or node $m$ as $k(m) = \text{argmax}_{k}\hat{p}_{mk}$