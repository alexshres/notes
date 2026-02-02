# Data Parallelism


__TLDR__: If the model can fit onto a single GPU then we send the model to each
server and split the data up across the servers


_Question_: How can we know that a model can fit on a GPU along with the entire
data? What goes into understanding if a model can fit - I'm guessing it's just
about the number of weight params in the model + some overhead but there
could be other things?

Side note: __model parallelism__ would be splitting up the model's layers
across servers and data is copied across but a hybrid model is also possible.


Node and GPU can be used interchangeably.

Setup:
* Cluster: 2 computers each having 2 GPUs - 4 nodes total
* Model's weights are initialized on one node and sent to all the other nodes
  via a __broadcast__
* Each node trains the same model with the same initial weights on a subset of
  dataset
* Every few batches, gradients of each node are accumulated on one node  -summed up (and averaged?) - and then sent back to all the other nodes (__all-reduce__)
* Each node updates the parameters of its local model with the gradients
  received using its own optimizer
* back to step 2

_Question_: can one server or even node have different proportions of the total
data? i.e. split is `[.25, .25. .10, .4]`?

Sequence of Reduce and Broadcast are implemented as a single operation
(All-Reduce)

### Point-To-Point (P2P) Communication

P2P communication means each node talks to each other node and if we want to
one node to communicate to all nodes then this happens sequentially (one node
at a time).

Let's go through an example. Say we have 7 frends and we want to send a file
to all our other friends. Let's number our friends from 0-7 where we are the
$0^{th}$ friend and our last friend is the $7^{th}$ friend - this will make
more sense in the next section. Let's say our internet communication is `1 MB/s` and the file is `5 MB` in size. Through P2P:

* we send the file to our first friend and that takes 5 seconds
* we send the file to our second friend which takes 5 seconds
* $\vdots$
* we send the file to our last friend which takes 5 seconds

In total this takes 35 seconds. Now we could ask why not send the file to all
our friends at once. This may work but we are still limited by our internet
connection/bandwidth. We can only send `1 MB/s` so we send `~143 KB/s` to all
our friends every second but to send the whole thing will still take 35
seconds.

### Collective Communication: Broadcast

Collective communication involves everyone in a group, where a single command
or operation issued affects all simultaneously. Sounds magical doesn't it.
Libraries like NCCL (Nvidia) assigns a unique ID to each node known as
__RANK__. Let's go back to our previous example of wanting to send `5 MB` of
data with an internet speed of `1 MB/s`. How would collective comm. work with
broadcast:

* We send our `5 MB` to say our fourth friend: __total_time__: `5 seconds`;
  __friends with file__: 1
_Note_: Now we have a copy and our fourth friend has a copy, what if now we
both are sending the file.
* We send the file to our second friend and our fourth friend sends to sixth
  friend: __total time__: `10 seconds`; __friends with file__: 3
* we send to 1, 2 sends to 3, 4 sends to 5, 6 sends to 7: __total time__: `15
  seconds`; __friends_with file__: 7 (all)

It took us `15 seconds` to send our files to all our friends with collective
communication (this strategy is also known as __Divid and Conquer__). We
exploit the interconnectivity between nodes to avoid idle times and reduce the
total communication times.

_Question_: How do we check if a node has received the data yet or not?


### Collective Communication: Reduce Operation


Going back to calculating gradients and passing it to a source node. Say we
have 8 nodes and each node has calculated its gradients, how can we use
collective comms to send the gradients back? We can have every other node send
their gradients to one of the remaining nodes and the remaining nodes job is to
sum up the gradients. From just the set of receiving nodes we recursively do
this process until only one node is left. Example with 8 nodes:

* 1 sends to 0 -> accumulates
* 3 sends to 2 -> accumulates
* 5 sends to 4 -> accumulates
* 7 sends to 6 -> accumulates

From the set `[0, 2, 4, 6]`
* 6 sends to 4 -> accumulates
* 2 sends to 0 -> accumulates

From the set `[0, 4]`:
* 4 sends to 0 -> accumulates

Now 0 has the sum of all gradients from the 8 nodes and can broadcast it back
to the rest of the nodes. Takes $\log n$ number of steps (basically same as
binary search)


### Collective Communication: All-Reduce

The sequence of __Reduce-Broadcast__ is implemented by another operator known
as __All-Reduce__ which is typically faster than the sequence of reduce
followed by broadcast.


### Failover: What if One Node Crashes?

Use of __checkpointing__: we save the weights of the model on a shared disk
every few iterations (maybe every epoch) and resume training from the last
checkpoint in case there's a crash.


#### Who saves the checkpoint?

PyTorch randomly assigned a unique ID or RANK to each GPU. We write our code in
such a way that whichever node is assigned the RANK 0 will be responsible for
saving the checkpoint - this way the other nodes do not overwrite each other's
files. Only one node is responsible for writing the checkpoints and all the
other files we need for training.

