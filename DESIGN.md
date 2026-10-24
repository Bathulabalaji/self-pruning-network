# Design

The network is an MLP, `784 → 128 → 64 → 10`, trained on MNIST with Adam (`lr = 1e-3`, batch 128, 10 epochs). Hidden layers are ReLU. The loss is fused log-softmax cross-entropy. Weights use He normal initialization, `std = sqrt(2 / fan_in)`, because a ReLU zeros about half of a symmetric pre-activation and the next layer would otherwise see half the variance. Biases start at zero. Biases are never pruned.

## Why the keep-score approximates the loss change

Let `w` be one connection and `e` the unit vector for that coordinate. Removing it replaces `w` with `0`, a step of `Δw = -w`. The first-order Taylor expansion of the loss is

```
L(w - w e) - L(w) = -g w + (1/2) w² H + higher-order terms
```

where `g = ∂L/∂w` and `H` is the corresponding diagonal Hessian entry. The linear term `-g w` is the predicted change in loss. Its absolute value `|w g|` is the size of that change when the sign of a minibatch gradient cannot be trusted.

`g` here is the gradient of the loss with respect to the masked product `W ⊙ M`, taken before the mask multiplies it again. That is the derivative through the value that actually entered the matmul. The parameter gradient `∂L/∂W = g ⊙ M` is the wrong tensor for this score: on a pruned connection it is identically zero, so `|w g_param|` cannot rank anything that is already gone, and it is not the sensitivity of the forward value.

The training score is an exponential moving average of `|w g|` with keep rate `0.9`, updated after every backward and before the optimizer step, so `w` is the weight that produced `g`. Masked positions are stored as zero and are not candidates to return. There is no regrowth.

The remainder `(1/2) w² H` is dropped. That term dominates when `g ≈ 0`, which is what happens at a minimum. A one-shot `|w g|` prune at the end of training would therefore rank connections by noise, while `|w|` would still mean something. The cubic schedule exists so the score is used while gradients are still large. Sparsity follows Zhu and Gupta,

```
s(t) = s_f + (s_i - s_f) (1 - (t - t_start) / (t_end - t_start))³
```

with `s_i = 0`, `t_start` at the end of epoch 1, and `t_end` at 75% of the steps. The mask is updated every 100 steps inside that window, globally across every weight entry, and then frozen. The last quarter of training only fine-tunes the survivors.

Magnitude, `|w|`, is the baseline. It ignores `g`. On this run it is the better rule. At 90% target sparsity, across seeds 0, 1, and 2:

| Criterion | Test accuracy |
| --- | --- |
| Taylor | 0.9758 ± 0.0004 |
| Magnitude | 0.9784 ± 0.0009 |
| No pruning | 0.9772 ± 0.0006 |

The Taylor–magnitude gap is larger than the sum of the two seed standard deviations, so it is not seed noise at `n = 3`. It is also small, about 0.26 percentage points. A likely reason the extra gradient term did not help: by the time the cubic schedule is removing most connections, `g` is already small, and multiplying by it mostly jitters the magnitude ranking. The global ranking also spends the budget unevenly. On the seed-0 Taylor model the first layer is 93.3% sparse, the second 54.8%, and the output layer 20.9%.

At 95% both criteria drop (Taylor 0.9723 ± 0.0004, magnitude 0.9744 ± 0.0015). Through 75% neither gives much up relative to the dense model.

## The gradient of a masked weight

The forward pass of a layer is `Y = X @ (W ⊙ M) + b`, with `M` a constant tensor of exact zeros and ones. `M` does not require grad. By the chain rule

```
∂L/∂W = ∂L/∂(W ⊙ M) ⊙ M
```

so a masked entry has parameter gradient 0. A finite-difference probe agrees: perturbing that stored weight does not change the loss. The pre-mask tensor `∂L/∂(W ⊙ M)` is saved on the weight as `saliency_grad` and is what the Taylor score reads. The optimizer never reads it.

Zeroing the stored weight without a mask is not the same thing. For `Y = X @ W` the gradient `Xᵀ ∂L/∂Y` does not depend on the value of `W`. A weight that happens to be zero still receives a nonzero gradient, and Adam will move it. The tests lock both facts.

While `M_ij = 0`, Adam does not update that index. The weight, the first moment, the second moment, and a per-connection step counter are set to zero. They are not decayed. Decay would be the wrong treatment of "this is not a parameter": `m ← β1 m` keeps a direction, and `v ← β2 v` keeps a scale that no longer matches it. On the next step those stale moments would be applied if the mask opened, or they would sit until a later revival and then fire.

Opening the mask does not restore the old moments. The connection is a new parameter. Its next update uses step `t = 1` and the bias correction `1 - β^1`, which is the same update Adam would apply to a freshly created scalar with the same gradient. A global clock would divide by `1 - β1^T` for a large `T`. After the moments have just been reset, that factor is about `1 - β1`, so the first revived step is several times too small. The revival test compares the pruned-then-revived coordinate with a one-step Adam on a new parameter, and with the closed form of that step.

After every masked update the weight is multiplied by the mask again, so a connection cannot drift off zero.

## Where the tape is slow

Each op allocates NumPy arrays and a Python closure. A backward walks those closures one at a time and builds more arrays to unbroadcast gradients. The matmuls themselves are small: the widest product in this network is `(batch, 784) @ (784, 128)`. At batch 128 the whole autodiff forward is about 1 ms, close to a dense GEMM, because the interpreter overhead is hidden behind a few BLAS calls. That will not stay true for a deeper net or for a fused kernel that should have been one launch.

What I would change, in order:

1. Drop the tape at inference. The forward values do not need parents. `prune/sparse.py` already does this.
2. Fuse the MLP backward into one pass that never materializes the ReLU and bias nodes as separate tensors.
3. Store pruned weights as blocked sparse or CSR and contract only those blocks. A Python loop over the 128 output units, each a short dot, is the kernel that was timed. It does the right flops and loses on the clock.

The cost file is the evidence. For the 90% Taylor nonzero pattern (10,918 of 109,184 weights), a batch of 128 is 2,795,008 sparse multiply-adds against 27,951,104 dense ones. That is a real 10× cut in arithmetic. Wall clock on this machine:

| Batch | Gather kernel | Dense GEMM, zeros included |
| --- | --- | --- |
| 1 | 1.47 ms | 0.20 ms |
| 128 | 3.80 ms | 0.85 ms |
| 1024 | 40.6 ms | 3.09 ms |

The dense number multiplies structural zeros. It is not a sparse speedup, and it is still faster. The layers are too narrow for this gather loop to beat BLAS. I would not ship the Python kernel as a latency win.

## Serving this in a multi-tenant inference service

Each tenant gets an immutable snapshot of their pruned weights, packed once at load into a blocked-sparse or CSR matrix per layer. The mask does not change on the request path, and the autodiff tape is not built. Activations stay dense. Requests are batched with other requests for the same weight snapshot and the same shape, not across tenants whose sparsity patterns differ, because a mixed batch cannot share one sparse kernel launch.

Admission control uses the theoretical FLOPs, `2 * nnz * batch` per layer, plus the dense activation traffic. That is the cost that actually scales with the prune. It is not a promise about milliseconds. The measurement above is the reason: a 10× FLOP cut raised latency with the kernel we have. A tenant SLO would be set from a benchmark of the production kernel on that tenant's nnz pattern, and a model would be rejected or sent to the dense GEMM path when the sparse kernel does not win. At a few thousand requests per second the win, if it comes, comes from a fused sparse GEMM and from packing many same-shaped requests into one launch, not from skipping zeros inside a Python loop.

Regrowth is not in the serving path. Turning a connection back on changes the sparsity pattern, the packed matrix, and the Adam state, which belongs in training. A loaded model is a fixed nonzero set.
