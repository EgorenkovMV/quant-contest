# High-Precision NVFP4 → HiF4 Quantization

## Implementation specification for `solution.py`

## 1. Goal

Implement the strongest possible single-file `solution.py` for the NVFP4 → HiF4 conversion challenge.

The submission must optimize the **actual downstream operation**, not merely minimize elementwise reconstruction error:

* **Linear**

  $$
  \hat{Y} = X_{\text{HiF4}} W_{\text{HiF4}}^T
  $$

  should approximate

  $$
  Y_{\text{ref}} = X_{\text{NVFP4}} W_{\text{NVFP4}}^T
  $$

* **Attention**

  $$
  \hat{O} =
  \operatorname{softmax}
  \left(
  Q_{\text{HiF4}}K_{\text{HiF4}}^T/\sqrt d
  \right)
  V_{\text{HiF4}}
  $$

  should approximate the same expression using dequantized NVFP4 Q/K/V.

The final score is based on improvement over the baseline MSE, so the design should prioritize **operation-aware error** rather than individual Q/W/K/V tensor MSE.

The strongest practical architecture is:

> **distribution transformation → task-aware HiF4 discrete search → calibration-driven state → cheap deterministic online quantization**

Do not treat this as a format-conversion problem in which an NVFP4 scale is somehow translated into HiF4 metadata. Reconstruct the FP32 statistical domain first, then perform a new HiF4 optimization.

---

# 2. Respect the evaluator's HiF4 representation exactly

Do not implement the physical packed HiF4 representation from the hardware paper unless the supplied template explicitly asks for it.

The public checker corresponding to this challenge expects five logical tensors:

```text
scale_factor
scale_lv2
scale_lv3
sign
mant
```

For an original tensor of shape:

```text
(*prefix, C)
```

with `C % 64 == 0`, the expected shapes are:

```text
scale_factor : (*prefix, C//64, 1, 1, 1)
scale_lv2    : (*prefix, C//64, 8, 1, 1)
scale_lv3    : (*prefix, C//64, 8, 2, 1)
sign         : (*prefix, C//64, 8, 2, 4)
mant         : (*prefix, C//64, 8, 2, 4)
```

and reconstruction is:

```text
x_hat = sign * mant * scale_lv3 * scale_lv2 * scale_factor
```

The legality requirements are:

```text
scale_factor ∈ exact E6M2 values in [2^-48, 49152]
scale_lv2    ∈ {1, 2}
scale_lv3    ∈ {1, 2}
sign         ∈ {-1, 0, 1}
mant         ∈ {0, 0.25, 0.50, ..., 1.75}
```

The checker explicitly validates exact E6M2 snapping and these shapes/values.

This logical form corresponds to HiF4's 64-element hierarchy: one global scale per 64 values, refinement per 8 values and per 4 values, and a 4-bit element representation.

### Critical consequence

The optimizer is allowed to select **any legal combination** of these values.

It is **not required** to reproduce the canonical max-based HiF4 Algorithm 1 if a different legal combination gives lower evaluator MSE.

That is the main optimization opportunity.

---

# 3. NVFP4 input handling

Follow the supplied challenge representation, not ordinary packed NVIDIA NVFP4 assumptions.

In the public implementation corresponding to this task, `quant_float` is already a numerical carrier with the same logical shape as the tensor. Dequantization is essentially:

```python
x = quant_float.float().reshape(..., C//16, 16)
x *= scale_float.float().unsqueeze(-1)
x = x.flatten(...)
```

The public helper returns BF16, but all optimization internals should remain FP32 to avoid adding another rounding error before HiF4 quantization.

Therefore:

```text
NVFP4 carrier + block scale
          ↓
FP32 reconstructed tensor
          ↓
statistics / transformations / HiF4 search
```

Never attempt to map an NVFP4 16-value scale onto a HiF4 64/8/4 hierarchy directly.

---

# 4. Submission/API rules

Keep the exact public function signatures supplied by the challenge.

The public checker corresponding to this task expects:

```python
hif4_calibration_and_quantize_weight(...)
hif4_dynamic_quantize_activation(...)
hif4_calibration_attention(...)
hif4_dynamic_quantize_q(...)
hif4_dynamic_quantize_k(...)
hif4_dynamic_quantize_v(...)
```

and the provided `dequantize_nvfp4(...)` helper if it is already present in the template.

Do not alter argument ordering or return structures.

Linear calibration may return only:

```python
{
    "weight_params": ...,
    "activation_state": ...
}
```

Attention calibration may return only:

```python
{
    "q_state": ...,
    "k_state": ...,
    "v_state": ...
}
```

Calibration state should contain only plain data:

* Python `None`, bool, int, finite float, str
* CPU dense Torch tensors
* list/tuple/dict containing these types

The public checker limits state nesting and checks that tensors are on CPU, finite, non-complex, dense, and non-gradient tensors.

Final submission requirements:

```text
one solution.py
no file reads
no file writes
no model-name detection
no test-set lookup
no environment-specific hidden state
no global mutable cache needed for correctness
no custom state classes
no autograd
deterministic output
```

Private numerical helpers inside `solution.py` are fine if the supplied template permits them. Do not add new public APIs.

---

# 5. Core HiF4 quantizer

Everything depends on getting this component right.

## 5.1 Work in blocks of 64

Represent each block internally as:

```text
[8 groups][2 microgroups][4 elements]
```

i.e.:

```python
block.reshape(..., 8, 2, 4)
```

For each 8-element group there are exactly eight legal local hierarchy configurations:

```text
lv2 ∈ {1,2}
left lv3  ∈ {1,2}
right lv3 ∈ {1,2}
```

so:

```text
2 × 2 × 2 = 8 choices
```

Enumerate all eight exactly.

Do not use heuristics for lv2/lv3 if exact enumeration costs only eight candidates.

---

## 5.2 Global E6M2 scale

A block's maximum possible magnitude for a given global scale is:

```text
1.75 × 2 × 2 = 7 × global_scale
```

so a useful initial scale is:

```text
base = max(abs(block)) / 7
```

But `base` itself is usually not MSE-optimal.

Search legal E6M2 values around it.

A proven practical candidate set from the public optimization history is approximately:

```text
base × {
    0.25, 0.35, 0.50, 0.70,
    0.85, 1.00, 1.20, 1.50,
    2.00, 2.50, 3.00, 4.00
}
```

snap every candidate to exact E6M2, evaluate all local hierarchy choices, then refine around the winner with:

```text
winner × {0.75, 0.875, 1.0, 1.125, 1.25}
```

Again snap to E6M2.

Only accept the refinement if it provides a meaningful improvement, e.g. ≥1%. This approach was used successfully in the public challenge experiments.

Do not implement a huge exhaustive search over the entire E6M2 range. It wastes the seven-minute budget.

---

## 5.3 Mantissa selection

For ordinary separable MSE:

```python
mant = round(abs(x) / total_scale * 4) * 0.25
mant = clamp(mant, 0, 1.75)
sign = sign(x)
sign[mant == 0] = 0
```

This is optimal once scale/local hierarchy is fixed for elementwise squared error.

For Hessian/task-aware objectives, nearest mantissa is no longer necessarily globally optimal. However, full signed-codebook coordinate descent should only be used where measurements show worthwhile benefit.

A public experiment added a full 15-value signed codebook sweep under a Hessian objective. It improved its local Hessian objective only about 0.03%, altered less than 1% of blocks, and did not beat the best Attention-focused server result.

Therefore:

> Use full codebook coordinate descent only as an optional late-stage experiment, not in the initial final submission.

---

# 6. Equivalent transformations: the biggest low-cost win

Because the evaluator scores downstream operations, use mathematically lossless FP32 transformations that make tensors easier to represent in HiF4.

## 6.1 Signed orthonormal Hadamard

Apply a fixed orthonormal signed 64×64 Hadamard transform independently inside each HiF4 block.

Conceptually:

```text
H = normalized Hadamard
D = fixed ±1 diagonal matrix
R = D H
```

with:

```text
R R^T = I
```

Use a deterministic sign pattern, not random runtime state.

The public strong implementation uses exactly this idea for Linear and Q/K.

Benefits:

* distributes isolated outliers across 64 dimensions;
* preserves inner products;
* costs only additions/subtractions and one normalization;
* does not require storing full matrices.

Do not materialize a 64×64 matrix. Implement FWHT using six butterfly stages for widths:

```text
1, 2, 4, 8, 16, 32
```

and multiply by `1/8`.

---

# 7. Linear scenario

# 7.1 Offline calibration

Reconstruct:

```text
W   = FP32 NVFP4-dequantized weight
A_i = FP32 NVFP4-dequantized calibration activations
```

Collect per-input-channel statistics:

```text
activation_amax[c]
weight_amax[c]
```

Use SmoothQuant-style reciprocal scaling:

```text
s[c] = activation_amax[c]^alpha /
       weight_amax[c]^(1-alpha)
```

with safe epsilon and clipping such as:

```text
[1/16, 16]
```

A strong public baseline uses:

```text
alpha = 0.65
```

for Linear.

Then transform:

```text
W' = Hadamard(W * s)
A' = Hadamard(A / s)
```

The same orthonormal transform on both sides gives, to FP32 numerical precision:

```text
A' W'^T = A W^T
```

Therefore the transform changes only the difficulty of quantization.

---

# 7.2 Build calibration Hessian

The Linear objective is output error, so weight error should be measured according to how calibration activations use each direction.

For every input-channel block `b` of 64 dimensions:

```text
H_b = A_b'^T A_b' / N
```

Normalize and damp:

```text
d = max(trace(H_b)/64, 1e-8)

Hreg_b = H_b / d + 0.01 I
```

Use FP32.

A block-diagonal 64×64 Hessian was one of the strongest successful Linear improvements in the public experiment history, raising the server score from 22024 to 23254 before the later Attention improvement.

---

# 7.3 Quantize W using the Hessian objective

For one weight-row block:

```text
e = W'_block - W_hat_block
```

optimize:

$$
L_W=e^T H_{\text{reg}} e
$$

instead of plain:

$$
\sum_i e_i^2.
$$

Efficient procedure:

1. Generate the normal weighted-MSE HiF4 candidate.
2. Use that candidate as the permanent fallback.
3. For every global scale candidate:

   * initialize the eight local hierarchy choices using independent group search;
   * perform coordinate descent over the eight 8-value groups;
   * enumerate all 8 local hierarchy configurations for one group;
   * compute the change in full `eᵀHe`;
   * update the chosen group.
4. Two forward passes over the eight groups are enough.
5. Select the best global-scale candidate.
6. Accept Hessian candidate only if it beats the fallback by a meaningful margin, e.g. 1%.

Do not enumerate `8^8`.

The public implementation uses this exact block-Hessian principle.

---

# 7.4 State for online activation quantization

Store at least:

```text
smooth_scale
activation/error importance information
```

A cheap useful activation importance is:

```text
g_diag[c] = Σ_output W_hat[:,c]^2
```

because activation error in a channel is amplified according to the norm of the corresponding weight column.

Better, if memory and runtime allow, store block-diagonal:

```text
G_b = W_hat_b^T W_hat_b
C_b = W_hat_b^T W_b
```

for each 64-channel block.

These give the more accurate online objective:

$$
L(q;x)=q^T G q - 2q^T Cx
$$

up to a constant independent of `q`.

This directly approximates:

```text
|| q W_hat^T - x W^T ||²
```

and therefore explicitly targets the Linear score.

### Recommended implementation decision

Start with `g_diag` only.

Only implement full 64×64 `G/C` online search if profiling shows substantial time headroom. The public diagonal joint-compensation experiment produced no changes on its public Linear sample and expected limited benefit.

Linear is therefore not where the remaining runtime budget should first be spent.

---

# 8. Dynamic Linear activation

For each test activation:

```text
X = dequantize NVFP4 → FP32

X' = Hadamard(X / smooth_scale)
```

Then quantize X' with the ordinary HiF4 candidate search, using:

```text
error_weights = Σ W_hat² over output rows
```

as channel weights.

This turns:

```text
elementwise activation MSE
```

into a cheap diagonal approximation of:

```text
Linear output MSE.
```

Do not perform expensive Hessian/global-coordinate search online unless benchmarks clearly justify it.

Online work occurs multiple times and therefore is much more expensive than calibration-only search.

---

# 9. Attention scenario: highest priority

The most important optimization work should be here.

A public implementation history for this exact-looking challenge reports:

```text
block-Hessian Linear:       23254
Attention V-aware version:  23305
```

while a more sophisticated Linear full-codebook refinement scored only `23255`. The Attention-aware V change therefore produced the best reported result among those versions.

Treat this as strong evidence that final optimization effort should prioritize Attention.

---

# 10. Attention Q/K equivalent transformation

For each KV head and all Q heads mapped to it, construct a common per-channel smooth scale.

If:

```text
q_per_kv = q_num_heads // kv_num_heads
```

then Q heads:

```text
kv_head*q_per_kv :
(kv_head+1)*q_per_kv
```

must share the K head's reciprocal scale.

Use:

```text
Q' = Q / s
K' = K * s
```

or the exact inverse convention consistently.

Then:

```text
Q' K'^T = Q K^T
```

in FP32.

After scaling, apply the **same signed Hadamard transform** to Q and K inside corresponding 64-dimensional head blocks:

```text
Q'' = Q' R
K'' = K' R
```

so:

```text
Q'' K''^T = Q' K'^T
```

Do not independently rotate Q and K.

---

# 11. Do not use one fixed Smooth-QK strength

This should be the principal improvement over the known strong baseline.

HiFA4 reports that Smooth-QK is valuable for models with concentrated K outliers but should be disabled on models without such behavior; mild and strong smoothing are useful in different regimes.

The public challenge optimization history independently proposed moving from fixed `alpha=0.25` to per-KV-head candidate selection.

Implement **calibration-selected Smooth-QK per KV head**.

Recommended candidates:

```text
C0: alpha = 0       / smoothing disabled
C1: alpha = 0.125
C2: alpha = 0.25
```

Add:

```text
C3: alpha = 0.50
```

only when K has a clearly severe stable outlier pattern.

A useful gate from HiFA4 is based on:

```text
rho_K = max(|K|) / P99(|K|)
T_KQ  = max(|K|) / max(|Q|)
```

with strong smoothing considered when roughly:

```text
rho_K >= 6
T_KQ  >= 2
```

Do not rely only on this analytical gate, though. The final candidate should be chosen by **actual calibration HiF4 error**.

---

# 12. Calibration objective for Smooth-QK

For every candidate and KV head:

1. Apply candidate Q/K reciprocal scaling.
2. Apply the common signed Hadamard.
3. Perform the **real HiF4 quantizer**.
4. Reconstruct candidate Q_hat/K_hat from the five output tensors.
5. Measure task-aware calibration errors.

Use three metrics.

## 12.1 Logit NMSE

$$
E_{\text{logit}}
=
\frac{
\|\hat Q \hat K^T-QK^T\|_F^2
}{
\|QK^T\|_F^2+\epsilon
}.
$$

Aggregate all Q heads mapped to the KV head.

---

## 12.2 Attention-output NMSE

Keep V in FP32 for this selection so Q/K quality is isolated.

Compute both:

```text
full attention
causal attention
```

and use approximately:

```text
E_attn = 0.5 * E_causal + 0.5 * E_full
```

Do not form enormous full matrices.

Sample at most roughly 128–256 uniformly spaced query positions per calibration sample, processing them in chunks of 32–64.

The public V implementation used a deterministic 256-query / 64-query-chunk approach and stayed inside the server budget.

---

## 12.3 Plain Q/K reconstruction guard

Also calculate ordinary Q/K MSE.

This prevents an overfitted equivalent transform that slightly improves one calibration attention realization while severely degrading tensor fidelity.

---

# 13. Candidate-selection guard

Never choose an aggressive candidate just because its calibration attention proxy is microscopically lower.

A robust rule is:

```text
candidate attention NMSE <= 0.99 × baseline attention NMSE
candidate logit NMSE     <= 1.0025 × baseline logit NMSE
candidate Q/K MSE        <= 1.01 × baseline Q/K MSE
```

If none pass, retain the fixed baseline.

The public proposed adaptive-head design uses essentially these types of guards.

Additionally, split calibration samples into two deterministic halves.

Require the selected candidate to be non-regressive on **both halves**.

This is much safer than selecting a head-specific parameter from only five calibration examples and trusting a single aggregate metric.

Store only the final selected smooth scale:

```text
q_smooth_scale : [q_num_heads, head_dim]
k_smooth_scale : [kv_num_heads, head_dim]
```

Dynamic Q/K must not rerun the candidate search.

---

# 14. Add task-aware Q/K channel importance if it validates

This is the next experiment after adaptive Smooth-QK.

Once the selected smooth transform is known, calculate statistics in the transformed domain.

For a Q error vector `e_q`, logit error is approximately controlled by:

$$
e_q K^T.
$$

Therefore:

$$
\|e_qK^T\|^2
=
e_q(K^TK)e_q^T.
$$

A cheap diagonal approximation gives:

```text
q_importance[c] = mean(K_transformed[:,c]^2)
```

For K:

```text
k_importance[c] =
mean over mapped transformed Q heads/tokens of Q²
```

Then dynamic Q/K can search HiF4 scales using these channel weights rather than unweighted tensor MSE.

### Safe strategy

Do not automatically enable it globally.

During calibration compare:

```text
unweighted Q/K quantizer
vs
opposite-side-weighted Q/K quantizer
```

using the same logit and attention-output proxy.

Enable weighted Q/K only for heads where calibration shows a robust improvement.

### Runtime rule

Prefer selecting **one quantizer mode in calibration** and running only that mode online.

Avoid generating both weighted and unweighted candidates online, because Q and K online functions are called repeatedly and doubling their search can break the 420-second limit.

---

# 15. Dynamic Q

Online Q should do only:

```text
1. NVFP4 → FP32
2. reshape [seq, q_heads, head_dim]
3. apply selected reciprocal smooth scale
4. apply selected common signed Hadamard
5. one HiF4 search
6. return five tensors
```

If calibration selected Q importance, feed that fixed importance into the search.

No attention matrices should be computed online inside `hif4_dynamic_quantize_q`.

---

# 16. Dynamic K

Exactly symmetric:

```text
1. NVFP4 → FP32
2. reshape
3. multiply by selected K smooth scale
4. identical corresponding Hadamard
5. one HiF4 search
6. return
```

Q/K FP32 equivalence should be tested locally:

```text
max relative error of
Q' K'^T vs Q K^T
```

should be approximately FP32 numerical noise, ideally below `1e-6` to `1e-5`.

---

# 17. V calibration: retain attention-aware weighting

Do not rotate V.

A free rotation of V changes:

```text
P V
```

and there is no inverse output-projection interface in this challenge.

Instead use the calibration attention distribution to determine which V information matters most.

For calibration Q/K:

```text
P = softmax(QK^T / sqrt(d))
```

accumulate a mixture of causal and full attention usage.

A proven public implementation uses:

```text
0.5 causal + 0.5 full
```

and approximately computes squared attention usage:

```text
usage[j] = Σ_queries P[q,j]^2
```

then combines it with V channel energy.

The public winning V-aware experiment applies a compressed/bounded importance approximately equivalent to:

```text
relative energy
→ power 0.25
→ clamp [0.5, 2.0]
→ blend 25% with 1
→ normalize each 64-channel block to mean 1
```

and improved the server result from `23254` to `23305`.

This should be retained unless your own measurements clearly beat it.

---

# 18. Dynamic V

A safe implementation can generate:

```text
baseline = normal HiF4 search
weighted = HiF4 search using V importance
```

and choose the weighted candidate per 64-value block only when:

```text
weighted_error < 0.99 × baseline_weighted_error
plain_error    <= 1.0025 × baseline_plain_error
```

This double search costs runtime, but it is already demonstrated to fit close to the challenge limit and improve score.

If adaptive Smooth-QK calibration pushes the runtime too close to 420 seconds, keep V's dynamic double search and reduce **calibration query count** first. Do not sacrifice the known-good V mechanism before profiling.

---

# 19. Things not to implement

## Do not implement P-Reordering

HiFA4's P-Reordering is promising, but this challenge exposes only Q/K/V quantization interfaces. The external evaluator owns softmax and attention execution.

You cannot change its probability normalizer through these APIs.

Use the Smooth-QK idea; ignore kernel-level P-Reordering.

## Do not start with general 64×64 affine transformations

An earlier public experiment tried richer affine Q/K and Linear transformations and achieved excellent public proxy numbers, but such approaches were later abandoned in favor of simpler Smooth + Hadamard + guarded search.

Rich affine matrices:

* are expensive;
* need much more state;
* risk calibration overfit;
* are numerically fragile;
* complicate GQA;
* can consume most of the time limit.

## Do not perform broad Hadamard seed search

Public experiments explicitly report that pure Hadamard-seed searching caused some Attention regressions.

Use one deterministic transform initially.

## Do not add full-codebook coordinate descent everywhere

Measured benefit was extremely small for the cost.

## Do not optimize only plain tensor MSE

The score is downstream Linear/Attention MSE.

Plain reconstruction is a guard, not the primary objective.

---

# 20. Optional high-risk Linear experiment

Only attempt this after the Attention design is stable and runtime is safely below the limit.

The block-Hessian Linear approximation ignores correlations between different 64-channel blocks.

A higher-order approximation is:

$$
H_{\text{approx}}
=
\operatorname{blockdiag}(H_0,\ldots,H_B)
+
U\Lambda U^T
$$

with a very small rank such as:

```text
rank = 4 or 8
```

For a row error `e`:

$$
L =
\sum_b e_b^T H_b e_b
+
\|\Lambda^{1/2}U^Te\|^2.
$$

This lets quantization decisions in one block account for error cancellation/correlation with other blocks without storing a complete `K×K` Hessian.

A public future-work design for this exact challenge proposes rank 8 and one forward scan.

Treat this as a separate branch.

Do **not** combine it with new Attention modifications in the first evaluation because attribution becomes impossible.

---

# 21. Performance engineering

The seven-minute limit is a first-class optimization objective.

## Required rules

### Vectorize over blocks

Never loop over:

```text
elements
tokens
weight rows
all blocks individually
```

in Python.

Chunk blocks into reasonably large Torch tensors.

Python loops are acceptable only over tiny fixed dimensions such as:

```text
8 local groups
2 Hessian passes
3 or 4 Smooth-QK candidates
small number of KV heads
```

### Keep FP32 only where needed

Use FP32 for:

```text
dequantized values
scale search
statistics
Hessian
attention proxy
loss comparison
```

State tensors can remain FP32 on CPU.

### Avoid unnecessary reconstruction

During search, compute candidate losses directly where possible.

Only materialize complete candidate tensors for the winner or for a small calibration proxy.

### Avoid synchronization

Do not call `.item()` repeatedly in inner loops.

### No autograd

Use detached tensors and optionally `torch.no_grad()` around the full public APIs.

### CPU state

Before returning calibration state:

```python
tensor.detach().cpu().contiguous()
```

### Determinism

No random sampling.

For calibration query subsampling use deterministic uniformly spaced positions.

### Time target

Do not aim for 419 seconds.

Target roughly:

```text
≤ 380–395 seconds
```

on the evaluation-equivalent environment.

The strongest reported public version already required about `369s`, leaving limited margin.

---

# 22. Suggested implementation order

The coding agent should implement and benchmark in this exact order.

## Stage A — correctness

Implement:

```text
NVFP4 FP32 reconstruction
E6M2 snapping
HiF4 reconstruction
legal hierarchy enumeration
basic global-scale search
exact output shapes
state validation
determinism
```

Pass every self-check before optimization.

---

## Stage B — strong base quantizer

Add:

```text
12 global scale multipliers
8 exact local hierarchy combinations
5-candidate scale refinement
guarded replacement
signed Hadamard
```

Validate that every output is legal.

---

## Stage C — Linear

Add:

```text
alpha=0.65 smooth
paired Hadamard transform
activation-second weighted baseline
64×64 block Hessian weight search
```

Keep dynamic activation relatively cheap.

---

## Stage D — Attention baseline

Add:

```text
GQA-aware reciprocal Q/K smoothing
alpha=0.25 baseline
paired per-head Hadamard
V calibration importance
guarded dynamic V
```

This gives a known strong architecture.

---

## Stage E — highest-priority improvement

Replace fixed Q/K smooth strength with:

```text
per-KV-head {off, 0.125, 0.25}
```

and optionally `0.5` behind the strong-outlier gate.

Select using actual HiF4 calibration Attention NMSE plus logit/plain guards.

This is the first extension I would expect to have a realistic chance of improving the strongest public result.

---

## Stage F — second Attention experiment

Add calibration-selected:

```text
Q importance from transformed K²
K importance from transformed Q²
```

but keep only **one** chosen online quantization mode per head.

Evaluate independently from any more complex Linear changes.

---

## Stage G — only if substantial runtime remains

Try:

```text
block-diagonal G/C output-aware Dynamic Activation
```

and finally the low-rank cross-block Linear compensation.

Do not start here.

---

# 23. Calibration cross-validation

Because there are few calibration samples, aggressively prevent overfitting.

For every strategy-selection mechanism:

```text
half A = first calibration subset
half B = second calibration subset
```

A nonbaseline strategy must:

```text
improve aggregate calibration objective
not regress materially on half A
not regress materially on half B
pass plain reconstruction guard
```

If statistics are degenerate or non-finite:

```text
fallback to baseline
```

Do not tune constants using the public test samples.

---

# 24. Local evaluation metrics

Your local coding agent should calculate, outside the final `solution.py`, at least:

## Linear

```text
MSE(Y_HiF4, Y_NVFP4)
relative improvement
per-test-case MSE
```

and optionally:

```text
Weight MSE
Activation MSE
Hessian loss
```

but the output MSE is the decision metric.

## Attention

Record separately:

```text
Q MSE
K MSE
V MSE
QK logit NMSE
softmax probability error
causal Attention output NMSE
full Attention output NMSE
```

Do not accept an optimization merely because Q/K/V MSE improves if Attention output becomes worse.

---

# 25. Ablation table to maintain

For every candidate submission maintain:

```text
Version
Linear output MSE
Attention causal MSE
Attention full MSE
worst test regression
self-check pass/fail
total runtime
```

A feature should remain in `solution.py` only if it has a clear isolated benefit.

Suggested experiment ladder:

```text
B0  plain legal HiF4
B1  + scale search
B2  + Smooth/Hadamard
B3  + Linear Hessian
B4  + V importance
B5  + adaptive Smooth-QK
B6  + Q/K opposite-side importance
B7  + output-aware activation G/C
B8  + low-rank Linear correction
```

Never introduce B6+B7+B8 simultaneously.

---

# 26. Safety/fallback behavior

Every advanced algorithm needs a baseline candidate.

For every 64-value block/head:

```text
baseline parameters must always remain available
```

Reject candidate when:

```text
loss is NaN/Inf
scale invalid
statistics degenerate
candidate improvement below threshold
plain MSE exceeds guard
cross-validation half regresses
```

For zero blocks:

```text
mant = 0
sign = 0
legal minimum/global scale
```

or preserve the baseline representation.

For exact-representable values, do not perturb them merely to satisfy a more complex optimization.

---

# 27. Important numerical details

Use epsilon around:

```text
1e-8 to 1e-12
```

depending on normalization.

Clamp Smooth scales:

```text
[1/16, 16]
```

unless local experiments demonstrate a safer fixed interval.

Ensure E6M2 snapping returns exact FP32-representable legal values.

After snapping, always evaluate the candidate using the **snapped value**, never the unsnapped theoretical scale.

Do all candidate comparisons using reconstructed legal HiF4 output.

Never score an internal continuous approximation and then emit a different snapped representation.

---

# 28. What the final `solution.py` should conceptually contain

The final file should remain conceptually simple:

```text
constants

NVFP4 reconstruction helper

E6M2 snap/search helpers

HiF4 search/reconstruction helpers

Hadamard transform helper

Linear calibration:
    dequant
    stats
    smooth
    transform
    Hessian
    quantize W
    create activation state

Dynamic activation:
    dequant
    transform
    one task-aware HiF4 quantization

Attention calibration:
    dequant calibration samples
    collect Q/K stats
    evaluate Smooth-QK candidates
    compute optional Q/K importance
    compute V attention importance
    return CPU states

Dynamic Q:
    dequant
    selected scale
    Hadamard
    one quantization

Dynamic K:
    dequant
    selected inverse scale
    Hadamard
    one quantization

Dynamic V:
    dequant
    baseline/importance quantization
    guarded selection
```

No external state or I/O should be necessary.

---

# 29. Recommended final strategy

If development time is limited, the version I would prioritize is:

```text
CORE
✓ exact legal HiF4 discrete search
✓ E6M2 neighborhood refinement
✓ FP32 reconstruction

LINEAR
✓ alpha=0.65 Smooth
✓ signed 64D Hadamard
✓ block-Hessian Weight quantization
✓ cheap Weight-aware Dynamic Activation

ATTENTION
✓ GQA-aware equivalent Q/K transform
✓ signed per-head Hadamard
✓ calibration-selected Smooth-QK per KV head
    alpha ∈ {0, 0.125, 0.25}
    alpha=0.5 only behind severe-outlier gate
✓ direct Attention-output candidate selection
✓ causal + full calibration proxy
✓ V attention-usage importance
✓ guarded Dynamic V

OPTIONAL
? calibration-selected Q/K opposite-side importance

DO NOT SHIP INITIALLY
✗ arbitrary affine matrices
✗ online attention candidate search
✗ broad Hadamard seed search
✗ full-codebook sweep everywhere
✗ low-rank Linear GPTQ before Attention is complete
```

The underlying principle should be:

> **Spend expensive computation during calibration, then make the online Q/K/Activation paths execute only one preselected transformation and one HiF4 search.**

That gives the best chance of increasing accuracy while remaining below the seven-minute execution limit.

---

# 30. Final acceptance checklist for the coding agent

Before producing the final `solution.py`, verify all of the following:

```text
[ ] exactly one submission file
[ ] no file I/O
[ ] no test-data access
[ ] exact public signatures
[ ] exact required top-level return keys
[ ] every HiF4 tensor has correct shape
[ ] every global scale is exact legal E6M2
[ ] lv2/lv3 contain only 1 or 2
[ ] mant is exact multiple of 0.25 in [0,1.75]
[ ] sign is -1/0/1
[ ] all states contain only allowed plain CPU data
[ ] all outputs finite
[ ] repeated execution is deterministic
[ ] FP32 equivalent transforms preserve Linear/QK products
[ ] GQA head mapping is correct
[ ] all advanced candidates have baseline fallbacks
[ ] calibration selection does not use test data
[ ] local Linear output MSE measured
[ ] local causal Attention MSE measured
[ ] local full Attention MSE measured
[ ] worst-case test regression inspected
[ ] full self-check passes
[ ] runtime has comfortable margin below 420 s
```

The most important experimentation order is:

> **Adaptive per-head Smooth-QK first, calibration-aware Q/K importance second, more expensive Linear improvements last.**

That ordering is supported both by the public challenge experiment history and by recent HiF4-specific Attention research.
