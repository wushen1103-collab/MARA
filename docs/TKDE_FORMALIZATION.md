# MARA Formalization for the TKDE Version

## 1. Structured reliability evidence

For an input instance `x`, MARA represents reliability evidence as

$$
\mathbf r(x) = [r_1(x),\ldots,r_K(x)]^\top \in [0,1]^K,
$$

where the generic axes are representation support, source/domain support,
supervision reliability, distribution frontier, and model disagreement. A
domain-specific feature, such as Tanimoto support in molecular prediction or
graph-neighborhood support in node classification, is an implementation of a
generic axis rather than a new axis definition.

The risk head is

$$
\hat p_f(x)=\sigma\left(b+\sum_{j=1}^{K} w_j r_j(x)\right),\qquad w_j\ge 0,
$$

with optional nonnegative pairwise interactions. Nonnegative weights make the
orientation explicit: increasing risk evidence cannot reduce the estimated
failure probability.

## 2. Local non-identifiability of scalar compression

**Proposition 1 (local mechanism non-identifiability).** Let `K > 1` and let
`f: U subset R^K -> R` be continuously differentiable on an open set `U`. At
any regular point `r_0` where `grad f(r_0) != 0`, the level set

$$
\mathcal L_{r_0}=\{\mathbf r\in U:f(\mathbf r)=f(\mathbf r_0)\}
$$

is locally a `(K-1)`-dimensional manifold. Therefore, infinitely many distinct
reliability-evidence vectors have the same scalar score in a neighborhood of
`r_0`.

**Justification.** This follows directly from the implicit function theorem.
At a regular point, one coordinate can be expressed locally as a differentiable
function of the remaining `K-1` coordinates while preserving the value of `f`.

For the common linear compression `f(r)=a^T r`, the ambiguity is explicit:
for every nonzero `v` in the null space of `a^T`,

$$
f(\mathbf r+t\mathbf v)=f(\mathbf r)
$$

whenever both points remain in the domain. This proposition does not claim that
every pathological mapping from `R^K` to `R` is non-injective. It states the
relevant result for continuous or differentiable scalar uncertainty functions
used in prediction systems.

**Implication.** A scalar score can rank overall failure risk, but it cannot in
general recover which combination of support, source, supervision, frontier,
and conflict evidence generated that risk. MARA retains the structured vector
and reports an axis decomposition in addition to the scalar failure estimate.

## 3. Monotonic invariance of equal-rank fusion

Let `s_j(x)` be component risk estimators and let `R_j(x)` be their empirical
fractional ranks over an evaluation batch. The frozen equal-rank score is

$$
S_{\mathrm{rank}}(x)=\frac{1}{J}\sum_{j=1}^{J}R_j(x).
$$

**Proposition 2 (strictly monotonic invariance).** If each component is
transformed by a strictly increasing function `g_j`, then equal-rank fusion is
unchanged:

$$
S_{\mathrm{rank}}(s_1,\ldots,s_J)
=S_{\mathrm{rank}}(g_1(s_1),\ldots,g_J(s_J)).
$$

**Proof.** A strictly increasing transformation preserves every pairwise order
within component `j`, and therefore preserves its empirical ranks `R_j`.
Averaging unchanged ranks gives the same fused score.

This property explains why frozen rank fusion can transfer across predictors
whose uncertainty estimators have incompatible numerical scales. It does not
make the score invariant to non-monotonic transformations or to changes in the
relative ordering of examples.

## 4. Complexity

Let `n` be the reference-set size, `q` the query-set size, `d` the
representation dimension, `p` the number of axis features, `K` the number of
axes, `J` the number of fused risk components, `m` the calibration-set size,
and `I` the number of optimizer iterations.

| Component | Time | Working memory | Notes |
|---|---:|---:|---|
| Exact support search | `O(n q d)` | `O(nd + qd)` with chunking | A dense all-pairs matrix would require `O(nq)` additional memory and is not used. |
| HNSW support index | empirical `O(n log n)` build, `O(q log n)` query | `O(nd + nM)` | Average empirical behavior; no worst-case logarithmic guarantee is claimed. `M` is graph connectivity. |
| Source-frequency axes | `O(n+q)` | `O(G)` | `G` is the number of source groups. |
| PCA/frontier axes | `O(ndc + qdc)` | `O((n+q)c + dc)` | `c <= d` retained components; randomized PCA is used in large runs. |
| Graph propagation views | `O(L |E| d)` | `O(|E| + nd)` | `L` sparse propagation steps. |
| Nonnegative MARA fitting | `O(I m p)` | `O(mp)` | The implemented L-BFGS-B head has five axis scores in the cross-domain suite. |
| MARA inference and attribution | `O(qp + qK)` | `O(qK)` | Linear in the number of records and evidence features. |
| Equal-rank fusion | `O(J q log q)` | `O(Jq)` | Sorting-based empirical ranks. |

The measured scalability experiment complements this analysis because HNSW
costs depend on data geometry and implementation constants. It reports wall
clock, peak resident memory, per-sample latency, exact-neighbor recall, and the
failure-AUROC difference between exact and approximate support estimates from
10,000 to 1,000,000 records.

## 5. Missing evidence

For an availability mask `m in {0,1}^K`, modular inference uses only observed
axes and rescales the observed evidence mass:

$$
z_m(x)=b+\frac{K}{\max(1,\sum_j m_j)}
\sum_{j=1}^{K}m_j w_j r_j(x).
$$

This estimator preserves the expected evidence magnitude under missing
completely at random, but it is not guaranteed to be optimal when missingness
is informative. The experiments therefore compare it with zero imputation,
explicit missing indicators, and axis-dropout training at 10%, 30%, 50%, and
70% missing evidence. Attribution scores are reported both against all injected
causes and against the subset of causes whose evidence remains observable.
