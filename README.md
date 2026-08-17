# Adaptive Parallel Insertion Merge (PIM)

A stable, adaptive parallel sorting engine for relational database and
bioinformatics workloads, written in Rust.

The engine is built on one observation: production sort keys are rarely random.
They arrive as long monotone stretches, as concatenations of already-sorted
batches, or as categorical values with few distinct keys. A general-purpose
sort discards that structure and pays `n log n` anyway. This engine measures the
structure first, then chooses a merge strategy that exploits it — and steps
aside when there is nothing to exploit.

Its largest gains are on **keys behind a pointer** (`&str`, `&[u8]`, indices
into an arena) with **prior grouping** — the shape of `ORDER BY` on a VARCHAR
column, of sequence identifiers, and of barcode or read-ID sorting.

---

## Academic background

The algorithm derives from:

> **Parallel Insertion Merge**
> Fernando Belmiro do Couto (Divisão de TI, BBDTVM) and Fábio Silva do Couto (UFRJ)
> PDPTA — *International Conference on Parallel and Distributed Processing
> Techniques and Applications*

Three ideas come directly from that paper, and all three were modified here.

**The initial scan (Task 0).** The paper maps the input into signed indices —
positive for an ascending run, negative for a descending one — and stitches
per-block indices together. This implementation keeps that representation
exactly: `detect_global_trend` on the paper's own example `[2 4 6 3 1 9 8 5 7 10]`
produces `[3, -2, -3, 2]`, the figure printed in the paper. What changed is the
stitching: blocks overlap by one element and their metadata is combined by an
associative operation, so the join is a monoid under a parallel reduce and the
array is never read twice.

The paper's average-case analysis states that a random distribution yields
about `2N/5` indices, with 100,000 Monte Carlo trials measuring 41.318%. That
constant has a closed form. Since `P(run ≥ k) = 2/k!`, the expected run length
is `1 + 2(e − 2) = 2e − 3`, so the index density converges to
**`1/(2e − 3) = 41.0415%`**. The engine measures 41.35% on 10M random `u64`.

**Block insertion by search (Section 6).** The paper inserts blocks of elements
by locating boundaries with binary search, skipping the elements between two
pointers that share a target. This implementation uses **exponential (galloping)
search** rather than plain binary search, so the cost is logarithmic in the
block length instead of in the array length, and it is **adaptive**: the kernel
starts in linear mode and switches to galloping only after a run of consecutive
wins, because galloping loses when the two runs interleave element by element.

**Merging from both ends.** The paper assigns positions from the head and the
tail of the subsets. This implementation formalises that as a theorem that
removes the partition computation entirely — see *Bidirectional merge* below.

Two elements of the paper are **not** implemented. Table II specialises the merge
into four routines according to the signs of adjacent runs; here descending runs
are normalised by reversal instead. And the paper partitions the *smaller* subset
across threads; this implementation cuts by output rank, which bounds the
imbalance that partitioning the smaller side does not.

---

## Architecture

### 1. Input classification

Before any pass over the whole array, the engine samples a set of windows spread
uniformly across the input and counts two independent quantities in the same
comparison:

- **Local inversion rate** — the fraction of comparisons where the direction
  changed. For i.i.d. data this tends to **2/3**, because the direction flips in
  four of the six orderings of three values. (This is not the `1/(2e−3)` run
  density above: that constant describes the *greedy run decomposition*, which
  restarts the direction at every run boundary.)
- **Tie rate** — the fraction of comparisons that were exact equalities.

The tie counter costs nothing extra: `Ord::cmp` already returns all three
answers, so equality is available from a comparison that had to happen anyway.
Direction alone cannot distinguish a sorted array of distinct values from a
sorted array of 32 distinct values — both report zero direction changes. The tie
rate separates them (0% vs 100%).

Measured cost of the full sample: **0.5 µs for `u64`, 4.2 µs for `&str`** — under
0.3% of even the fastest scenario. Sampling is not a budget constraint.

**The sample is a guess, and it is checked.** Eight windows of 96 comparisons see
0.0004% of a 100M-element array. Once `detect_global_trend` has run, the exact
run count is available for free, and a mean run length far below what the sample
predicted overrides the guess. This matters: a reversed array containing repeated
keys looks monotone to the sampler — equal pairs count as ascending — but the
metadata reveals runs averaging under ten elements.

### 2. Run detection and the metadata monoid

The array is divided into blocks that **overlap by exactly one element**. Each
block is scanned in parallel for monotone runs; each run becomes one signed
integer. Nothing is moved; the pass is read-only.

The one-element overlap is what makes the scheme work. With blocks `[0,11)`,
`[10,21)`, `[20,30)`, every adjacent-pair comparison belongs to exactly one
block — no gaps, no duplicates. Joining two blocks is therefore a pure operation
on the metadata:

- Same sign — the runs are contiguous through the shared element and merge into
  one of length `a + b − 1`.
- Opposite signs — the shared element is a peak or valley; it stays with the left
  run and the right run loses one.
- A block reporting a single element is the shared element and is absorbed.

The operation is associative, so it composes under a parallel reduce, and the
same function stitches micro-blocks, macro-blocks, and the final result.

If the array collapses to a single run, the answer is immediate: positive means
already sorted and nothing is written; negative means one parallel reverse.

**One asymmetry is load-bearing.** Ascending runs use `<=`, descending runs use a
strict `>`. Equal elements therefore always join ascending runs, which makes every
descending run *strictly* decreasing — and reversing a strictly decreasing
sequence cannot swap two equal elements. That is what keeps the reverse path
stable.

### 3. Bidirectional merge

Two workers merge the same pair of runs at once: one forward from the start, one
backward from the end. No partition point is computed, because of this:

> A forward merge that breaks ties toward the left run emits exactly the `k`
> smallest elements. A backward merge that breaks ties toward the right run emits
> exactly the `total − k` largest. For **any** `k`, those two sets partition the
> multiset.

Each side counts only its own output. The complementary tie-breaking rules —
left-wins forward, right-wins backward — are what make the theorem hold and what
preserve stability at the same time.

In Rust this falls out of the borrow checker: both halves borrow the sources
immutably and write into disjoint halves of the destination via `split_at_mut`,
so the absence of a data race is proved with no `unsafe`.

**The two directions do not cost the same.** Measured on `u64`, the backward
kernel costs about **1.6–1.8×** per element what the forward kernel costs; on
`&str` the ratio is **1.0**. A 2×2 experiment isolates the cause: a kernel that
reads backward but writes forward is fast, and one that reads forward but writes
backward is fast. Only the combination is slow, and a control that moves the same
data with a precomputed selection shows a penalty of just 10%. It is neither the
read direction nor the write direction alone — it is three simultaneously
descending streams (`a`, `b`, and the destination) exhausting the hardware
prefetcher's stream trackers. Galloping erases the asymmetry entirely
(`B/F = 0.95–1.09`) because `copy_from_slice` compiles to `memmove`, which works
in ascending blocks internally.

The consequence is a policy, not a fix: for embedded keys the output split is
biased toward the forward worker, so the two finish together. The optimal
fraction is `B/(F+B)`, which for a measured `B/F = 1.6` gives **60/40** — and
60/40 was independently the best of the fractions tested. For pointer keys, where
`B/F = 1.0`, the split stays at 50/50.

### 4. P-way merge

The bidirectional merge generalises: the two **ends** remain free by the theorem
above, and every **interior** task pays one rank cut. `P` tasks therefore need
`P − 2` cuts, one fewer than a plain rank partition.

Each cut is a binary search **on the diagonal** `i + j = k`. Because `i + j` is
fixed there is one degree of freedom, so `j` follows from `i` and each probe is a
single comparison rather than a search into the other array. Searching for an
element *in* the other array would give the rank *of that element*, which is the
inverse of the question, and finding the right `i` by trial would cost
`O(log m · log n)` instead of `O(log(m+n))`.

Cost is negligible: 96 cuts of roughly 20 comparisons each is under 2,000
comparisons against a million elements moved, and the cuts run in parallel.

`P` is derived from the thread count, not from `N` — measured saturation is
around four tasks per thread — with a floor on segment size so that each task
carries enough work to dilute its scheduling cost.

### 5. Cache-blocked k-way merge

A binary merge tree over `k` runs costs `log₂(k)` full passes over memory. On a
bandwidth-saturated machine the pass count, not the comparison count, sets the
runtime. An 8-way merge costs `log₈(k)` passes: for 30,000 runs, 5 levels instead
of 15.

It is **not** implemented with a loser tree. A loser tree indexes its stream heads
with a runtime value, which prevents the compiler from holding them in registers;
measured against the plain binary kernel it cost roughly an order of magnitude
more per element, erasing the benefit of fewer passes. Instead each output tile
is partitioned across all `k` runs by multi-sequence selection and merged by an
ordinary binary tree that fits **entirely in L1**. Only the last level writes to
the destination, so DRAM sees one read and one write per k-way level while the
inner loop keeps the register-resident binary kernel.

### 6. Galloping insertion kernel

At the merge leaf, the linear kernel can be replaced by one that locates block
boundaries by exponential search and moves whole blocks with `copy_from_slice`.
Its comparison count approaches the Hwang–Lin bound for merging lists of unequal
length, which is optimal.

Measured in isolation, on runs of low cardinality:

| key type | gain over linear |
| :--- | ---: |
| `u64` | 1.2–1.6× |
| `[u8; 32]`, 28-byte common prefix | 1.5× |
| `&str` into a shuffled arena | **2.9–3.1×** |
| `&str`, perfectly interleaved runs | **0.79–0.95×** |

The last row is why **galloping is not the default**. When adjacent runs
interleave element by element, every block has length one and the kernel pays two
searches per element for nothing — a 3.4× loss on `&str` and 4× on `u64`. The
crossover is around block length 8.

### 7. Dispatch by key type

The optimal configuration depends on the key, and a configuration that is good
for one is bad for the other. The dividing line is **not** numeric versus textual:

| | comparison | bottleneck | galloping |
| :--- | :--- | :--- | :--- |
| `u64`, `i32`, `[u8; N]` | contiguous bytes, clean prefetch | **memory bandwidth** | little to gain |
| `&str`, `&[u8]`, arena index | dereference into scattered heap | **comparison latency** | large gain |

`[u8; 32]` is textual and behaves like a wide integer. `&str` is limited by cache
misses on dereference. The proof is direct: cutting comparisons by 45× on `u64`
produced **zero** time saving; the same reduction on `&str` produced **3.1×**.

The `Chave` trait carries this as `COMPARACAO_CARA`, with implementations for the
primitives, for `[u8; N]`, and for `&str` / `&[u8]`. An escape hatch exists because
the type does not always tell the truth — a `u64` that is an index into an
external arena, compared through a dereferencing `Ord`, has an expensive
comparison despite being eight bytes.

### 8. Cache-aware leaf sizing

The merge leaf is sized as `L1 / size_of::<T>()`, with a floor low enough that the
calculation actually applies. A floor of 4096 elements silently defeats the
computation for every type wider than 8 bytes: `u64` computes 4096 and stays
there, but `[u8; 32]` computes 1024 and is clamped back up to 4096 — a 128 KB leaf
in a 32 KB L1. Correcting the floor alone was worth 24% on grouped `[u8; 32]` and
turned the engine's only losing scenario into a 12% win.

The same reasoning applies to the block size used on the chaotic path, where the
budget is the **shared** cache divided by the number of threads that sort
concurrently — which is bounded by physical cores, not by the configured thread
count.

### Stability

The engine is stable end to end. Strictly-decreasing runs make reversal safe;
complementary tie-breaking makes the bidirectional and P-way merges safe;
multi-sequence selection distributes tied elements in stream order so the
parallel partition and the sequential kernel agree on the same total order; and
the galloping kernel's two searches use `<=` on one side and `<` on the other,
which is the same tie rule the linear kernel applies.

---

## Benchmarks

Criterion, 8 threads, `u64` and `&'static str` keys, against `rayon::par_sort`
(stable). Negative is faster.

### `&str` — 32-byte keys, separately allocated, shuffled in memory

| Scenario | 1M | 10M | 100M |
| :--- | ---: | ---: | ---: |
| Low cardinality (32 keys) | **−37%** | **−34%** | **−28%** |
| Random | **−12%** | **−14%** | **−13%** |
| Sorted | −9% | +11% | +15% |
| Reversed | −9% | +71% | +30% |
| Sawtooth | +41% | +33% | +23% |

### `u64`

| Scenario | 1M | 10M | 100M |
| :--- | ---: | ---: | ---: |
| Low cardinality (32 keys) | **−22%** | **−21%** | **−26%** |
| Reversed | **−14%** | −2% | **−13%** |
| Sorted | −4% | 0% | +9% |
| Random | +17% | +11% | +18% |
| Sawtooth | +23% | +24% | +23% |

### What the numbers say

**Low cardinality is the strongest case, in both key types and at every scale.**
This is the categorical-key workload: `GROUP BY` columns, status flags, foreign
keys with few distinct values, secondary sort keys.

**Random `&str` is a genuine win and random `u64` is not.** Both are chaotic; the
difference is the bottleneck. Pointer keys are latency-bound, and threads waiting
on cache misses cover for each other; `par_sort` has nothing that compensates for
a dereference. Embedded keys are bandwidth-bound, and once the structure has to
be *created* rather than exploited, the engine is calling the same `sort()` the
baseline calls, then paying for a merge on top.

**Sawtooth loses consistently.** The pattern `i % 1000` makes every run cover the
*same* value range, so adjacent runs interleave completely. Measured by
comparison count it is the worst case for an adaptive merge, not a typical one —
the shortcut that turns an already-ordered merge into a copy never fires, and the
rank cuts of the P-way tree buy nothing while still costing.

**Reversed `&str` at scale is a routing failure, not an algorithmic one.** With
repeated keys, a reversed array is not one descending run: the strict `>` in the
detector breaks the run at every equal pair, producing 458,000 runs averaging 8.7
elements in a 4M array. The sampler classifies it as structured, the P-way tree
takes it, and short runs are exactly where rank cutting does not pay.

---

## Known limitations

- **Sawtooth and reversed-with-duplicates route to the wrong merge tree.** Both
  produce runs too short for rank cutting to pay. A measured sweep shows the
  k-way tree ahead across the entire range tested, with the crossover near 8,000
  elements per run — routing by mean run length is the open fix.
- **Random embedded keys.** With no structure to exploit and bandwidth already
  saturated, the engine cannot beat a well-tuned parallel sort. It delegates.
- **Thread scaling turns over at four threads.** Gains of 32–44% over the
  baseline at one, two and four threads become losses at eight and above.
  Oversubscription (16 and 32 threads on 8 cores) helps in neither regime.
- **Tuning constants were calibrated on one machine.** The L1 and shared-cache
  budgets, the block size, and the galloping crossover should be re-measured on
  the target hardware.

---

## Correctness

- Unit and integration tests covering the metadata monoid, cut-point invariants
  at **every** `k` from 0 to `m+n`, the bidirectional theorem at every possible
  split, and leaf folding of descending runs.
- A stability fuzz over every public entry point using `(key, index)` pairs, so
  any reordering of equal keys is caught — not just a broken sort order.
- Cut algorithms are cross-validated against each other and against a reference
  stable merge on cardinalities down to two distinct values, which saturates
  tie-breaking.

---

## Usage

```rust
use adaptive_parallel_insertion_merge::despacho;

let mut keys: Vec<u64> = load_keys();
despacho::sort(&mut keys);

let mut names: Vec<&str> = load_names();
despacho::sort(&mut names);
```

`despacho::sort` selects the path from the key type. To override the profile —
for a key whose `Ord` dereferences despite a small footprint:

```rust
despacho::sort_com_perfil(&mut arena_indices, true);
```

For a database or work-queue context, `PimExecutor` confines the sort to its own
thread pool and `try_sort` returns a `Result` instead of panicking when the
auxiliary buffer cannot be reserved:

```rust
let exec = PimExecutor::new(8)?;
exec.try_sort(&mut keys, PimConfig::default())?;
```

---

## Bibliography

**Foundation**

- Couto, F. B. and Couto, F. S. *Parallel Insertion Merge*. PDPTA — International
  Conference on Parallel and Distributed Processing Techniques and Applications.

**Adaptive merging and galloping**

- Peters, T. *Timsort*. `Objects/listsort.txt`, CPython, 2002. Natural run
  detection, minimum run length, and adaptive galloping with a credit counter.
- Hwang, F. K. and Lin, S. *A Simple Algorithm for Merging Two Disjoint Linearly
  Ordered Sets*. SIAM Journal on Computing 1(1), 1972, pp. 31–39.
  DOI: 10.1137/0201004. The comparison bound that block insertion approaches.
- Estivill-Castro, V. and Wood, D. *A Survey of Adaptive Sorting Algorithms*. ACM
  Computing Surveys 24(4), 1992.
- Knuth, D. E. *The Art of Computer Programming, Vol. 3: Sorting and Searching*,
  2nd ed. Addison-Wesley, 1998. Natural runs and run statistics.

**Parallel merge partitioning**

- Odeh, S., Green, O., Mwassi, Z., Shmueli, O. and Birk, Y. *Merge Path — Parallel
  Merging Made Simple*. IPDPS Workshops 2012, pp. 1611–1618.
  DOI: 10.1109/IPDPSW.2012.202. The diagonal rank cut used by the P-way merge.
- Green, O., McColl, R. and Bader, D. A. *GPU Merge Path: A GPU Merging
  Algorithm*. ICS 2012, pp. 331–340.

**Multiway merging and cache behaviour**

- Singler, J., Sanders, P. and Putze, F. *MCSTL: The Multi-Core Standard Template
  Library*. Euro-Par 2007, LNCS 4641, pp. 682–694.
  DOI: 10.1007/978-3-540-74466-5_72. Multiway merge with multi-sequence
  selection.
- Varman, P. J., Scheufler, S. D., Iyer, B. R. and Ricard, G. R. *Merging Multiple
  Lists on Hierarchical-Memory Multiprocessors*. Journal of Parallel and
  Distributed Computing 12(2), 1991, pp. 171–177.
- Frigo, M., Leiserson, C. E., Prokop, H. and Ramachandran, S. *Cache-Oblivious
  Algorithms*. FOCS 1999, pp. 285–297.

**Baselines**

- Peters, O. *Pattern-Defeating Quicksort*, 2021. The unstable sort underlying
  `slice::sort_unstable`.
- Bergdoll, L. and Peters, O. *driftsort*. The stable sort underlying
  `slice::sort` and, through Rayon, `par_sort`.

---

## License

Licensed under the Apache License, Version 2.0.

Copyright © Fernando Belmiro do Couto and Fábio Silva do Couto

You may not use this project except in compliance with the License. You may
obtain a copy of the License at:

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed
under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
