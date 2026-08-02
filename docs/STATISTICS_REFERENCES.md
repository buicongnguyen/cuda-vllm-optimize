# Benchmark statistics: definitions, sources, and repo conventions

This is the GitHub-readable source map for
[Learning module 10](https://buicongnguyen.github.io/cuda-vllm-optimize/learn.html#m10).
The web module contains the formulas, worked example, and visual decision rules.

## Source labels

- **STANDARD**: an established statistical concept with an independent primary
  reference.
- **REPO PROTOCOL**: a procedure or decision label chosen for this benchmark.
  It is not claimed to be a universally named statistical test.
- **DERIVED / UNVERIFIED**: computed from the ERS expression in the supplied
  contest narrative. No public official evaluator contract has been found for
  verification.

## 1. Paired request R0/B — STANDARD, adapted to the benchmark

Two measurements are paired when observation `i` in one sample corresponds to
observation `i` in the other sample. Here, R0 and B must contain the same
`request_id`, prompt, conversation turn, seed, and output policy.

For latency, this repo defines:

```text
delta_i = latency_i(B) - latency_i(R0)
```

A negative delta means the candidate was faster for that request. Pairing
reduces variation caused by request difficulty; it does not remove time drift
between runs.

- [NIST: Paired Observations](https://www.itl.nist.gov/div898/handbook/prc/section3/prc311.htm)
- [Repo implementation: scripts/rtx4080_compare.py](https://github.com/buicongnguyen/cuda-vllm-optimize/blob/main/scripts/rtx4080_compare.py)

## 2. A/B/A and baseline-return R0-prime — REPO PROTOCOL

The sequence is baseline R0, candidate B, then the same baseline R0-prime.
The repo observes baseline drift as:

```text
C = metric(B) - metric(R0)          # candidate change
D = metric(R0-prime) - metric(R0)   # observed baseline drift
```

R0-prime brackets the candidate in time and makes drift observable. It does not
mathematically remove drift, and A/B/A is not presented here as the official
name of a universal statistical test. Its design rationale comes from blocking
known nuisance factors and inspecting trends in sequential process data.

- [NIST: Randomized Block Designs](https://www.itl.nist.gov/div898/handbook/pri/section3/pri332.htm)
- [NIST: Trends in Sequential Process Data](https://www.itl.nist.gov/div898/handbook/prc/section1/prc17.htm)
- [Repo run matrix](https://buicongnguyen.github.io/cuda-vllm-optimize/reproduce-rtx4080.html#matrix)

## 3. Mean and median — STANDARD

For observations `x_1 ... x_n`, the arithmetic mean is:

```text
mean(x) = (1/n) * sum(x_i)
```

The median is the middle sorted observation for odd `n`, or the average of the
two middle observations for even `n`. Mean uses every magnitude and is more
sensitive to tail values. Median describes the center but does not describe
tail latency.

- [NIST: Measures of Location](https://www.itl.nist.gov/div898/handbook/eda/section3/eda351.htm)
- [Python statistics: mean and median](https://docs.python.org/3/library/statistics.html)

## 4. p95 and p99 — STANDARD, method-sensitive

p95 and p99 are the 95th and 99th percentiles of a distribution. They are not
“95% confidence” and “99% confidence.” In a finite sample, several legitimate
quantile definitions exist and can produce different values.

This repo uses linear/type-7 interpolation:

```text
h = (n - 1) * q
j = floor(h)
g = h - j
Q(q) = (1 - g) * x[j] + g * x[j + 1]
```

Always report the sample size and percentile method.

- [NIST: Percentiles](https://www.itl.nist.gov/div898/handbook/prc/section2/prc262.htm)
- [NumPy: percentile methods](https://numpy.org/doc/stable/reference/generated/numpy.percentile.html)
- [Repo implementation: scripts/rtx4080_compare.py](https://github.com/buicongnguyen/cuda-vllm-optimize/blob/main/scripts/rtx4080_compare.py)

## 5. Bootstrap 95% confidence interval — STANDARD method, repo parameters

The repo resamples the paired deltas with replacement, calculates the mean for
each resample, then takes the 2.5th and 97.5th percentiles of those bootstrap
means:

```text
delta*_b = sample_with_replacement(delta, n)
m_b      = mean(delta*_b)
CI_95    = [percentile_2.5(m), percentile_97.5(m)]
```

The implementation uses percentile bootstrap, 2,000 resamples, and a fixed
seed. SciPy currently defaults to BCa, so a SciPy default call is not identical
to this repo's procedure. A 95% confidence interval describes long-run coverage
of the procedure; it is not a 95% probability statement about this fixed
interval containing the true effect.

- [SciPy: bootstrap](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.bootstrap.html)
- [Repo implementation: scripts/rtx4080_compare.py](https://github.com/buicongnguyen/cuda-vllm-optimize/blob/main/scripts/rtx4080_compare.py)

## 6. faster, slower, or uncertain — REPO DECISION RULE

Because latency delta is `candidate - baseline`, the repo classifies each
latency metric independently:

```text
CI.high < 0              => faster
CI.low  > 0              => slower
CI.low <= 0 <= CI.high   => uncertain
```

This label reports the sign of the estimated mean latency change under the
chosen interval procedure. It does not establish output correctness, stability,
or a system-wide win. For example, TTFT can be faster while TPOT is slower.

- [Confidence interval basis: SciPy bootstrap](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.bootstrap.html)
- [Repo classifier: scripts/rtx4080_compare.py](https://github.com/buicongnguyen/cuda-vllm-optimize/blob/main/scripts/rtx4080_compare.py)

## 7. ERS delta — DERIVED / UNVERIFIED

Using the expression quoted in the supplied narrative:

```text
S(T,F) = 100 * [0.5 * ((400-T)/390)^2 + 0.5 * ((10-F)/9)^2]
delta_ERS = S(T_B, F_B) - S(T_R0, F_R0)
```

`T` is aggregate TTFT and `F` is aggregate TPOT. A positive delta is better
under this expression, but the same aggregation must be used for R0 and B.

There is no official evaluator/spec link in this repository. Clipping,
aggregation, quality rules, and hidden evaluator behavior remain unverified.

- [Evidence status and open questions](https://buicongnguyen.github.io/cuda-vllm-optimize/problem.html#contract)
- [Repo formula: src/racebench/score.py](https://github.com/buicongnguyen/cuda-vllm-optimize/blob/main/src/racebench/score.py)

## 8. Warning when R0-prime is missing — EXPERIMENT DESIGN CAUTION

Without R0-prime, the experiment does not measure how much the baseline changed
after B ran. The R0/B result should remain **provisional** because thermal,
clock, cache, or background-load drift may be confounded with the candidate
effect. Paired requests solve a different problem: variation between requests.

- [NIST: Trends in Sequential Process Data](https://www.itl.nist.gov/div898/handbook/prc/section1/prc17.htm)
- [Repo promotion gate](https://buicongnguyen.github.io/cuda-vllm-optimize/reproduce-rtx4080.html#decide)

## Minimal report fields

Record at least: workload identity, paired sample count, sign convention,
mean/median/p95/p99 for raw runs and paired deltas, percentile method, bootstrap
method/resample count/seed, confidence interval, R0-to-R0-prime drift, failures,
correctness result, ERS delta, and the source status of the ERS contract.
