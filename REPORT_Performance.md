# C++ Parallel Performance Report

## Methodology

- Images loaded from `input_images/`; outputs disabled to avoid I/O skew.
- Benchmarks executed by `benchmark_plot_C.py` with counts from 50 to 1000 (step 50).
- Parallelism controlled via environment:
  - `threads.exe`: `THREADS={1,4,8,16}`
  - `openmp.exe`: `OMP_NUM_THREADS={1,4,8,16}`
- Each point is the average of `--repeats` runs (1 in these results). Timing reported is the total for all filters.
- Build flags: `-O2`, `-std=c++17`; `openmp.exe` additionally `-fopenmp`.

## Metrics

- Speedup: $S(N) = \dfrac{T_{serial}}{T_{parallel}(N)}$ where $T_{serial}$ is the 1-thread baseline.
- Efficiency: $E(N) = \dfrac{S(N)}{N}$ (fraction of ideal linear speedup).

## Test Cases

- Thread counts: 1, 4, 8, 16 for both paradigms (`std::thread` and OpenMP).
- Image counts: 50..1000 (step 50). Tables below highlight 200, 600, and 1000 as representative.

## C++ Performance Results

### Execution Time Table (milliseconds)

Serial baseline uses OpenMP with 1 thread (closest to a pure single-thread run). Corresponding `threads.exe` 1-thread times are similar but slightly higher; they are noted in parentheses.

| Images |      Serial (OpenMP 1) | std::thread 4 | std::thread 8 | std::thread 16 | OpenMP 4 | OpenMP 8 | OpenMP 16 |
| -----: | ---------------------: | ------------: | ------------: | -------------: | -------: | -------: | --------: |
|    200 |   2427 (threads: 2682) |           835 |           806 |            902 |      681 |      410 |       403 |
|    600 |   7256 (threads: 7812) |          2491 |          2045 |           2436 |     2063 |     1238 |      1181 |
|   1000 | 11954 (threads: 12946) |          4144 |          3378 |           4023 |     3327 |     2449 |      1949 |

Data source: `plots/v2_C_runtime_openmp_counts.csv` and `plots/v2_C_runtime_threads_counts.csv`.

### Speedup Graphs

- Threads baseline vs 1 thread: see `plots/v2_C_speedup_threads_vs1.png`.
- OpenMP baseline vs 1 thread: see `plots/v2_C_speedup_openmp_vs1.png`.

### Efficiency Analysis

- OpenMP achieved higher speedups overall; at 1000 images, speedup ≈ 6.13× (16 threads) and ≈ 4.93× (8 threads), giving efficiencies of ~$E(16) \approx 0.38$ and ~$E(8) \approx 0.62$.
- std::thread achieved ≈ 3.83× (8 threads) and ≈ 3.22× (16 threads) at 1000 images, with efficiencies ~$E(8) \approx 0.48$ and ~$E(16) \approx 0.20$. The 16-thread run is not the fastest due to overheads.
- Likely causes of sub-linear scaling and 16-thread slowdowns:
  - Synchronization and scheduling overhead (thread launch/join, work distribution).
  - Memory bandwidth saturation and cache contention at higher core counts.
  - Row-chunk partitioning in `threads.cpp` can cause load imbalance and more contention on shared memory.
  - OS scheduling and NUMA effects; OpenMP’s runtime may bind threads and schedule loops more efficiently than manual `std::thread` partitioning.

### Notes and Tuning Ideas

- For `std::thread`:
  - Increase chunk size (process more rows per thread) or use a work-stealing queue to reduce imbalance.
  - Consider `std::vector<std::thread>::reserve(nt)` and reusing threads across images.
  - Pin threads to cores using platform APIs to reduce migration.
- For OpenMP:
  - Try `schedule(static, chunk)` and environment `OMP_PROC_BIND=close`, `OMP_PLACES=cores`.
- Verify 1-thread baselines are consistent; both serial paths include the same filter code but different drivers, hence small differences.

## Reproduce

- Build:
  - `g++ -std=c++17 -O2 -I third_party/stb -I src threads.cpp -o threads.exe`
  - `g++ -std=c++17 -O2 -fopenmp -I third_party/stb -I src openmp.cpp -o openmp.exe`
- Run:
  - `python benchmark_plot_C.py --thread-counts 1,4,8,16 --start 50 --end 1000 --step 50`
- Outputs:
  - CSVs and PNGs in `plots/` as referenced above.

## Discussion

### Amdahl's Law

- Formula: $S(N) = \dfrac{1}{(1-p) + \dfrac{p}{N}}$, where $p$ is the parallelizable fraction.
- Estimating $p$ from actual results:

  - OpenMP at $N=16$ achieved $S\approx 6.13$ (from `v2_C_runtime_openmp_counts.csv` at 1000 images).
    - $\dfrac{1}{S} = (1-p) + \dfrac{p}{16} \Rightarrow 1 - p\cdot\frac{15}{16} \approx 0.163$.
    - $p \approx 0.893$.
    - Theoretical maximum speedup as $N\to\infty$: $S_{\max} = \dfrac{1}{1-p} \approx 9.3$.
  - std::thread at $N=8$ achieved $S\approx 3.83$ (threads baseline at 1000 images).
    - $\dfrac{1}{S} = (1-p) + \dfrac{p}{8} \Rightarrow 1 - p\cdot\frac{7}{8} \approx 0.261$.
    - $p \approx 0.845$, giving $S_{\max} \approx \dfrac{1}{1-0.845} \approx 6.45$.

- Comparison (theoretical vs actual):
  - OpenMP: Actual $S(16)\approx 6.13$ vs theoretical $\approx 9.3$ — gap attributed to runtime overhead, memory bandwidth saturation, and cache effects.
  - std::thread: Actual $S(8)\approx 3.83$ vs theoretical $\approx 6.45$ — larger gap indicates heavier per-image thread creation/join overhead and less efficient scheduling.

### Trade-offs

- **C++ manual threads:**
  - Pros: Full control over partitioning, portability without extra runtime.
  - Cons: Higher complexity (lifecycle management, load balancing, affinity), risk of overhead when creating/joining threads per image; tuning requires bespoke work-stealing or pools.
- **OpenMP:**
  - Pros: Simple pragmas (`#pragma omp parallel for`), efficient team management, easy control of thread count and scheduling, often better default scaling.
  - Cons: Dependency on compiler/runtime support; tuning requires understanding of scheduling and binding options.
- **Overhead perspective:**
  - Thread/process creation, synchronization, and scheduling add fixed costs that don’t shrink with more cores; for small workloads per thread (few rows per image), overhead can dominate.
  - Computation time scales with data size; larger images or batching more rows per task improve arithmetic intensity and amortize overhead.

## Scalability and Bottlenecks

### Bottlenecks

- **I/O (image read/write):** Reads use `stbi_load` per image; writes are disabled in both C++ programs, so disk output is not part of runtime. Decoding JPEG/PNG is partly CPU-bound but, at these scales, total runtime is dominated by filter computation.
- **CPU compute (filters):** Row-wise filters (Gaussian blur, Sobel, sharpen, brightness) are memory-intensive. Access patterns are contiguous per row, but throughput is limited by memory bandwidth. As threads increase, contention grows; scaling becomes bandwidth-bound.
- **Threading overhead:** In `threads.exe`, threads are created and joined for each image processed. This per-image creation cost accumulates and reduces efficiency at high thread counts. OpenMP typically reuses a thread team within a `parallel for`, amortizing this cost.
- **Scheduling and cache effects:** `schedule(static)` in OpenMP is a good fit for uniform row costs, but very high thread counts can suffer from cache thrash and reduced locality. The manual row-striping in `threads.cpp` can also yield imbalance near image tails and increases pressure on shared caches.

### Scalability

- **Speedup ($S$) and Efficiency ($E$):** Using the 1-thread baseline for each paradigm: $S(N)=T_{serial}/T_{parallel}(N)$ and $E(N)=S(N)/N$.
  - At 1000 images:
    - OpenMP: $S(4)\approx 3.59$ ($E\approx 0.90$), $S(8)\approx 4.88$ ($E\approx 0.61$), $S(16)\approx 6.13$ ($E\approx 0.38$).
    - std::thread: $S(4)\approx 3.13$ ($E\approx 0.78$), $S(8)\approx 3.83$ ($E\approx 0.48$), $S(16)\approx 3.22$ ($E\approx 0.20$).
- **Observation:** Adding more vCPUs does not linearize performance. Both implementations show diminishing returns; OpenMP continues to improve up to 16 threads but sublinearly, while `threads.exe` peaks near 8 threads and degrades at 16.
- **Why 16 threads isn’t always fastest:** Amdahl’s Law and bandwidth limits dominate at high concurrency. Extra threads add overhead (creation, scheduling, synchronization) and can saturate memory bandwidth, lowering per-thread progress. OpenMP’s runtime generally manages teams more efficiently than repeatedly launching `std::thread`s.

### Improvement Ideas

- **Amortize thread overhead:** Convert `threads.exe` to use a persistent thread pool or a work queue that processes all rows across all images, avoiding per-image creation/join.
- **Tune chunking:** Increase row chunk sizes or use dynamic work-stealing for `threads.exe` to balance tail rows. For OpenMP, try `schedule(static, chunk)` with `chunk` sized to several rows.
- **Bind threads:** Experiment with `OMP_PROC_BIND=close` and `OMP_PLACES=cores` to improve locality; for `std::thread`, consider OS-level affinity APIs.
- **Improve memory locality:** Tile operations (process blocks rather than full rows) and explore vectorization to better utilize cache and SIMD units.
