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
