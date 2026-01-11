import argparse
import os
import re
import subprocess
import sys
from pathlib import Path
from statistics import mean
from typing import Dict, List, Optional, Tuple

# Optional import for plotting
try:
    import matplotlib.pyplot as plt
except ImportError as e:
    plt = None


def find_executable(name: str, workspace: Path) -> Optional[Path]:
    cand = workspace / f"{name}.exe"
    return cand if cand.exists() else None


def count_available_images(input_dir: Path) -> int:
    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    return sum(1 for p in input_dir.iterdir() if p.is_file() and p.suffix.lower() in exts)


def run_program(
    exe: Path,
    input_dir: Path,
    n: int,
    timeout: int = 600,
    env_override: Optional[Dict[str, str]] = None,
) -> Tuple[Optional[float], str]:
    """
    Run the program and parse its reported time in milliseconds.
    Returns (time_ms, stdout_text). time_ms is None if parse failed.
    """
    cmd = [str(exe), str(input_dir), str(n)]
    try:
        completed = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            cwd=str(exe.parent),
            timeout=timeout,
            shell=False,
            check=False,
            env=(
                {**os.environ, **env_override}
                if env_override is not None
                else None
            ),
        )
    except subprocess.TimeoutExpired:
        return None, f"TIMEOUT running: {' '.join(cmd)}"

    out = completed.stdout or ""

    # Try to parse lines like:
    #   Threads time (all filters): 123 ms
    #   OpenMP time (all filters): 78 ms
    # Be tolerant about spacing/case/units.
    ms = parse_time_ms(out)
    return ms, out


def parse_time_ms(text: str) -> Optional[float]:
    # First try specific prefixes for better accuracy
    patterns = [
        r"(?i)\bthreads\b.*?time.*?:\s*([0-9]+(?:\.[0-9]+)?)\s*(ms|milliseconds|s|sec|seconds)\b",
        r"(?i)\bopenmp\b.*?time.*?:\s*([0-9]+(?:\.[0-9]+)?)\s*(ms|milliseconds|s|sec|seconds)\b",
        r"(?i)time\s*\(.*?filters.*?\)\s*:\s*([0-9]+(?:\.[0-9]+)?)\s*(ms|milliseconds|s|sec|seconds)\b",
    ]
    for pat in patterns:
        m = re.search(pat, text, flags=re.DOTALL)
        if m:
            val = float(m.group(1))
            unit = m.group(2).lower()
            if unit.startswith("ms"):
                return val
            else:
                # seconds to ms
                return val * 1000.0
    return None


def averaged_time(
    exe: Path,
    input_dir: Path,
    n: int,
    repeats: int,
    env_override: Optional[Dict[str, str]] = None,
) -> Tuple[Optional[float], List[str]]:
    logs = []
    times: List[float] = []
    for _ in range(max(1, repeats)):
        t_ms, out = run_program(exe, input_dir, n, env_override=env_override)
        logs.append(out)
        if t_ms is not None:
            times.append(t_ms)
    if times:
        return mean(times), logs
    return None, logs


def ensure_plots_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def main():
    parser = argparse.ArgumentParser(description="Benchmark threads.exe and openmp.exe vs number of images and thread counts.")
    parser.add_argument("--input-dir", default="input_images", help="Folder containing input images.")
    parser.add_argument("--start", type=int, default=50, help="Starting image count.")
    parser.add_argument("--end", type=int, default=250, help="Ending image count (inclusive).")
    parser.add_argument("--step", type=int, default=50, help="Step size for image count.")
    parser.add_argument("--repeats", type=int, default=1, help="Number of runs to average per point.")
    parser.add_argument("--save", default="plots/v1_C_runtime_vs_images.png", help="Path to save the plot PNG.")
    parser.add_argument("--csv", default="plots/v1_C_runtime_data.csv", help="Path to save the CSV results.")
    parser.add_argument("--save-speedup", default="plots/v1_C_speedup_vs_images.png", help="Path to save the speedup plot PNG.")
    parser.add_argument("--csv-speedup", default="plots/v1_C_speedup_data.csv", help="Path to save the speedup CSV results.")
    parser.add_argument("--thread-counts", default="1,2,4,8,12,16", help="Comma-separated list of thread counts to test (applies to both OpenMP and threads.exe via env).")
    parser.add_argument("--skip-threads", action="store_true", help="Skip running threads.exe.")
    parser.add_argument("--skip-openmp", action="store_true", help="Skip running openmp.exe.")
    parser.add_argument("--ignore-limit", action="store_true", help="Do not cap counts by available images.")

    args = parser.parse_args()

    workspace = Path(__file__).resolve().parent
    input_dir = (workspace / args.input_dir).resolve()

    if not input_dir.exists():
        print(f"Input directory not found: {input_dir}")
        sys.exit(1)

    threads_exe = find_executable("threads", workspace)
    openmp_exe = find_executable("openmp", workspace)

    if not args.skip_threads and threads_exe is None:
        print("threads.exe not found in workspace root. Build it first.")
        print("Hint (PowerShell): g++ -std=c++17 -O2 -pthread -I third_party/stb threads.cpp -o threads.exe")
        sys.exit(2)
    if not args.skip_openmp and openmp_exe is None:
        print("openmp.exe not found in workspace root. Build it first.")
        print("Hint (PowerShell): g++ -std=c++17 -O2 -fopenmp -I third_party/stb openmp.cpp -o openmp.exe")
        sys.exit(2)

    counts = list(range(args.start, args.end + 1, args.step))

    available = count_available_images(input_dir)
    if not args.ignore_limit and available > 0:
        capped = [c for c in counts if c <= available]
        if len(capped) < len(counts):
            print(f"Only {available} image(s) available in {input_dir}. Capping counts to <= {available}.")
        counts = capped

    if not counts:
        print("No counts to run. Check --start/--end/--step and available images.")
        sys.exit(0)

    print("Benchmark configuration:")
    print(f"  Input dir:     {input_dir}")
    print(f"  Counts:        {counts[0]}..{counts[-1]} step {args.step} ({len(counts)} points)")
    print(f"  Repeats:       {args.repeats}")
    print(f"  Execs:         threads={'yes' if not args.skip_threads else 'no'}, openmp={'yes' if not args.skip_openmp else 'no'}")

    # Parse thread counts
    try:
        thread_counts = [int(x.strip()) for x in args.thread_counts.split(",") if x.strip()]
        thread_counts = [tc for tc in thread_counts if tc >= 1]
    except Exception:
        thread_counts = [1, 4, 8, 16]

    results = []  # legacy single-run results (kept for compatibility)
    results_threads: Dict[int, List[Tuple[int, Optional[float]]]] = {}
    results_openmp: Dict[int, List[Tuple[int, Optional[float]]]] = {}

    for n in counts:
        print(f"\n=== Running n={n} ===")
        t_ms = None
        o_ms = None
        o1_ms = None

        if not args.skip_threads and threads_exe:
            results_threads.setdefault(n, [])
            for tc in thread_counts:
                print(f"Running threads.exe with THREADS={tc}...")
                t_ms, logs = averaged_time(
                    threads_exe,
                    input_dir,
                    n,
                    args.repeats,
                    env_override={"THREADS": str(tc)},
                )
                if t_ms is None:
                    print(f"  WARNING: Could not parse time from threads.exe (THREADS={tc}) output.")
                else:
                    print(f"  threads.exe time (avg) @ {tc} threads: {t_ms:.2f} ms")
                results_threads[n].append((tc, t_ms))

        if not args.skip_openmp and openmp_exe:
            results_openmp.setdefault(n, [])
            for tc in thread_counts:
                print(f"Running openmp.exe with OMP_NUM_THREADS={tc}...")
                o_ms, logs = averaged_time(
                    openmp_exe,
                    input_dir,
                    n,
                    args.repeats,
                    env_override={"OMP_NUM_THREADS": str(tc)},
                )
                if o_ms is None:
                    print(f"  WARNING: Could not parse time from openmp.exe (OMP_NUM_THREADS={tc}) output.")
                else:
                    print(f"  openmp.exe time (avg) @ {tc} threads: {o_ms:.2f} ms")
                results_openmp[n].append((tc, o_ms))

        results.append((n, t_ms, o_ms, o1_ms))

    # Save CSVs for per-thread-count runs
    csv_thr_path = (workspace / "plots/v2_C_runtime_threads_counts.csv").resolve()
    ensure_plots_dir(csv_thr_path)
    with open(csv_thr_path, "w", newline="", encoding="utf-8") as f:
        header = "count," + ",".join([f"threads_{tc}_ms" for tc in thread_counts]) + "\n"
        f.write(header)
        for n in counts:
            row = [str(n)]
            vals: List[str] = []
            entries = {tc: ms for (tc, ms) in results_threads.get(n, [])}
            for tc in thread_counts:
                ms = entries.get(tc)
                vals.append(f"{float(ms):.4f}" if ms is not None else "")
            row.extend(vals)
            f.write(",".join(row) + "\n")
    print(f"\nSaved threads CSV: {csv_thr_path}")

    csv_omp_path = (workspace / "plots/v2_C_runtime_openmp_counts.csv").resolve()
    ensure_plots_dir(csv_omp_path)
    with open(csv_omp_path, "w", newline="", encoding="utf-8") as f:
        header = "count," + ",".join([f"openmp_{tc}_ms" for tc in thread_counts]) + "\n"
        f.write(header)
        for n in counts:
            row = [str(n)]
            vals: List[str] = []
            entries = {tc: ms for (tc, ms) in results_openmp.get(n, [])}
            for tc in thread_counts:
                ms = entries.get(tc)
                vals.append(f"{float(ms):.4f}" if ms is not None else "")
            row.extend(vals)
            f.write(",".join(row) + "\n")
    print(f"Saved OpenMP CSV: {csv_omp_path}")

    # Plot
    if plt is None:
        print("matplotlib not installed. Install with: pip install matplotlib")
        return

    # Plot: threads.exe for multiple thread counts
    if plt is not None and not args.skip_threads:
        import math
        plt.figure(figsize=(10, 6))
        for tc in thread_counts:
            ys = []
            for n in counts:
                entries = {t: ms for (t, ms) in results_threads.get(n, [])}
                ms = entries.get(tc)
                ys.append(float(ms) if ms is not None else math.nan)
            plt.plot(counts, ys, label=f"threads.exe ({tc} threads)", linewidth=2)
        plt.xlabel("Images processed (n)")
        plt.ylabel("Time (ms)")
        plt.title("threads.exe: Runtime vs Number of Images by thread count")
        plt.grid(True, linestyle="--", alpha=0.4)
        plt.legend()
        png_thr = (workspace / "plots/v2_C_runtime_vs_images_threads.png").resolve()
        ensure_plots_dir(png_thr)
        plt.tight_layout()
        plt.savefig(png_thr, dpi=150)
        print(f"Saved plot: {png_thr}")

    # Plot: openmp.exe for multiple thread counts
    if plt is not None and not args.skip_openmp:
        import math
        plt.figure(figsize=(10, 6))
        for tc in thread_counts:
            ys = []
            for n in counts:
                entries = {t: ms for (t, ms) in results_openmp.get(n, [])}
                ms = entries.get(tc)
                ys.append(float(ms) if ms is not None else math.nan)
            plt.plot(counts, ys, label=f"openmp.exe ({tc} threads)", linewidth=2)
        plt.xlabel("Images processed (n)")
        plt.ylabel("Time (ms)")
        plt.title("openmp.exe: Runtime vs Number of Images by thread count")
        plt.grid(True, linestyle="--", alpha=0.4)
        plt.legend()
        png_omp = (workspace / "plots/v2_C_runtime_vs_images_openmp.png").resolve()
        ensure_plots_dir(png_omp)
        plt.tight_layout()
        plt.savefig(png_omp, dpi=150)
        plt.tight_layout()
        plt.savefig(png_omp, dpi=150)
        print(f"Saved plot: {png_omp}")

    # Calculate baselines if available
    baseline_omp = {}
    if not args.skip_openmp:
        for n in counts:
            for (tc, ms) in results_openmp.get(n, []):
                if tc == 1:
                    baseline_omp[n] = ms
                    break

    baseline_thr = {}
    if not args.skip_threads:
        for n in counts:
            for (tc, ms) in results_threads.get(n, []):
                if tc == 1:
                    baseline_thr[n] = ms
                    break

    # Optional: compute and save speedup vs baseline (1 thread) for each thread count
    if plt is not None and not args.skip_openmp:
        # Save speedup CSV for openmp.exe
        csv_omp_speed_path = (workspace / "plots/v2_C_speedup_openmp_vs1.csv").resolve()
        ensure_plots_dir(csv_omp_speed_path)
        with open(csv_omp_speed_path, "w", newline="", encoding="utf-8") as f:
            f.write("count,baseline_ms," + ",".join([f"openmp_{tc}_speedup" for tc in thread_counts if tc != 1]) + "\n")
            for n in counts:
                b = baseline_omp.get(n)
                row = [str(n), (f"{float(b):.4f}" if b is not None else "")]
                vals = []
                for tc in thread_counts:
                    if tc == 1:
                        continue
                    ms = None
                    for (tcc, val) in results_openmp.get(n, []):
                        if tcc == tc:
                            ms = val
                            break
                    if b is not None and b > 0 and ms is not None and ms > 0:
                        vals.append(f"{float(b)/float(ms):.4f}")
                    else:
                        vals.append("")
                row.extend(vals)
                f.write(",".join(row) + "\n")
        print(f"Saved openmp speedup CSV: {csv_omp_speed_path}")

        # Speedup plots for OpenMP
        import math
        plt.figure(figsize=(10, 6))
        for tc in thread_counts:
            if tc == 1:
                continue
            ys = []
            for n in counts:
                b = baseline_omp.get(n)
                ms = None
                for (tcc, val) in results_openmp.get(n, []):
                    if tcc == tc:
                        ms = val
                        break
                if b is not None and b > 0 and ms is not None and ms > 0:
                    ys.append(float(b) / float(ms))
                else:
                    ys.append(math.nan)
            plt.plot(counts, ys, label=f"openmp.exe speedup vs 1 thread ({tc})", linewidth=2)
        plt.xlabel("Images processed (n)")
        plt.ylabel("Speedup (×)")
        plt.title("OpenMP Speedup vs 1 thread baseline")
        plt.grid(True, linestyle="--", alpha=0.4)
        plt.legend()
        png_speed_path = (workspace / "plots/v2_C_speedup_openmp_vs1.png").resolve()
        ensure_plots_dir(png_speed_path)
        plt.tight_layout()
        plt.savefig(png_speed_path, dpi=150)
        print(f"Saved speedup plot: {png_speed_path}")

    # Speedup graph for threads.exe (baseline: THREADS=1)
    if plt is not None and not args.skip_threads:
        # Save speedup CSV for threads.exe
        csv_thr_speed_path = (workspace / "plots/v2_C_speedup_threads_vs1.csv").resolve()
        ensure_plots_dir(csv_thr_speed_path)
        with open(csv_thr_speed_path, "w", newline="", encoding="utf-8") as f:
            f.write("count,baseline_ms," + ",".join([f"threads_{tc}_speedup" for tc in thread_counts if tc != 1]) + "\n")
            for n in counts:
                b = baseline_thr.get(n)
                row = [str(n), (f"{float(b):.4f}" if b is not None else "")]
                vals = []
                for tc in thread_counts:
                    if tc == 1:
                        continue
                    ms = None
                    for (tcc, val) in results_threads.get(n, []):
                        if tcc == tc:
                            ms = val
                            break
                    if b is not None and b > 0 and ms is not None and ms > 0:
                        vals.append(f"{float(b)/float(ms):.4f}")
                    else:
                        vals.append("")
                row.extend(vals)
                f.write(",".join(row) + "\n")
        print(f"Saved threads speedup CSV: {csv_thr_speed_path}")

        # Plot threads.exe speedup vs 1 thread baseline
        import math
        plt.figure(figsize=(10, 6))
        for tc in thread_counts:
            if tc == 1:
                continue
            ys = []
            for n in counts:
                b = baseline_thr.get(n)
                ms = None
                for (tcc, val) in results_threads.get(n, []):
                    if tcc == tc:
                        ms = val
                        break
                if b is not None and b > 0 and ms is not None and ms > 0:
                    ys.append(float(b) / float(ms))
                else:
                    ys.append(math.nan)
            plt.plot(counts, ys, label=f"threads.exe speedup vs 1 thread ({tc})", linewidth=2)
        plt.xlabel("Images processed (n)")
        plt.ylabel("Speedup (×)")
        plt.title("threads.exe Speedup vs 1 thread baseline")
        plt.grid(True, linestyle="--", alpha=0.4)
        plt.legend()
        png_thr_speed = (workspace / "plots/v2_C_speedup_threads_vs1.png").resolve()
        ensure_plots_dir(png_thr_speed)
        plt.savefig(png_thr_speed)
        print(f"Saved threads speedup plot: {png_thr_speed}")
        plt.close()

        # Save speedup CSV for openmp.exe
        if not args.skip_openmp:
            csv_omp_speed_path = (workspace / "plots/v2_C_speedup_openmp_vs1.csv").resolve()
            ensure_plots_dir(csv_omp_speed_path)
            with open(csv_omp_speed_path, "w", newline="", encoding="utf-8") as f:
                f.write("count,baseline_ms," + ",".join([f"openmp_{tc}_speedup" for tc in thread_counts if tc != 1]) + "\n")
                for n in counts:
                    b = baseline_omp.get(n)
                    row = [str(n), (f"{float(b):.4f}" if b is not None else "")]
                    vals = []
                    for tc in thread_counts:
                        if tc == 1:
                            continue
                        ms = None
                        for (tcc, val) in results_openmp.get(n, []):
                            if tcc == tc:
                                ms = val
                                break
                        if b is not None and b > 0 and ms is not None and ms > 0:
                            vals.append(f"{float(b)/float(ms):.4f}")
                        else:
                            vals.append("")
                    row.extend(vals)
                    f.write(",".join(row) + "\n")
            print(f"Saved openmp speedup CSV: {csv_omp_speed_path}") 


    # Optionally show interactively if running locally
    if os.environ.get("SHOW_PLOT", "0") == "1":
        plt.show()


if __name__ == "__main__":
    main()
