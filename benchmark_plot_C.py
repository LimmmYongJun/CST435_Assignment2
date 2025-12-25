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
    parser = argparse.ArgumentParser(description="Benchmark threads.exe and openmp.exe vs number of images.")
    parser.add_argument("--input-dir", default="input_images", help="Folder containing input images.")
    parser.add_argument("--start", type=int, default=50, help="Starting image count.")
    parser.add_argument("--end", type=int, default=1000, help="Ending image count (inclusive).")
    parser.add_argument("--step", type=int, default=50, help="Step size for image count.")
    parser.add_argument("--repeats", type=int, default=1, help="Number of runs to average per point.")
    parser.add_argument("--save", default="plots/v1_C_runtime_vs_images.png", help="Path to save the plot PNG.")
    parser.add_argument("--csv", default="plots/v1_C_runtime_data.csv", help="Path to save the CSV results.")
    parser.add_argument("--save-speedup", default="plots/v1_C_speedup_vs_images.png", help="Path to save the speedup plot PNG.")
    parser.add_argument("--csv-speedup", default="plots/v1_C_speedup_data.csv", help="Path to save the speedup CSV results.")
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

    results = []  # list of tuples: (count, threads_ms or None, openmp_ms or None, omp1_ms or None)

    for n in counts:
        print(f"\n=== Running n={n} ===")
        t_ms = None
        o_ms = None
        o1_ms = None

        if not args.skip_threads and threads_exe:
            print("Running threads.exe...")
            t_ms, logs = averaged_time(threads_exe, input_dir, n, args.repeats)
            if t_ms is None:
                print("  WARNING: Could not parse time from threads.exe output.")
            else:
                print(f"  threads.exe time (avg): {t_ms:.2f} ms")

        if not args.skip_openmp and openmp_exe:
            print("Running openmp.exe...")
            o_ms, logs = averaged_time(openmp_exe, input_dir, n, args.repeats)
            if o_ms is None:
                print("  WARNING: Could not parse time from openmp.exe output.")
            else:
                print(f"  openmp.exe time (avg): {o_ms:.2f} ms")

            print("Running openmp.exe with OMP_NUM_THREADS=1 (serial baseline)...")
            o1_ms, _ = averaged_time(
                openmp_exe,
                input_dir,
                n,
                args.repeats,
                env_override={"OMP_NUM_THREADS": "1"},
            )
            if o1_ms is None:
                print("  WARNING: Could not parse time from openmp.exe (1 thread) output.")
            else:
                print(f"  openmp.exe (1 thread) time (avg): {o1_ms:.2f} ms")

        results.append((n, t_ms, o_ms, o1_ms))

    # Save CSV
    csv_path = (workspace / args.csv).resolve()
    ensure_plots_dir(csv_path)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        f.write("count,threads_ms,openmp_ms,openmp_1thread_ms\n")
        for n, t_ms, o_ms, o1_ms in results:
            t_str = f"{t_ms:.4f}" if t_ms is not None else ""
            o_str = f"{o_ms:.4f}" if o_ms is not None else ""
            o1_str = f"{o1_ms:.4f}" if o1_ms is not None else ""
            f.write(f"{n},{t_str},{o_str},{o1_str}\n")
    print(f"\nSaved CSV: {csv_path}")

    # Plot
    if plt is None:
        print("matplotlib not installed. Install with: pip install matplotlib")
        return

    xs = [n for (n, _, __, ___) in results]
    thr = [t for (_, t, __, ___) in results]
    omp = [o for (_, _, o, ___) in results]
    omp1 = [o1 for (_, _, __, o1) in results]

    # Filter out None for plotting by masking with NaN
    import math
    thr_plot = [float(t) if t is not None else math.nan for t in thr]
    omp_plot = [float(o) if o is not None else math.nan for o in omp]

    plt.figure(figsize=(10, 6))
    if not args.skip_threads:
        plt.plot(xs, thr_plot, label="threads.exe", linewidth=2)
    if not args.skip_openmp:
        plt.plot(xs, omp_plot, label="openmp.exe", linewidth=2)

    plt.xlabel("Images processed (n)")
    plt.ylabel("Time (ms)")
    plt.title("Runtime vs Number of Images")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend()

    png_path = (workspace / args.save).resolve()
    ensure_plots_dir(png_path)
    plt.tight_layout()
    plt.savefig(png_path, dpi=150)
    print(f"Saved plot: {png_path}")

    # Speedup plot (using OpenMP 1-thread as serial baseline)
    if not args.skip_openmp:
        # Compute speedups where both baseline and parallel time are present
        speed_thr = []
        speed_omp = []
        for t, o, b in zip(thr, omp, omp1):
            if b is not None and t is not None and t > 0:
                speed_thr.append(b / t)
            else:
                speed_thr.append(float("nan"))
            if b is not None and o is not None and o > 0:
                speed_omp.append(b / o)
            else:
                speed_omp.append(float("nan"))

        # Save speedup CSV
        csv_speed_path = (workspace / args.csv_speedup).resolve()
        ensure_plots_dir(csv_speed_path)
        with open(csv_speed_path, "w", newline="", encoding="utf-8") as f:
            f.write("count,baseline_ms,threads_speedup,openmp_speedup\n")
            for n, b, st, so in zip(xs, omp1, speed_thr, speed_omp):
                b_str = f"{float(b):.4f}" if b is not None else ""
                st_str = "" if (st != st) else f"{st:.4f}"  # NaN check
                so_str = "" if (so != so) else f"{so:.4f}"
                f.write(f"{n},{b_str},{st_str},{so_str}\n")
        print(f"Saved speedup CSV: {csv_speed_path}")

        # Plot speedup
        plt.figure(figsize=(10, 6))
        import math
        sp_thr_plot = [float(s) if s == s else math.nan for s in speed_thr]
        sp_omp_plot = [float(s) if s == s else math.nan for s in speed_omp]
        if not args.skip_threads:
            plt.plot(xs, sp_thr_plot, label="Speedup vs threads.exe", linewidth=2)
        plt.plot(xs, sp_omp_plot, label="Speedup vs openmp.exe", linewidth=2)
        plt.xlabel("Images processed (n)")
        plt.ylabel("Speedup (×)")
        plt.title("Speedup vs Number of Images (baseline: OpenMP 1 thread)")
        plt.grid(True, linestyle="--", alpha=0.4)
        plt.legend()
        png_speed_path = (workspace / args.save_speedup).resolve()
        ensure_plots_dir(png_speed_path)
        plt.tight_layout()
        plt.savefig(png_speed_path, dpi=150)
        print(f"Saved speedup plot: {png_speed_path}")

    # Optionally show interactively if running locally
    if os.environ.get("SHOW_PLOT", "0") == "1":
        plt.show()


if __name__ == "__main__":
    main()
