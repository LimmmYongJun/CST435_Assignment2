import argparse
import os
from pathlib import Path
from statistics import mean
from typing import List, Tuple, Optional

# Import project modules (safe: their main() runs only under __main__)
import main_mp as mp_mod
import main_futures as fut_mod

# Optional import for plotting
try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None


def list_images(input_dir: Path) -> List[str]:
    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    return [p.name for p in sorted(input_dir.iterdir()) if p.is_file() and p.suffix.lower() in exts]


def averaged_time_mp(input_dir: Path, output_dir: Path, image_files: List[str], num_processes: int, repeats: int) -> float:
    times: List[float] = []
    for _ in range(max(1, repeats)):
        t, _ = mp_mod.process_images_parallel(image_files, str(input_dir), str(output_dir), num_processes)
        times.append(t)
    return float(mean(times))


def averaged_time_futures(input_dir: Path, output_dir: Path, image_files: List[str], workers: int, repeats: int) -> float:
    times: List[float] = []
    for _ in range(max(1, repeats)):
        t, _ = fut_mod.process_images_parallel(image_files, str(input_dir), str(output_dir), workers)
        times.append(t)
    return float(mean(times))


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def main():
    parser = argparse.ArgumentParser(description="Benchmark Python multiprocessing and concurrent.futures vs number of images.")
    parser.add_argument("--input-dir", default="input_images", help="Folder containing input images.")
    parser.add_argument("--output-dir", default="output_images/_py_bench", help="Folder to write outputs (to avoid clutter).")
    parser.add_argument("--start", type=int, default=50, help="Starting image count.")
    parser.add_argument("--end", type=int, default=1000, help="Ending image count (inclusive).")
    parser.add_argument("--step", type=int, default=50, help="Step size for image count.")
    parser.add_argument("--repeats", type=int, default=1, help="Number of runs to average per point.")
    parser.add_argument("--mp-proc", type=int, default=os.cpu_count() or 4, help="Processes for multiprocessing version.")
    parser.add_argument("--fut-workers", type=int, default=os.cpu_count() or 4, help="Workers for concurrent.futures version.")
    parser.add_argument("--save", default="plots/runtime_vs_images_python.png", help="Path to save the plot PNG.")
    parser.add_argument("--csv", default="plots/runtime_data_python.csv", help="Path to save the CSV results.")
    parser.add_argument("--skip-mp", action="store_true", help="Skip multiprocessing benchmark.")
    parser.add_argument("--skip-futures", action="store_true", help="Skip concurrent.futures benchmark.")
    parser.add_argument("--ignore-limit", action="store_true", help="Do not cap counts by available images.")

    args = parser.parse_args()

    workspace = Path(__file__).resolve().parent
    input_dir = (workspace / args.input_dir).resolve()
    out_root = (workspace / args.output_dir).resolve()
    ensure_dir(out_root)

    if not input_dir.exists():
        print(f"Input directory not found: {input_dir}")
        raise SystemExit(1)

    all_images = list_images(input_dir)
    if not all_images:
        print(f"No images found in: {input_dir}")
        raise SystemExit(1)

    counts = list(range(args.start, args.end + 1, args.step))

    available = len(all_images)
    if not args.ignore_limit and available > 0:
        capped = [c for c in counts if c <= available]
        if len(capped) < len(counts):
            print(f"Only {available} image(s) available in {input_dir}. Capping counts to <= {available}.")
        counts = capped

    if not counts:
        print("No counts to run. Check --start/--end/--step and available images.")
        raise SystemExit(0)

    print("Benchmark configuration (Python):")
    print(f"  Input dir:       {input_dir}")
    print(f"  Output dir:      {out_root}")
    print(f"  Counts:          {counts[0]}..{counts[-1]} step {args.step} ({len(counts)} points)")
    print(f"  Repeats:         {args.repeats}")
    print(f"  mp processes:    {args.mp_proc} (skip={args.skip_mp})")
    print(f"  futures workers: {args.fut_workers} (skip={args.skip_futures})")

    results: List[Tuple[int, Optional[float], Optional[float]]] = []

    for n in counts:
        print(f"\n=== Running n={n} ===")
        subset = all_images[:n]
        mp_time = None
        fut_time = None

        # Use distinct subfolders to reduce cross-run overwrite contention
        out_dir_n = out_root / f"n_{n}"
        ensure_dir(out_dir_n)

        if not args.skip_mp:
            print("Running multiprocessing version...")
            mp_time = averaged_time_mp(input_dir, out_dir_n, subset, args.mp_proc, args.repeats)
            print(f"  multiprocessing time (avg): {mp_time:.3f} s")

        if not args.skip_futures:
            print("Running concurrent.futures version...")
            fut_time = averaged_time_futures(input_dir, out_dir_n, subset, args.fut_workers, args.repeats)
            print(f"  futures time (avg): {fut_time:.3f} s")

        results.append((n, mp_time, fut_time))

    # Save CSV
    csv_path = (workspace / args.csv).resolve()
    ensure_parent(csv_path)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        f.write("count,multiprocessing_s,concurrent_futures_s\n")
        for n, mp_time, fut_time in results:
            mp_str = f"{mp_time:.6f}" if mp_time is not None else ""
            fu_str = f"{fut_time:.6f}" if fut_time is not None else ""
            f.write(f"{n},{mp_str},{fu_str}\n")
    print(f"\nSaved CSV: {csv_path}")

    # Plot
    if plt is None:
        print("matplotlib not installed. Install with: pip install matplotlib")
        return

    xs = [n for (n, _, __) in results]
    mp_vals = [t for (_, t, __) in results]
    fu_vals = [t for (_, _, t) in results]

    import math
    mp_plot = [float(t) if t is not None else math.nan for t in mp_vals]
    fu_plot = [float(t) if t is not None else math.nan for t in fu_vals]

    import matplotlib
    plt.figure(figsize=(10, 6))
    if not args.skip_mp:
        plt.plot(xs, mp_plot, label=f"multiprocessing (p={args.mp_proc})", linewidth=2)
    if not args.skip_futures:
        plt.plot(xs, fu_plot, label=f"concurrent.futures (w={args.fut_workers})", linewidth=2)

    plt.xlabel("Images processed (n)")
    plt.ylabel("Time (s)")
    plt.title("Python Runtime vs Number of Images")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend()

    png_path = (workspace / args.save).resolve()
    ensure_parent(png_path)
    plt.tight_layout()
    plt.savefig(png_path, dpi=150)
    print(f"Saved plot: {png_path}")

    # Show interactively if requested
    if os.environ.get("SHOW_PLOT", "0") == "1":
        plt.show()


if __name__ == "__main__":
    main()
