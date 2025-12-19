import os
import time
import cv2
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Tuple, Dict
from filters import apply_all_filters

def process_single_image(args: Tuple[str, str, str]) -> Tuple[str, bool, int]:
    """
    Process a single image by applying all filters.
    This function will be executed in parallel by worker processes.
    
    Args:
        args: Tuple containing (input_path, output_dir, image_filename)
        
    Returns:
        Tuple of (image_filename, success_flag, num_filters_applied)
    """
    input_path, output_dir, image_filename = args
    
    try:
        # Load the image
        image = cv2.imread(input_path)
        
        if image is None:
            print(f"⚠️  Warning: Could not read {image_filename}")
            return (image_filename, False, 0)
        
        # Apply all filters
        results = apply_all_filters(image)
        
        # Save all filtered images
        base_name = os.path.splitext(image_filename)[0]
        
        for filter_name, filtered_image in results.items():
            output_filename = f'{base_name}_{filter_name}.jpg'
            output_path = os.path.join(output_dir, output_filename)
            cv2.imwrite(output_path, filtered_image)
        
        return (image_filename, True, len(results))
        
    except Exception as e:
        print(f"❌ Error processing {image_filename}: {e}")
        return (image_filename, False, 0)

def process_images_parallel_map(input_dir: str,output_dir: str, max_workers: int = None) -> List[Tuple[str, bool, int]]:
    """
    Process all images in input_dir using ThreadPoolExecutor.map()
    """

    os.makedirs(output_dir, exist_ok=True)

    image_filenames = [
        f for f in os.listdir(input_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]

    args_list = [
        (os.path.join(input_dir, f), output_dir, f)
        for f in image_filenames
    ]

    print(f"🚀 Processing {len(args_list)} images using threads...")

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(executor.map(process_single_image, args_list))

    return results

def print_performance_metrics(num_images: int, sequential_time: float, 
                             parallel_times: dict, num_cores: int):
    """
    Print detailed performance metrics including speedup and efficiency.
    
    Args:
        num_images: Number of images processed
        sequential_time: Time taken for sequential processing
        parallel_times: Dictionary mapping max_workers to execution time
        num_cores: Total number of CPU cores available
    """
    print("\n" + "="*70)
    print("📊 PERFORMANCE ANALYSIS - concurrent.futures Module")
    print("="*70)
    
    print(f"\n🖥️  System Information:")
    print(f"   • Total CPU cores available: {num_cores}")
    print(f"   • Images processed: {num_images}")
    print(f"   • Filters per image: 6")
    print(f"   • Total operations: {num_images * 6}")
    
    print(f"\n⏱️  Execution Times:")
    print(f"   • Sequential (baseline): {sequential_time:.3f} seconds")
    
    for num_workers in sorted(parallel_times.keys()):
        print(f"   • Parallel ({num_workers} workers): {parallel_times[num_workers]:.3f} seconds")
    
    print(f"\n🚀 Speedup (compared to sequential):")
    for num_workers in sorted(parallel_times.keys()):
        speedup = sequential_time / parallel_times[num_workers]
        print(f"   • {num_workers} workers: {speedup:.2f}x")
    
    print(f"\n⚡ Efficiency (Speedup / Number of Workers):")
    for num_workers in sorted(parallel_times.keys()):
        speedup = sequential_time / parallel_times[num_workers]
        efficiency = (speedup / num_workers) * 100
        print(f"   • {num_workers} workers: {efficiency:.2f}%")
    
    print("\n" + "="*70)

def main():
    """
    Main function to run parallel image processing with performance analysis.
    Demonstrates concurrent.futures module capabilities.
    """
    print("=" * 70)
    print("🚀 Parallel Image Processing - concurrent.futures Module")
    print("=" * 70)

    # Configuration
    INPUT_DIR = "input_images"
    OUTPUT_DIR = "output_images"

    # Check if input directory exists
    if not os.path.exists(INPUT_DIR):
        print(f"❌ Error: '{INPUT_DIR}' directory not found!")
        return

    # Get all image files
    image_files = [
        f for f in os.listdir(INPUT_DIR)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]

    if not image_files:
        print(f"❌ Error: No images found in '{INPUT_DIR}'!")
        return

    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # System information
    num_cores = os.cpu_count()

    num_images_to_process = len(image_files)
    selected_images = image_files[:num_images_to_process]

    print(f"\n📸 Configuration:")
    print(f"   • Total images available: {len(image_files)}")
    print(f"   • Images to process: {num_images_to_process}")
    print(f"   • CPU cores available: {num_cores}")

    # ─────────────────────────────────────────────
    # 1️⃣ SEQUENTIAL processing
    # ─────────────────────────────────────────────
    print(f"\n{'─' * 70}")
    print("1️⃣  Running SEQUENTIAL processing (baseline)...")
    print(f"{'─' * 70}")

    start = time.perf_counter()

    total_filters = 0
    for filename in selected_images:
        input_path = os.path.join(INPUT_DIR, filename)
        _, success, num_filters = process_single_image(
            (input_path, OUTPUT_DIR, filename)
        )
        if success:
            total_filters += num_filters

    sequential_time = time.perf_counter() - start

    print(f"✅ Sequential processing complete!")
    print(f"   Time: {sequential_time:.3f} seconds")
    print(f"   Filters applied: {total_filters}")

    # ─────────────────────────────────────────────
    # 2️⃣ PARALLEL processing (threads)
    # ─────────────────────────────────────────────
    parallel_times = {}
    worker_counts = [2, 4, 8, 16]

    for num_workers in worker_counts:
        print(f"\n{'─' * 70}")
        print(f"2️⃣  Running PARALLEL processing with {num_workers} workers...")
        print(f"{'─' * 70}")

        start = time.perf_counter()

        results = process_images_parallel_map(
            INPUT_DIR,
            OUTPUT_DIR,
            max_workers=num_workers
        )

        parallel_time = time.perf_counter() - start
        parallel_times[num_workers] = parallel_time

        total_filters = sum(n for _, ok, n in results if ok)

        print(f"✅ Parallel processing ({num_workers} workers) complete!")
        print(f"   Time: {parallel_time:.3f} seconds")
        print(f"   Filters applied: {total_filters}")

        speedup = sequential_time / parallel_time
        print(f"   Speedup: {speedup:.2f}x")

    # ─────────────────────────────────────────────
    # 3️⃣ PERFORMANCE SUMMARY
    # ─────────────────────────────────────────────
    print_performance_metrics(
        num_images_to_process,
        sequential_time,
        parallel_times,
        num_cores
    )

    print(f"\n📁 All processed images saved to '{OUTPUT_DIR}/'")
    print("\n✅ Processing complete!")

if __name__ == "__main__":
    main()

