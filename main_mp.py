"""
Parallel Image Processing using multiprocessing module
CST435 Assignment 2

This implementation uses Python's multiprocessing module to process
multiple images in parallel across different CPU cores.

Features:
- Process multiple images concurrently
- Apply all 5 filters to each image
- Performance metrics (speedup, efficiency)
- Configurable number of processes

Author: [Your Name]
Date: December 2024
"""

import os
import time
import cv2
import multiprocessing as mp
from typing import List, Tuple
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


def process_images_parallel(image_files: List[str], input_dir: str, 
                           output_dir: str, num_processes: int) -> Tuple[float, int]:
    """
    Process multiple images in parallel using multiprocessing.Pool
    
    Args:
        image_files: List of image filenames to process
        input_dir: Directory containing input images
        output_dir: Directory to save processed images
        num_processes: Number of parallel processes to use
        
    Returns:
        Tuple of (execution_time, total_filters_applied)
    """
    # Prepare arguments for each image
    args_list = [
        (os.path.join(input_dir, img), output_dir, img) 
        for img in image_files
    ]
    
    # Start timing
    start_time = time.time()
    
    # Create a pool of worker processes
    with mp.Pool(processes=num_processes) as pool:
        # Map the processing function to all images
        # pool.map distributes work across processes
        results = pool.map(process_single_image, args_list)
    
    # End timing
    end_time = time.time()
    execution_time = end_time - start_time
    
    # Count successful operations
    successful = sum(1 for _, success, _ in results if success)
    total_filters = sum(num_filters for _, _, num_filters in results)
    
    return execution_time, total_filters


def process_images_sequential(image_files: List[str], input_dir: str, 
                             output_dir: str) -> Tuple[float, int]:
    """
    Process images sequentially (one at a time) for baseline comparison.
    
    Args:
        image_files: List of image filenames to process
        input_dir: Directory containing input images
        output_dir: Directory to save processed images
        
    Returns:
        Tuple of (execution_time, total_filters_applied)
    """
    # Prepare arguments
    args_list = [
        (os.path.join(input_dir, img), output_dir, img) 
        for img in image_files
    ]
    
    # Start timing
    start_time = time.time()
    
    # Process each image sequentially
    results = [process_single_image(args) for args in args_list]
    
    # End timing
    end_time = time.time()
    execution_time = end_time - start_time
    
    # Count successful operations
    total_filters = sum(num_filters for _, _, num_filters in results)
    
    return execution_time, total_filters


def print_performance_metrics(num_images: int, sequential_time: float, 
                             parallel_times: dict, num_cores: int):
    """
    Print detailed performance metrics including speedup and efficiency.
    
    Args:
        num_images: Number of images processed
        sequential_time: Time taken for sequential processing
        parallel_times: Dictionary mapping num_processes to execution time
        num_cores: Total number of CPU cores available
    """
    print("\n" + "="*70)
    print("📊 PERFORMANCE ANALYSIS - Multiprocessing Module")
    print("="*70)
    
    print(f"\n🖥️  System Information:")
    print(f"   • Total CPU cores available: {num_cores}")
    print(f"   • Images processed: {num_images}")
    print(f"   • Filters per image: 6")
    print(f"   • Total operations: {num_images * 6}")
    
    print(f"\n⏱️  Execution Times:")
    print(f"   • Sequential (baseline): {sequential_time:.3f} seconds")
    
    for num_proc in sorted(parallel_times.keys()):
        print(f"   • Parallel ({num_proc} processes): {parallel_times[num_proc]:.3f} seconds")
    
    print(f"\n🚀 Speedup (compared to sequential):")
    for num_proc in sorted(parallel_times.keys()):
        speedup = sequential_time / parallel_times[num_proc]
        print(f"   • {num_proc} processes: {speedup:.2f}x")
    
    print(f"\n⚡ Efficiency (Speedup / Number of Processes):")
    for num_proc in sorted(parallel_times.keys()):
        speedup = sequential_time / parallel_times[num_proc]
        efficiency = (speedup / num_proc) * 100
        print(f"   • {num_proc} processes: {efficiency:.2f}%")
    
    print("\n" + "="*70)


def main():
    """
    Main function to run parallel image processing with performance analysis.
    """
    print("="*70)
    print("🚀 Parallel Image Processing - Multiprocessing Module")
    print("="*70)
    
    # Configuration
    INPUT_DIR = 'input_images'
    OUTPUT_DIR = 'output_images'
    
    # Check if input directory exists
    if not os.path.exists(INPUT_DIR):
        print(f"❌ Error: '{INPUT_DIR}' directory not found!")
        return
    
    # Get all image files
    image_files = [f for f in os.listdir(INPUT_DIR) 
                   if f.endswith(('.jpg', '.jpeg', '.png'))]
    
    if not image_files:
        print(f"❌ Error: No images found in '{INPUT_DIR}'!")
        return
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Get system information
    num_cores = mp.cpu_count()
    
    # Select subset of images for testing (adjust as needed)
    # Process all available images (or adjust number as needed)
    # Note: Process overhead is significant for small workloads
    # Recommendation: Use at least 20-50 images to see speedup benefits
    num_images_to_process = len(image_files)  # Process all images
    selected_images = image_files[:num_images_to_process]
    
    print(f"\n📸 Configuration:")
    print(f"   • Total images available: {len(image_files)}")
    print(f"   • Images to process: {num_images_to_process}")
    print(f"   • CPU cores available: {num_cores}")
    
    # Step 1: Sequential processing (baseline)
    print(f"\n{'─'*70}")
    print("1️⃣  Running SEQUENTIAL processing (baseline)...")
    print(f"{'─'*70}")
    
    sequential_time, total_filters = process_images_sequential(
        selected_images, INPUT_DIR, OUTPUT_DIR
    )
    
    print(f"✅ Sequential processing complete!")
    print(f"   Time: {sequential_time:.3f} seconds")
    print(f"   Filters applied: {total_filters}")
    
    # Step 2: Parallel processing with different process counts
    parallel_times = {}
    
    # Test with different numbers of processes: 2, 4, 8
    # Note: Using too many processes can reduce efficiency due to overhead
    process_counts = [2, 4, 8]
    
    for num_proc in process_counts:
        print(f"\n{'─'*70}")
        print(f"2️⃣  Running PARALLEL processing with {num_proc} processes...")
        print(f"{'─'*70}")
        
        parallel_time, total_filters = process_images_parallel(
            selected_images, INPUT_DIR, OUTPUT_DIR, num_proc
        )
        
        parallel_times[num_proc] = parallel_time
        
        print(f"✅ Parallel processing ({num_proc} processes) complete!")
        print(f"   Time: {parallel_time:.3f} seconds")
        print(f"   Filters applied: {total_filters}")
        
        # Calculate and show immediate speedup
        speedup = sequential_time / parallel_time
        print(f"   Speedup: {speedup:.2f}x")
    
    # Step 3: Print comprehensive performance metrics
    print_performance_metrics(
        num_images_to_process, 
        sequential_time, 
        parallel_times, 
        num_cores
    )
    
    print(f"\n📁 All processed images saved to '{OUTPUT_DIR}/'")
    print("\n✅ Processing complete!")


if __name__ == "__main__":
    # Required for Windows to prevent recursive process spawning
    mp.freeze_support()
    
    main()

