"""
Image Processing Filters for CST435 Assignment 2

This module implements five image processing filters:
1. Grayscale Conversion
2. Gaussian Blur
3. Edge Detection (Sobel)
4. Image Sharpening
5. Brightness Adjustment

All filters are designed to work with OpenCV images (NumPy arrays).

Author: [Your Name]
Date: December 2024
"""

import cv2
import numpy as np
from typing import Tuple


def apply_grayscale(image: np.ndarray) -> np.ndarray:
    """
    Convert RGB image to grayscale using luminance formula.
    
    Uses the standard luminance formula:
    Gray = 0.299*R + 0.587*G + 0.114*B
    
    Args:
        image: Input BGR image (OpenCV format)
        
    Returns:
        Grayscale image as 3-channel BGR for consistency
    """
    # Convert BGR to grayscale using OpenCV's optimized function
    # OpenCV uses the luminance formula internally
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Convert back to 3-channel for consistency with other filters
    gray_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    
    return gray_bgr


def apply_gaussian_blur(image: np.ndarray, kernel_size: Tuple[int, int] = (3, 3), 
                        sigma: float = 0) -> np.ndarray:
    """
    Apply Gaussian blur filter for image smoothing.
    
    Uses a 3×3 Gaussian kernel by default as specified in requirements.
    
    Args:
        image: Input BGR image
        kernel_size: Size of the Gaussian kernel (must be odd numbers)
        sigma: Standard deviation for Gaussian kernel (0 = auto-calculate)
        
    Returns:
        Blurred image
    """
    # Apply Gaussian blur using OpenCV's optimized implementation
    blurred = cv2.GaussianBlur(image, kernel_size, sigma)
    
    return blurred


def apply_edge_detection(image: np.ndarray) -> np.ndarray:
    """
    Apply Sobel edge detection filter.
    
    Uses Sobel operator to detect edges in both X and Y directions,
    then combines them to get the final edge map.
    
    Args:
        image: Input BGR image
        
    Returns:
        Edge-detected image (3-channel for consistency)
    """
    # Convert to grayscale first (edge detection works on intensity)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply Sobel operator in X direction
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    
    # Apply Sobel operator in Y direction
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    
    # Combine both gradients using magnitude
    sobel_combined = np.sqrt(sobel_x**2 + sobel_y**2)
    
    # Normalize to 0-255 range
    sobel_combined = np.uint8(np.clip(sobel_combined, 0, 255))
    
    # Convert to 3-channel BGR for consistency
    edges_bgr = cv2.cvtColor(sobel_combined, cv2.COLOR_GRAY2BGR)
    
    return edges_bgr


def apply_sharpening(image: np.ndarray) -> np.ndarray:
    """
    Apply image sharpening filter to enhance edges and details.
    
    Uses an unsharp masking technique:
    Sharpened = Original + (Original - Blurred) * amount
    
    Args:
        image: Input BGR image
        
    Returns:
        Sharpened image
    """
    # Method 1: Using a sharpening kernel
    # This kernel enhances edges and details
    kernel = np.array([
        [0, -1, 0],
        [-1, 5, -1],
        [0, -1, 0]
    ])
    
    # Apply the sharpening kernel using filter2D
    sharpened = cv2.filter2D(image, -1, kernel)
    
    return sharpened


def apply_brightness_adjustment(image: np.ndarray, beta: int = 50) -> np.ndarray:
    """
    Adjust image brightness by adding/subtracting a constant value.
    
    Brightness adjustment formula:
    New_pixel = Original_pixel + beta
    
    Args:
        image: Input BGR image
        beta: Brightness adjustment value
              Positive values increase brightness
              Negative values decrease brightness
              Range: typically -100 to +100
              
    Returns:
        Brightness-adjusted image
    """
    # Add beta to all pixels using OpenCV's convertScaleAbs
    # This automatically handles overflow/underflow (clips to 0-255)
    adjusted = cv2.convertScaleAbs(image, alpha=1.0, beta=beta)
    
    return adjusted


def apply_all_filters(image: np.ndarray, output_prefix: str = "filtered") -> dict:
    """
    Apply all five filters to an image and return results.
    
    This is a convenience function for testing or batch processing.
    
    Args:
        image: Input BGR image
        output_prefix: Prefix for output filenames (not used in processing)
        
    Returns:
        Dictionary containing all filtered images with descriptive keys
    """
    results = {
        'grayscale': apply_grayscale(image),
        'gaussian_blur': apply_gaussian_blur(image),
        'edge_detection': apply_edge_detection(image),
        'sharpening': apply_sharpening(image),
        'brightness_increase': apply_brightness_adjustment(image, beta=50),
        'brightness_decrease': apply_brightness_adjustment(image, beta=-50)
    }
    
    return results


# Filter mapping for easy access by name
FILTER_FUNCTIONS = {
    'grayscale': apply_grayscale,
    'blur': apply_gaussian_blur,
    'edges': apply_edge_detection,
    'sharpen': apply_sharpening,
    'brighten': lambda img: apply_brightness_adjustment(img, beta=50),
    'darken': lambda img: apply_brightness_adjustment(img, beta=-50)
}


if __name__ == "__main__":
    """
    Simple test to verify filters work correctly.
    This section runs only when filters.py is executed directly.
    For batch processing and performance testing, use main_mp.py or main_futures.py
    """
    import os
    
    # Check if input_images directory exists
    if not os.path.exists('input_images'):
        print("❌ Error: 'input_images' directory not found!")
        print("   Please create the directory and add some images.")
        exit(1)
    
    # Get image files from input_images directory
    image_files = [f for f in os.listdir('input_images') if f.endswith(('.jpg', '.jpeg', '.png'))]
    
    if not image_files:
        print("❌ Error: No images found in 'input_images' directory!")
        exit(1)
    
    # Use first image for testing
    test_image = image_files[0]
    test_image_path = os.path.join('input_images', test_image)
    
    print(f"📸 Testing filters on: {test_image}")
    
    # Load image
    image = cv2.imread(test_image_path)
    
    if image is None:
        print(f"❌ Error: Could not read {test_image}")
        exit(1)
    
    print(f"✅ Image loaded: {image.shape[1]}x{image.shape[0]} pixels")
    
    # Create output directory if it doesn't exist
    os.makedirs('output_images', exist_ok=True)
    
    # Apply all filters and save results
    print("\n🔧 Applying filters...")
    results = apply_all_filters(image)
    
    base_name = os.path.splitext(test_image)[0]
    
    for filter_name, filtered_image in results.items():
        output_filename = f'test_{base_name}_{filter_name}.jpg'
        output_path = os.path.join('output_images', output_filename)
        cv2.imwrite(output_path, filtered_image)
        print(f"   ✓ {filter_name}")
    
    print(f"\n✅ All filters applied successfully!")
    print(f"📁 Check 'output_images' folder for results.")
    print(f"\n💡 Use main_mp.py or main_futures.py for batch processing.")

