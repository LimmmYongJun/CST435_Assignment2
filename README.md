# CST435 Assignment 2 - Parallel Image Processing

A parallel image processing system that applies various filters to images using Python (multiprocessing and concurrent.futures) and C++ (std::thread and OpenMP). The C++ implementation processes images in-place (no file outputs) and reports timing for both approaches.

---

## 📋 Project Overview

This project implements 5 image processing filters and compares the performance of two parallel programming approaches:

### **Image Filters Implemented:**

1. **Grayscale Conversion** - Convert RGB to grayscale using luminance formula
2. **Gaussian Blur** - 3×3 Gaussian kernel for smoothing
3. **Edge Detection** - Sobel filter to detect edges
4. **Image Sharpening** - Enhance edges and details
5. **Brightness Adjustment** - Increase or decrease brightness

### **Parallel Paradigms:**

- **multiprocessing module** (`main_mp.py`) - Process-based parallelism
- **concurrent.futures module** (`main_futures.py`) - Higher-level parallelism interface

### **Purpose:**

- Demonstrate parallel processing techniques
- Compare performance across different process counts (2, 4, 8)
- Calculate speedup and efficiency metrics
- Deploy on Google Cloud Platform for scalability analysis

---

## 🚀 Quick Setup

### **Prerequisites:**

- **Conda** (Miniconda or Anaconda) - [Download Miniconda](https://docs.conda.io/en/latest/miniconda.html)

### **Setup Steps:**

```bash
# 1. Clone the repository
git clone <your-repo-url>
cd CST435_Assignment2

# 2. Create the conda environment
conda env create -f environment.yml

# 3. Activate the environment
conda activate cst435_assign2

```

That's it! You're ready to run the code.

---

## 🎯 How to Run (Python)

### **1. Test Individual Filters**

```bash
python filters.py
```

**What it does:**

- Tests all 5 filters on a single sample image
- Saves results to `output_images/` folder
- Verifies filters work correctly

### **2. Run Multiprocessing Version**

```bash
python main_mp.py
```

**What it does:**

- Processes all 50 images using `multiprocessing` module
- Tests with 2, 4, and 8 processes
- Shows performance metrics (execution time, speedup, efficiency)
- Compares parallel vs sequential performance

**Expected output:**

```
🚀 Parallel Image Processing - Multiprocessing Module
======================================================================

📸 Configuration:
   • Total images available: 50
   • Images to process: 50
   • CPU cores available: 16

──────────────────────────────────────────────────────────────────────
1️⃣  Running SEQUENTIAL processing (baseline)...
──────────────────────────────────────────────────────────────────────
✅ Sequential processing complete!
   Time: 1.123 seconds

──────────────────────────────────────────────────────────────────────
2️⃣  Running PARALLEL processing with 2 processes...
──────────────────────────────────────────────────────────────────────
✅ Parallel processing (2 processes) complete!
   Time: 0.621 seconds
   Speedup: 1.81x

📊 PERFORMANCE ANALYSIS - Multiprocessing Module
======================================================================
🚀 Speedup (compared to sequential):
   • 2 processes: 1.81x
   • 4 processes: 2.94x
   • 8 processes: 3.75x

⚡ Efficiency:
   • 2 processes: 90.50%
   • 4 processes: 73.50%
   • 8 processes: 46.88%
```

### **3. Run concurrent.futures Version**

```bash
python main_futures.py
```

**What it does:**

- Same as `main_mp.py` but uses `concurrent.futures` module
- Provides cleaner API and better exception handling
- Allows comparison between the two parallel paradigms

---

## 📁 Project Structure

```
CST435_Assignment2/
│
├── environment.yml          # Conda environment configuration
├── .gitignore              # Git ignore rules
├── README.md               # This file
│
├── filters.py              # Image filter implementations (5 filters)
├── main_mp.py              # Multiprocessing paradigm implementation
├── main_futures.py         # Concurrent.futures paradigm implementation
├── CMakeLists.txt          # C++ build configuration
├── src/                    # C++ sources (image loader + main)
│   ├── image_loader.h
│   ├── image_loader.cpp
│   └── main.cpp
├── third_party/
│   └── stb/                # (optional) vendored headers
│
├── input_images/           # 50 sample images (tracked in Git)
│   └── *.jpg
│
└── output_images/          # Processed images (NOT tracked in Git)
    └── (generated files)
```

---

## 🧰 C++ Build & Run (Windows)

Two standalone programs with no project files:

- `threads.cpp` — std::thread implementation
- `openmp.cpp` — OpenMP implementation

### Compile

```powershell
g++ -std=c++17 -O2 -pthread -I third_party/stb threads.cpp -o threads.exe
g++ -std=c++17 -O2 -fopenmp -I third_party/stb openmp.cpp -o openmp.exe
```

### Run

You can optionally pass how many images to process. If omitted, all images in the folder are processed. The `filter` argument can be provided in any order with the numeric count (the program detects which is which).

Examples (Windows PowerShell):

```powershell
# Process all images
.\threads.exe input_images
.\openmp.exe input_images

# Process only 10 images (from input_images)
.\threads.exe input_images 10
.\openmp.exe input_images 10

# Run a specific filter on only 10 images
.\threads.exe input_images edges 10
.\openmp.exe input_images gaussian 10

# Order-insensitive for filter vs count
.\threads.exe input_images 10 sharpen
.\openmp.exe input_images 5 grayscale
```

### Compile and run (Linux on GCP)

```
g++ -std=c++17 -O2 -pthread -I third_party/stb threads.cpp -o threads
./threads input_images
./threads input_images 10
```

```
g++ -std=c++17 -O2 -fopenmp -I third_party/stb openmp.cpp -o openmp
./openmp input_images
./openmp input_images 10
```

Output includes counts and times, e.g.:

```
Found 50 image(s), processing 10.
Saved outputs to: output_images/grayscale
Saved outputs to: output_images/gaussian_blur
Saved outputs to: output_images/edge_detection
Saved outputs to: output_images/sharpen
Saved outputs to: output_images/brightness_adjustment
Threads checksum: 123456
Threads time (all filters): 123 ms

Found 50 image(s), processing 10.
Saved outputs to: output_images/grayscale
Saved outputs to: output_images/gaussian_blur
Saved outputs to: output_images/edge_detection
Saved outputs to: output_images/sharpen
Saved outputs to: output_images/brightness_adjustment
OpenMP enabled: yes
OpenMP checksum: 123456
OpenMP time (all filters): 78 ms
```

---

## 💻 Development

### **Dependencies:**

| Package       | Version   | Purpose          |
| ------------- | --------- | ---------------- |
| Python        | 3.10      | Base interpreter |
| NumPy         | Latest    | Array operations |
| OpenCV-Python | 4.12.0.88 | Image processing |
