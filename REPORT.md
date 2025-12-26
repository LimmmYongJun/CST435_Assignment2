# C++ Implementation Report (Option 1)

## Core Logic

The C++ implementation relies on the **stb** single-file public domain libraries for image Input/Output operations, specifically:

- `stb_image.h`: Used for loading images from disk. It handles various formats (PNG, JPG, etc.) and provides raw pixel data. In this project, images are loaded and converted to 8-bit grayscale vectors.
- `stb_image_write.h`: Used for saving the processed results back to disk, primarily in PNG format.

The core image processing logic (Gaussian blur, Sobel edge detection, etc.) is implemented in `src/filters.h`, operating directly on raw 1D arrays of pixel data (`std::vector<uint8_t>`). This approach avoids the overhead of heavy frameworks like OpenCV for the core algorithms, ensuring the performance measurements reflect the raw computation and parallelization efficiency.

## Paradigm A: std::thread

### Task Division

The implementation utilizes **Data Parallelism**. The image processing workload is parallelized by decomposing the image domain spatially. Specifically, the image is divided horizontally into strips or chunks of rows.

The total height of the image ($h$) is divided by the number of available hardware threads ($nt$). Each thread is assigned a specific range of rows $[y_{start}, y_{end})$ to process independently. Since the filters (like convolution) primarily depend on local neighborhoods, threads can write to disjoint sections of the output buffer without race conditions.

### Thread Management

The application queries the hardware for the optimal number of concurrent threads using `std::thread::hardware_concurrency()`.

1.  **Calculation**: The number of rows per thread is calculated: `rows_per = (h + nt - 1) / nt`.
2.  **Launch**: A loop creates `std::thread` objects. Each thread executes a lambda function that calls the specific filter function for its assigned row range (`y0` to `y1`).
3.  **Synchronization**: The main thread waits for all worker threads to complete using `join()`.

**Code Snippet (Thread Management):**

```cpp
unsigned int nt = std::thread::hardware_concurrency();
int rows_per = (h + static_cast<int>(nt) - 1) / static_cast<int>(nt);

std::vector<std::thread> threads;
for (unsigned int t = 0; t < nt; ++t)
{
    int y0 = static_cast<int>(t) * rows_per;
    int y1 = std::min(h, y0 + rows_per);
    if (y0 >= h) break;

    // Launch thread with lambda capturing specific row range
    threads.emplace_back([&img, &out, w, h, y0, y1, &filter, beta]()
                         { apply_filter_rows(filter, img.gray.data(), out.data(), w, h, y0, y1, beta); });
}

// Wait for all threads to finish
for (auto &th : threads)
    th.join();
```

## Paradigm B: OpenMP

### Directives Used

The OpenMP implementation simplifies the parallelization significantly using compiler directives. The primary directive used is:
`#pragma omp parallel for schedule(static)`

This directive is placed immediately before the `for` loops iterating over the image rows (`y` from 0 to `h`). It instructs the compiler to automatically fork a team of threads and distribute the loop iterations (rows) among them.

### Scheduling Choice

**`schedule(static)`** was chosen for this implementation.

- **Reasoning**: The workload for image processing filters (like Gaussian blur or brightness adjustment) is highly **uniform**. Processing row $N$ takes almost exactly the same amount of time as processing row $N+1$, as the number of pixels and the mathematical operations per pixel are constant.
- **Benefit**: Static scheduling divides the loop iterations into equal-sized chunks at the beginning of the loop. This has significantly **lower runtime overhead** compared to `dynamic` or `guided` scheduling, which require threads to request new chunks of work during execution. Since load balancing is not a major concern with uniform workloads, static scheduling offers the best performance.

**Code Snippet:**

```cpp
if (filter == "gaussian" || filter == "blur")
{
#pragma omp parallel for schedule(static)
    for (int y = 0; y < h; ++y)
        filters::gaussian_blur_row_gray(img.gray.data(), out.data(), w, h, y);
}
```
