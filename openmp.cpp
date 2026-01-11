#include <filesystem>
#include <vector>
#include <string>
#include <algorithm>
#include <iostream>
#include <chrono>
#include <cstdlib>
#include <fstream>
#include <cctype>
#include <cstring>

#ifdef _OPENMP
#include <omp.h>
#endif

#undef STBI_NO_STDIO
#define STB_IMAGE_IMPLEMENTATION
#include "third_party/stb/stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "third_party/stb/stb_image_write.h"
#include "src/filters.h"
namespace fs = std::filesystem;

struct Image
{
    int width = 0;
    int height = 0;
    std::vector<uint8_t> rgb; // 3-channel RGB
};

struct PipelineTimings
{
    long long grayscale_us = 0;
    long long gaussian_us = 0;
    long long edges_us = 0;
    long long sharpen_us = 0;
    long long brightness_us = 0;
};

// Reusable buffers to avoid repeated allocation
struct SharedBuffers
{
    std::vector<uint8_t> gray;
    std::vector<uint8_t> blur;
    std::vector<uint8_t> edges;
    std::vector<uint8_t> sharp;
    std::vector<uint8_t> brightness;
    std::vector<uint8_t> final_out;
    PipelineTimings current_times;
    
    // Helper to resizing all buffers to match the current image size
    void resize(size_t num_pixels)
    {
        if (gray.size() != num_pixels) gray.resize(num_pixels);
        if (blur.size() != num_pixels) blur.resize(num_pixels);
        if (edges.size() != num_pixels) edges.resize(num_pixels);
        if (sharp.size() != num_pixels) sharp.resize(num_pixels);
        if (brightness.size() != num_pixels) brightness.resize(num_pixels);
        if (final_out.size() != num_pixels) final_out.resize(num_pixels);
    }
};

static bool load_image(const std::string &path, Image &out)
{
    int w = 0, h = 0, ch = 0;
    stbi_uc *data = stbi_load(path.c_str(), &w, &h, &ch, 3); // load 3-channel
    if (!data || w <= 0 || h <= 0)
        return false;
    out.width = w;
    out.height = h;
    size_t num_pixels = static_cast<size_t>(w) * static_cast<size_t>(h);
    out.rgb.resize(num_pixels * 3);
    std::memcpy(out.rgb.data(), data, num_pixels * 3);
    stbi_image_free(data);
    return true;
}

static bool has_image_ext(const fs::path &p)
{
    static const std::vector<std::string> exts = {".png", ".jpg", ".jpeg", ".bmp", ".gif", ".tga", ".ppm", ".pgm"};
    std::string ext = p.extension().string();
    std::transform(ext.begin(), ext.end(), ext.begin(), [](unsigned char c)
                   { return static_cast<char>(std::tolower(c)); });
    return std::find(exts.begin(), exts.end(), ext) != exts.end();
}

static std::vector<std::string> list_image_files(const std::string &folder)
{
    std::vector<std::string> files;
    try
    {
        for (const auto &entry : fs::directory_iterator(folder))
        {
            if (entry.is_regular_file() && has_image_ext(entry.path()))
            {
                files.push_back(entry.path().string());
            }
        }
    }
    catch (...)
    {
    }
    std::sort(files.begin(), files.end());
    return files;
}

static bool write_jpg(const std::string &filepath, const uint8_t *data, int w, int h)
{
    return stbi_write_jpg(filepath.c_str(), w, h, 1, data, 100) != 0;
}

// NOTE: This function must be called from WITHIN a parallel region.
static void run_pipeline_with_openmp(const Image &img, float gamma, SharedBuffers &bufs, 
                                     std::vector<uint8_t> &final_out, const std::string &return_stage,
                                     PipelineTimings &times)
{
    int w = img.width;
    int h = img.height;
    
    // Local timers for the master thread
    std::chrono::high_resolution_clock::time_point t0, t1;

    // 1. Grayscale
    // Only Master records start time (very cheap)
    if (omp_get_thread_num() == 0) t0 = std::chrono::high_resolution_clock::now();

    // The WORK (nowait allows threads to finish without waiting for laggards immediately)
    #pragma omp for schedule(static) nowait
    for(int y=0; y<h; ++y) 
        filters::rgb_to_grayscale_row(img.rgb.data() + y*w*3, bufs.gray.data() + y*w, w, /*bgr=*/false);
    
    // Only Master records end time immediately (Pure Math Time)
    if (omp_get_thread_num() == 0) {
        t1 = std::chrono::high_resolution_clock::now();
        times.grayscale_us = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
        t0 = std::chrono::high_resolution_clock::now(); // Start clock for next stage
    }

    // Explicit barrier to sync before next stage
    #pragma omp barrier

    // 2. Gaussian
    if (omp_get_thread_num() == 0) t0 = std::chrono::high_resolution_clock::now(); // Restart clock after barrier
    #pragma omp for schedule(static) nowait
    for(int y=0; y<h; ++y)
        filters::gaussian_blur_row_gray(bufs.gray.data(), bufs.blur.data(), w, h, y);

    if (omp_get_thread_num() == 0) {
        t1 = std::chrono::high_resolution_clock::now();
        times.gaussian_us = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
        t0 = std::chrono::high_resolution_clock::now();
    }
    #pragma omp barrier

    // 3. Edges
    if (omp_get_thread_num() == 0) t0 = std::chrono::high_resolution_clock::now();
    #pragma omp for schedule(static) nowait
    for(int y=0; y<h; ++y)
        filters::sobel_row_gray(bufs.blur.data(), bufs.edges.data(), w, h, y);

    if (omp_get_thread_num() == 0) {
        t1 = std::chrono::high_resolution_clock::now();
        times.edges_us = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
        t0 = std::chrono::high_resolution_clock::now();
    }
    #pragma omp barrier

    // 4. Sharpen
    if (omp_get_thread_num() == 0) t0 = std::chrono::high_resolution_clock::now();
    #pragma omp for schedule(static) nowait
    for(int y=0; y<h; ++y)
        filters::sharpen_row_gray(bufs.edges.data(), bufs.sharp.data(), w, h, y);

    if (omp_get_thread_num() == 0) {
        t1 = std::chrono::high_resolution_clock::now();
        times.sharpen_us = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
        t0 = std::chrono::high_resolution_clock::now();
    }
    #pragma omp barrier

    // 5. Brightness
    if (omp_get_thread_num() == 0) t0 = std::chrono::high_resolution_clock::now();
    #pragma omp for schedule(static) nowait
    for(int y=0; y<h; ++y)
        filters::brightness_row_gray(bufs.sharp.data(), bufs.brightness.data(), w, h, y, gamma);

    if (omp_get_thread_num() == 0) {
        t1 = std::chrono::high_resolution_clock::now();
        times.brightness_us = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
    }
    #pragma omp barrier

    // Copy to final output (Only master needs to do this logic)
    #pragma omp single
    {
        if (return_stage == "grayscale") final_out = bufs.gray;
        else if (return_stage == "gaussian") final_out = bufs.blur;
        else if (return_stage == "edges") final_out = bufs.edges;
        else if (return_stage == "sharpen") final_out = bufs.sharp;
        else final_out = bufs.brightness; 
    }
}

int main(int argc, char **argv)
{
    std::string folder = "input_images";
    std::string filter = "all";
    float gamma = 0.9f;
    size_t max_images = 0; // 0 means process all
    if (argc > 1)
        folder = argv[1];
    if (argc > 2)
    {
        std::string a2 = argv[2];
        bool a2_is_num = !a2.empty() && std::all_of(a2.begin(), a2.end(), [](unsigned char c)
                                                    { return std::isdigit(c) != 0; });
        if (a2_is_num)
            max_images = static_cast<size_t>(std::stoull(a2));
        else
            filter = a2;
    }
    if (argc > 3)
    {
        std::string a3 = argv[3];
        bool a3_is_num = !a3.empty() && std::all_of(a3.begin(), a3.end(), [](unsigned char c)
                                                    { return std::isdigit(c) != 0; });
        if (a3_is_num)
            max_images = static_cast<size_t>(std::stoull(a3));
        else
            filter = a3;
    }

    auto files = list_image_files(folder);
    if (files.empty())
    {
        std::cerr << "No images found in folder: " << folder << std::endl;
        return 1;
    }
    size_t to_process = files.size();
    if (max_images > 0 && max_images < to_process)
        to_process = max_images;
    std::cout << "Found " << files.size() << " image(s), processing " << to_process << "." << std::endl;

    std::vector<Image> images;
    images.reserve(files.size());
    for (size_t i = 0; i < to_process; ++i)
    {
        const auto &path = files[i];
        Image img;
        if (load_image(path, img))
            images.push_back(std::move(img));
        else
            std::cerr << "Failed to load: " << path << std::endl;
    }
    if (images.empty())
    {
        std::cerr << "No images could be loaded." << std::endl;
        return 1;
    }

    // If specific filter requested, run only that; else run all five
    std::vector<std::string> filters_to_run;
    if (filter == "all")
        filters_to_run = {"grayscale", "gaussian", "edges", "sharpen", "brightness"};
    else
        filters_to_run = {filter};

    uint64_t checksum = 0;
    long long total_us = 0;
    
    // Create output directory for the specific filter
    std::string out_subdir = (filter == "all") ? "pipeline_final" : filter;
    fs::path out_dir = fs::path("output_images") / out_subdir;
    fs::create_directories(out_dir);

    // Prepare shared resources
    SharedBuffers bufs;

    // --- PARALLEL REGION STARTS HERE ---
    // The thread team is created ONCE and reused for all images.
    #pragma omp parallel
    {
        for (size_t i = 0; i < images.size(); ++i)
        {
            const auto &img = images[i];
            size_t num_pixels = static_cast<size_t>(img.width) * img.height;

            // 1. Resize buffers if needed (only one thread needs to do this)
            #pragma omp single
            {
                bufs.resize(num_pixels);
                // Reset timings for this image
                bufs.current_times = PipelineTimings{};
            }
            // Barrier implied by omp single? Yes. But keep safe.
            
            // Run pipeline
            run_pipeline_with_openmp(img, gamma, bufs, bufs.final_out, filter, bufs.current_times);

            // Accumulate timings and write (Master only)
            #pragma omp single
            {
                // Add to total
                 for (const auto &filt : filters_to_run) {
                    long long val = 0;
                    if(filt == "grayscale") val = bufs.current_times.grayscale_us;
                    else if(filt == "gaussian") val = bufs.current_times.gaussian_us;
                    else if(filt == "edges") val = bufs.current_times.edges_us;
                    else if(filt == "sharpen") val = bufs.current_times.sharpen_us;
                    else if(filt == "brightness") val = bufs.current_times.brightness_us;
                    total_us += val;
                }
                
                // Checksum
                for (uint8_t v : bufs.final_out) checksum += v;

                // Write
                fs::path in_path = files[i];
                std::string stem = in_path.empty() ? ("img_" + std::to_string(i)) : fs::path(in_path).stem().string();
                std::string fname = stem + "_" + out_subdir + ".jpg";
                fs::path out_path = out_dir / fname;
                (void)write_jpg(out_path.string(), bufs.final_out.data(), img.width, img.height);
            }
        }
    } // End Parallel

    std::cout << "Saved outputs to: " << out_dir.string() << std::endl;
    std::cout << "OpenMP enabled: yes" << std::endl;
    int max_threads = omp_get_max_threads();
    const char* env_threads = std::getenv("OMP_NUM_THREADS");
    if (env_threads) max_threads = std::atoi(env_threads);
    std::cout << "OpenMP threads: " << max_threads << std::endl; // Approximation
    
    std::cout << "OpenMP checksum: " << checksum << std::endl;
    auto ms = total_us / 1000;
    
    if (filter == "all") std::cout << "OpenMP time (all filters): " << ms << " ms" << std::endl;
    if(filter == "grayscale") std::cout << "OpenMP time (grayscale): " << ms << " ms" << std::endl;
    if(filter == "gaussian") std::cout << "OpenMP time (gaussian): " << ms << " ms" << std::endl;
    if(filter == "edges") std::cout << "OpenMP time (edges): " << ms << " ms" << std::endl;
    if(filter == "sharpen") std::cout << "OpenMP time (sharpen): " << ms << " ms" << std::endl;
    if(filter == "brightness") std::cout << "OpenMP time (brightness): " << ms << " ms" << std::endl;

    return 0;
}
