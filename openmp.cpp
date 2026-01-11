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

static PipelineTimings run_pipeline_with_openmp(const Image &img, float gamma, std::vector<uint8_t> &final_out, const std::string &return_stage)
{
    PipelineTimings times;
    int w = img.width;
    int h = img.height;
    size_t num_pixels = static_cast<size_t>(w) * h;

    // Buffers for intermediate stages
    std::vector<uint8_t> gray(num_pixels);
    std::vector<uint8_t> blur(num_pixels);
    std::vector<uint8_t> edges(num_pixels);
    std::vector<uint8_t> sharp(num_pixels);
    std::vector<uint8_t> brightness(num_pixels);
    final_out.resize(num_pixels);

    // 1. Grayscale
    auto t0 = std::chrono::high_resolution_clock::now();
    #pragma omp parallel for schedule(static)
    for(int y=0; y<h; ++y) 
        filters::rgb_to_grayscale_row(img.rgb.data() + y*w*3, gray.data() + y*w, w, /*bgr=*/false);
    auto t1 = std::chrono::high_resolution_clock::now();
    times.grayscale_us = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();

    // 2. Gaussian (Input: gray, Output: blur)
    t0 = std::chrono::high_resolution_clock::now();
    #pragma omp parallel for schedule(static)
    for(int y=0; y<h; ++y)
        filters::gaussian_blur_row_gray(gray.data(), blur.data(), w, h, y);
    t1 = std::chrono::high_resolution_clock::now();
    times.gaussian_us = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();

    // 3. Edges (Input: blur, Output: edges)
    t0 = std::chrono::high_resolution_clock::now();
    #pragma omp parallel for schedule(static)
    for(int y=0; y<h; ++y)
        filters::sobel_row_gray(blur.data(), edges.data(), w, h, y);
    t1 = std::chrono::high_resolution_clock::now();
    times.edges_us = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();

    // 4. Sharpen (Input: edges, Output: sharp)
    t0 = std::chrono::high_resolution_clock::now();
    #pragma omp parallel for schedule(static)
    for(int y=0; y<h; ++y)
        filters::sharpen_row_gray(edges.data(), sharp.data(), w, h, y);
    t1 = std::chrono::high_resolution_clock::now();
    times.sharpen_us = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();

    // 5. Brightness (Input: sharp, Output: brightness)
    t0 = std::chrono::high_resolution_clock::now();
    #pragma omp parallel for schedule(static)
    for(int y=0; y<h; ++y)
        filters::brightness_row_gray(sharp.data(), brightness.data(), w, h, y, gamma);
    t1 = std::chrono::high_resolution_clock::now();
    times.brightness_us = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();

    // Copy requested stage to final_out
    if (return_stage == "grayscale") final_out = gray;
    else if (return_stage == "gaussian") final_out = blur;
    else if (return_stage == "edges") final_out = edges;
    else if (return_stage == "sharpen") final_out = sharp;
    else final_out = brightness; // "brightness" or "all"

    return times;
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

    for (size_t i = 0; i < images.size(); ++i)
    {
        std::vector<uint8_t> out;
        PipelineTimings times = run_pipeline_with_openmp(images[i], gamma, out, filter);
        
        long long img_total = 0;
        // Accumulate timing based on what user asked to measure
        for (const auto &filt : filters_to_run) {
           long long t = 0;
           if(filt == "grayscale") t = times.grayscale_us;
           else if(filt == "gaussian" || filt == "blur") t = times.gaussian_us;
           else if(filt == "edges" || filt == "sobel") t = times.edges_us;
           else if(filt == "sharpen") t = times.sharpen_us;
           else if(filt == "brightness") t = times.brightness_us;
           total_us += t;
           img_total += t;
        }

        // Just use simple checksum on final output
        for (uint8_t v : out) checksum += v;
        
        // Save output
        fs::path in_path = files[i];
        std::string stem = in_path.empty() ? ("img_" + std::to_string(i)) : fs::path(in_path).stem().string();
        std::string fname = stem + "_" + out_subdir + ".jpg";
        fs::path out_path = out_dir / fname;
        (void)write_jpg(out_path.string(), out.data(), images[i].width, images[i].height);
    }
    std::cout << "Saved outputs to: " << out_dir.string() << std::endl;

    auto ms = total_us / 1000;

    std::cout << "OpenMP enabled: "
#ifdef _OPENMP
              << "yes" << std::endl;
    std::cout << "OpenMP threads: " << omp_get_max_threads() << std::endl;
#else
              << "no (compile with /openmp or -fopenmp)" << std::endl;
#endif
    std::cout << "OpenMP checksum: " << checksum << std::endl;
    std::cout << "OpenMP time (all filters): " << ms << " ms" << std::endl;

    return 0;
}
