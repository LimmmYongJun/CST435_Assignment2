#include <filesystem>
#include <vector>
#include <string>
#include <algorithm>
#include <iostream>
#include <chrono>
#include <thread>
#include <cstdlib>
#include "src/filters.h"
#include <fstream>
#include <cctype>
#include <cstring>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "third_party/stb/stb_image_write.h"

#undef STBI_NO_STDIO
#define STB_IMAGE_IMPLEMENTATION
#include "third_party/stb/stb_image.h"

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

static unsigned int determine_threads()
{
    unsigned int nt = std::thread::hardware_concurrency();
    if (nt == 0)
        nt = 4;
    if (const char *env = std::getenv("THREADS"))
    {
        try
        {
            int val = std::stoi(env);
            if (val >= 1)
                nt = static_cast<unsigned int>(val);
        }
        catch (...)
        {
        }
    }
    return nt;
}

static bool load_image(const std::string &path, Image &out)
{
    int w = 0, h = 0, ch = 0;
    stbi_uc *data = stbi_load(path.c_str(), &w, &h, &ch, 3); // load 3-channel (BGR)
    if (!data || w <= 0 || h <= 0)
        return false;

    out.width = w;
    out.height = h;
    size_t num_pixels = static_cast<size_t>(w) * h;
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
    // write grayscale PNG; stride = w bytes
    int ok = stbi_write_jpg(filepath.c_str(), w, h, 1, data, 100);
    return ok != 0;
}

// Helper to apply all filters on a range of rows
static void apply_pipeline_rows(const Image &img, 
                                std::vector<uint8_t> &gray,
                                std::vector<uint8_t> &blur,
                                std::vector<uint8_t> &edges,
                                std::vector<uint8_t> &sharp,
                                std::vector<uint8_t> &brightness,
                                int w, int h, int y0, int y1, float gamma,
                                PipelineTimings &t_local)
{
    // 1. Grayscale
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int y = y0; y < y1; ++y)
        filters::rgb_to_grayscale_row(img.rgb.data() + y*w*3, gray.data() + y*w, w, /*bgr=*/true);
    auto t1 = std::chrono::high_resolution_clock::now();
    t_local.grayscale_us += std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();

    // 2. Gaussian
    t0 = std::chrono::high_resolution_clock::now();
    filters::gaussian_blur_rows_gray(gray.data(), blur.data(), w, h, y0, y1);
    t1 = std::chrono::high_resolution_clock::now();
    t_local.gaussian_us += std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();

    // 3. Edges
    t0 = std::chrono::high_resolution_clock::now();
    filters::sobel_rows_gray(blur.data(), edges.data(), w, h, y0, y1);
    t1 = std::chrono::high_resolution_clock::now();
    t_local.edges_us += std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();

    // 4. Sharpen
    t0 = std::chrono::high_resolution_clock::now();
    filters::sharpen_rows_gray(edges.data(), sharp.data(), w, h, y0, y1);
    t1 = std::chrono::high_resolution_clock::now();
    t_local.sharpen_us += std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();

    // 5. Brightness
    t0 = std::chrono::high_resolution_clock::now();
    filters::brightness_rows_gray(sharp.data(), brightness.data(), w, h, y0, y1, gamma);
    t1 = std::chrono::high_resolution_clock::now();
    t_local.brightness_us += std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
}

static PipelineTimings run_pipeline_with_threads(const Image &img, float gamma, std::vector<uint8_t> &final_out, const std::string &return_stage)
{
    PipelineTimings total_times;
    const int w = img.width;
    const int h = img.height;
    size_t num_pixels = static_cast<size_t>(w) * h;
    
    // Allocate intermediate buffers
    std::vector<uint8_t> gray(num_pixels);
    std::vector<uint8_t> blur(num_pixels);
    std::vector<uint8_t> edges(num_pixels);
    std::vector<uint8_t> sharp(num_pixels);
    std::vector<uint8_t> brightness(num_pixels);
    final_out.resize(num_pixels);

    unsigned int nt = determine_threads();
    int rows_per = (h + static_cast<int>(nt) - 1) / static_cast<int>(nt);

    std::vector<std::thread> threads;
    std::vector<PipelineTimings> thread_times(nt);

    for (unsigned int t = 0; t < nt; ++t)
    {
        int y0 = static_cast<int>(t) * rows_per;
        int y1 = std::min(h, y0 + rows_per);
        if (y0 >= h)
            break;
        threads.emplace_back([&, t, y0, y1]() {
            apply_pipeline_rows(img, gray, blur, edges, sharp, brightness, w, h, y0, y1, gamma, thread_times[t]);
        });
    }
    for (auto &th : threads)
        th.join();
    
    for (const auto &t : thread_times) {
        if(t.grayscale_us > total_times.grayscale_us) total_times.grayscale_us = t.grayscale_us;
        if(t.gaussian_us > total_times.gaussian_us) total_times.gaussian_us = t.gaussian_us;
        if(t.edges_us > total_times.edges_us) total_times.edges_us = t.edges_us;
        if(t.sharpen_us > total_times.sharpen_us) total_times.sharpen_us = t.sharpen_us;
        if(t.brightness_us > total_times.brightness_us) total_times.brightness_us = t.brightness_us;
    }

    // Copy requested stage to final_out
    if (return_stage == "grayscale") final_out = gray;
    else if (return_stage == "gaussian") final_out = blur;
    else if (return_stage == "edges") final_out = edges;
    else if (return_stage == "sharpen") final_out = sharp;
    else final_out = brightness; // "brightness" or "all"

    return total_times;
}

int main(int argc, char **argv)
{
    std::string folder = "input_images";
    std::string filter = "all"; // default: run all 5 filters
    float gamma = 0.9f;              // (unused for percentage brightness)
    size_t max_images = 0;      // 0 means process all
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

    // If a specific filter requested, process only that; else run all
    std::vector<std::string> filters_to_run;
    if (filter == "all")
    {
        filters_to_run = {"grayscale", "gaussian", "edges", "sharpen", "brightness"};
    }
    else
    {
        filters_to_run = {filter};
    }

    uint64_t checksum_total = 0;
    long long total_us = 0;
    
    // Create output directory for the specific filter
    std::string out_subdir = (filter == "all") ? "pipeline_final" : filter;
    fs::path out_dir = fs::path("output_images") / out_subdir;
    fs::create_directories(out_dir);

    for (size_t i = 0; i < images.size(); ++i)
    {
        std::vector<uint8_t> out;
        PipelineTimings times = run_pipeline_with_threads(images[i], gamma, out, filter);
        
        long long img_total = 0;
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

        for (uint8_t v : out) checksum_total += v;

        fs::path in_path = files[i];
        std::string stem = in_path.empty() ? ("img_" + std::to_string(i)) : fs::path(in_path).stem().string();
        std::string fname = stem + "_" + out_subdir + ".jpg";
        fs::path out_path = out_dir / fname;
        (void)write_jpg(out_path.string(), out.data(), images[i].width, images[i].height);
    }
    std::cout << "Saved outputs to: " << out_dir.string() << std::endl;

    auto ms = total_us / 1000;
    std::cout << "Threads used: " << determine_threads() << std::endl;

    std::cout << "Threads checksum: " << checksum_total << std::endl;

    if (filter == "all") std::cout << "Threads time (all filters): " << ms << " ms" << std::endl;
    if(filter == "grayscale") std::cout << "Threads time (grayscale): " << ms << " ms" << std::endl;
    if(filter == "gaussian") std::cout << "Threads time (gaussian): " << ms << " ms" << std::endl;
    if(filter == "edges") std::cout << "Threads time (edges): " << ms << " ms" << std::endl;
    if(filter == "sharpen") std::cout << "Threads time (sharpen): " << ms << " ms" << std::endl;
    if(filter == "brightness") std::cout << "Threads time (brightness): " << ms << " ms" << std::endl;

    return 0;
}
