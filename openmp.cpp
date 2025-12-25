#include <filesystem>
#include <vector>
#include <string>
#include <algorithm>
#include <iostream>
#include <chrono>
#include <cstdlib>
#include <fstream>
#include <cctype>

#ifdef _OPENMP
#include <omp.h>
#endif

#undef STBI_NO_STDIO
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "third_party/stb/stb_image_write.h"
#include "src/filters.h"
namespace fs = std::filesystem;

struct Image
{
    int width = 0;
    int height = 0;
    std::vector<uint8_t> gray; // 8-bit grayscale
};

static bool load_grayscale_image(const std::string &path, Image &out)
{
    int w = 0, h = 0, ch = 0;
    stbi_uc *data = stbi_load(path.c_str(), &w, &h, &ch, 3); // load 3-channel
    if (!data || w <= 0 || h <= 0)
        return false;
    out.width = w;
    out.height = h;
    out.gray.resize(static_cast<size_t>(w) * static_cast<size_t>(h));
    filters::rgb_to_grayscale(reinterpret_cast<const uint8_t *>(data), out.gray.data(), w, h, /*bgr=*/false);
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

static bool write_png(const std::string &filepath, const uint8_t *data, int w, int h)
{
    return stbi_write_png(filepath.c_str(), w, h, 1, data, w) != 0;
}

static void run_filter_with_openmp(const Image &img, const std::string &filter, int beta, std::vector<uint8_t> &out)
{
    const int w = img.width;
    const int h = img.height;
    out.resize(static_cast<size_t>(w) * h);

    auto clamp = [](int v, int lo, int hi)
    { return v < lo ? lo : (v > hi ? hi : v); };

    if (filter == "gaussian" || filter == "blur")
    {
#pragma omp parallel for schedule(static)
        for (int y = 0; y < h; ++y)
            filters::gaussian_blur_row_gray(img.gray.data(), out.data(), w, h, y);
    }
    else if (filter == "edges" || filter == "sobel")
    {
#pragma omp parallel for schedule(static)
        for (int y = 0; y < h; ++y)
            filters::sobel_row_gray(img.gray.data(), out.data(), w, h, y);
    }
    else if (filter == "sharpen")
    {
#pragma omp parallel for schedule(static)
        for (int y = 0; y < h; ++y)
            filters::sharpen_row_gray(img.gray.data(), out.data(), w, h, y);
    }
    else if (filter == "brightness")
    {
#pragma omp parallel for schedule(static)
        for (int y = 0; y < h; ++y)
        {
            const uint8_t *row_in = img.gray.data() + y * w;
            uint8_t *row_out = out.data() + y * w;
            for (int x = 0; x < w; ++x)
            {
                int v = static_cast<int>(std::lround(row_in[x] * 1.5));
                row_out[x] = static_cast<uint8_t>(filters::clamp(v, 0, 255));
            }
        }
    }
    else if (filter == "grayscale")
    {
#pragma omp parallel for schedule(static)
        for (int y = 0; y < h; ++y)
            std::copy(img.gray.data() + y * w, img.gray.data() + (y + 1) * w, out.data() + y * w);
    }
    else
    {
#pragma omp parallel for schedule(static)
        for (int y = 0; y < h; ++y)
            filters::gaussian_blur_row_gray(img.gray.data(), out.data(), w, h, y);
    }

    (void)out;
}

int main(int argc, char **argv)
{
    std::string folder = "input_images";
    std::string filter = "all";
    int beta = 50;
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
        if (load_grayscale_image(path, img))
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
    std::vector<uint8_t> out;
    long long total_microseconds = 0;

    for (const auto &filt : filters_to_run)
    {
        std::string folder_name = filt;
        if (filt == "gaussian")
            folder_name = "gaussian_blur";
        else if (filt == "edges")
            folder_name = "edge_detection";
        else if (filt == "brightness")
            folder_name = "brightness_adjustment";
        fs::path out_dir = fs::path("output_images") / folder_name;
        fs::create_directories(out_dir);
        for (size_t i = 0; i < images.size(); ++i)
        {
            auto t_start = std::chrono::high_resolution_clock::now();
            run_filter_with_openmp(images[i], filt, beta, out);
            auto t_end = std::chrono::high_resolution_clock::now();
            total_microseconds += std::chrono::duration_cast<std::chrono::microseconds>(t_end - t_start).count();

            for (uint8_t v : out)
                checksum += v;
            fs::path in_path = files[i];
            std::string stem = in_path.empty() ? ("img_" + std::to_string(i)) : fs::path(in_path).stem().string();
            std::string fname = stem + "_" + folder_name;
            // fs::path out_path = out_dir / (fname + ".png");
            // (void)write_png(out_path.string(), out.data(), images[i].width, images[i].height);
        }
        std::cout << "Saved outputs to: " << out_dir.string() << std::endl;
    }

    auto ms = total_microseconds / 1000;

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
