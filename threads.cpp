#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
#include <filesystem>
#include <chrono>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <cstdlib>
#include <cctype>
#include <cstring>

#define STB_IMAGE_IMPLEMENTATION
#include "third_party/stb/stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "third_party/stb/stb_image_write.h"
#include "src/filters.h"
#include <fstream>

// Use a namespace for file system to handle different C++ versions/compilers
namespace fs = std::filesystem;

// -------------------------------------------------------------------------
// Structures
// -------------------------------------------------------------------------

struct PipelineTimings
{
    long long grayscale_us = 0;
    long long gaussian_us = 0;
    long long edges_us = 0;
    long long sharpen_us = 0;
    long long brightness_us = 0;
};

// Simple Barrier for C++17
class Barrier {
public:
    explicit Barrier(std::size_t count) : threshold_(count), count_(count), generation_(0) {}

    void wait() {
        std::unique_lock<std::mutex> lock(mutex_);
        auto gen = generation_;
        if (--count_ == 0) {
            generation_++;
            count_ = threshold_;
            cond_.notify_all();
        } else {
            cond_.wait(lock, [this, gen] { return gen != generation_; });
        }
    }

private:
    std::mutex mutex_;
    std::condition_variable cond_;
    std::size_t threshold_;
    std::size_t count_;
    std::size_t generation_;
};

struct SharedBuffers
{
    std::vector<uint8_t> gray;
    std::vector<uint8_t> blur;
    std::vector<uint8_t> edges;
    std::vector<uint8_t> sharp;
    std::vector<uint8_t> brightness;
    std::vector<uint8_t> final_out;
    std::vector<PipelineTimings> thread_times;

    void resize(size_t num_pixels, size_t num_threads) {
        if (gray.size() != num_pixels) {
            gray.resize(num_pixels);
            blur.resize(num_pixels);
            edges.resize(num_pixels);
            sharp.resize(num_pixels);
            brightness.resize(num_pixels);
            final_out.resize(num_pixels);
        }
        if (thread_times.size() != num_threads) {
            thread_times.resize(num_threads);
        }
        // Reset timings
        for(auto &t : thread_times) t = {};
    }
};

struct Image
{
    int width = 0;
    int height = 0;
    int channels = 0;
    std::vector<uint8_t> rgb;
};

// -------------------------------------------------------------------------
// Helper Functions
// -------------------------------------------------------------------------

// Helper to determine thread count
static unsigned int determine_threads()
{
    // Check environment variable first
    const char* env_threads = std::getenv("THREADS");
    if (env_threads)
    {
        try {
            int val = std::stoi(env_threads);
            if (val > 0) return static_cast<unsigned int>(val);
        } catch (...) {}
    }
    // Fallback to hardware concurrency or 4
    unsigned int n = std::thread::hardware_concurrency();
    return (n == 0) ? 4 : n;
}

static bool load_image(const fs::path &path, Image &img)
{
    int w, h, c;
    unsigned char *data = stbi_load(path.string().c_str(), &w, &h, &c, 3);
    if (!data) return false;
    img.width = w;
    img.height = h;
    img.channels = 3;
    img.rgb.assign(data, data + w * h * 3);
    stbi_image_free(data);
    return true;
}

static bool write_jpg(const std::string &filename, const uint8_t *data, int w, int h)
{
    int ok = stbi_write_jpg(filename.c_str(), w, h, 1, data, 90);
    return ok != 0;
}

static std::vector<fs::path> list_image_files(const std::string &folder)
{
    std::vector<fs::path> files;
    if (!fs::exists(folder) || !fs::is_directory(folder))
        return files;

    for (const auto &entry : fs::directory_iterator(folder))
    {
        if (entry.is_regular_file())
        {
            auto ext = entry.path().extension().string();
            // Basic check for jpg/png/bmp
            std::string ext_lower = ext;
            std::transform(ext_lower.begin(), ext_lower.end(), ext_lower.begin(), [](unsigned char c){ return static_cast<char>(std::tolower(c)); });
            
            if (ext_lower == ".jpg" || ext_lower == ".jpeg" || ext_lower == ".png" || ext_lower == ".bmp")
            {
                files.push_back(entry.path());
            }
        }
    }
    // Sort to ensure deterministic order
    std::sort(files.begin(), files.end());
    return files;
}

// -------------------------------------------------------------------------
// Core Pipeline
// -------------------------------------------------------------------------

// Helper to apply all filters on a range of rows
static void apply_pipeline_rows(const Image &img, 
                                SharedBuffers &bufs,
                                int w, int h, int y0, int y1, float gamma,
                                PipelineTimings &t_local,
                                Barrier &barrier)
{
    // 1. Grayscale
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int y = y0; y < y1; ++y)
        filters::rgb_to_grayscale_row(img.rgb.data() + y*w*3, bufs.gray.data() + y*w, w, /*bgr=*/false);
    auto t1 = std::chrono::high_resolution_clock::now();
    t_local.grayscale_us += std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();

    // Barrier: Wait for everyone to finish Grayscale
    barrier.wait();

    // 2. Gaussian
    t0 = std::chrono::high_resolution_clock::now();
    filters::gaussian_blur_rows_gray(bufs.gray.data(), bufs.blur.data(), w, h, y0, y1);
    t1 = std::chrono::high_resolution_clock::now();
    t_local.gaussian_us += std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();

    // Barrier: Wait for everyone to finish Gaussian
    barrier.wait();

    // 3. Edges
    t0 = std::chrono::high_resolution_clock::now();
    filters::sobel_rows_gray(bufs.blur.data(), bufs.edges.data(), w, h, y0, y1);
    t1 = std::chrono::high_resolution_clock::now();
    t_local.edges_us += std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();

    // Barrier: Wait for everyone to finish Edges
    barrier.wait();

    // 4. Sharpen
    t0 = std::chrono::high_resolution_clock::now();
    filters::sharpen_rows_gray(bufs.edges.data(), bufs.sharp.data(), w, h, y0, y1);
    t1 = std::chrono::high_resolution_clock::now();
    t_local.sharpen_us += std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();

    // Barrier: Wait for everyone to finish Sharpen
    barrier.wait();

    // 5. Brightness
    t0 = std::chrono::high_resolution_clock::now();
    filters::brightness_rows_gray(bufs.sharp.data(), bufs.brightness.data(), w, h, y0, y1, gamma);
    t1 = std::chrono::high_resolution_clock::now();
    t_local.brightness_us += std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
}

// -------------------------------------------------------------------------
// Main
// -------------------------------------------------------------------------

// --- Global Sync Variables ---
std::mutex mtx_job;
std::condition_variable cv_job;
int global_job_id = 0; // Monotonic counter to prevent races
bool time_to_quit = false;

// Sync for Master to know when image is done
std::mutex mtx_done;
std::condition_variable cv_done;
int threads_completed_count = 0;

// Global pointers for current job context
const Image* current_img_ptr = nullptr;
SharedBuffers* current_bufs_ptr = nullptr;
float current_gamma = 1.0f;

// Worker Function
void worker_thread_func(int thread_id, int num_threads, Barrier& stage_barrier) {
    int last_job_id = 0;
    while (true) {
        // 1. Wait for New Job
        {
            std::unique_lock<std::mutex> lock(mtx_job);
            cv_job.wait(lock, [&] { return global_job_id > last_job_id || time_to_quit; });
            if (time_to_quit) return;
            last_job_id = global_job_id;
        }

        // 2. Setup Work
        int w = current_img_ptr->width;
        int h = current_img_ptr->height;
        int rows_per = (h + num_threads - 1) / num_threads;
        int y0 = thread_id * rows_per;
        int y1 = std::min(h, y0 + rows_per);

        // 3. Run Pipeline 
        if (y0 < h) {
             apply_pipeline_rows(*current_img_ptr, *current_bufs_ptr, 
                                w, h, y0, y1, current_gamma, 
                                current_bufs_ptr->thread_times[thread_id], stage_barrier);
        } else {
             // Inactive threads still need to hit barriers to let active peers proceed
             for(int k=0; k<5; ++k) stage_barrier.wait();
        }

        // 4. Report "I am done"
        {
            std::lock_guard<std::mutex> lk(mtx_done);
            threads_completed_count++;
            if (threads_completed_count == num_threads) {
                cv_done.notify_one(); // Wake up Master
            }
        }
        
        // 5. Wait for loop synchronization 
        // Sync with peers before checking for next job
        stage_barrier.wait(); 
    }
}

int main(int argc, char **argv)
{
    std::string folder = "input_images";
    std::string filter = "all"; 
    float gamma = 0.9f; 
    size_t max_images = 0;
    if (argc > 1) folder = argv[1];
    if (argc > 2)
    {
        std::string a2 = argv[2];
        bool a2_is_num = !a2.empty() && std::all_of(a2.begin(), a2.end(), [](unsigned char c) { return std::isdigit(c) != 0; });
        if (a2_is_num) max_images = static_cast<size_t>(std::stoull(a2));
        else filter = a2;
    }
    if (argc > 3)
    {
        std::string a3 = argv[3];
        bool a3_is_num = !a3.empty() && std::all_of(a3.begin(), a3.end(), [](unsigned char c) { return std::isdigit(c) != 0; });
        if (a3_is_num) max_images = static_cast<size_t>(std::stoull(a3));
        else filter = a3;
    }

    auto files = list_image_files(folder);
    if (files.empty()) {
        std::cerr << "No images found in folder: " << folder << std::endl;
        return 1;
    }
    size_t to_process = files.size();
    if (max_images > 0 && max_images < to_process) to_process = max_images;
    std::cout << "Found " << files.size() << " image(s), processing " << to_process << "." << std::endl;

    std::vector<Image> images;
    images.reserve(files.size());
    for (size_t i = 0; i < to_process; ++i) {
        const auto &path = files[i];
        Image img;
        if (load_image(path, img)) images.push_back(std::move(img));
        else std::cerr << "Failed to load: " << path << std::endl;
    }
    if (images.empty()) {
        std::cerr << "No images could be loaded." << std::endl;
        return 1;
    }

    // Determine threads
    unsigned int nt = determine_threads();
    std::cout << "Threads used: " << nt << std::endl;

    // Initialize Barrier for WORKERS ONLY
    Barrier stage_barrier(nt);
    SharedBuffers bufs;
    std::vector<std::thread> threads;
    
    // Launch Persistent Threads
    for (unsigned int t = 0; t < nt; ++t) {
        threads.emplace_back(worker_thread_func, t, nt, std::ref(stage_barrier));
    }

    uint64_t checksum = 0;
    PipelineTimings total_times{};
    
    std::string out_subdir = (filter == "all") ? "pipeline_final" : filter;
    fs::path out_dir = fs::path("output_images") / out_subdir;
    fs::create_directories(out_dir);

    // Filter list
    std::vector<std::string> filters_to_run;
    if (filter == "all") filters_to_run = {"grayscale", "gaussian", "edges", "sharpen", "brightness"};
    else filters_to_run = {filter};

    // Master Loop
    for (size_t i = 0; i < images.size(); ++i) {
        const auto &img = images[i];
        
        // 1. Prepare Shared State
        current_img_ptr = &img;
        current_bufs_ptr = &bufs;
        current_gamma = gamma;
        
        // Resize buffers (Serial, safe because workers are waiting)
        bufs.resize(static_cast<size_t>(img.width) * img.height, nt);

        // 2. Reset Counter
        {
            std::lock_guard<std::mutex> lk(mtx_done);
            threads_completed_count = 0;
        }

        // 3. Wake Workers
        {
            std::lock_guard<std::mutex> lk(mtx_job);
            global_job_id++;
        }
        cv_job.notify_all();

        // 4. Wait for Completion
        {
            std::unique_lock<std::mutex> lk(mtx_done);
            cv_done.wait(lk, [&]{ return threads_completed_count == (int)nt; });
        }

        // 5. Workers are looping back to wait for next job_id increment
        // No manual reset needed for job_id
        
        // 6. Aggregate Results
        PipelineTimings img_times{};
        for(const auto &loc : bufs.thread_times) {
                if(loc.grayscale_us > img_times.grayscale_us) img_times.grayscale_us = loc.grayscale_us;
                if(loc.gaussian_us > img_times.gaussian_us) img_times.gaussian_us = loc.gaussian_us;
                if(loc.edges_us > img_times.edges_us) img_times.edges_us = loc.edges_us;
                if(loc.sharpen_us > img_times.sharpen_us) img_times.sharpen_us = loc.sharpen_us;
                if(loc.brightness_us > img_times.brightness_us) img_times.brightness_us = loc.brightness_us;
        }
        total_times.grayscale_us += img_times.grayscale_us;
        total_times.gaussian_us += img_times.gaussian_us;
        total_times.edges_us += img_times.edges_us;
        total_times.sharpen_us += img_times.sharpen_us;
        total_times.brightness_us += img_times.brightness_us;

        // Checksum & Write
        if (filter == "grayscale") bufs.final_out = bufs.gray;
        else if (filter == "gaussian") bufs.final_out = bufs.blur;
        else if (filter == "edges") bufs.final_out = bufs.edges;
        else if (filter == "sharpen") bufs.final_out = bufs.sharp;
        else bufs.final_out = bufs.brightness; 

        for (uint8_t v : bufs.final_out) checksum += v;

        fs::path in_path = files[i];
        std::string stem = in_path.empty() ? ("img_" + std::to_string(i)) : fs::path(in_path).stem().string();
        std::string fname = stem + "_" + out_subdir + ".jpg";
        fs::path out_path = out_dir / fname;
        (void)write_jpg(out_path.string(), bufs.final_out.data(), img.width, img.height);
    }

    // Shutdown
    {
        std::lock_guard<std::mutex> lk(mtx_job);
        time_to_quit = true;
    }
    cv_job.notify_all();
    for (auto &th : threads) th.join();

    std::cout << "Threads checksum: " << checksum << std::endl;
    long long total_us_all = total_times.grayscale_us + total_times.gaussian_us + total_times.edges_us + total_times.sharpen_us + total_times.brightness_us;
    auto ms = total_us_all / 1000;
    
    if (filter == "all") std::cout << "Threads time (all filters): " << ms << " ms" << std::endl;
    if (filter == "grayscale") std::cout << "Threads time (grayscale): " << ms << " ms" << std::endl;
    if (filter == "gaussian") std::cout << "Threads time (gaussian): " << ms << " ms" << std::endl;
    if (filter == "edges") std::cout << "Threads time (edges): " << ms << " ms" << std::endl;
    if (filter == "sharpen") std::cout << "Threads time (sharpen): " << ms << " ms" << std::endl;
    if (filter == "brightness") std::cout << "Threads time (brightness): " << ms << " ms" << std::endl;

    return 0;
}
