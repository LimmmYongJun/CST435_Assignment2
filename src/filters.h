#pragma once
#include <cstdint>
#include <algorithm>
#include <cmath>

namespace filters
{

    inline int clamp(int v, int lo, int hi)
    {
        return v < lo ? lo : (v > hi ? hi : v);
    }

    // Convert 3-channel RGB (or BGR) to grayscale using luminance
    // Assumes input layout: [y*w*3 + x*3 + 0..2]
    inline void rgb_to_grayscale(const uint8_t *rgb, uint8_t *gray, int w, int h, bool bgr = true)
    {
        for (int y = 0; y < h; ++y)
        {
            const uint8_t *row = rgb + y * w * 3;
            uint8_t *gout = gray + y * w;
            for (int x = 0; x < w; ++x)
            {
                uint8_t c0 = row[x * 3 + 0];
                uint8_t c1 = row[x * 3 + 1];
                uint8_t c2 = row[x * 3 + 2];
                double R, G, B;
                if (bgr)
                {
                    B = c0;
                    G = c1;
                    R = c2;
                }
                else
                {
                    R = c0;
                    G = c1;
                    B = c2;
                }
                int v = static_cast<int>(0.299 * R + 0.587 * G + 0.114 * B + 0.5);
                gout[x] = static_cast<uint8_t>(clamp(v, 0, 255));
            }
        }
    }

    // 3x3 Gaussian blur on grayscale image
    inline void gaussian_blur_row_gray(const uint8_t *in, uint8_t *out, int w, int h, int y)
    {
        for (int x = 0; x < w; ++x)
        {
            int sum = 0;
            for (int dy = -1; dy <= 1; ++dy)
            {
                int yy = clamp(y + dy, 0, h - 1);
                const uint8_t *row = in + yy * w;
                for (int dx = -1; dx <= 1; ++dx)
                {
                    int xx = clamp(x + dx, 0, w - 1);
                    int weight = 1;
                    if (dx == 0 && dy == 0)
                        weight = 4; // center
                    else if (dx == 0 || dy == 0)
                        weight = 2; // axial neighbors
                    // corners remain 1
                    sum += weight * row[xx];
                }
            }
            out[y * w + x] = static_cast<uint8_t>(sum / 16);
        }
    }

    inline void gaussian_blur_rows_gray(const uint8_t *in, uint8_t *out, int w, int h, int y0, int y1)
    {
        for (int y = y0; y < y1; ++y)
            gaussian_blur_row_gray(in, out, w, h, y);
    }

    // Sobel edge detection on grayscale; output is magnitude (0..255)
    inline void sobel_row_gray(const uint8_t *in, uint8_t *out, int w, int h, int y)
    {
        for (int x = 0; x < w; ++x)
        {
            int gx = 0, gy = 0;
            for (int dy = -1; dy <= 1; ++dy)
            {
                int yy = clamp(y + dy, 0, h - 1);
                const uint8_t *row = in + yy * w;
                for (int dx = -1; dx <= 1; ++dx)
                {
                    int xx = clamp(x + dx, 0, w - 1);
                    int p = row[xx];
                    // Sobel kernels
                    int kx = 0, ky = 0;
                    if (dy == -1)
                    {
                        kx = -dx;
                        ky = -1;
                    }
                    else if (dy == 0)
                    {
                        kx = -2 * dx;
                        ky = 0;
                    }
                    else
                    { /* dy == 1 */
                        kx = dx;
                        ky = 1;
                    }
                    gy += ky * p;
                    gx += kx * p;
                }
            }
            int mag = std::abs(gx) + std::abs(gy); // integer approximation to gradient magnitude
            out[y * w + x] = static_cast<uint8_t>(clamp(mag, 0, 255));
        }
    }

    inline void sobel_rows_gray(const uint8_t *in, uint8_t *out, int w, int h, int y0, int y1)
    {
        for (int y = y0; y < y1; ++y)
            sobel_row_gray(in, out, w, h, y);
    }

    // Sharpening with kernel [[0,-1,0],[-1,5,-1],[0,-1,0]]
    inline void sharpen_row_gray(const uint8_t *in, uint8_t *out, int w, int h, int y)
    {
        for (int x = 0; x < w; ++x)
        {
            int acc = 0;
            for (int dy = -1; dy <= 1; ++dy)
            {
                int yy = clamp(y + dy, 0, h - 1);
                const uint8_t *row = in + yy * w;
                for (int dx = -1; dx <= 1; ++dx)
                {
                    int xx = clamp(x + dx, 0, w - 1);
                    int k = 0;
                    if (dy == 0 && dx == 0)
                        k = 5;
                    else if (dy == 0 || dx == 0)
                        k = -1;
                    else
                        k = 0;
                    acc += k * row[xx];
                }
            }
            out[y * w + x] = static_cast<uint8_t>(clamp(acc, 0, 255));
        }
    }

    inline void sharpen_rows_gray(const uint8_t *in, uint8_t *out, int w, int h, int y0, int y1)
    {
        for (int y = y0; y < y1; ++y)
            sharpen_row_gray(in, out, w, h, y);
    }

    // Brightness adjust on grayscale
    inline void brightness_row_gray(const uint8_t *in, uint8_t *out, int w, int h, int y, int beta)
    {
        const uint8_t *row_in = in + y * w;
        uint8_t *row_out = out + y * w;
        for (int x = 0; x < w; ++x)
        {
            row_out[x] = static_cast<uint8_t>(clamp(static_cast<int>(row_in[x]) + beta, 0, 255));
        }
    }

    inline void brightness_rows_gray(const uint8_t *in, uint8_t *out, int w, int h, int y0, int y1, int beta)
    {
        for (int y = y0; y < y1; ++y)
            brightness_row_gray(in, out, w, h, y, beta);
    }

} // namespace filters
