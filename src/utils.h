#pragma once

#include <filesystem>
#include <vector>
#include <iostream>

#ifndef SERVER
    #include "matplotlibcpp.h"
    namespace plt = matplotlibcpp;
#endif

#define GRAYSCALE_RANGE 256


namespace fs = std::filesystem;

#ifndef SERVER
void plotHist(const uint32_t* histogram)
{
    std::vector<int> histogram_vec(histogram, histogram + GRAYSCALE_RANGE);

    plt::bar(histogram_vec);
    plt::show();
}

void plotHistImg(const uint8_t* img, int width, int height)
{
    uint32_t histogram[GRAYSCALE_RANGE] = {0};
    uint32_t pixels_num = width * height;
    for(uint32_t pixel_idx=0; pixel_idx<pixels_num; pixel_idx++)
    {
        histogram[img[pixel_idx]]++;
    }

    plotHist(histogram);
}
#endif

void convert_to_grayscale(uint8_t *input_img, uint8_t *output_img, uint32_t height, uint32_t width, uint32_t channels)
{
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            // Convert RGB to grayscale using the formula: grayscale = 0.299*R + 0.587*G + 0.114*B
            int index = (y * width + x) * channels;
            output_img[y * width + x] = static_cast<uint8_t>(
                0.299 * input_img[index] +
                0.587 * input_img[index + 1] +
                0.114 * input_img[index + 2]
            );
        }
    }
}

void mirror_img_borders(uint8_t* in_mirrored_image, uint8_t* in_image,
                        uint32_t mirrored_height, uint32_t mirrored_width,
                        uint32_t height, uint32_t width,
                        uint32_t tileRadius)
{
    for(uint32_t pixel_idx_y = 0; pixel_idx_y < mirrored_height; pixel_idx_y++)
    {
        for(uint32_t pixel_idx_x = 0; pixel_idx_x < mirrored_width; pixel_idx_x++)
        {
            uint32_t abs_pixel_idx = pixel_idx_y * mirrored_width + pixel_idx_x;

            // Copy the image if pixel is inside of border
            if(pixel_idx_y >= tileRadius && pixel_idx_y < (mirrored_height - tileRadius) && 
               pixel_idx_x >= tileRadius && pixel_idx_x < (mirrored_width- tileRadius) )
            {
                in_mirrored_image[abs_pixel_idx] = in_image[(pixel_idx_y - tileRadius) * width + (pixel_idx_x - tileRadius)];
            }
            else
            {
                // Handle different border location and mirror the pixel
                if(pixel_idx_y < tileRadius)
                {
                    if(pixel_idx_x < tileRadius)    // TOP-LEFT
                    {
                        in_mirrored_image[abs_pixel_idx] = in_image[(tileRadius - pixel_idx_y - 1) * width + (tileRadius - pixel_idx_x - 1)];
                    }
                    else if(pixel_idx_x < mirrored_width - tileRadius)    // TOP-CENTER
                    {
                        in_mirrored_image[abs_pixel_idx] = in_image[(tileRadius - pixel_idx_y - 1) * width + (pixel_idx_x - tileRadius)];
                    }
                    else    // TOP-RIGHT
                    {
                        in_mirrored_image[abs_pixel_idx] = in_image[(tileRadius - pixel_idx_y - 1) * width + (2 * mirrored_width - 3 * tileRadius - pixel_idx_x - 1)];
                    }
                }
                else if(pixel_idx_y < (mirrored_height - tileRadius))
                {
                    if(pixel_idx_x < tileRadius)    // CENTER-LEFT
                    {
                        in_mirrored_image[abs_pixel_idx] = in_image[(pixel_idx_y - tileRadius) * width + (tileRadius - pixel_idx_x - 1)];
                    }
                    else    // CENTER-RIGHT
                    {
                        in_mirrored_image[abs_pixel_idx] = in_image[(pixel_idx_y - tileRadius) * width + (2 * mirrored_width - 3 * tileRadius - pixel_idx_x - 1)];
                    }
                }
                else
                {
                    if(pixel_idx_x < tileRadius)    // BOTTOM-LEFT
                    {
                        in_mirrored_image[abs_pixel_idx] = in_image[(2 * mirrored_height - 3 * tileRadius - pixel_idx_y - 1) * width + (tileRadius - pixel_idx_x - 1)];
                    }
                    else if(pixel_idx_x < mirrored_width - tileRadius)    // BOTTOM-CENTER
                    {
                        in_mirrored_image[abs_pixel_idx] = in_image[(2 * mirrored_height - 3 * tileRadius - pixel_idx_y - 1) * width + (pixel_idx_x - tileRadius)];
                    }
                    else    // BOTTOM-RIGHT
                    {
                        in_mirrored_image[abs_pixel_idx] = in_image[(2 * mirrored_height - 3 * tileRadius - pixel_idx_y - 1) * width + (2 * mirrored_width - 3 * tileRadius - pixel_idx_x - 1)];
                    }
                }
            }

            // std::cout << (int)in_mirrored_image[abs_pixel_idx] << " ";
        }
        // std::cout << "\n";
    }
}