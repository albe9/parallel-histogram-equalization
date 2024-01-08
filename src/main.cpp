#include <iostream>
#include <filesystem>
#include "utils.h"


#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

namespace plt = matplotlibcpp;
namespace fs = std::filesystem;

// TODO wrap this function into another one that allows to repeat clahe on the same image , also add checking radius and dimensions
void clahe(uint8_t* in_image, uint8_t* out_image, int width, int height, int clip_limit, int tileRadius)
{
    //Border mirroring
    uint32_t mirrored_width = 2 * tileRadius + width,
             mirrored_height = 2 * tileRadius + height;

    uint8_t* in_mirrored_image = (uint8_t*)malloc((mirrored_height) * (mirrored_width) * sizeof(uint8_t)); 
    memset(in_mirrored_image, 0, (mirrored_height) * (mirrored_width) * sizeof(uint8_t));

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

    float pixel_for_window = (2 * tileRadius + 1) * (2 * tileRadius + 1);
    // Iterate through each pixel (need offset because of the mirrored border)
    for(uint32_t main_pixel_y = tileRadius; main_pixel_y < mirrored_height - tileRadius; main_pixel_y++)
    {
        for(uint32_t main_pixel_x = tileRadius; main_pixel_x < mirrored_width - tileRadius; main_pixel_x++)
        {
            // Calculate the histogram for the current tile
            uint32_t histogram[GRAYSCALE_RANGE] = {0};
            uint32_t start_tile_y = main_pixel_y - tileRadius, 
                     end_tile_y = main_pixel_y + tileRadius,
                     start_tile_x = main_pixel_x - tileRadius, 
                     end_tile_x = main_pixel_x + tileRadius;


            // Iterate through pixels in the current tile and calculate histogram
            for(uint32_t tile_pixel_y = start_tile_y; tile_pixel_y < end_tile_y + 1; tile_pixel_y++)
            {
                for(uint32_t tile_pixel_x = start_tile_x; tile_pixel_x < end_tile_x + 1; tile_pixel_x++) 
                {
                    uint32_t pixelValue = in_mirrored_image[tile_pixel_y * mirrored_width + tile_pixel_x];
                    histogram[pixelValue]++;
                }
            }

            // plotHist(histogram);

            // Calculate max and min value in histogram to convert clip limit
            uint32_t max_level_value = 0, min_level_value = 1000000;
            for(uint32_t level = 0; level < GRAYSCALE_RANGE; level++)
            {
                uint32_t level_value = histogram[level];
                if(level_value < min_level_value) min_level_value = level_value;
                if(level_value > max_level_value) max_level_value = level_value;
            }

            uint32_t scaled_clip_limit = min_level_value + (clip_limit * (max_level_value - min_level_value) / 10); 

            // Clip histogram values above the clip limit
            uint32_t excess = 0;
            for(uint32_t level = 0; level < GRAYSCALE_RANGE; level++)
            {
                if (histogram[level] > scaled_clip_limit)
                {
                    excess += histogram[level] - scaled_clip_limit;
                    histogram[level] = scaled_clip_limit;
                }
            }

            // Distribute the excess counts uniformly among all histogram bins
            uint32_t tile_cdf[GRAYSCALE_RANGE] = {0}, cdf_counter = 0;
            uint32_t bin_increment = excess / GRAYSCALE_RANGE;
            uint32_t remainder = excess % GRAYSCALE_RANGE;
            for(uint32_t level = 0; level < GRAYSCALE_RANGE; level++)
            {
                histogram[level] += bin_increment;
            }
            for(uint32_t level = 0; level < remainder; level++)
            {
                histogram[level]++;
            }

            // calculate tile cdf
            uint32_t min_cdf = 0;
            for(uint32_t level = 0; level < GRAYSCALE_RANGE; level++)
            {
                cdf_counter += histogram[level];
                tile_cdf[level] = cdf_counter;
                if(min_cdf == 0 && cdf_counter != 0) min_cdf = cdf_counter;
            }

            // plotHist(tile_cdf);

            // Apply the contrast-limited histogram equalization to the main pixel
            uint32_t main_pixel_abs_pos = main_pixel_y * mirrored_width + main_pixel_x;
            uint8_t new_pixel_value = std::round(static_cast<double>(tile_cdf[in_mirrored_image[main_pixel_abs_pos]] - min_cdf) / (pixel_for_window - min_cdf) * (GRAYSCALE_RANGE - 1));
            out_image[(main_pixel_y - tileRadius) * width + (main_pixel_x - tileRadius)] = new_pixel_value;
        }
    }
}

int main()
{

    // std::string in_img_path = "./../media/test2.jpg";
    // std::string out_img_path = "./../media/test2_output.jpg";

    // std::string in_img_path = "./../media/test1.png";
    // std::string out_img_path = "./../media/test1_output.png";

    std::string in_img_path = "./../media/test3.jpg";
    std::string out_img_path = "./../media/test3_output.jpg";

    int width, height, channels;
    uint8_t *input_img = stbi_load(in_img_path.c_str(), &width, &height, &channels, 0);
    if(input_img == NULL) {
        std::cout << "Error loading image\n";
    }
    else
    {
        std::cout << "width: " <<width << " height: " << height << " channels: " << channels << "\n";
    }


    uint8_t *output_img = (uint8_t* )malloc(width * height * sizeof(uint8_t));
    memset(output_img, 0, width * height * sizeof(uint8_t));

    // Convert to grayscale
    // for (int y = 0; y < height; y++) {
    //     for (int x = 0; x < width; x++) {
    //         // Convert RGB to grayscale using the formula: grayscale = 0.299*R + 0.587*G + 0.114*B
    //         int index = (y * width + x) * channels;
    //         output_img[y * width + x] = static_cast<uint8_t>(
    //             0.299 * input_img[index] +
    //             0.587 * input_img[index + 1] +
    //             0.114 * input_img[index + 2]
    //         );
    //     }
    // }


    // uint8_t debug_img[] =
    // {
    //     1,  1,  1,
    //     1,  0,  1,
    //     1,  0,  1,
    //     1,  0,  1,
    //     1,  0,  1,
    //     1,  1,  1
    // }; 

    // uint8_t width = 3, height = 6;
    // uint8_t *output_img = (uint8_t* )malloc(width * height * sizeof(uint8_t));
    // memset(output_img, 0, width * height * sizeof(uint8_t));

    // clahe(debug_img, output_img, 3, 6, 2, 1);


    plotHistImg(input_img, width, height);
    clahe(input_img, output_img, width, height, 4, 20);
    plotHistImg(output_img, width, height);

    stbi_write_png(out_img_path.c_str(), width, height, 1, output_img, width);
}