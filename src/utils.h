#pragma once

#include <filesystem>
#include <vector>
#include <map>
#include <iostream>
#include <fstream>

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

void parallel_clahe(uint32_t clip_limit, uint32_t tileRadius, std::string in_img_path , std::string out_img_path, bool save_img = false);

void parallel_clahe_shared_mem(uint32_t clip_limit, uint32_t tileRadius, std::string in_img_path , std::string out_img_path, bool save_img = false);

void single_core_clahe(uint32_t clip_limit, uint32_t tileRadius, std::string in_img_path , std::string out_img_path, bool save_img = false);

typedef struct
{
    uint32_t iter_for_reliability;
    std::pair<bool, std::vector<uint32_t>> cpu_version;
    std::pair<bool, std::vector<uint32_t>> gpu_version;
    std::pair<bool, std::vector<uint32_t>> gpu_mem_shared_version;
}benchmark_config;

typedef struct
{
    uint32_t iter_for_reliability;
    std::string benchmark_type;
    std::map<uint32_t, std::vector<double>> elapsed_times;

    void toJson(std::string json_out_path)
    {
        std::ofstream json_file(json_out_path, std::ios::out);

        json_file << "{\n";
        json_file << "\t\"" << benchmark_type << "\": {\n";
        json_file << "\t\t\"iter_for_reliability\": " << iter_for_reliability << ",\n";
        json_file << "\t\t\"test_performed\": [\n";

        for(auto pair : elapsed_times)
        {
            json_file << "\t\t\t{\n";
            json_file << "\t\t\t\t\"img_n\": " << pair.first << ",\n";
            json_file << "\t\t\t\t\"elapsed_times\": [";
            for(auto time : pair.second)
            {
                json_file << time;
                if (time != *pair.second.rbegin())
                {
                    json_file << ",";
                } 
            }     
            json_file << "]\n";
            json_file << "\t\t\t}";
            if (pair != *elapsed_times.rbegin())
            {
                json_file << ",";
            } 
            json_file << "\n";
        }
        json_file << "\t\t]\n";
        json_file << "\t}\n";
        json_file << "}\n";

    }
}benchmark_data;

void make_single_benchmark(std::string benchmark_type,std::string media_out_prefix, std::string json_out_path, std::vector<uint32_t> img_limits, uint32_t iter_for_reliability)
{
    std::chrono::high_resolution_clock::time_point start_time, end_time;
    std::chrono::duration<double> elapsed_time;
    uint32_t img_count = std::distance(fs::directory_iterator("./../media/input/grayscale_images/"), fs::directory_iterator{});
    uint32_t img_counter = 0;
    uint32_t bench_counter = 0;

    benchmark_data timing_data={
        .iter_for_reliability = iter_for_reliability,
        .benchmark_type = benchmark_type
    };

    for(auto img_limit : img_limits)
    {
        if (img_limit > img_count)
        {
            img_limit = img_count;
        }
        for(uint32_t reliability_index=0; reliability_index<iter_for_reliability; reliability_index++)
        {
            start_time = std::chrono::high_resolution_clock::now();
            img_counter = 0;
            for (const auto& entry : fs::directory_iterator("./../media/input/grayscale_images/")) {
                    std::string img_name =  entry.path().filename().string();
                    std::string in_img_path = "./../media/input/grayscale_images/" + img_name;
                    std::string out_img_path = "./../media/output/" + media_out_prefix + img_name;

                    if(benchmark_type == "cpu_version")
                    {
                        single_core_clahe(4, 40, in_img_path, out_img_path);
                    }
                    else if(benchmark_type == "gpu_version")
                    {
                        parallel_clahe(4, 40, in_img_path, out_img_path);
                    }
                    else if(benchmark_type == "gpu_mem_shared_version")
                    {
                        parallel_clahe_shared_mem(4, 40, in_img_path, out_img_path);
                    }
                    
                    if(img_counter == img_count - 1)
                    {
                        std::cout << "Imgs:["<< img_counter + 1 << "/" << img_limit << "] " 
                        << "reliability:["<< reliability_index + 1  <<"/" << iter_for_reliability << "] "
                        << "bench_n:[" << bench_counter + 1 << "/" << img_limits.size() << "] " << "\n";
                    }
                    else
                    {
                        std::cout << "Imgs:["<< img_counter + 1 << "/" << img_limit << "] " 
                        << "reliability:["<< reliability_index + 1  <<"/" << iter_for_reliability << "] "
                        << "bench_n:[" << bench_counter + 1 << "/" << img_limits.size() << "] " << std::flush << "\r";
                    }
                    img_counter++;
                    if(img_counter >= img_limit)break;
            }
            end_time = std::chrono::high_resolution_clock::now();
            elapsed_time = std::chrono::duration_cast<std::chrono::duration<double>>(end_time - start_time);
            timing_data.elapsed_times[img_limit].push_back(elapsed_time.count());
        }
        bench_counter++;
    }
    timing_data.toJson(json_out_path);
}