#define SERVER
#include "utils.h"
#include <cuda_runtime.h>
#include <chrono>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

__global__ void print_array_kernel(uint8_t* array, uint32_t idx) {
    printf("thread : %d, array[%d] = %d\n", blockIdx.x * blockDim.x + threadIdx.x, idx, array[idx]);
    array[idx] = 5;
}

void test_cuda()
{
    // Size of the array
    size_t arraySize = 1000;

    // Host array
    uint8_t* hostArray = new uint8_t[arraySize];
    hostArray[3] = 20;

    // CUDA device array
    uint8_t* deviceArray;

    // Allocate GPU memory
    cudaMalloc((void**)&deviceArray, arraySize * sizeof(uint8_t));

    // Copy data from host to GPU
    cudaMemcpy(deviceArray, hostArray, arraySize * sizeof(uint8_t), cudaMemcpyHostToDevice);

    // Launch the kernel
    print_array_kernel<<<1, 1>>>(deviceArray, 3);
    cudaDeviceSynchronize();
    cudaMemcpy(hostArray, deviceArray, arraySize * sizeof(uint8_t), cudaMemcpyDeviceToHost);
    printf("CPU : array[3] = %d\n", hostArray[3]);


    // Free GPU memory
    cudaFree(deviceArray);
}

__global__ void clahe_kernel(uint8_t* in_image, uint8_t* out_image, uint32_t width, uint32_t height, uint32_t clip_limit, uint32_t tileRadius)
{
    uint32_t center_pixel_x = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t center_pixel_y = blockIdx.y * blockDim.y + threadIdx.y;

    uint32_t mirrored_width = 2 * tileRadius + width,
             mirrored_height = 2 * tileRadius + height;

    float pixel_for_window = (2 * tileRadius + 1) * (2 * tileRadius + 1);
    
    // Check if thread's pixel belongs to the center, if not do nothing
    if(center_pixel_y >= tileRadius && center_pixel_y < mirrored_height - tileRadius && center_pixel_x >= tileRadius && center_pixel_x < mirrored_width - tileRadius)
    {
        // Calculate the histogram for the current tile
        uint32_t histogram[GRAYSCALE_RANGE] = {0};
        uint32_t start_tile_y = center_pixel_y - tileRadius, 
                 end_tile_y = center_pixel_y + tileRadius,
                 start_tile_x = center_pixel_x - tileRadius, 
                 end_tile_x = center_pixel_x + tileRadius;


        // Iterate through pixels in the current tile and calculate histogram
        for(uint32_t tile_pixel_y = start_tile_y; tile_pixel_y < end_tile_y + 1; tile_pixel_y++)
        {
            for(uint32_t tile_pixel_x = start_tile_x; tile_pixel_x < end_tile_x + 1; tile_pixel_x++) 
            {
                uint32_t pixelValue = in_image[tile_pixel_y * mirrored_width + tile_pixel_x];
                histogram[pixelValue]++;
            }
        }

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

        // Apply the contrast-limited histogram equalization to the main pixel
        uint32_t main_pixel_abs_pos = center_pixel_y * mirrored_width + center_pixel_x;
        uint8_t new_pixel_value = std::round(static_cast<double>(tile_cdf[in_image[main_pixel_abs_pos]] - min_cdf) / (pixel_for_window - min_cdf) * (GRAYSCALE_RANGE - 1));
        out_image[(center_pixel_y - tileRadius) * width + (center_pixel_x - tileRadius)] = new_pixel_value;
    }
}

void parallel_clahe(uint32_t clip_limit, uint32_t tileRadius, uint32_t iter_n)
{
    std::string in_img_path = "./../media/test3.jpg";
    std::string out_img_path = "./../media/test3_output_cuda.jpg";

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

    //Border mirroring
    uint32_t mirrored_width = 2 * tileRadius + width,
             mirrored_height = 2 * tileRadius + height;

    uint8_t* in_mirrored_image = (uint8_t*)malloc((mirrored_height) * (mirrored_width) * sizeof(uint8_t)); 
    memset(in_mirrored_image, 0, (mirrored_height) * (mirrored_width) * sizeof(uint8_t));

    mirror_img_borders(in_mirrored_image, input_img, mirrored_height, mirrored_width, height, width, tileRadius);

    // CUDA device imgs
    uint8_t* gpu_input_img;
    uint8_t* gpu_output_img;

    // Allocate GPU memory
    cudaMalloc((void**)&gpu_input_img, (mirrored_height) * (mirrored_width) * sizeof(uint8_t));
    cudaMalloc((void**)&gpu_output_img, width * height * sizeof(uint8_t));

    // Copy data from host to GPU
    cudaMemcpy(gpu_input_img, in_mirrored_image, (mirrored_height) * (mirrored_width) * sizeof(uint8_t), cudaMemcpyHostToDevice);

    // Launch the kernel
    dim3 blockDim(16, 16);
    dim3 gridDim((mirrored_width + blockDim.x - 1) / blockDim.x, (mirrored_height + blockDim.y - 1) / blockDim.y);

    for(uint32_t iter_idx=0; iter_idx<iter_n; iter_idx++)
    {
        clahe_kernel<<<gridDim, blockDim>>>(gpu_input_img, gpu_output_img, width, height, clip_limit, tileRadius);
        cudaDeviceSynchronize();
        if(iter_idx == iter_n - 1)
        {
            std::cout << "Iteration : ["<< iter_idx + 1 << "/" << iter_n << "]\n";
        }
        else
        {
            std::cout << "Iteration : ["<< iter_idx + 1 << "/" << iter_n << "]" << std::flush << "\r";
        }
        
    }
    
    cudaMemcpy(output_img, gpu_output_img, width * height * sizeof(uint8_t), cudaMemcpyDeviceToHost);

    // Free GPU memory
    cudaFree(gpu_input_img);
    cudaFree(gpu_output_img);

    stbi_write_png(out_img_path.c_str(), width, height, 1, output_img, width);
}

void cpu_clahe(uint8_t* in_image, uint8_t* out_image, uint32_t width, uint32_t height, uint32_t clip_limit, uint32_t tileRadius)
{
    uint32_t mirrored_width = 2 * tileRadius + width,
             mirrored_height = 2 * tileRadius + height;
    
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
                    uint32_t pixelValue = in_image[tile_pixel_y * mirrored_width + tile_pixel_x];
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
            uint8_t new_pixel_value = std::round(static_cast<double>(tile_cdf[in_image[main_pixel_abs_pos]] - min_cdf) / (pixel_for_window - min_cdf) * (GRAYSCALE_RANGE - 1));
            out_image[(main_pixel_y - tileRadius) * width + (main_pixel_x - tileRadius)] = new_pixel_value;
        }
    }
}

void single_core_clahe(uint32_t clip_limit, uint32_t tileRadius)
{
    std::string in_img_path = "./../media/test3.jpg";
    std::string out_img_path = "./../media/test3_output_cpu.jpg";

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

    //Border mirroring
    uint32_t mirrored_width = 2 * tileRadius + width,
             mirrored_height = 2 * tileRadius + height;

    uint8_t* in_mirrored_image = (uint8_t*)malloc((mirrored_height) * (mirrored_width) * sizeof(uint8_t)); 
    memset(in_mirrored_image, 0, (mirrored_height) * (mirrored_width) * sizeof(uint8_t));

    mirror_img_borders(in_mirrored_image, input_img, mirrored_height, mirrored_width, height, width, tileRadius);

    cpu_clahe(in_mirrored_image, output_img, width, height, clip_limit, tileRadius);

    stbi_write_png(out_img_path.c_str(), width, height, 1, output_img, width);
}

int main()
{
    // Gpu
    auto start_time = std::chrono::high_resolution_clock::now();
    single_core_clahe(4, 20);
    auto end_time = std::chrono::high_resolution_clock::now();
    auto elapsed_time = std::chrono::duration_cast<std::chrono::duration<double>>(end_time - start_time);
    std::cout << "Cpu version time elapsed : " << elapsed_time.count() << "\n";

    // Cpu
    start_time = std::chrono::high_resolution_clock::now();
    parallel_clahe(4, 20, 85);
    end_time = std::chrono::high_resolution_clock::now();
    elapsed_time = std::chrono::duration_cast<std::chrono::duration<double>>(end_time - start_time);
    std::cout << "Gpu version time elapsed : " << elapsed_time.count() << "\n";
}