#pragma once

#include <filesystem>
#include <vector>
#include <iostream>
#include "matplotlibcpp.h"

#define GRAYSCALE_RANGE 256

namespace plt = matplotlibcpp;
namespace fs = std::filesystem;

void test()
{
    std::cout << "test\n";
}

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