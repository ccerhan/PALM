#include <iostream>
#include <opencv2/highgui.hpp>
#include "PALM.h"

int main()
{
    // Type an image path to compute PALM descriptor
    std::string testImagePath = "test.png";

    // Type an image path to save the resulting pattern image
    std::string patternImagePath = "pattern_image.png";

    // Load the test image
    cv::Mat image = cv::imread(testImagePath, cv::IMREAD_GRAYSCALE);
    if (image.empty())
    {
        std::cout << "Please type an image path to compute PALM descriptor!" << std::endl;
        return -1;
    }

    // Adjust the parameters of PALM
    palm::PALMConfig config;
    config.momentOrder = 2;
    config.gridSize = 5;
    config.stepSize = 8;
    config.patchSize = 32;
    config.filterType = palm::FilterType::Approximated;
    config.applyInsidePartitioning = true;

    // Create PALM object as a smart pointer
    cv::Ptr<palm::PALM> palm = new palm::PALM(config, false);
    palm->initialize(); // Manual initialization. Pass TRUE to the second argument of the constructor to skip this line

    // Compute the descriptor
    cv::Mat descriptor = palm->compute(image);
    cv::Mat patternImage = palm->lastPatternImage();

    // Calculate distance between two descriptors
    double distance = palm->distance(descriptor, descriptor); // Must be zero, because the descriptors are the same

    // Print the descriptor
    std::cout << descriptor << std::endl;

    // Save the pattern image
    cv::Mat normalizedPatternImage;
    cv::normalize(patternImage, normalizedPatternImage, 0, 255, cv::NORM_MINMAX, CV_8U);
    cv::resize(normalizedPatternImage, normalizedPatternImage, image.size(), 0, 0, cv::INTER_AREA);
    cv::imwrite(patternImagePath, normalizedPatternImage, {CV_IMWRITE_PNG_COMPRESSION, 0});

    return 0;
}