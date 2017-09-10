#include "HistogramBuilder.h"

using namespace palm;


HistogramBuilder::HistogramBuilder(cv::Size gridSize, int binCount, bool applyInsidePartitioning)
{
    setGridSize(gridSize);
    setBinCount(binCount);
    setApplyInsidePartitioning(applyInsidePartitioning);
}

void HistogramBuilder::setGridSize(cv::Size gridSize)
{
    CV_Assert(gridSize.height > 0 && gridSize.width > 0);

    _gridSize = gridSize;
}

void HistogramBuilder::setApplyInsidePartitioning(bool applyInsidePartitioning)
{
    _applyInsidePartitioning = applyInsidePartitioning;
}

void HistogramBuilder::setBinCount(int binCount)
{
    CV_Assert(binCount > 1);

    _binCount = binCount;
}

cv::Mat HistogramBuilder::build(const cv::Mat &image)
{
    return compute(image, getGridSize(), getBinCount(), isInsidePartitioningApplied());
}

cv::Mat HistogramBuilder::getGaussianKernel(cv::Size size, double sigma) const
{
    CV_Assert(sigma > 0);

    int kernelSize = (size.height % 2 != 0) ? size.height : size.height + 1;

    cv::Mat kernel = cv::getGaussianKernel(kernelSize, sigma);
    kernel = kernel * kernel.t();

    if (kernel.size() != size)
    {
        cv::resize(kernel, kernel, size);
    }

    return kernel;
}

cv::Mat HistogramBuilder::getRegionHistogram(const cv::Mat &region, int binCount, const cv::Mat &gaussianKernel)
{
    cv::Mat histogram = cv::Mat::zeros(1, binCount, CV_64F);

    for (int i = 0; i < region.rows; i++)
    {
        for (int j = 0; j < region.cols; j++)
        {
            int bin = region.at<uchar>(i, j);
            histogram.at<double>(0, bin) += gaussianKernel.at<double>(i, j);
        }
    }

    histogram = histogram / (cv::norm(histogram, cv::NORM_L2) + std::numeric_limits<double>::epsilon());

    return histogram;
}

int HistogramBuilder::histogramLength()
{
    cv::Size gridSize = getGridSize();
    int binCount = getBinCount();

    int length = gridSize.area() * binCount;
    if (isInsidePartitioningApplied())
    {
        length += (gridSize.height - 1) * (gridSize.width - 1) * binCount;
    }

    return length;
}

cv::Mat HistogramBuilder::compute(const cv::Mat &image, cv::Size gridSize, int binCount, bool applySlidedGrid)
{
    CV_Assert(!image.empty() && image.rows > 0 && image.cols > 0);
    CV_Assert(gridSize.width > 0 && gridSize.height > 0);

    cv::Size regionSize = cv::Size(image.cols / gridSize.width, image.rows / gridSize.height);
    cv::Mat gaussianKernel = getGaussianKernel(regionSize, 8);

    int descriptorSize = histogramLength();
    cv::Mat histogram(1, descriptorSize, CV_64F);

    // Compute the histograms for the complete grid
    for (int i = 0; i < gridSize.width; i++)
    {
        for (int j = 0; j < gridSize.height; j++)
        {
            cv::Rect roi = cv::Rect(i * regionSize.width, j * regionSize.height, regionSize.width, regionSize.height);
            cv::Mat region = image(roi);

            cv::Mat regionHistogram = getRegionHistogram(region, binCount, gaussianKernel);

            cv::Rect targetLocation = cv::Rect(binCount * (j * gridSize.width + i), 0, binCount, 1);
            cv::Mat targetRegion = histogram(targetLocation);
            regionHistogram.copyTo(targetRegion);
        }
    }

    if (applySlidedGrid)
    {
        // Compute the histograms for the slided grid

        int numRegionsX = gridSize.width - 1;
        int numRegionsY = gridSize.height - 1;
        int numRegions = gridSize.width * gridSize.height;

        for (uint i = 0; i < numRegionsX; i++)
        {
            for (uint j = 0; j < numRegionsY; j++)
            {
                cv::Range rowRange = cv::Range(j * regionSize.height + regionSize.height / 2,
                                               (j + 1) * regionSize.height + regionSize.height / 2);
                cv::Range colRange = cv::Range(i * regionSize.width + regionSize.width / 2,
                                               (i + 1) * regionSize.width + regionSize.width / 2);
                cv::Mat region = image(rowRange, colRange).clone();

                cv::Mat regionHistogram = getRegionHistogram(region, binCount, gaussianKernel);

                cv::Rect targetLocation = cv::Rect(binCount * (numRegions + j * numRegionsX + i), 0, binCount, 1);
                cv::Mat targetRegion = histogram(targetLocation);
                regionHistogram.copyTo(targetRegion);
            }
        }
    }

    return histogram;
}