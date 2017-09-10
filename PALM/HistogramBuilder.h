#ifndef PALM_HISTOGRAMBUILDER_H
#define PALM_HISTOGRAMBUILDER_H

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>


namespace palm
{
    class HistogramBuilder
    {
    public:
        HistogramBuilder(cv::Size gridSize, int binCount, bool applyInsidePartitioning);
        virtual ~HistogramBuilder() { };

        cv::Size getGridSize() const { return _gridSize; }
        void setGridSize(cv::Size gridSize);

        bool isInsidePartitioningApplied() const { return _applyInsidePartitioning; }
        void setApplyInsidePartitioning(bool applyInsidePartitioning);

        int getBinCount() const { return _binCount; }
        void setBinCount(int binCount);

        virtual int histogramLength();
        virtual cv::Mat build(const cv::Mat &image);

    protected:
        cv::Mat getGaussianKernel(cv::Size size, double sigma) const;
        cv::Mat getRegionHistogram(const cv::Mat &region, int binCount, const cv::Mat &gaussianKernel);
        cv::Mat compute(const cv::Mat &image, cv::Size gridSize, int binCount, bool applySlidedGrid);

    private:
        cv::Size _gridSize;
        bool _applyInsidePartitioning;
        int _binCount;
    };
}

#endif //PALM_HISTOGRAMBUILDER_H
