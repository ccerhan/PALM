#include "PALM.h"

using namespace palm;


PALMConfig::PALMConfig()
{
    patchSize = 32;
    gridSize = 5;
    stepSize = 8;
    momentOrder = 2;
    filterType = FilterType::Approximated;
    applyInsidePartitioning = true;
}


PALM::PALM(bool initialize)
{
    PALMConfig config;
    setConfig(config, initialize);
}

PALM::PALM(PALMConfig config, bool initialize)
{
    setConfig(config, initialize);
}

void PALM::setConfig(PALMConfig config, bool initialize)
{
    _config = config;

    if (initialize)
    {
        this->initialize();
    }
}

void PALM::initialize()
{
    _PatternImageExtractor = PatternImageExtractor::create(_config.filterType, _config.patchSize, _config.stepSize,
                                                           _config.momentOrder);

    cv::Size gridSize = cv::Size(_config.gridSize, _config.gridSize);
    int binCount = (int) std::pow(2, _PatternImageExtractor->filters().size());

    _HistogramBuilder = new HistogramBuilder(gridSize, binCount, _config.applyInsidePartitioning);
}

bool PALM::isInitialized() const
{
    return _HistogramBuilder != nullptr &&
           _PatternImageExtractor != nullptr;
}

int PALM::descriptorSize() const
{
    CV_Assert(isInitialized());

    return _HistogramBuilder->histogramLength();
}

cv::Mat PALM::lastPatternImage()
{
    CV_Assert(!_LastPatternImage.empty());

    return _LastPatternImage;
}

std::vector<cv::Mat> PALM::filters() const
{
    return _PatternImageExtractor->filters();
}

cv::Mat PALM::compute(const cv::Mat &image)
{
    CV_Assert(isInitialized());

    cv::Mat patterns = _PatternImageExtractor->extract(image);
    _LastPatternImage = patterns;

    cv::Mat desc = _HistogramBuilder->build(patterns);

    return desc;
}

cv::Mat PALM::compute(const std::vector<cv::Mat> &images, bool rowStack)
{
    CV_Assert(isInitialized());
    CV_Assert(images.size() > 0);

    int size = descriptorSize();

    cv::Mat descs;
    if (rowStack)
    {
        descs = cv::Mat::zeros((int) images.size(), size, CV_64F);
    }
    else
    {
        descs = cv::Mat::zeros(1, size * (int) images.size(), CV_64F);
    }

    for (int i = 0; i < images.size(); i++)
    {
        cv::Mat desc = compute(images[i]);

        cv::Rect location = rowStack ? cv::Rect(0, i, size, 1) : cv::Rect(i * size, 0, size, 1);
        desc.copyTo(descs(location));
    }

    return descs;
}

double PALM::distance(const cv::Mat &desc1, const cv::Mat &desc2) const
{
    CV_Assert(desc1.cols > 0 && desc1.rows == 1 && desc2.cols > 0 && desc2.rows == 1 && desc1.cols == desc2.cols);

    return cv::norm(desc1, desc2, cv::NORM_L1);
}