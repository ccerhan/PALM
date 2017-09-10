#ifndef PALM_PATTERNIMAGEEXTRACTOR_H
#define PALM_PATTERNIMAGEEXTRACTOR_H

#include "ZernikeBaseGenerator.h"


namespace palm
{
    enum class FilterType
    {
        Regular,
        Approximated
    };


    class PatternImageExtractor
    {
    public:
        PatternImageExtractor(FilterType filterType, int patchSize, int stepSize, int momentOrder);
        virtual ~PatternImageExtractor() { };

        static PatternImageExtractor* create(FilterType filterType, int patchSize, int stepSize, int momentOrder);

        int getPatchSize() const { return _patchSize; }
        void setPatchSize(int patchSize);

        int getOverlapDensity() const;
        int getStepSize() const { return _stepSize; }
        void setStepSize(int stepSize);

        int getMomentOrder() const { return _momentOrder; }
        void setMomentOrder(int momentOrder);

        FilterType filterType() const;
        virtual std::vector<cv::Mat> filters() const;
        virtual cv::Mat extract(const cv::Mat &image);

    protected:
        std::vector<cv::Mat> _Filters;

        virtual std::vector<cv::Mat> createFilters(const cv::Ptr<ZernikeBaseGenerator> &baseGenerator, int momentOrder);
        virtual uchar applyFilters(int patchSize, const cv::Mat &src, const std::vector<cv::Mat> &filters);
        virtual cv::Mat compute(const cv::Mat &input, int patchSize, int stepSize, std::vector<cv::Mat> filters);

    private:
        FilterType _filterType;
        int _patchSize;
        int _stepSize;
        int _momentOrder;
    };


    class RegularPatternImageExtractor : public PatternImageExtractor
    {
    public:
        RegularPatternImageExtractor(int patchSize, int stepSize, int momentOrder);

    protected:

    private:

    };


    class ApproximatedPatternImageExtractor : public PatternImageExtractor
    {
    public:
        ApproximatedPatternImageExtractor(int patchSize, int stepSize, int momentOrder);

        static const int FILTER_CORE_SIZE = 4;

        virtual cv::Mat extract(const cv::Mat &image) override;

    protected:
        uchar applyFilters(int patchSize, const cv::Mat &src, const std::vector<cv::Mat> &filters) override;

    private:

    };
}

#endif //PALM_PATTERNIMAGEEXTRACTOR_H
