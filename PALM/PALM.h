#ifndef PALM_H
#define PALM_H

#include "ZernikeBaseGenerator.h"
#include "PatternImageExtractor.h"
#include "HistogramBuilder.h"
#include "IlluminationFilter.h"


namespace palm
{
    class PALMConfig
    {
    public:
        PALMConfig();
        virtual ~PALMConfig() { };

        int patchSize;
        int gridSize;
        int stepSize;
        int momentOrder;
        FilterType filterType;
        bool applyInsidePartitioning;
    };


    class PALM
    {
    public:
        PALM(bool initialize = true);
        PALM(PALMConfig config, bool initialize = true);
        virtual ~PALM() { }

        void setConfig(PALMConfig config, bool initialize = true);
        PALMConfig getConfig() { return _config; }

        virtual void initialize();
        virtual bool isInitialized() const;
        virtual int descriptorSize() const;
        virtual std::vector<cv::Mat> filters() const;
        virtual cv::Mat lastPatternImage();
        virtual cv::Mat compute(const cv::Mat &image);
        virtual cv::Mat compute(const std::vector<cv::Mat> &images, bool rowStack = false);
        virtual double distance(const cv::Mat &desc1, const cv::Mat &desc2) const;

    protected:
        cv::Ptr<PatternImageExtractor> _PatternImageExtractor;
        cv::Ptr<HistogramBuilder> _HistogramBuilder;
        cv::Mat _LastPatternImage;

    private:
        PALMConfig _config;
    };
}

#endif //PALM_H
