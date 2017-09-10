#ifndef PALM_FILTER_H
#define PALM_FILTER_H

#include <opencv2/core.hpp>


namespace palm
{
    class IlluminationFilter
    {
    public:
        IlluminationFilter(double alpha = 0.3, int returnType = CV_8U);
        virtual ~IlluminationFilter() { }

        double getAlpha() { return _alpha; }
        void setAlpha(double alpha);

        int getReturnType() { return _returnType; }
        void setReturnType(int returnType);

        cv::Mat apply(const cv::Mat &image);

    protected:

    private:
        double _alpha;
        int _returnType;
    };
}

#endif //PALM_FILTER_H
