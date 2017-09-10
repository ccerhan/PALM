#include "IlluminationFilter.h"

palm::IlluminationFilter::IlluminationFilter(double alpha, int returnType)
{
    setAlpha(alpha);
    setReturnType(returnType);
}

void palm::IlluminationFilter::setAlpha(double alpha)
{
    CV_Assert(alpha > 0 && alpha < 1);

    _alpha = alpha;
}

void palm::IlluminationFilter::setReturnType(int returnType)
{
    CV_Assert(returnType == CV_8U || returnType == CV_64F);

    _returnType = returnType;
}

cv::Mat palm::IlluminationFilter::apply(const cv::Mat &image)
{
    CV_Assert(!image.empty() && image.type() == CV_8UC3);

    cv::Mat out = cv::Mat::zeros(image.rows, image.cols, _returnType);
    for (int i = 0; i < out.rows; i++)
    {
        for (int j = 0; j < out.cols; j++)
        {
            cv::Vec3b value = image.at<cv::Vec3b>(i, j);
            double b = value.val[0] / 255.0;
            double g = value.val[1] / 255.0;
            double r = value.val[2] / 255.0;

            double temp = 0.5 + std::log(g) - _alpha * std::log(b) - (1 - _alpha) * std::log(r);
            if (temp > 1) temp = 1;
            if (temp < 0) temp = 0;

            if (_returnType == CV_8U)
            {
                out.at<uchar>(i, j) = (uchar) (temp * 255.0);
            }
            else if (_returnType == CV_64F)
            {
                out.at<double>(i, j) = temp;
            }
        }
    }

    return out;
}
