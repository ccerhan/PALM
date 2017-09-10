#include "ZernikeBaseGenerator.h"


using namespace palm;


ZernikeBase::ZernikeBase(int n, int m, cv::Mat reel, cv::Mat imag)
        : _n(n), _m(m), _reel(reel), _imag(imag)
{

}

ZernikeBaseGenerator::ZernikeBaseGenerator(int size)
{
    setSize(size);
}

void ZernikeBaseGenerator::setSize(int size)
{
    CV_Assert(size > 1);

    _size = size;
}

ZernikeBase ZernikeBaseGenerator::generate(int n, int m)
{
    CV_Assert(n >= 0 && m >= 0);
    CV_Assert(nmRelation(n, m));

    cv::Mat reel, imag;
    compute(n, m, getSize(), reel, imag);

    ZernikeBase base(n, m, reel, imag);
    return base;
}

std::vector<ZernikeBase> ZernikeBaseGenerator::generate(int n)
{
    CV_Assert(n >= 0);

    std::vector<ZernikeBase> bases;
    for (int m = 0; m <= n; m++)
    {
        if (nmRelation(n, m))
        {
            ZernikeBase base = generate(n, m);
            bases.push_back(base);
        }
    }

    return bases;
}

bool ZernikeBaseGenerator::nmRelation(int n, int m)
{
    int m_ = std::abs(m);
    return m_ <= n && (n - m_) % 2 == 0;
}

double ZernikeBaseGenerator::factorial(int x)
{
    double result = 1.0;
    while (x > 0)
    {
        result = result * x;
        x--;
    }

    return result;
}

void ZernikeBaseGenerator::compute(int n, int m, int size, cv::Mat &reel, cv::Mat &imag)
{
    reel = cv::Mat::zeros(size, size, CV_64F);
    imag = cv::Mat::zeros(size, size, CV_64F);

    for (int y = 0; y < size; y++)
    {
        for (int x = 0; x < size; x++)
        {
            std::complex<double> value;

            double D = size * std::sqrt(2.);
            double xn = (2. * x + 1. - size) / D;
            double yn = (2. * y + 1. - size) / D;

            for (int s = 0; s <= (n - m) / 2; s++)
            {
                // theta must be between the range of (0,2PI)
                double theta = std::atan2(yn, xn);
                if (theta < 0)
                {
                    theta = 2 * M_PI + theta;
                }

                value += (pow(-1., (double) s)) * (factorial(n - s)) /
                         (factorial(s) * (factorial((n - 2 * s + m) / 2)) *
                          (factorial((n - 2 * s - m) / 2))) *
                         (pow(sqrt(xn * xn + yn * yn), (n - 2. * s))) *
                         4. / (D * D) * std::polar(1., m * theta);
            }

            reel.at<double>(y, x) = std::real(std::conj(value));
            imag.at<double>(y, x) = std::imag(std::conj(value));
        }
    }
}


ApproximatedZernikeBaseGenerator::ApproximatedZernikeBaseGenerator(int size, int coreSize)
        : ZernikeBaseGenerator(size)
{
    setCoreSize(coreSize);
}

void ApproximatedZernikeBaseGenerator::setCoreSize(int coreSize)
{
    CV_Assert(coreSize > 1);
    CV_Assert(coreSize <= getSize());

    _coreSize = coreSize;
}

ZernikeBase ApproximatedZernikeBaseGenerator::generate(int n, int m)
{
    CV_Assert(n >= 0 && m >= 0);
    CV_Assert(nmRelation(n, m));

    cv::Mat reel, imag;
    compute(n, m, getCoreSize(), reel, imag);

    cv::Size size = cv::Size(getSize(), getSize());
    cv::resize(reel, reel, size, 0, 0, cv::INTER_AREA);
    cv::resize(imag, imag, size, 0, 0, cv::INTER_AREA);

    ZernikeBase base(n, m, reel, imag);
    return base;
}