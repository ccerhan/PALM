#include "PatternImageExtractor.h"

using namespace palm;


PatternImageExtractor::PatternImageExtractor(FilterType filterType, int patchSize, int stepSize, int momentOrder)
        : _filterType(filterType)
{
    setPatchSize(patchSize);
    setStepSize(stepSize);
    setMomentOrder(momentOrder);
}

PatternImageExtractor *PatternImageExtractor::create(FilterType filterType, int patchSize, int stepSize,
                                                     int momentOrder)
{
    PatternImageExtractor *instance = nullptr;

    switch (filterType)
    {
        case FilterType::Regular:
            instance = new RegularPatternImageExtractor(patchSize, stepSize, momentOrder);
            break;

        case FilterType::Approximated:
            instance = new ApproximatedPatternImageExtractor(patchSize, stepSize, momentOrder);
            break;
    }

    return instance;
}

void PatternImageExtractor::setPatchSize(int patchSize)
{
    CV_Assert(patchSize > 3);

    _patchSize = patchSize;
}

void PatternImageExtractor::setStepSize(int stepSize)
{
    CV_Assert(stepSize > 0 && stepSize < _patchSize);

    _stepSize = stepSize;
}

int PatternImageExtractor::getOverlapDensity() const
{
    CV_Assert(_patchSize % _stepSize == 0 && _stepSize > 0 && _stepSize < _patchSize && _patchSize > 3);

    return _patchSize / _stepSize;
}

void PatternImageExtractor::setMomentOrder(int momentOrder)
{
    CV_Assert(momentOrder > 0 && momentOrder < 4);

    _momentOrder = momentOrder;
}

FilterType PatternImageExtractor::filterType() const
{
    return _filterType;
}

std::vector<cv::Mat> PatternImageExtractor::filters() const
{
    return _Filters;
}

cv::Mat PatternImageExtractor::extract(const cv::Mat &image)
{
    CV_Assert(!image.empty());
    CV_Assert(image.channels() == 1);
    CV_Assert(image.rows > 0 && image.cols > 0);

    cv::Mat input;
    image.convertTo(input, CV_64F);

    return compute(input, _patchSize, _stepSize, _Filters);
}

uchar PatternImageExtractor::applyFilters(int patchSize, const cv::Mat &src, const std::vector<cv::Mat> &filters)
{
    uchar value = 0;
    for (int k = 0; k < filters.size(); k++)
    {
        double sum = 0;
        for (int ii = 0; ii < patchSize; ii++)
        {
            for (int jj = 0; jj < patchSize; jj++)
            {
                sum += (src.at<double>(ii, jj) * filters[k].at<double>(ii, jj));
            }
        }

        value |= (uchar) (sum > 0) << k;
    }

    return value;
}

cv::Mat PatternImageExtractor::compute(const cv::Mat &input, int patchSize, int stepSize, std::vector<cv::Mat> filters)
{
    int rows = (input.rows - patchSize) / stepSize + 1;
    int cols = (input.cols - patchSize) / stepSize + 1;

    cv::Mat patterns = cv::Mat::zeros(rows, cols, CV_8U);

    for (int i = 0; i < patterns.rows; i++)
    {
        for (int j = 0; j < patterns.cols; j++)
        {
            cv::Mat src = input(cv::Rect(j * stepSize, i * stepSize, patchSize, patchSize));

            patterns.at<uchar>(i, j) = applyFilters(patchSize, src, filters);
        }
    }

    return patterns;
}

std::vector<cv::Mat> PatternImageExtractor::createFilters(const cv::Ptr<ZernikeBaseGenerator> &baseGenerator,
                                                          int momentOrder)
{
    CV_Assert(baseGenerator != nullptr);

    std::vector<cv::Mat> filters;
    for (int n = 0; n <= momentOrder; n++)
    {
        std::vector<ZernikeBase> bases = baseGenerator->generate(n);
        for (int i = 0; i < bases.size(); i++)
        {
            if (bases[i].m() != 0)
            {
                cv::Mat reel, imag;
                cv::normalize(bases[i].reel(), reel, -1, 1, cv::NORM_MINMAX, CV_64F);
                cv::normalize(bases[i].imag(), imag, -1, 1, cv::NORM_MINMAX, CV_64F);

                filters.push_back(reel);
                filters.push_back(imag);
            }
        }
    }

    return filters;
}


RegularPatternImageExtractor::RegularPatternImageExtractor(int patchSize, int stepSize, int momentOrder)
        : PatternImageExtractor(FilterType::Regular, patchSize, stepSize, momentOrder)
{
    cv::Ptr<ZernikeBaseGenerator> baseGenerator = new ZernikeBaseGenerator(patchSize);
    _Filters = createFilters(baseGenerator, momentOrder);
}


ApproximatedPatternImageExtractor::ApproximatedPatternImageExtractor(int patchSize, int stepSize, int momentOrder)
        : PatternImageExtractor(FilterType::Approximated, patchSize, stepSize, momentOrder)
{
    int overlapDensity = getOverlapDensity(); // Assert if overlap density could not calculated correctly

    cv::Ptr<ZernikeBaseGenerator> baseGenerator = new ApproximatedZernikeBaseGenerator(patchSize, FILTER_CORE_SIZE);
    _Filters = createFilters(baseGenerator, momentOrder);
}

cv::Mat ApproximatedPatternImageExtractor::extract(const cv::Mat &image)
{
    CV_Assert(!image.empty());
    CV_Assert(image.channels() == 1);
    CV_Assert(image.rows > 0 && image.cols > 0);

    cv::Mat input;
    image.convertTo(input, CV_64F);

    int patchSize = getPatchSize();
    int stepSize = getStepSize();

    int patch = patchSize / FILTER_CORE_SIZE;
    int step = stepSize / patch;

    cv::Mat values;
    cv::resize(input, values, cv::Size(), 1.0 / patch, 1.0 / patch, cv::INTER_AREA);

    return compute(values, FILTER_CORE_SIZE, step, _Filters);
}

uchar ApproximatedPatternImageExtractor::applyFilters(int patchSize, const cv::Mat &src,
                                                      const std::vector<cv::Mat> &filters)
{
    double v[FILTER_CORE_SIZE][FILTER_CORE_SIZE];
    for (int i = 0; i < FILTER_CORE_SIZE; i++)
    {
        for (int j = 0; j < FILTER_CORE_SIZE; j++)
        {
            v[i][j] = src.at<double>(i, j);
        }
    }

    int momentOrder = getMomentOrder();

    // Manually typed filter values for speed performance
    const double C_333 = 0.333333;
    const double C_111 = 0.111111;
    const double C_569 = 0.568627;
    const double C_294 = 0.294118;
    const double C_481 = 0.481481;
    const double C_037 = 0.037037;

    uchar value = 0;

    if (momentOrder > 0)
    {
        double v1 = -v[0][0] - v[0][1] * C_333 + v[0][2] * C_333 + v[0][3]
                    - v[1][0] - v[1][1] * C_333 + v[1][2] * C_333 + v[1][3]
                    - v[2][0] - v[2][1] * C_333 + v[2][2] * C_333 + v[2][3]
                    - v[3][0] - v[3][1] * C_333 + v[3][2] * C_333 + v[3][3];
        value |= (uchar) (v1 > 0) << 0;

        double v2 = v[0][0] + v[0][1] + v[0][2] + v[0][3]
                    + v[1][0] * C_333 + v[1][1] * C_333 + v[1][2] * C_333 + v[1][3] * C_333
                    - v[2][0] * C_333 - v[2][1] * C_333 - v[2][2] * C_333 - v[2][3] * C_333
                    - v[3][0] - v[3][1] - v[3][2] - v[3][3];
        value |= (uchar) (v2 > 0) << 1;
    }

    if (momentOrder > 1)
    {
        double v3 = -v[0][1] - v[0][2] - v[3][1] - v[3][2] + v[1][0] + v[2][0] + v[1][3] + v[2][3];
        value |= (uchar) (v3 > 0) << 2;

        double v4 = -v[0][0] - v[0][1] * C_333 + v[0][2] * C_333 + v[0][3]
                    - v[1][0] * C_333 - v[1][1] * C_111 + v[1][2] * C_111 + v[1][3] * C_333
                    + v[2][0] * C_333 + v[2][1] * C_111 - v[2][2] * C_111 - v[2][3] * C_333
                    + v[3][0] + v[3][1] * C_333 - v[3][2] * C_333 - v[3][3];
        value |= (uchar) (v4 > 0) << 3;
    }

    if (momentOrder > 2)
    {
        double v5 = v[0][0] * C_294 + v[0][1] * C_333 - v[0][2] * C_333 - v[0][3] * C_294
                    + v[1][0] + v[1][1] * C_569 - v[1][2] * C_569 - v[1][3]
                    + v[2][0] + v[2][1] * C_569 - v[2][2] * C_569 - v[2][3]
                    + v[3][0] * C_294 + v[3][1] * C_333 - v[3][2] * C_333 - v[3][3] * C_294;
        value |= (uchar) (v5 > 0) << 4;

        double v6 = -v[0][0] * C_294 - v[0][1] - v[0][2] - v[0][3] * C_294
                    - v[1][0] * C_333 - v[1][1] * C_569 - v[1][2] * C_569 - v[1][3] * C_333
                    + v[2][0] * C_333 + v[2][1] * C_569 + v[2][2] * C_569 + v[2][3] * C_333
                    + v[3][0] * C_294 + v[3][1] + v[3][2] + v[3][3] * C_294;
        value |= (uchar) (v6 > 0) << 5;

        double v7 = v[0][0] + v[0][1] * C_481 - v[0][2] * C_481 - v[0][3]
                    - v[1][0] * C_333 + v[1][1] * C_037 - v[1][2] * C_037 + v[1][3] * C_333
                    - v[2][0] * C_333 + v[2][1] * C_037 - v[2][2] * C_037 + v[2][3] * C_333
                    + v[3][0] + v[3][1] * C_481 - v[3][2] * C_481 - v[3][3];
        value |= (uchar) (v7 > 0) << 6;

        double v8 = v[0][0] - v[0][1] * C_333 - v[0][2] * C_333 + v[0][3]
                    + v[1][0] * C_481 + v[1][1] * C_037 + v[1][2] * C_037 + v[1][3] * C_481
                    - v[2][0] * C_481 - v[2][1] * C_037 - v[2][2] * C_037 - v[2][3] * C_481
                    - v[3][0] + v[3][1] * C_333 + v[3][2] * C_333 - v[3][3];
        value |= (uchar) (v8 > 0) << 7;
    }

    return value;
}