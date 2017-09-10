#ifndef PALM_ZERNIKEBASEGENERATOR_H
#define PALM_ZERNIKEBASEGENERATOR_H

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>


namespace palm
{
    class ZernikeBase
    {
    public:
        ZernikeBase(int n, int m, cv::Mat reel, cv::Mat imag);

        int n() { return _n; }
        int m() { return _m; }
        cv::Mat reel() { return _reel; }
        cv::Mat imag() { return _imag; }

    private:
        int _n, _m;
        cv::Mat _reel, _imag;
    };


    class ZernikeBaseGenerator
    {
    public:
        ZernikeBaseGenerator(int size);
        virtual ~ZernikeBaseGenerator() { };

        void setSize(int size);
        int getSize() const { return _size; }

        virtual ZernikeBase generate(int n, int m);
        virtual std::vector<ZernikeBase> generate(int n);

    protected:
        bool nmRelation(int n, int m);
        double factorial(int x);
        void compute(int n, int m, int size, cv::Mat &reel, cv::Mat &imag);

    private:
        int _size;
    };


    class ApproximatedZernikeBaseGenerator : public ZernikeBaseGenerator
    {
    public:
        ApproximatedZernikeBaseGenerator(int size, int coreSize = 4);

        void setCoreSize(int coreSize);
        int getCoreSize() const { return _coreSize; }

        ZernikeBase generate(int n, int m) override;

    private:
        int _coreSize;
    };
}

#endif //PALM_ZERNIKEBASEGENERATOR_H
