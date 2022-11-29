#pragma once
#include<chrono>
using namespace std::chrono;

class CELLTime
{
public:
    //获取当前时间戳 (毫秒)
    static time_t getNowInMilliSec()
    {
        return duration_cast<milliseconds>(high_resolution_clock::now().time_since_epoch()).count();
    }
};

class CELLTimestamp
{
public:
    CELLTimestamp()
    {
        update();
    }
    ~CELLTimestamp()
    {}

    void    update()
    {
        _begin = high_resolution_clock::now();
    }
    /**
    *   获取当前秒
    */
    double getElapsedSecond()
    {
        return  getElapsedTimeInMicroSec() * 0.000001;
    }
    /**
    *   获取毫秒
    */
    double getElapsedTimeInMilliSec()
    {
        return this->getElapsedTimeInMicroSec() * 0.001;
    }
    /**
    *   获取微妙
    */
    long long getElapsedTimeInMicroSec()
    {
        return duration_cast<microseconds>(high_resolution_clock::now() - _begin).count();
    }
protected:
    time_point<high_resolution_clock> _begin;
};
