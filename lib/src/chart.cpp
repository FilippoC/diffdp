#include <stdexcept>
#include <iostream>
#include "diffdp/chart.h"

#define MAX_FLOAT_SIZE 101*101*101*16
#define MAX_UNSIGNED_SIZE 101*101*16
#define MAX_SIGNED_LOG_SIZE 101*101*16
#define N_FLOAT 16 // 64
#define N_UNSIGNED 16 // 64
#define N_SIGNED_LOG 16 // 64

namespace diffdp
{

ChartMemory& ChartMemory::getInstance()
{
    static ChartMemory instance;
    return instance;
}

ChartMemory::~ChartMemory()
{
    delete[] _float_mem;
    delete[] _unsigned_mem;
}
ChartMemory::ChartMemory() :
    _float_mem(new float[MAX_FLOAT_SIZE * N_FLOAT]),
    _unsigned_mem(new unsigned[MAX_UNSIGNED_SIZE * N_UNSIGNED]),
    _float_usage(N_FLOAT, false),
    _unsigned_usage(N_UNSIGNED, false)
{
}

float* ChartMemory::get_float(unsigned size)
{
    if (size > MAX_FLOAT_SIZE)
        throw std::runtime_error("Asking for a too big chart");
    for (unsigned i = 0u ; i < _float_usage.size() ; ++i)
    {
        if (!_float_usage.at(i))
        {
            _float_usage.at(i) = true;
            float* adr = _float_mem + i * MAX_FLOAT_SIZE;
            //std::fill(adr, adr + size, float{});
            //std::cerr << "get_float: " << adr << "\n";
            return adr;
        }
    }
    throw std::runtime_error("No more memory");
}

void ChartMemory::release(float* start)
{
    //std::cerr << "release_float: " << start << "\n";
    float* ptr = _float_mem;
    for (unsigned i = 0u ; i < _float_usage.size() ; ++i)
    {
        if (ptr == start)
        {
            if (!_float_usage.at(i))
                throw std::runtime_error("Trying to release unused memory");
            _float_usage.at(i) = false;
            return;
        }
        ptr += MAX_FLOAT_SIZE;
    }
    throw std::runtime_error("Unknown memory address");
}

unsigned* ChartMemory::get_unsigned(unsigned size)
{
    if (size > MAX_UNSIGNED_SIZE)
        throw std::runtime_error("Asking for a too big chart");
    for (unsigned i = 0u ; i < _unsigned_usage.size() ; ++i)
    {
        if (!_unsigned_usage.at(i))
        {
            _unsigned_usage.at(i) = true;
            unsigned* adr = _unsigned_mem + i * MAX_UNSIGNED_SIZE;
            //std::fill(adr, adr + size, unsigned{});
            //std::cerr << "get_unsigned: " << adr << "\n";
            return adr;
        }
    }
    throw std::runtime_error("No more memory");
}

void ChartMemory::release(unsigned* start)
{
    //std::cerr << "release_unsigned: " << start << "\n";
    unsigned* ptr = _unsigned_mem;
    for (unsigned i = 0u ; i < _unsigned_usage.size() ; ++i)
    {
        if (ptr == start)
        {
            if (!_unsigned_usage.at(i))
                throw std::runtime_error("Trying to release unused memory");
            _unsigned_usage.at(i) = false;
            return;
        }
        ptr += MAX_UNSIGNED_SIZE;
    }
    throw std::runtime_error("Unknown memory address");
}


ChartMatrix3D::ChartMatrix3D(const unsigned size) :
    _size(size),
    _free_data(true)
{
    _data= new float[required_cells(size)]();
}

ChartMatrix3D::ChartMatrix3D(const unsigned size, float* _data) :
    _size(size),
    _free_data(false),
    _data(_data)
{}

ChartMatrix3D::~ChartMatrix3D()
{
    if (_free_data)
        delete[] _data;
}



}
