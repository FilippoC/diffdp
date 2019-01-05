#pragma once

#include <vector>

namespace diffdp
{

class ChartMemory
{
public:

    float* _float_mem;
    unsigned *_unsigned_mem;

    std::vector<bool> _float_usage;
    std::vector<bool> _unsigned_usage;

    static ChartMemory& getInstance();
    ChartMemory(ChartMemory const&) = delete;
    void operator=(ChartMemory const&) = delete;
    ~ChartMemory();

    float* get_float(unsigned size);
    void release(float* start);
    unsigned* get_unsigned(unsigned size);
    void release(unsigned* start);

private:
    ChartMemory();
};

struct ChartMatrix3D
{
    unsigned _size;
    bool _free_data;
    float* _data;

    ChartMatrix3D(const unsigned size);
    ChartMatrix3D(const unsigned size, float* _data);
    ~ChartMatrix3D();

    inline static
    std::size_t required_memory(const unsigned size)
    {
        return required_cells(size) * sizeof(float);
    }

    inline static
    unsigned required_cells(const unsigned size)
    {
        return size * size * size;
    }

    inline
    float& operator()(const unsigned i, const unsigned j, const unsigned k) noexcept
    {
        return _data[i * _size * _size + j * _size + k];
    }

    inline
    float operator()(const unsigned i, const unsigned j, const unsigned k) const noexcept
    {
        return _data[i * _size * _size + j * _size + k];
    }

    inline
    float* iter3(const unsigned i, const unsigned j, const unsigned k) noexcept
    {
        return _data + i * _size *_size + j * _size + k;
    }
};

template<class T>
struct ChartMatrix2D;

template<class T>
struct RowIt
{
    ChartMatrix2D<T>* chart;
    T* current;

    RowIt(ChartMatrix2D<T>* chart, float* current) :
        chart(chart),
        current(current)
    {}

    // copy constructor
    RowIt<T>(const RowIt<T>& o) :
        chart(o.chart),
        current(o.current)
    {}

    //float& operator*() noexcept;
    //RowIt& operator++() noexcept;
    T& operator*()
    {
        return *current;
    }

    RowIt<T>& operator++()
    {
        current += chart->_size;
        return *this;
    }

    bool operator!=(const RowIt<T>& o) const
    {
        return !(chart == o.chart && current == o.current);
    }
};

template<class T>
struct ChartMatrix2D
{
    unsigned _size;
    bool _free_data;
    T* _data;

    ChartMatrix2D(const unsigned size) :
        _size(size),
        _free_data(true),
        _data(new T[required_cells(size)])
    {
        std::fill(_data, _data + required_cells(size), T{});
        std::cerr << "MALLOC 3\n";
    }

    ChartMatrix2D(const unsigned size, T* _data) :
        _size(size),
        _free_data(false),
        _data(_data)
    {
        std::fill(_data, _data + required_cells(size), T{});
    }

    inline static
    std::size_t required_memory(const unsigned size)
    {
        return required_cells(size) * sizeof(T);
    }

    inline static
    unsigned required_cells(const unsigned size)
    {
        return size * size;
    }

    ~ChartMatrix2D()
    {
        if (_free_data)
            delete[] _data;
    }

    inline
    T& operator()(const unsigned i, const unsigned j) noexcept
    {
        return _data[i * _size + j];
    }

    inline
    T operator()(const unsigned i, const unsigned j) const noexcept
    {
        return _data[i * _size + j];
    }

    inline
    RowIt<T> iter1(const unsigned i, const unsigned j) noexcept
    {
        return {this, _data + i * _size + j};
    }

    inline
    T* iter2(const unsigned i, const unsigned j) noexcept
    {
        return _data + i * _size + j;
    }
};

}
