#pragma once

#include <vector>

namespace diffdp
{

template<class T>
struct Tensor3D
{
    unsigned _size;
    bool _free_data;
    T* _data;

    Tensor3D(const unsigned size);
    Tensor3D(const unsigned size, T* _data);
    ~Tensor3D();

    static std::size_t required_memory(const unsigned size);
    static unsigned required_cells(const unsigned size);

    inline T& operator()(const unsigned i, const unsigned j, const unsigned k) noexcept;
    inline T operator()(const unsigned i, const unsigned j, const unsigned k) const noexcept;

    inline
    T* iter3(const unsigned i, const unsigned j, const unsigned k) noexcept;
};

template<class T>
struct Matrix;

template<class T>
struct MatrixRowIterator
{
    Matrix<T>* chart;
    T* current;

    MatrixRowIterator(Matrix<T>* chart, T* current);
    MatrixRowIterator<T>(const MatrixRowIterator<T>& o);

    T& operator*();
    MatrixRowIterator<T>& operator++();
    bool operator!=(const MatrixRowIterator<T>& o) const;
};

template<class T>
struct Matrix
{
    unsigned _size;
    bool _free_data;
    T* _data;

    Matrix(const unsigned size);
    Matrix(const unsigned size, T* _data);
    ~Matrix();

    inline static std::size_t required_memory(const unsigned size);
    inline static unsigned required_cells(const unsigned size);

    inline T& operator()(const unsigned i, const unsigned j) noexcept;
    inline T operator()(const unsigned i, const unsigned j) const noexcept;

    inline MatrixRowIterator<T> iter1(const unsigned i, const unsigned j) noexcept;
    inline T* iter2(const unsigned i, const unsigned j) noexcept;
};


// Template implementations
template <class T>
Tensor3D<T>::Tensor3D(const unsigned size) :
    _size(size),
    _free_data(true)
{
    _data = new T[required_cells(size)];
}

template <class T>
Tensor3D<T>::Tensor3D(const unsigned size, T* _data) :
    _size(size),
    _free_data(false),
    _data(_data)
{}

template <class T>
Tensor3D<T>::~Tensor3D()
{
    if (_free_data)
        delete[] _data;
}

template <class T>
std::size_t Tensor3D<T>::required_memory(const unsigned size)
{
    return required_cells(size) * sizeof(T);
}

template <class T>
unsigned Tensor3D<T>::required_cells(const unsigned size)
{
    return size * size * size;
}


template <class T>
T& Tensor3D<T>::operator()(const unsigned i, const unsigned j, const unsigned k) noexcept
{
    return _data[i * _size * _size + j * _size + k];
}


template <class T>
T Tensor3D<T>::operator()(const unsigned i, const unsigned j, const unsigned k) const noexcept
{
    return _data[i * _size * _size + j * _size + k];
}


template <class T>
T* Tensor3D<T>::iter3(const unsigned i, const unsigned j, const unsigned k) noexcept
{
    return _data + i * _size *_size + j * _size + k;
}


template <class T>
MatrixRowIterator<T>::MatrixRowIterator(Matrix<T>* chart, T* current) :
        chart(chart),
        current(current)
{}

template <class T>
MatrixRowIterator<T>::MatrixRowIterator(const MatrixRowIterator<T>& o) :
        chart(o.chart),
        current(o.current)
{}

template <class T>
T& MatrixRowIterator<T>::operator*()
{
    return *current;
}

template <class T>
MatrixRowIterator<T>& MatrixRowIterator<T>::operator++()
{
    current += chart->_size;
    return *this;
}

template <class T>
bool MatrixRowIterator<T>::operator!=(const MatrixRowIterator<T>& o) const
{
    return !(chart == o.chart && current == o.current);
}



template<class T>
Matrix<T>::Matrix(const unsigned size) :
        _size(size),
        _free_data(true),
        _data(new T[required_cells(size)])
{}

template<class T>
Matrix<T>::Matrix(const unsigned size, T* _data) :
        _size(size),
        _free_data(false),
        _data(_data)
{}

template<class T>
std::size_t Matrix<T>::required_memory(const unsigned size)
{
    return required_cells(size) * sizeof(T);
}

template<class T>
unsigned Matrix<T>::required_cells(const unsigned size)
{
    return size * size;
}

template<class T>
Matrix<T>::~Matrix()
{
    if (_free_data)
        delete[] _data;
}

template<class T>
T& Matrix<T>::operator()(const unsigned i, const unsigned j) noexcept
{
    return _data[i * _size + j];
}

template<class T>
T Matrix<T>::operator()(const unsigned i, const unsigned j) const noexcept
{
    return _data[i * _size + j];
}

template<class T>
MatrixRowIterator<T> Matrix<T>::iter1(const unsigned i, const unsigned j) noexcept
{
    return {this, _data + i * _size + j};
}

template<class T>
T* Matrix<T>::iter2(const unsigned i, const unsigned j) noexcept
{
    return _data + i * _size + j;
}


}
