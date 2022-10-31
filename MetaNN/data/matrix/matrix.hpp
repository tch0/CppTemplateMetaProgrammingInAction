#pragma once
#ifndef METANN_MATRIX_HPP_
#define METANN_MATRIX_HPP_

#include <data/tags.hpp>
#include <data/allocator.hpp>
#include <data/lower_access.hpp>
#include <type_traits>
#include <cassert>

namespace MetaNN
{

template<typename TElem, typename TDevice = DeviceTags::CPU>
class Matrix;

// 提供底层访问接口
template<typename TElem>
struct LowerAccessImpl<Matrix<TElem, DeviceTags::CPU>>;

// 为Batch提供前向声明
template<typename TElem, typename TDevice, typename TCategory>
class Batch;

template<typename TElem>
class Matrix<TElem, DeviceTags::CPU>
{
    static_assert(std::is_same_v<std::remove_cvref_t<TElem>, TElem>, "TElem is not an available type");
    friend struct LowerAccessImpl<Matrix<TElem, DeviceTags::CPU>>;
    friend struct Batch<TElem, DeviceTags::CPU, CategoryTags::Matrix>;
public:
    using Category = CategoryTags::Matrix;
    using ElementType = TElem;
    using DeviceType = DeviceTags::CPU;
public:
    Matrix(std::size_t row = 0, std::size_t col = 0)
        : m_mem(row * col)
        , m_rowNum(row)
        , m_colNum(col)
        , m_rowLen(col)
    {
    }

    // 访问接口
    std::size_t rowNum() const
    {
        return m_rowNum;
    }
    std::size_t colNum() const
    {
        return m_colNum;
    }
    // 写操作，需要可写才能调用
    void setValue(std::size_t row, std::size_t col, ElementType val)
    {
        assert(availableForWrite());
        assert(row < m_rowNum && col < m_colNum);
        m_mem.rawMemory()[row * m_rowLen + col] = val;
    }
    // 读操作，返回副本而非引用
    const auto operator()(std::size_t row, std::size_t col) const
    {
        assert(row < m_rowNum && col < m_colNum);
        return m_mem.rawMemory()[row * m_rowLen + col];
    }
    bool availableForWrite() const
    {
        return m_mem.useCount() == 1;
    }

    // 子矩阵接口，浅拷贝，共享存储空间，区间前闭后开
    Matrix subMatrix(std::size_t rowBegin, std::size_t rowEnd, std::size_t colBegin, std::size_t colEnd)
    {
        assert(rowBegin < m_rowNum && colBegin < m_colNum);
        assert(rowEnd <= m_rowNum && colEnd <= m_colNum);
        TElem* pos = m_mem.rawMemory() + rowBegin * m_rowLen + colBegin;
        return Matrix(m_mem.sharedPtr(), pos, rowEnd - rowBegin, colEnd - colBegin, m_rowLen);
    }

    // 求值接口: todo

private:
    // 为构造子矩阵准备
    Matrix(std::shared_ptr<ElementType> spMem, ElementType* pMemStart,
        std::size_t row, std::size_t col, std::size_t rowLen)
        : m_mem(spMem, pMemStart)
        , m_rowNum(row)
        , m_colNum(col)
        , m_rowLen(rowLen)
    {
    }
private:
    ContinuousMemory<ElementType, DeviceType> m_mem;
    std::size_t m_rowNum;
    std::size_t m_colNum;
    std::size_t m_rowLen;
};

// 底层访问
template<typename TElem>
struct LowerAccessImpl<Matrix<TElem, DeviceTags::CPU>>
{
    LowerAccessImpl(Matrix<TElem, DeviceTags::CPU> p)
        : m_matrix(p) {}
    
    // 使用这个接口提供的指针进行写操作具有一定安全性隐患，因为不会检查共享数量
    // 所以这个只应该提供给库作者使用，提供性能更高的操作，相当于一个后门，使用时应当非常注意
    auto mutableRawMemory()
    {
        return m_matrix.m_mem.rawMemory();
    }

    const auto rawMemory() const
    {
        return m_matrix.m_mem.rawMemory();
    }

    std::size_t rowLen() const
    {
        return m_matrix.m_rowLen;
    }

private:
    Matrix<TElem, DeviceTags::CPU> m_matrix;
};

} // namespace MetaNN

#endif // METANN_MATRIX_HPP_