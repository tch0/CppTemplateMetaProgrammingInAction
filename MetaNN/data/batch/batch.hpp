#pragma once

#include <data/tags.hpp>
#include <data/lower_access.hpp>
#include <data/allocator.hpp>
#include <data/matrix/matrix.hpp>
#include <cassert>

namespace MetaNN
{

template<typename TElem, typename TDevice, typename TCategory>
class Batch;

// 别名
template<typename TElem>
using CpuBatchScalar = Batch<TElem, DeviceTags::CPU, CategoryTags::Scalar>;
template<typename TElem>
using CpuBatchMatix = Batch<TElem, DeviceTags::CPU, CategoryTags::Matrix>;

// 底层访问
template<typename TElem>
struct LowerAccessImpl<Batch<TElem, DeviceTags::CPU, CategoryTags::Scalar>>;
template<typename TElem>
struct LowerAccessImpl<Batch<TElem, DeviceTags::CPU, CategoryTags::Matrix>>;

// 标量列表
template<typename TElem>
class Batch<TElem, DeviceTags::CPU, CategoryTags::Scalar>
{
    static_assert(std::is_same_v<std::remove_cvref_t<TElem>, TElem>, "TElem is not an available type");
    friend struct LowerAccessImpl<Batch<TElem, DeviceTags::CPU, CategoryTags::Scalar>>;
public:
    using Category = CategoryTags::BatchScalar;
    using ElementType = TElem;
    using DeviceType = DeviceTags::CPU;
public:
    Batch(std::size_t length = 0)
        : m_mem(length)
        , m_len(length)
    {
    }

    // 查询接口
    size_t batchNum() const
    {
        return m_len;
    }
    bool availableForWrite() const
    {
        return m_mem.useCount() == 1;
    }
    // 写入接口
    void setValue(std::size_t index, ElementType val)
    {
        assert(availableForWrite());
        assert(index < m_len);
        m_mem.rawMemory()[index] = val;
    }
    // 读取接口
    const auto operator[](std::size_t index) const
    {
        assert(index < m_len);
        return m_mem.rawMemory()[index];
    }

    // 求值接口: todo
private:
    ContinuousMemory<ElementType, DeviceType> m_mem;
    std::size_t m_len;
};

// 标量列表底层访问
template<typename TElem>
struct LowerAccessImpl<Batch<TElem, DeviceTags::CPU, CategoryTags::Scalar>>
{
    LowerAccessImpl(Batch<TElem, DeviceTags::CPU, CategoryTags::Scalar> p)
        : m_data(std::move(p))
    {
    }
    auto mutableRawMemory()
    {
        return m_data.m_mem.rawMemory();
    }
    const auto rawMemory() const
    {
        return m_data.m_mem.rawMemory();
    }
private:
    Batch<TElem, DeviceTags::CPU, CategoryTags::Scalar> m_data;
};


// 矩阵列表
template<typename TElem>
class Batch<TElem, DeviceTags::CPU, CategoryTags::Matrix>
{
    static_assert(std::is_same_v<std::remove_cvref_t<TElem>, TElem>, "TElem is not an available type");
    friend struct LowerAccessImpl<Batch<TElem, DeviceTags::CPU, CategoryTags::Matrix>>;
public:
    using Category = CategoryTags::BatchMatrix;
    using ElementType = TElem;
    using DeviceType = DeviceTags::CPU;
public:
    Batch(std::size_t batchNum = 0, std::size_t row = 0, std::size_t col = 0)
        : m_mem(row * col * batchNum)
        , m_rowNum(row)
        , m_colNum(col)
        , m_batchNum(batchNum)
        , m_rowLen(col)
        , m_rawMatrixSize(row * col)
    {
    }
    // 查询接口
    std::size_t rowNum() const
    {
        return m_rowNum;
    }
    std::size_t colNum() const
    {
        return m_colNum;
    }
    std::size_t batchNum() const
    {
        return m_batchNum;
    }
    bool availableForWrite() const
    {
        return m_mem.useCount() == 1;
    }
    // 写入接口：写入具体的某个矩阵的某个值
    void setValue(std::size_t batchId, std::size_t row, std::size_t col, ElementType val)
    {
        assert(availableForWrite());
        assert(row < m_rowNum && col < m_colNum && batchId < m_batchNum);
        m_mem.rawMemory()[batchId * m_rawMatrixSize + row * m_rowLen + col] = val;
    }
    // 读取接口：返回一个临时矩阵，共享存储，仅用于访问
    const auto operator[](std::size_t batchId) const
    {
        assert(batchId < m_batchNum);
        auto pos = m_mem.rawMemory() + batchId * m_rawMatrixSize;
        return Matrix<ElementType, DeviceType>(m_mem.sharedPtr(), pos, m_rowNum, m_colNum, m_rowLen);
    }

    // 子矩阵列表接口，浅拷贝，共享存储，区间前闭后开
    auto subBatchMatrix(std::size_t rowBegin, std::size_t rowEnd, std::size_t colBegin, std::size_t colEnd)
    {
        assert(rowBegin < m_rowNum && colBegin < m_colNum);
        assert(rowend <= m_rowNum && colEnd <= m_colNum);
        auto pos = m_mem.rawMemory() + rowBegin * m_rowLen + colBegin;
        return Matrix<ElementType, DeviceType>(m_mem.sharedPtr(), pos, rowEnd - rowBegin, colEnd - colBegin, m_rowLen);
    }

    // 求值接口: todo

private:
    ContinuousMemory<ElementType, DeviceType> m_mem; // 内部数据存储于一维数组，并使用ContinuousMemory维护
    std::size_t m_rowNum;
    std::size_t m_colNum;
    std::size_t m_batchNum;
    std::size_t m_rowLen;
    std::size_t m_rawMatrixSize; // 原始的矩阵大小，也就是原始的矩阵行列之积
};

// 矩阵列表底层访问
template<typename TElem>
struct LowerAccessImpl<Batch<TElem, DeviceTags::CPU, CategoryTags::Matrix>>
{
    LowerAccessImpl(Batch<TElem, DeviceTags::CPU, CategoryTags::Matrix> p)
        : m_data(std::move(p))
    {
    }
    auto mutableRawMemory()
    {
        return m_data.m_mem.rawMemory();
    }
    const auto rawMemory() const
    {
        return m_data.m_mem.rawMemory();
    }
    std::size_t rowLen() const
    {
        return m_data.m_mem.m_rowLen;
    }
    std::size_t rawMatrixSize() const
    {
        return m_data.m_mem.m_rawMatrixSize;
    }
private:
    Batch<TElem, DeviceTags::CPU, CategoryTags::Matrix> m_data;
};

} // namespace MetaNN