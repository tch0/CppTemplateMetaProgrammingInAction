#pragma once

#include <data/tags.hpp>
#include <data/traits.hpp>
#include <cassert>

namespace MetaNN
{

// TOpTag是操作标签，TCategory是结果类别，针对不同类别进行偏特化
// 如果有运算不满足这里的默认行为，需要针对特定运算进行偏特化或者全特化
template<typename TOpTag, typename TCategory>
class OpOrganizer;

template<typename TOpTag>
class OpOrganizer<TOpTag, CategoryTags::Scalar>
{
public:
    template<ScalarC THead, ScalarC... TRemain>
    OpOrganizer(const THead& head, const TRemain&... remain)
    {
    }
};

template<typename TOpTag>
class OpOrganizer<TOpTag, CategoryTags::Matrix>
{
public:
    template<MatrixC THead, MatrixC... TRemain>
    OpOrganizer(const THead& head, const TRemain&... remain)
        : m_rowNum(head.rowNum())
        , m_colNum(head.colNum())
    {
        assert((true && ... && (head.rowNum() == remain.rowNum())));
        assert((true && ... && (head.colNum() == remain.colNum())));
    }
    std::size_t rowNum() const
    {
        return m_rowNum;
    }
    std::size_t colNum() const
    {
        return m_colNum;
    }
private:
    std::size_t m_rowNum;
    std::size_t m_colNum;
};

template<typename TOpTag>
class OpOrganizer<TOpTag, CategoryTags::BatchScalar>
{
public:
    template<BatchScalarC THead, BatchScalarC... TRemain>
    OpOrganizer(const THead& head, const TRemain&... remain)
        : m_batchNum(head.batchNum)
    {
        assert((true && ... && (head.batchNum() == remain.batchNum())));
    }
    std::size_t batchNum() const
    {
        return m_batchNum;
    }
private:
    std::size_t m_batchNum;
};

template<typename TOpTag>
class OpOrganizer<TOpTag, CategoryTags::BatchMatrix>
{
public:
    template<BatchMatrixC THead, BatchMatrixC... TRemain>
    OpOrganizer(const THead& head, const TRemain&... remain)
        : m_rowNum(head.rowNum())
        , m_colNum(head.colNum())
        , m_batchNum(head.batchNum)
    {
        assert((true && ... && (head.rowNum() == remain.rowNum())));
        assert((true && ... && (head.colNum() == remain.colNum())));
        assert((true && ... && (head.batchNum() == remain.batchNum())));
    }
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
private:
    std::size_t m_rowNum;
    std::size_t m_colNum;
    std::size_t m_batchNum;
};

} // namespace MetaNN
