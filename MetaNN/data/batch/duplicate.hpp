#pragma once

#include <data/tags.hpp>
#include <data/traits.hpp>
#include <cassert>

namespace MetaNN
{

// 将矩阵或者标量转换为包含相同值的矩阵列表或者标量列表
// 比如用于矩阵与矩阵列表（或类似场景）的操作，首先将矩阵转换为矩阵的重复列表。
// 最终的操作形式都是列表与列表或者非列表与非列表，可大幅减小冗余代码
template<typename TData>
class Duplicate;

// 标量重复列表
template<ScalarC TData>
class Duplicate<TData>
{
    static_assert(std::is_same_v<std::remove_cvref_t<TData>, TData>, "TData is not an available type");
public:
    using Category = CategoryTags::BatchScalar;
    using ElementType = typename TData::ElementType;
    using DeviceType = typename TData::DeviceType;
public:
    Duplicate(TData data, std::size_t batchNum)
        : m_data(std::move(data))
        , m_batchNum(batchNum)
    {
        assert(m_batchNum != 0);
    }

    // 查询接口
    std::size_t batchNum() const
    {
        return m_batchNum;
    }
    const TData& element() const
    {
        return m_data;
    }

    // 求值接口: todo
private:
    TData m_data;
    std::size_t m_batchNum;
};

// 矩阵重复列表
template<MatrixC TData>
class Duplicate<TData>
{
    static_assert(std::is_same_v<std::remove_cvref_t<TData>, TData>, "TData is not an available type");
public:
    using Category = CategoryTags::BatchMatrix;
    using ElementType = typename TData::ElementType;
    using DeviceType = typename TData::DeviceType;
public:
    Duplicate(TData data, std::size_t batchNum)
        : m_data(std::move(data))
        , m_batchNum(batchNum)
    {
        assert(m_batchNum != 0);
    }
    
    // 查询接口
    std::size_t rowNum() const
    {
        return m_data.rowNum();
    }
    std::size_t colNum() const
    {
        return m_data.colNum();
    }
    std::size_t batchNum() const
    {
        return m_batchNum;
    }
    const TData& element() const
    {
        return m_data;
    }

    // 求值接口: todo

private:
    TData m_data;
    std::size_t m_batchNum;
    // 求值缓存: todo
};

// 快捷构造Duplicate
template<typename TData> requires ScalarC<TData> || MatrixC<TData>
auto makeDuplicate(std::size_t batchNum, TData&& data) 
{
    using RawDataType = std::remove_cvref_t<TData>;
    return Duplicate<RawDataType>(std::forward<TData>(data), batchNum);
}

template<typename TData, typename... Args> requires ScalarC<TData> || MatrixC<TData>
auto makeDupliate(std::size_t batchNum, Args&&... args) 
{
    using RawDataType = std::remove_cvref_t<TData>;
    RawDataType tmp(std::forward<Args>(args)...);
    return Dupliate<RawDataType>(std::move(tmp), batchNum);
}

} // namespace MetaNN