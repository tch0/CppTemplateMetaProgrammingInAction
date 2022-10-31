#pragma once
#ifndef METANN_ONE_HOT_VECTOR_HPP_
#define METANN_ONE_HOT_VECTOR_HPP_

#include <data/tags.hpp>

namespace MetaNN
{

template<typename TElem, typename TDevice = DeviceTags::CPU>
class OneHotVector;

template<typename TElem>
class OneHotVector<TElem, DeviceTags::CPU>
{
    static_assert(std::is_same_v<std::remove_cvref_t<TElem>, TElem>, "TElem is not an available type");
public:
    using Category = CategoryTags::Matrix;
    using ElementType = TElem;
    using DeviceType = DeviceTags::CPU;
public:
    // 行向量
    OneHotVector(std::size_t col, std::size_t hotPos)
        : m_colNum(col)
        , m_hotPos(hotPos)
    {
    }
    // 访问接口
    std::size_t rowNum() const
    {
        return 1;
    }
    std::size_t colNum() const
    {
        return m_colNum;
    }
    auto hotPos() const
    {
        return m_hotPos;
    }

    // 求值接口: todo
private:
    std::size_t m_colNum;
    std::size_t m_hotPos;
    // 求值结果缓存: todo
};

} // namespace MetaNN

#endif // METANN_ONE_HOT_VECTOR_HPP_