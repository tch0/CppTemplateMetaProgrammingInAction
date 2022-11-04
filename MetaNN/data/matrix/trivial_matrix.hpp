#pragma once

#include <data/tags.hpp>
#include <data/traits.hpp>
#include <data/scalar.hpp>
#include <type_traits>

namespace MetaNN
{

// 平凡矩阵：所有元素值都一样的矩阵
template<typename TElem, typename TDevice = DeviceTags::CPU, typename TScalar = Scalar<TElem, TDevice>>
class TrivialMatrix;

template<typename TElem, typename TScalar>
class TrivialMatrix<TElem, DeviceTags::CPU, TScalar>
{
    static_assert(std::is_same_v<std::remove_cvref_t<TElem>, TElem>, "TElem is not an available type");
public:
    using Category = CategoryTags::Matrix;
    using ElementType = TElem;
    using DeviceType = DeviceTags::CPU;
public:
    TrivialMatrix(std::size_t row, std::size_t col, TScalar val)
        : m_rowNum(row)
        , m_colNum(col)
        , m_val(val)
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
    // 读访问接口
    auto elementValue() const
    {
        return m_val;
    }
    
    // 求值接口: todo

private:
    std::size_t m_rowNum;
    std::size_t m_colNum;
    TScalar m_val;
    // 求值结果缓存: todo
};

// 创建平凡矩阵，简化构造过程
template<typename TElem, typename TDevice, typename TVal>
auto makeTrivialMatrix(std::size_t row, std::size_t col, TVal&& val)
{
    using RawVal = std::remove_cvref_t<TVal>;
    if constexpr (IsScalarC<RawVal>)
    {
        static_assert(std::is_same_v<typename RawVal::DeviceType, TDevice> ||
                      std::is_same_v<typename RawVal::DeviceType, DeviceTags::CPU>);
        return TrivialMatrix<TElem, TDevice, RawVal>(row, col, val);
    }
    else
    {
        TElem tmpElem = static_cast<TElem>(val);
        Scalar<TElem, DeviceTags::CPU> scalar(std::move(tmpElem));
        return TrivialMatrix<TElem, TDevice, Scalar<TElem, DeviceTags::CPU>>(row, col, std::move(scalar));
    }
}

} // namespace MetaNN