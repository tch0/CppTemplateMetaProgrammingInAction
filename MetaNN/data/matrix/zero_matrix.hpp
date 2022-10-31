#pragma once
#ifndef METANN_ZERO_MATRIX_HPP_
#define METANN_ZERO_MATRIX_HPP_

#include <data/tags.hpp>
#include <type_traits>

namespace MetaNN
{

// 全零矩阵：即矩阵中元素全为0的平凡矩阵
template<typename TElem, typename TDevice = DeviceTags::CPU>
class ZeroMatrix;

template<typename TElem>
class ZeroMatrix<TElem, DeviceTags::CPU>
{
    static_assert(std::is_same_v<std::remove_cvref_t<TElem>, TElem>, "TElem is not an available type");
public:
    using Category = CategoryTags::Matrix;
    using ElementType = TElem;
    using DeviceType = DeviceTags::CPU;
public:
    ZeroMatrix(std::size_t row, std::size_t col)
        : m_rowNum(row)
        , m_colNum(col)
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

    // 求值接口: todo
private:
    std::size_t m_rowNum;
    std::size_t m_colNum;
    // 求值结果缓存: todo
};

} // namespace MetaNN

#endif // METANN_ZERO_MATRIX_HPP_