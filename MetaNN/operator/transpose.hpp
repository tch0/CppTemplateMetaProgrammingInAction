#pragma once

#include <operator/operators.hpp>

namespace MetaNN
{

// 转置操作修改了OpOrganizer的默认行为，需要对OpOrganizer进行特化

// 矩阵转置
template<>
class OpOrganizer<UnaryOpTags::Transpose, CategoryTags::Matrix>
{
public:
    template<MatrixC TData>
    OpOrganizer(const TData& data)
        : m_rowNum(data.colNum())
        , m_colNum(data.rowNum())
    {
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

// 矩阵列表转置：转置其中每一个矩阵
template<>
class OpOrganizer<UnaryOpTags::Transpose, CategoryTags::BatchMatrix>
    : public OpOrganizer<UnaryOpTags::Transpose, CategoryTags::Matrix>
{
    using BaseType = OpOrganizer<UnaryOpTags::Transpose, CategoryTags::Matrix>;
public:
    template<BatchMatrixC TData>
    OpOrganizer(const TData& data)
        : BaseType(data)
        , m_batchNum(data.batchNum())
    {
    }

    std::size_t batchNum() const
    {
        return m_batchNum;
    }

private:
    std::size_t m_batchNum;
};

// 转置运算
template<typename T>
class OpTranspose
{
    using RawT = std::remove_cvref_t<T>;
public:
    static auto eval(T&& data)
    {
        using ResType = UnaryOp<UnaryOpTags::Transpose, RawT>;
        return ResType(std::forward<T>(data));
    }
};

template<typename T> requires MatrixC<T> || BatchMatrixC<T>
auto transpose(T&& data)
{
    return OpTranspose<T>::eval(std::forward<T>(data));
}

} // namespace MetaNN
