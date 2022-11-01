#pragma once

#include <operator/operators.hpp>
#include <cassert>

namespace MetaNN
{

// 矩阵乘法
// 支持类型：
//      矩阵与矩阵
//      矩阵与矩阵列表
//      矩阵列表与矩阵
//      矩阵列表与矩阵列表

// 重载OpOrganizer定义结果矩阵的行数和列数
template<>
class OpOrganizer<BinaryOpTags::Dot, CategoryTags::Matrix>
{
public:
    template<MatrixC T1, MatrixC T2>
    OpOrganizer(const T1& data1, const T2& data2)
        : m_rowNum(data1.rowNum())
        , m_colNum(data2.colNum())
    {
        assert(data1.colNum() == data2.rowNum());
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

template<>
class OpOrganizer<BinaryOpTags::Dot, CategoryTags::BatchMatrix>
{
public:
    template<BatchMatrixC T1, BatchMatrixC T2>
    OpOrganizer(const T1& data1, const T2& data2)
        : m_rowNum(data1.rowNum())
        , m_colNum(data2.colNum())
        , m_batchNum(data1.batchNum())
    {
        assert(data1.colNum() == data2.rowNum());
        assert(data1.batchNum() == data2.batchNum());
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
    std::size_t m_batchNum;
};

// 矩阵乘法运算
template<typename T1, typename T2>
class OpDot
{
    using RawT1 = std::remove_cvref_t<T1>;
    using RawT2 = std::remove_cvref_t<T2>;
public:
    // 类别相同：矩阵与矩阵、矩阵列表与矩阵列表
    static auto eval(T1&& data1, T2&& data2) requires (MatrixC<T1> && MatrixC<T2>) || (BatchMatrixC<T1> && BatchMatrixC<T2>)
    {
        static_assert(std::is_same_v<typename RawT1::ElementType, typename RawT2::ElementType>, "Matrices with different element types can not dot directly");
        static_assert(std::is_same_v<typename RawT1::DeviceType, typename RawT2::DeviceType>, "Matrices with different device types can not dot directly");

        using ResType = BinaryOp<BinaryOpTags::Dot, RawT1, RawT2>;
        return ResType(std::forward<T1>(data1), std::forward<T2>(data2));
    }
    // 矩阵与矩阵列表
    static auto eval(T1&& data1, T2&& data2) requires MatrixC<T1> && BatchMatrixC<T2>
    {
        static_assert(std::is_same_v<typename RawT1::ElementType, typename RawT2::ElementType>, "Matrices with different element types can not dot directly");
        static_assert(std::is_same_v<typename RawT1::DeviceType, typename RawT2::DeviceType>, "Matrices with different device types can not dot directly");

        Duplicate<RawT1> tmpDuplicateMatrix(std::forward<T1>(data1), data2.batchNum());
        using ResType = BinaryOp<BinaryOpTags::Dot, Duplicate<RawT1>, RawT2>;
        return ResType(std::move(tmpDuplicateMatrix), std::forward<T2>(data2));
    }
    // 矩阵列表与矩阵
    static auto eval(T1&& data1, T2&& data2) requires BatchMatrixC<T1> && MatrixC<T2>
    {
        static_assert(std::is_same_v<typename RawT1::ElementType, typename RawT2::ElementType>, "Matrices with different element types can not dot directly");
        static_assert(std::is_same_v<typename RawT1::DeviceType, typename RawT2::DeviceType>, "Matrices with different device types can not dot directly");
        
        Duplicate<RawT2> tmpDuplicateMatrix(std::forward<T2>(data2), data1.batchNum());
        using ResType = BinaryOp<BinaryOpTags::Dot, RawT1, Duplicate<RawT2>>;
        return ResType(std::forward<T1>(data1), std::move(tmpDuplicateMatrix));

    }
};

template<typename T1, typename T2>
    requires (MatrixC<T1> && MatrixC<T2>) ||
             (MatrixC<T1> && BatchMatrixC<T2>) ||
             (BatchMatrixC<T1> && MatrixC<T2>) ||
             (BatchMatrixC<T1> && BatchMatrixC<T2>)
auto dot(T1&& data1, T2&& data2)
{
    return OpDot<T1, T2>::eval(std::forward<T1>(data1), std::forward<T2>(data2));
}

} // namespace MetaNN