#pragma once

#include <operator/operators.hpp>

namespace MetaNN
{

// 减法操作：类似于加法
// 支持类型：
//      标量与矩阵
//      标量与矩阵列表
//      矩阵与矩阵
//      矩阵与矩阵列表
//      矩阵列表与矩阵列表

template<typename T1, typename T2>
class OpSubtract
{
    using RawT1 = std::remove_cvref_t<T1>;
    using RawT2 = std::remove_cvref_t<T2>;
public:
    // 类别相同：矩阵与矩阵、矩阵列表与矩阵列表，平凡实现
    static auto eval(T1&& data1, T2&& data2) requires (MatrixC<T1> && MatrixC<T2>) || (BatchMatrixC<T1> && BatchMatrixC<T2>)
    {
        static_assert(std::is_same_v<typename RawT1::ElementType, typename RawT2::ElementType>, "Matrices with different element types can not subtract directly");
        static_assert(std::is_same_v<typename RawT1::DeviceType, typename RawT2::DeviceType>, "Matrices with different device types can not subtract directly");
        using ResType = BinaryOp<BinaryOpTags::Subtract, RawT1, RawT2>;
        return ResType(std::forward<T1>(data1), std::forward<T2>(data2));
    }
    // 标量与矩阵：将标量构造为平凡矩阵，转换为矩阵与矩阵操作
    static auto eval(T1&& data1, T2&& data2) requires (ScalarC<T1> && MatrixC<T2>) || (MatrixC<T1> && ScalarC<T2>)
    {
        if constexpr (ScalarC<T1> && MatrixC<T2>)
        {
            using ElementType = typename T2::ElementType;
            using DeviceType = typename T2::DeviceType;
            auto tmpTrivialMatix = makeTrivialMatrix<ElementType, DeviceType>(data2.rowNum(), data2.colNum(), data1);
            using ResType = BinaryOp<BinaryOpTags::Subtract, std::remove_cvref_t<decltype(tmpTrivialMatix)>, RawT2>;
            return ResType(std::move(tmpTrivialMatix), std::forward<T2>(data2));
        }
        else // MatrixC<T1> && ScalarC<T2>
        {
            return eval(std::forward<T2>(data2), std::forward<T1>(data1));
        }
    }
    // 标量与矩阵列表：将标量构造为平凡矩阵的重复列表，转换为矩阵列表与矩阵列表操作
    static auto eval(T1&& data1, T2&& data2) requires (ScalarC<T1> && BatchMatrixC<T2>) || (BatchMatrixC<T1> && ScalarC<T2>)
    {
        if constexpr (ScalarC<T1> && BatchMatrixC<T2>)
        {
            using ElementType = typename T2::ElementType;
            using DeviceType = typename T2::DeviceType;
            auto tmpTrivialMatrix = makeTrivialMatrix<ElementType, DeviceType>(data2.rowNum(), data2.colNum(), data1);
            auto tmpDuplicateTrivialMatrix = makeDuplicate(data2.batchNum(), std::move(tmpTrivialMatrix));
            using ResType = BinaryOp<BinaryOpTags::Subtract, std::remove_cvref_t<decltype(tmpDuplicateTrivialMatrix)>, RawT2>;
            return ResType(std::move(tmpDuplicateTrivialMatrix), std::forward<T2>(data2));
        }
        else // BatchMatrixC<T1> && ScalarC<T2>
        {
            return eval(std::forward<T2>(data2), std::forward<T1>(data1));
        }
    }
    // 矩阵与矩阵列表：将矩阵构造为重复矩阵列表，转换为矩阵列表与矩阵列表操作
    static auto eval(T1&& data1, T2&& data2) requires (MatrixC<T1> && BatchMatrixC<T2>) || (BatchMatrixC<T1> && MatrixC<T2>)
    {
        static_assert(std::is_same_v<typename RawT1::ElementType, typename RawT2::ElementType>, "Matrices with different element types can not subtract directly");
        static_assert(std::is_same_v<typename RawT1::DeviceType, typename RawT2::DeviceType>, "Matrices with different device types can not subtract directly");
        if constexpr (MatrixC<T1> && BatchMatrixC<T2>)
        {
            auto tmpDuplicateMatrix = makeDuplicate(data2.batchNum(), std::move(data1));
            using ResType = BinaryOp<BinaryOpTags::Subtract, std::remove_cvref_t<decltype(tmpDuplicateMatrix)>, RawT2>;
            return ResType(std::move(tmpDuplicateMatrix), std::forward<T2>(data2));
        }
        else // BatchMatrixC<T1> && MatrixC<T2>
        {
            return eval(std::forward<T2>(data2), std::forward<T1>(data1));
        }
    }
};

template<typename T1, typename T2>
    requires (ScalarC<T1> && MatrixC<T2>) ||
             (MatrixC<T1> && ScalarC<T2>) ||
             (ScalarC<T1> && BatchMatrixC<T2>) ||
             (BatchMatrixC<T1> && ScalarC<T2>) ||
             (MatrixC<T1> && MatrixC<T2>) ||
             (MatrixC<T1> && BatchMatrixC<T2>) ||
             (BatchMatrixC<T1> && MatrixC<T2>) ||
             (BatchMatrixC<T1> && BatchMatrixC<T2>)
auto operator-(T1&& data1, T2&& data2)
{
    return OpSubtract<T1, T2>::eval(std::forward<T1>(data1), std::forward<T2>(data2));
}

} // namespace MetaNN