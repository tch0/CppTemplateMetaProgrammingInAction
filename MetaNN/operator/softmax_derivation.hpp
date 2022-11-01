#pragma once

#include <operator/operators.hpp>

namespace MetaNN
{

// VecSoftmaxDerivation operation
// 支持类型：
//      矩阵与矩阵
//      矩阵列表与矩阵列表

template<typename T1, typename T2>
class OpVecSoftmaxDerivation
{
    using RawT1 = std::remove_cvref_t<T1>;
    using RawT2 = std::remove_cvref_t<T2>;
public:
    static auto eval(T1&& data1, T2&& data2)
    {
        static_assert(std::is_same_v<typename RawT1::ElementType, typename RawT2::ElementType>, "Matrices with different element types can not do VecSoftmaxDerivation directly");
        static_assert(std::is_same_v<typename RawT1::DeviceType, typename RawT2::DeviceType>, "Matrices with different device types can not do VecSoftmaxDerivation directly");

        using ResType = BinaryOp<BinaryOpTags::VecSoftmaxDerivation, RawT1, RawT2>;
        return ResType(std::forward<T1>(data1), std::forward<T2>(data2));
    }
};

template<typename T1, typename T2>
    requires (MatrixC<T1> && MatrixC<T2>) || (BatchMatrixC<T1> && BatchMatrixC<T2>)
auto vecSoftmaxDerivation(T1&& data1, T2&& data2)
{
    return OpVecSoftmaxDerivation<T1, T2>::eval(std::forward<T1>(data1), std::forward<T2>(data2));
}

} // namespace MetaNN