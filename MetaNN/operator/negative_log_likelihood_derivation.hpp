#pragma once

#include <operator/operators.hpp>

namespace MetaNN
{

// NegativeLogLikelihoodDerivation operation：三元运算符
// 支持类型：
//      矩阵、矩阵、矩阵
//      矩阵列表、矩阵列表、矩阵列表

template<typename T1, typename T2, typename T3>
class OpNegativeLogLikelihoodDerivation
{
    using RawT1 = std::remove_cvref_t<T1>;
    using RawT2 = std::remove_cvref_t<T2>;
    using RawT3 = std::remove_cvref_t<T3>;
public:
    static auto eval(T1&& data1, T2&& data2, T3&& data3)
    {
        static_assert(std::is_same_v<typename RawT1::ElementType, typename RawT2::ElementType>, "Matrices with different element types can not do NegativeLogLikelihoodDerivation directly");
        static_assert(std::is_same_v<typename RawT1::ElementType, typename RawT3::ElementType>, "Matrices with different element types can not do NegativeLogLikelihoodDerivation directly");
        static_assert(std::is_same_v<typename RawT1::DeviceType, typename RawT2::DeviceType>, "Matrices with different device types can not do NegativeLogLikelihoodDerivation directly");
        static_assert(std::is_same_v<typename RawT1::DeviceType, typename RawT3::DeviceType>, "Matrices with different device types can not do NegativeLogLikelihoodDerivation directly");

        using ResType = TernaryOp<TernaryOpTags::NegativeLogLikelihoodDerivation, RawT1, RawT2, RawT3>;
        return ResType(std::forward<T1>(data1), std::forward<T2>(data2), std::forward<T3>(data3));
    }
};

template<typename T1, typename T2, typename T3>
    requires (MatrixC<T1> && MatrixC<T2> && MatrixC<T3>) ||
             (BatchMatrixC<T1> && BatchMatrixC<T2> && BatchMatrixC<T3>)
auto negativeLogLikelihoodDerivation(T1&& data1, T2&& data2, T3&& data3)
{
    return OpNegativeLogLikelihoodDerivation<T1, T2, T3>::eval(std::forward<T1>(data1), std::forward<T2>(data2), std::forward<T3>(data3));
}

} // namespace MetaNN