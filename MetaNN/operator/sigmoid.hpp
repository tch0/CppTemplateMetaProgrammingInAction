#pragma once

#include <operator/operators.hpp>

// sigmoid function : S(x) = 1/(1+e^(-x))
// map (-infinity, +inifinity) to (0, 1)

namespace MetaNN
{

// Sigmoid：仅支持矩阵或者矩阵列表
template<typename T>
class OpSigmoid
{
private:
    using RawT = std::remove_cvref_t<T>;
public:
    static auto eval(T&& data)
    {
        using ResType = UnaryOp<UnaryOpTags::Sigmoid, RawT>;
        return ResType(std::forward<T>(data));
    }
};

template<typename T> requires MatrixC<T> || BatchMatrixC<T>
auto sigmoid(T&& data)
{
    return OpSigmoid<T>::eval(std::forward<T>(data));
}

} // namespace MetaNN