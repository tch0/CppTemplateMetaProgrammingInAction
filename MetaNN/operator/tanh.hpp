#pragma once

#include <operator/operators.hpp>

namespace MetaNN
{

// tanh：双曲正切，仅支持矩阵或者矩阵列表
template<typename T>
class OpTanh
{
private:
    using RawT = std::remove_cvref_t<T>;
public:
    static auto eval(T&& data)
    {
        using ResType = UnaryOp<UnaryOpTags::Tanh, RawT>;
        return ResType(std::forward<T>(data));
    }
};

template<typename T> requires MatrixC<T> || BatchMatrixC<T>
auto tanh(T&& data)
{
    return OpTanh<T>::eval(std::forward<T>(data));
}

} // namespace MetaNN