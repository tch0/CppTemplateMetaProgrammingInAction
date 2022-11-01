#pragma once

#include <operator/operators.hpp>

namespace MetaNN
{

// 绝对值：仅针对矩阵和矩阵列表
template<typename T>
class OpAbs
{
    using RawT = std::remove_cvref_t<T>;
public:
    static auto eval(T&& data)
    {
        using ResType = UnaryOp<UnaryOpTags::Abs, RawT>;
        return ResType(std::forward<T>(data));
    }
};

template<typename T> requires MatrixC<T> || BatchMatrixC<T>
auto abs(T&& data)
{
    return OpAbs<T>::eval(std::forward<T>(data));
}

} // namespace MetaNN