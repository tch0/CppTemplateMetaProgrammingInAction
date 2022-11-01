#pragma once

#include <operator/operators.hpp>

namespace MetaNN
{

// sign函数：用于矩阵和矩阵列表
template<typename T>
class OpSign
{
private:
    using RawT = std::remove_cvref_t<T>;
public:
    static auto eval(T&& data)
    {
        using ResType = UnaryOp<UnaryOpTags::Sign, RawT>;
        return ResType(std::forward<T>(data));
    }
};

template<typename T> requires MatrixC<T> || BatchMatrixC<T>
auto sign(T&& data)
{
    return OpSign<T>::eval(std::forward<T>(data));
}

} // namespace MetaNN