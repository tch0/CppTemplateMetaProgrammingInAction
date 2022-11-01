#pragma once

#include <operator/operators.hpp>

namespace MetaNN
{

// VecSoftMax：将输入矩阵归一化，用于矩阵和矩阵列表
template<typename T>
class OpVecSoftmax
{
private:
    using RawT = std::remove_cvref_t<T>;
public:
    static auto eval(T&& data)
    {
        using ResType = UnaryOp<UnaryOpTags::VecSoftmax, RawT>;
        return ResType(std::forward<T>(data));
    }
};

template<typename T> requires MatrixC<T> || BatchMatrixC<T>
auto vecSoftmax(T&& data)
{
    return OpVecSoftmax<T>::eval(std::forward<T>(data));
}


} // namespace MetaNN