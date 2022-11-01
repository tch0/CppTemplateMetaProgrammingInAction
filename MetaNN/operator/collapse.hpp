#pragma once

#include <operator/operators.hpp>

namespace MetaNN
{

// 折叠运算：对一个矩阵列表求和，生成一个矩阵，输入输出类型不一样，需要特化OpCategory
template<>
struct OpCategory_<UnaryOpTags::Collapse, CategoryTags::BatchMatrix>
{
    using type = CategoryTags::Matrix;
};

template<typename T>
class OpCollapse
{
    using RawT = std::remove_cvref_t<T>;
public:
    static auto eval(T&& data)
    {
        using ResType = UnaryOp<UnaryOpTags::Collapse, RawT>;
        return ResType(std::forward<T>(data));
    }
};

template<typename T>
auto collapse(T&& data) requires BatchMatrixC<T>
{
    return OpCollapse<T>::eval(std::forward<T>(data));
}

} // namespace MetaNN