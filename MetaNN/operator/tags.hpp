#pragma once

namespace MetaNN
{

// 一元运算
struct UnaryOpTags
{
    struct Abs;
    struct Sigmoid;
    struct Sign;
    struct Tanh;
    struct Transpose;
    struct Collapse;
    struct VecSoftmax;
};

// 二元运算
struct BinaryOpTags
{
    struct Add;
    struct Subtract;
    struct ElementMul;
    struct Divide;
    struct Dot;
    struct NegativeLogLikelihood;
    struct SigmoidDerivation;
    struct TanhDerivation;
    struct VecSoftmaxDerivation;
};

// 三元运算
struct TernaryOpTags
{
    struct Interpolation;
    struct NegativeLogLikelihoodDerivation;
};

} // namespace MetaNN