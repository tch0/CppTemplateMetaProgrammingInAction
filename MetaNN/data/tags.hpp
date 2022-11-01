#pragma once

namespace MetaNN
{

// 数据类型分类
// 注意：MetaNN并不区分向量和矩阵，向量被视为行或者列数为1的矩阵，涉及的运算也使用矩阵运算来表示，并且标签之间不存在层次包含关系，他们是互斥的
struct CategoryTags
{
    struct Scalar;      // 标量
    struct Matrix;      // 矩阵
    struct BatchScalar; // 标量列表
    struct BatchMatrix; // 矩阵列表
};

// 硬件设备标签：当前仅支持使用CPU计算，但支持自行扩展
struct DeviceTags
{
    struct CPU;
};

} // namespace MetaNN