#pragma once

#include <data/tags.hpp>
#include <data/traits.hpp>
#include <type_traits>

namespace MetaNN
{

// 获取运算结果元素类型，如果一个新运算具有不同行为，那么需要对其进行特化
template<typename TOpTag, typename TOp1, typename... TOperands>
struct OpElementType_
{
    using type = typename TOp1::ElementType;
};

template<typename TOpTag, typename TOp1, typename... TOperands>
using OpElementType = typename OpElementType_<TOpTag, TOp1, TOperands...>::type;

// 获取运算结果设备类型
template<typename TOpTag, typename TOp1, typename... TOperands>
struct OpDeviceType_
{
    using type = typename TOp1::DeviceType;
};

template<typename TOpTag, typename TOp1, typename... TOperands>
using OpDeviceType = typename OpDeviceType_<TOpTag, TOp1, TOperands...>::type;


// 当参与运算的所有类型都是一个类别时，直接定义结果类别为其共同类别（比如矩阵和矩阵那么结果就是矩阵）
// 对不同类别的运算则需要特化（比如标量和矩阵，那么结果就由具体运算的特化决定）
template<typename TOpTag, typename THeadCategory, typename... TRemainCategory>
struct OpCategory_
{
    static_assert((true && ... && std::is_same_v<THeadCategory, TRemainCategory>), "Data category mismatch.");
    using type = THeadCategory;
};

template<typename TOpTag, typename THead, typename... TRemain>
using OpCateCal = typename OpCategory_<TOpTag, DataCategory<THead>, DataCategory<TRemain>...>::type;

// 求值逻辑：需要对具体类型特化
template<typename... TCases>
struct OpSeqContainer;

template<typename TOpTag>
struct OpSeq_;

} // namespace MetaNN