#pragma once

#include <type_traits>

namespace MetaNN
{

// 通过一个中间层，提供更底层的访问，比如矩阵级的访问
// 不直接在矩阵的数据访问接口中提供，接口中不提供以保证用户普通使用场景的安全性，这个中间层仅暴露给库实现者

// 为需要暴露的任何类提供LowerAccessImpl特化
template<typename TData>
struct LowerAccessImpl;

template<typename TData>
auto lowerAccess(TData&& p)
{
    using RawType = std::remove_cvref_t<TData>;
    return LowerAccessImpl<RawType>(std::forward<TData>(p));
}

// 是否具有底层访问：满足能够通过该对象构造LowerAccessImpl的要求
template<typename TData>
concept LowerAccessC = requires
{
    LowerAccessImpl<TData>(std::declval<TData>());
};

} // namespace MetaNN