#pragma once

#include <policy/policy_container.hpp>
#include <type_traits>

namespace MetaNN
{

namespace NsPolicySelect
{

// 组合多个策略的结果
template<typename TPolicyContainer>
struct PolicySelectResult;

// 直接多继承即可组合，按道理来说经过了检查之后不会存在成员有冲突
template<typename... TPolicies>
struct PolicySelectResult<PolicyContainer<TPolicies...>> : public TPolicies... {};

// 过滤出所有特定MajorClass的策略对象
template<typename TResult, typename TMajorClass, typename... TRestPolicies>
struct MajorFilter_
{
    using type = TResult;
};

template<typename... TFilteredPolicies, typename TMajorClass, typename TCurPolicy, typename... TRestPolicies>
struct MajorFilter_<PolicyContainer<TFilteredPolicies...>, TMajorClass, TCurPolicy, TRestPolicies...>
{
    using type = typename MajorFilter_<std::conditional_t<std::is_same_v<TMajorClass, typename TCurPolicy::MajorClass>,
                                                          PolicyContainer<TFilteredPolicies..., TCurPolicy>,
                                                          PolicyContainer<TFilteredPolicies...>>,
                                       TMajorClass,
                                       TRestPolicies...>::type;
};

// 检查策略容器中是否有互斥（相同的MinorClass）的策略对象，有的话返回false
template<typename TPolicyContainer>
struct MinorCheck_
{
    static constexpr bool value = true;
};

template<typename TCurPolicy, typename... TRestPolicies>
struct MinorCheck_<PolicyContainer<TCurPolicy, TRestPolicies...>>
{
    static constexpr bool current = (true && ... && (!std::is_same_v<typename TCurPolicy::MinorClass, typename TRestPolicies::MinorClass>));
    static constexpr bool value = current && MinorCheck_<TRestPolicies...>::value;
};

// 从策略容器中选择出所有相同MajorClass的策略对象
template<typename TMajorClass, typename TPolicyContainer>
struct Selector_;

template<typename TMajorClass, typename... TPolicies>
struct Selector_<TMajorClass, PolicyContainer<TPolicies...>>
{
    using TMF = typename MajorFilter_<PolicyContainer<>, TMajorClass, TPolicies...>::type;
    static_assert(MinorCheck_<TMF>::value, "Minor class set conflict!");

    using type = std::conditional_t<std::is_same_v<TMF, PolicyContainer<>>, // 筛选结果是否为空
                                    TMajorClass, // 为空则使用默认策略，也就是TMajorClass
                                    PolicySelectResult<TMF>>; // 不为空则将这些策略与默认策略组合起来得到结果
};

} // namespace NsPolicySelect

template<typename TMajorClass, typename TPolicyContainer>
using PolicySelect = typename NsPolicySelect::Selector_<TMajorClass, TPolicyContainer>::type;

} // namespace MetaNN