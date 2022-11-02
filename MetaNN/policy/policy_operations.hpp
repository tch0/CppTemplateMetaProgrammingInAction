#pragma once

#include <policy/policy_container.hpp>
#include <type_traits>

namespace MetaNN
{

//  =============================== poliy existence check ===============================
template<typename TContainer, typename TPolicy>
struct PolicyExist_;

template<typename T1, typename... Ts, typename TPolicy>
struct PolicyExist_<PolicyContainer<T1, Ts...>, TPolicy>
{
    static constexpr bool value = (std::is_same_v<typename T1::MajorClass, typename TPolicy::MajorClass> &&
                                   std::is_same_v<typename T1::MinorClass, typename TPolicy::MinorClass>) ||
                                  PolicyExist_<PolicyContainer<Ts...>, TPolicy>::value;
};

// 跳过SubPolicyContainer
template<typename TLayerName, typename... Ts1, typename... Ts2, typename TPolicy>
struct PolicyExist_<PolicyContainer<SubPolicyContainer<TLayerName, Ts1...>, Ts2...>, TPolicy>
{
    static constexpr bool value = PolicyExist_<PolicyContainer<Ts2...>, TPolicy>::value;
};

template<typename TPolicy>
struct PolicyExist_<PolicyContainer<>, TPolicy>
{
    static constexpr bool value = false;
};

template<typename TContainer, typename TPolicy>
constexpr bool PolicyExist = PolicyExist_<TContainer, TPolicy>::value;


//  =============================== poliy derivation ===============================
namespace NsPolicyDerive
{



} // namespace NsPolicyDerive


} // namespace MetaNN