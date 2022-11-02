#pragma once

namespace MetaNN
{

template<typename... TPolicies>
struct PolicyContainer;

template<typename T>
constexpr bool IsPolicyContainer = false;

template<typename... Ts>
constexpr bool IsPolicyContainer<PolicyContainer<Ts...>> = true;

template<typename TLayerName, typename... TPolicies>
struct SubPolicyContainer;

template<typename T>
constexpr bool IsSubPolicyContainer = false;

template<typename TLayer, typename... Ts>
constexpr bool IsSubPolicyContainer<SubPolicyContainer<TLayer, Ts...>> = true;

} // namespace MetaNN