#pragma once

#include <random>
#include <policy/policy_macro_begin.hpp>

namespace MetaNN
{

struct InitPolicy
{
    using MajorClass = InitPolicy;
    
    struct OverallTypeCategory;
    struct WeightTypeCategory;
    struct BiasTypeCategory;

    using Overall = void;
    using Weight = void;
    using Bias = void;

    struct RandomEngineTypeCategory;
    using RandomEngine = std::mt19937;
};

TypePolicyTemplate(PInitializerIs,          InitPolicy, Overall);       // 设置默认初始化器
TypePolicyTemplate(PWeightInitializerIs,    InitPolicy, Weight);        // 设置权重初始化器
TypePolicyTemplate(PBiasInitializerIs,      InitPolicy, Bias);          // 设置偏置初始化器
TypePolicyTemplate(PRandomGeneratorIs,      InitPolicy, RandomEngine);  // 设置随机数引擎


} // namespace MetaNN

#include <policy/policy_macro_end.hpp>