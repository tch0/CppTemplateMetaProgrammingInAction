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

// VarScaleFiller的策略
struct VarScaleFillerPolicy
{
    using MajorClass = VarScaleFillerPolicy;

    struct DistributionTypeCategory
    {
        struct Uniform;
        struct Normal;
    };
    using Distribution = DistributionTypeCategory::Uniform;

    struct ScaleModeTypeCategory
    {
        struct FanIn;
        struct FanOut;
        struct FanAvg;
    };
    using ScaleMode = ScaleModeTypeCategory::FanAvg;
};

TypePolicyObj(PNormalVarScale,      VarScaleFillerPolicy, Distribution, Normal);
TypePolicyObj(PUniformVarScale,     VarScaleFillerPolicy, Distribution, Uniform);
TypePolicyObj(PVarScaleFanIn,       VarScaleFillerPolicy, ScaleMode,    FanIn);
TypePolicyObj(PVarScaleFanOut,      VarScaleFillerPolicy, ScaleMode,    FanOut);
TypePolicyObj(PVarScaleFanAvg,      VarScaleFillerPolicy, ScaleMode,    FanAvg);

} // namespace MetaNN

#include <policy/policy_macro_end.hpp>