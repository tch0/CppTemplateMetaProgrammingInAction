#pragma once

#include <data/traits.hpp>
#include <policy/policy_container.hpp>
#include <policy/policy_selector.hpp>
#include <data/matrix/matrix.hpp>
#include <param_initializer/init_policy.hpp>
#include <param_initializer/fill_with_distribution.hpp>
#include <random>
#include <stdexcept>
#include <type_traits>
#include <cmath>

namespace MetaNN
{

// TensorFlow中的variance_scaling_filler，并由此构造出XavierFiller和MSRAFiller(todo yet!)
template<typename TPolicyContainer = PolicyContainer<>>
class VarScaleFiller
{
    using TRandomEngine = typename PolicySelect<InitPolicy, TPolicyContainer>::RandomEngine;
public:
    VarScaleFiller(double factor, unsigned seed = std::random_device{}())
        : m_engine(seed)
        , m_factor(factor)
    {
    }
    template<typename TData>
    void fill(TData& data, std::size_t fanIn, std::size_t fanOut)
    {
        using ScaleMode = typename PolicySelect<VarScaleFillerPolicy, TPolicyContainer>::ScaleMode;
        double fan_factor = 0;
        if constexpr (std::is_same_v<ScaleMode, VarScaleFillerPolicy::ScaleModeTypeCategory::FanIn>)
        {
            fan_factor = fanIn;
        }
        else if constexpr (std::is_same_v<ScaleMode, VarScaleFillerPolicy::ScaleModeTypeCategory::FanOut>)
        {
            fan_factor = fanOut;
        }
        else if constexpr (std::is_same_v<ScaleMode, VarScaleFillerPolicy::ScaleModeTypeCategory::FanAvg>)
        {
            fan_factor = (fanIn + fanOut) / 2;
        }
        else
        {
            static_assert(DependencyFalse<ScaleMode>);
        }

        using DistType = typename PolicySelect<VarScaleFillerPolicy, TPolicyContainer>::Distribution;
        using ElementType = typename TData::ElementType;
        if constexpr (std::is_same_v<DistType, VarScaleFillerPolicy::DistributionTypeCategory::Uniform>)
        {
            double limit = std::sqrt(3.0 * m_factor / fan_factor);
            std::uniform_int_distribution<ElementType> dist(-limit, limit);
            NsInitializer::fillWithDistribution(data, dist, m_engine);
        }
        else if constexpr (std::is_same_v<DistType, VarScaleFillerPolicy::DistributionTypeCategory::Normal>)
        {
            double stdDeviation = std::sqrt(m_factor / fan_factor);
            std::normal_distribution<ElementType> dist(0, stdDeviation);
            NsInitializer::fillWithDistribution(data, dist, m_engine);
        }
        else
        {
            static_assert(DependencyFalse<DistType>);
        }
    }
private:
    TRandomEngine m_engine;
    double m_factor;
};

} // namespace MetaNN