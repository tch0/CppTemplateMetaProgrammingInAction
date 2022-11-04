#pragma once

#include <policy/policy_container.hpp>
#include <policy/policy_selector.hpp>
#include <data/matrix/matrix.hpp>
#include <param_initializer/init_policy.hpp>
#include <param_initializer/fill_with_distribution.hpp>
#include <random>
#include <stdexcept>
#include <type_traits>

namespace MetaNN
{

// 使用均匀分布初始化参数矩阵：提供最大最小值和一个随机数种子
template<typename TPolicyContainer = PolicyContainer<>>
class UniformFiller
{
    using TRandomEngine = typename PolicySelect<InitPolicy, TPolicyContainer>::RandomEngine;
public:
    UniformFiller(double min, double max, unsigned seed = std::random_device{}())
        : m_engine(seed)
        , m_min(min)
        , m_max(max)
    {
        if (min >= max)
        {
            throw std::runtime_error("Min if larger or equal than max for uniform ditribution.");
        }
    }
    template<typename TData>
    void fill(TData& data, std::size_t /*fan-in*/, std::size_t/*fan-out*/)
    {
        using ElementType = typename TData::ElementType;
        using DistType = std::conditional_t<std::is_integral_v<ElementType>,
                                            std::uniform_int_distribution<ElementType>,
                                            std::uniform_real_distribution<ElementType>>;
        DistType dist(m_min, m_max);
        NsInitializer::fillWithDistribution(data, dist, m_engine);
    }
private:
    TRandomEngine m_engine;
    double m_min;
    double m_max;
};

} // namespace MetaNN