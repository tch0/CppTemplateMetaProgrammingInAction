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

// 使用正态分布初始化参数矩阵：提供平均值与标准差与一个随机数种子
template<typename TPolicyContainer = PolicyContainer<>>
class GaussianFiller
{
    using TRandomEngine = typename PolicySelect<InitPolicy, TPolicyContainer>::RandomEngine;
public:
    GaussianFiller(double meanVal, double standardDeviation, unsigned seed = std::random_device{}())
        : m_engine(seed)
        , m_meanVal(meanVal)
        , m_stdDeviation(standardDeviation)
    {
        if (standardDeviation <= 0)
        {
            throw std::runtime_error("Invalid standard derivation for gaussian ditribution.");
        }
    }
    template<typename TData>
    void fill(TData& data, std::size_t /*fan-in*/, std::size_t/*fan-out*/)
    {
        using ElementType = typename TData::ElementType;
        std::normal_distribution<ElementType> dist(m_meanVal, m_stdDeviation);
        NsInitializer::fillWithDistribution(data, dist, m_engine);
    }
private:
    TRandomEngine m_engine;
    double m_meanVal;
    double m_stdDeviation;
};

} // namespace MetaNN