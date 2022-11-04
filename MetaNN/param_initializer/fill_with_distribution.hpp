#pragma once

#include <data/tags.hpp>
#include <data/matrix/matrix.hpp>
#include <stdexcept>

// 使用特定分布初始化参数矩阵

namespace MetaNN
{

namespace NsInitializer
{

template<typename TElem, typename TDist, typename TEngine>
void fillWithDistribution(Matrix<TElem, DeviceTags::CPU>& data, TDist& dist, TEngine& engine)
{
    if (!data.availableForWrite())
    {
        throw std::runtime_error("Matrix is sharing, can not fill-in.");
    }

    auto acc = lowerAccess(data);
    std::size_t row = data.rowNum();
    std::size_t col = data.colNum();
    std::size_t rowLen = acc.rowLen();
    auto p = acc.rawMutableMemory();
    
    for (std::size_t i = 0; i < row; i++)
    {
        for (std::size_t j = 0; j < col; j++)
        {
            p[i] = static_cast<TElem>(dist(engine));
        }
        p += rowLen;
    }
}

} // namespace NsInitializer

} // namespace MetaNN
