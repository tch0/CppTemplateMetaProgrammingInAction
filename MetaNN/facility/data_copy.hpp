#pragma once

#include <data/tags.hpp>
#include <data/matrix/matrix.hpp>
#include <stdexcept>
#include <algorithm>

namespace MetaNN
{

template<typename TElem>
void dataCopy(const Matrix<TElem, DeviceTags::CPU>& src, Matrix<TElem, DeviceTags::CPU>& dest)
{
    std::size_t rowNum = src.rowNum();
    std::size_t colNum = src.colNum();
    if (rowNum != dest.rowNum() || colNum != dest.colNum())
    {
        throw std::runtime_error("Error in dataCopy: matrix dimension mismatch!");
    }
    const auto memSrc = lowerAccess(src);
    auto memDest = lowerAccess(dest);

    std::size_t srcRowLen = memSrc.rowLen();
    std::size_t destRowLen = memDest.rowLen();

    const TElem* pSrc = memSrc.rawMemory();
    TElem* pDest = memDest.mutableRawMemory();

    if (srcRowLen == colNum && destRowLen == colNum)
    {
        std::copy(pSrc, pSrc + rowNum * colNum, pDest);
    }
    else
    {
        for (std::size_t i = 0; i < rowNum; ++i)
        {
            std::copy(pSrc, pSrc + colNum, pDest);
            pDest += destRowLen;
            pSrc += srcRowLen;
        }
    }
}

} // namespace MetaNN