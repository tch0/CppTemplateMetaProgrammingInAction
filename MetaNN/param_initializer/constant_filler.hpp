#pragma once

#include <data/tags.hpp>
#include <data/matrix/matrix.hpp>
#include <stdexcept>

namespace MetaNN
{

namespace NsConstantFiller
{

template<typename TElem>
void fill(Matrix<TElem, DeviceTags::CPU>& mat, const double& val)
{
    if (!mat.avilableForWrite())
    {
        throw std::runtime_error("Matrix is string weight, can not fill in.");
    }

    auto acc = lowerAccess(mat);
    std::size_t row = mat.rowNum();
    std::size_t col = mat.colNum();
    std::size_t rowLen = acc.rowLen();
    auto p = acc.mutableMemory();
    for (std::size_t i = 0; i < row; i++)
    {
        for (std::size_t j = 0; j < col; j ++)
        {
            p[j] = static_cast<TElem>(val);
        }
        p += rowLen;
    }
}

} // NsConstantFiller

class ConstantFiller
{
public:
    ConstantFiller(double val = 0) : m_val(val) {}
    template<typename TData>
    void fill(TData& data, std::size_t /*fan-in*/, std::size_t/*fan-out*/)
    {
        NsConstantFiller::fill(data, m_val);
    }
private:
    double m_val;
};


} // namespace MetaNN
