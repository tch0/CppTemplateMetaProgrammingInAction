#pragma once
#ifndef METANN_SCALAR_HPP_
#define METANN_SCALAR_HPP_

#include <data/tags.hpp>
#include <type_traits>

namespace MetaNN
{

template<typename TElem, typename TDevice = DeviceTags::CPU>
class Scalar;

template<typename TElem>
class Scalar<TElem, DeviceTags::CPU>
{
    static_assert(std::is_same_v<std::remove_cvref_t<TElem>, TElem>, "TElem is not an available type");
public:
    using Category = CategoryTags::Scalar;
    using ElementType = TElem;
    using DeviceType = DeviceTags::CPU;
public:
    Scalar(ElementType elem = {})
        : m_elem(elem) {}
    // 拷贝构造、拷贝赋值、移动构造、移动赋值由编译器合成
    
    auto& value() { return m_elem; }

    auto value() const { return m_elem; }

    // 求值相关接口: todo
    bool operator==(const Scalar& rhs) const;
    
    template<typename TOtherType>
    bool operator==(const TOtherType& rhs) const;

    template<typename TData>
    bool operator!=(const TData& rhs) const;
private:
    ElementType m_elem;
};

} // namespace MetaNN

#endif // METANN_SCALAR_HPP_