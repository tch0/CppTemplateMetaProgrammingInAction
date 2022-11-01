#pragma once

#include <data/tags.hpp>
#include <memory>
#include <type_traits>

namespace MetaNN
{

template<typename TDevice>
struct Allocator;

template<>
struct Allocator<DeviceTags::CPU>
{
    template<typename TElem>
    static std::shared_ptr<TElem> allocate(size_t elementSize)
    {
        return std::shared_ptr<TElem>(new TElem[elementSize], [](TElem* ptr) { delete [] ptr; });
    }
};

// 维护Allocator分配的内存
// 传递内存时同时传递智能指针，确保引用计数的正确性
// 使用时只使用底层的内存，通常指向智能指针维护内存的开始，但也可能指向中间（比如涉及子矩阵的情况）
// 该对象拷贝是浅拷贝，以避免大量数据的深拷贝
// 读操作可以任意时候进行，但是写操作只能在引用计数为1，也就是无其他地方引用时进行，防止修改了共享了底层内存的其他数据造成错误。
template<typename TElem, typename TDevice>
class ContinuousMemory
{
    static_assert(std::is_same_v<std::remove_cvref_t<TElem>, TElem>, "TElem is not an available type"); // 内存中保存的类型不应该有CVRef限定
    using ElementType = TElem;
public:
    explicit ContinuousMemory(size_t size)
        : m_sp(Allocator<TDevice>::template allocate<ElementType>(size))
        , m_pMemStart(m_sp.get())
    {
    }
    ContinuousMemory(std::shared_ptr<ElementType> spMem, ElementType* pMemStart)
        : m_sp(std::move(spMem))
        , m_pMemStart(pMemStart)
    {
    }
    auto rawMemory() const
    {
        return m_pMemStart;
    }
    const std::shared_ptr<ElementType> sharedPtr() const
    {
        return m_sp;
    }
    bool operator==(const ContinuousMemory& rhs) const
    {
        return m_sp == rhs.m_sp && m_pMemStart == rhs.m_pMemStart;
    }
    bool operator!=(const ContinuousMemory& rhs) const
    {
        return !(operator==(rhs));
    }
    size_t useCount() const
    {
        return m_sp.use_count();
    }
private:
    std::shared_ptr<ElementType> m_sp;
    ElementType* m_pMemStart;
};

} // namespace MetaNN