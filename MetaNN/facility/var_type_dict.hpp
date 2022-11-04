#pragma once

#include <cstddef>
#include <type_traits>
#include <memory>

// 异类类型词典，为了实现命名参数
// 原理：通过下标建立从可变参数外层类模板到其中的嵌套可变参数类模板的参数之间的关系
// 键是编译期常量，值是运行时对象。

namespace MetaNN
{

struct NullParameter {};

namespace NsVarTypeDict
{

// 将N个NullParameter占位类型添加到类型容器左端，主要用于创建保存N个NullParameter的类型容器
template<size_t N, template<typename...> class TCont, typename... Ts>
struct Create_
{
    using type = typename Create_<N-1, TCont, NullParameter, Ts...>::type;
};

template<template<typename...> class TCont, typename... Ts>
struct Create_<0, TCont, Ts...>
{
    using type = TCont<Ts...>;
};

// 替换typelist中的指定位置的类型为TVal，M是辅助变量，表示已经扫描过的类型数量
template<typename TVal, size_t N, size_t M, typename TProcessedTypes, typename... TRemainTypes>
struct NewTupleType_;
// N!=M的情况，继续扫描
template<typename TVal, size_t N, size_t M, template<typename...> class TCont, typename... TModifiedTypes, typename TCurType, typename... TRemainTypes>
struct NewTupleType_<TVal, N, M, TCont<TModifiedTypes...>, TCurType, TRemainTypes...>
{
    using type = typename NewTupleType_<TVal, N, M+1, TCont<TModifiedTypes..., TCurType>, TRemainTypes...>::type;
};
// N==M的情况，替换后直接返回
template<typename TVal, size_t N, template<typename...> class TCont, typename... TModifiedTypes, typename TCurType, typename... TRemainTypes>
struct NewTupleType_<TVal, N, N, TCont<TModifiedTypes...>, TCurType, TRemainTypes...>
{
    using type = TCont<TModifiedTypes..., TVal, TRemainTypes...>;
};

template<typename TVal, size_t TagPos, typename TProcessedTypes, typename... TRemainTypes>
using NewTupleType = typename NewTupleType_<TVal, TagPos, 0, TProcessedTypes, TRemainTypes...>::type;

// 在多个类型中查找指定类型的下标，从左到右，从0开始
template<typename TTag, size_t Count, typename... Ts>
struct Tag2Id_;
template<typename TTag, size_t Count, typename TCurType, typename... TRemainTypes>
struct Tag2Id_<TTag, Count, TCurType, TRemainTypes...>
{
    static constexpr size_t value = std::conditional_t<std::is_same_v<TTag, TCurType>,
                                                       std::integral_constant<size_t, Count>,
                                                       Tag2Id_<TTag, Count+1, TRemainTypes...>>::value; // 不能直接用?:，需要阻止不满足条件的情况下的继续实例化
};

template<typename TTag, typename... Ts>
constexpr size_t Tag2Id = Tag2Id_<TTag, 0, Ts...>::value;

// 在类型容器中查找指定下标的类型
template<typename TypeList, size_t TagPos>
struct ContPosType_
{
    using type = void; // 这个必须要有，因为在递归终点std::conditional_t中使用了这个类型，所以也会被实例化
};
template<template<typename...> class TCont, typename TCurType, typename... TRemainTypes, size_t TagPos>
struct ContPosType_<TCont<TCurType, TRemainTypes...>, TagPos>
{
    using type = std::conditional_t<TagPos == 0, TCurType, typename ContPosType_<TCont<TRemainTypes...>, TagPos - 1>::type>;
};

template<typename TypeList, size_t TagPos>
using ContPosType = typename ContPosType_<TypeList, TagPos>::type;

} // namespace NsVarTypeDict


template<typename... Ts>
struct VarTypeDict
{
    template<typename... TTypes>
    struct Values
    {
    private:
        std::shared_ptr<void> m_tuple[sizeof...(TTypes)];
    public:
        Values() = default;
        Values(std::shared_ptr<void>(&&input)[sizeof...(TTypes)])
        {
            for (size_t i = 0; i < sizeof...(TTypes); i++)
            {
                m_tuple[i] = std::move(input[i]);
            }
        }
        // TTag作为类型键值数组的键，每一次Set调用会使用TVal去替代TTag位置的类型，返回结果类型的值
        // 当前已保存的值会被移动到返回结果中，所有TTag都被替代完成才能通过编译
        template<typename TTag, typename TVal>
        auto set(TVal&& val) &&
        {
            using namespace NsVarTypeDict;
            constexpr static size_t TagPos = Tag2Id<TTag, Ts...>;
            using RawVal = std::decay_t<TVal>;
            RawVal* tmp = new RawVal(std::forward<TVal>(val));
            m_tuple[TagPos] = std::shared_ptr<void>(tmp,
                [](void* ptr) {
                    RawVal* nptr = static_cast<RawVal*>(ptr);
                    delete nptr;
                }
            );
            using new_type = NewTupleType<RawVal, TagPos, Values<>, TTypes...>;
            return new_type(std::move(m_tuple));
        }
        template<typename TTag>
        const auto& get() const
        {
            using namespace NsVarTypeDict;
            constexpr static size_t TagPos = Tag2Id<TTag, Ts...>;
            using TupleType = std::decay_t<decltype(*this)>;
            return *static_cast<ContPosType<TupleType, TagPos>*>(m_tuple[TagPos].get());
        }
    };
    // 返回类型是Values<>，实参列表是sizeof...(Ts)个NullParameter
    static auto create()
    {
        using namespace NsVarTypeDict;
        using type = typename Create_<sizeof...(Ts), Values>::type;
        return type();
    }
};

} // MetaNN