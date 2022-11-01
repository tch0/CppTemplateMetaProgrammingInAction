#pragma once

#include <data/traits.hpp>
#include <operator/tags.hpp>
#include <operator/traits.hpp>
#include <operator/organizer.hpp>
#include <utility>

namespace MetaNN
{

// 表达式模板：不提供写接口

// 一元运算
template<typename TOpTag, typename TData>
class UnaryOp : public OpOrganizer<TOpTag, OpCateCal<TOpTag, TData>>
{
    static_assert(std::is_same_v<std::remove_cvref_t<TData>, TData>, "TData is not an available type");
public:
    using Category = OpCateCal<TOpTag, TData>;
    using ElementType = OpElementType<TOpTag, TData>;
    using DeviceType = OpDeviceType<TOpTag, TData>;
public:
    UnaryOp(TData data)
        : OpOrganizer<TOpTag, Category>(data)
        , m_data(std::move(data))
    {
    }

    const TData& operand() const
    {
        return m_data;
    }

    // 求值接口: todo

private:
    TData m_data;
    using TPrincipal = PrincipalDataType<Category, ElementType, DeviceType>;
};

// 二元运算
template<typename TOpTag, typename TData1, typename TData2>
class BinaryOp : public OpOrganizer<TOpTag, OpCateCal<TOpTag, TData1, TData2>>
{
    static_assert(std::is_same_v<std::remove_cvref_t<TData1>, TData1>, "TData1 is not an available type");
    static_assert(std::is_same_v<std::remove_cvref_t<TData2>, TData2>, "TData2 is not an available type");
public:
    using Category = OpCateCal<TOpTag, TData1, TData2>;
    using ElementType = OpElementType<TOpTag, TData1, TData2>;
    using DeviceType = OpDeviceType<TOpTag, TData1, TData2>;
public:
    BinaryOp(TData1 data1, TData2 data2)
        : OpOrganizer<TOpTag, Category>(data1, data2)
        , m_data1(std::move(data1))
        , m_data2(std::move(data2))
    {
    }

    const TData1& operand1() const
    {
        return m_data1;
    }
    const TData2& operand2() const
    {
        return m_data2;
    }

    // 求值接口: todo

private:
    TData1 m_data1;
    TData2 m_data2;
    using TPrincipal = PrincipalDataType<Category, ElementType, DeviceType>;
};

// 三元运算
template<typename TOpTag, typename TData1, typename TData2, typename TData3>
class TernaryOp : public OpOrganizer<TOpTag, OpCateCal<TOpTag, TData1, TData2, TData3>>
{
    static_assert(std::is_same_v<std::remove_cvref_t<TData1>, TData1>, "TData1 is not an available type");
    static_assert(std::is_same_v<std::remove_cvref_t<TData2>, TData2>, "TData2 is not an available type");
    static_assert(std::is_same_v<std::remove_cvref_t<TData3>, TData3>, "TData3 is not an available type");
public:
    using Category = OpCateCal<TOpTag, TData1, TData2, TData3>;
    using ElementType = OpElementType<TOpTag, TData1, TData2, TData3>;
    using DeviceType = OpDeviceType<TOpTag, TData1, TData2, TData3>;
public:
    TernaryOp(TData1 data1, TData2 data2, TData3 data3)
        : OpOrganizer<TOpTag, Category>(data1, data2, data3)
        , m_data1(std::move(data1))
        , m_data2(std::move(data2))
        , m_data3(std::move(data3))
    {
    }

    const TData1& operand1() const
    {
        return m_data1;
    }
    const TData2& operand2() const
    {
        return m_data2;
    }
    const TData3& operand3() const
    {
        return m_data3;
    }

    // 求值接口: todo

private:
    TData1 m_data1;
    TData2 m_data2;
    TData3 m_data3;
    using TPrincipal = PrincipalDataType<Category, ElementType, DeviceType>;
};

} // namespace MetaNN