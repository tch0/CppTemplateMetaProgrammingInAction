#pragma once

#include <data/traits.hpp>
#include <type_traits>
#include <concepts>
#include <vector>
#include <stdexcept>
#include <iterator>
#include <cassert>

namespace MetaNN
{

// Batch是不可变的列表，而Array是可变列表（话说用Array命名可变列表会不会有点怪？）
template<typename TData>
class Array;

// 标量数组
template<typename TData> requires std::same_as<DataCategory<TData>, CategoryTags::Scalar>
class Array<TData>
{
    static_assert(std::is_same_v<std::remove_cvref_t<TData>, TData>, "TData is not an available type");
public:
    using Category = CategoryTags::BatchScalar;
    using ElementType = typename TData::ElementType;
    using DeviceType = typename TData::DeviceType;
public:
    Array(std::size_t = 0, std::size_t = 0)
        : m_buffer(new std::vector<TData>())
    {
    }
    template<std::input_iterator TIterator>
    Array(TIterator begin, TIterator end)
        : m_buffer(new std::vector<TData>(begin, end))
    {
    }

    // 访问接口
    std::size_t batchNum() const
    {
        return m_buffer->size();
    }
    std::size_t size() const
    {
        return m_buffer->size();
    }
    bool availableForWrite() const
    {
        return m_buffer.use_count() == 1; // todo
    }

    // STL兼容接口
    void push_back(TData val)
    {
        assert(availableForWrite());
        m_buffer->emplace_back(std::move(val));
    }
    template<typename... Args>
    void emplace_back(Args&&... args)
    {
        assert(availableForWrite());
        TData tmp(std::forward<Args>(args)...);
        m_buffer->emplace_back(std::move(tmp));
    }
    void reserve(std::size_t num)
    {
        assert(availableForWrite());
        m_buffer->reserve(num);
    }
    void clear()
    {
        assert(availableForWrite());
        m_buffer->clear();
    }
    bool empty() const
    {
        return m_buffer->empty();
    }
    const auto& operator[](std::size_t idx) const
    {
        return (*m_buffer)[idx];
    }
    auto& operator[](std::size_t idx)
    {
        return (*m_buffer)[idx];
    }
    auto begin()
    {
        return m_buffer->begin();
    }
    auto begin() const
    {
        return m_buffer->begin();
    }
    auto end()
    {
        return m_buffer->end();
    }
    auto end() const
    {
        return m_buffer->end();
    }

    // 求值接口: todo
private:
    std::shared_ptr<std::vector<TData>> m_buffer;
    // 求值缓存: todo
};

// 矩阵数组
template<typename TData> requires std::same_as<DataCategory<TData>, CategoryTags::Matrix>
class Array<TData>
{
    static_assert(std::is_same_v<std::remove_cvref_t<TData>, TData>, "TData is not an available type");
public:
    using Category = CategoryTags::BatchMatrix;
    using ElementType = typename TData::ElementType;
    using DeviceType = typename TData::DeviceType;
public:
    Array(std::size_t row = 0, std::size_t col = 0)
        : m_rowNum(row)
        , m_colNum(col)
        , m_buffer(new std::vector<TData>())
    {
    }
    template<std::input_iterator TIterator>
    Array(TIterator begin, TIterator end)
        : m_rowNum(0)
        , m_colNum(0)
        , m_buffer(new std::vector<TData>(begin, end))
    {
        const auto& buffer = *m_buffer;
        if (!buffer.empty())
        {
            m_rowNum = buffer[0].rowNum();
            m_colNum = buffer[1].colNum();
            for (std::size_t i = 1; i < buffer.size(); ++i)
            {
                if (buffer[i].rowNum() != m_rowNum || buffer[i].colNum() != m_colNum)
                {
                    throw std::runtime_error("Dimension mismatch");
                }
            }
        }
    }
    // 访问接口
    std::size_t rowNum() const
    {
        return m_rowNum;
    }
    std::size_t colNum() const
    {
        return m_colNum;
    }
    std::size_t batchNum() const
    {
        return m_buffer->size();
    }
    std::size_t size() const
    {
        return m_buffer->size();
    }
    bool availableForWrite() const
    {
        return m_buffer.use_count() == 1; // todo
    }
    // STL兼容接口
    void push_back(TData mat)
    {
        assert(availableForWrite());
        if (mat.rowNum() != m_rowNum || mat.colNum() != m_colNum)
        {
            throw std::runtime_error("Dimension mismatch");
        }
        m_buffer->push_back(std::move(mat));
    }
    template<typename... Args>
    void emplace_back(Args&&... args)
    {
        assert(availableForWrite());
        TData tmp(std::forward<Args>(args)...);
        if (tmp.rowNum() != m_rowNum || tmp.colNum() != m_colNum)
        {
            throw std::runtime_error("Dimension mismatch");
        }
        m_buffer->emplace_back(std::move(tmp));
    }
    void reserve(std::size_t num)
    {
        assert(availableForWrite());
        m_buffer->reserve(num);
    }
    void clear()
    {
        assert(availableForWrite());
        m_buffer->clear();
    }
    bool empty() const
    {
        return m_buffer->empty();
    }
    const auto& operator[](std::size_t idx) const
    {
        return (*m_buffer)[idx];
    }
    auto& operator[](std::size_t idx)
    {
        return (*m_buffer)[idx];
    }
    auto begin()
    {
        return m_buffer->begin();
    }
    auto begin() const
    {
        return m_buffer->begin();
    }
    auto end()
    {
        return m_buffer->end();
    }
    auto end() const
    {
        return m_buffer->end();
    }

    // 求值接口: todo

private:
    std::size_t m_rowNum;
    std::size_t m_colNum;
    std::shared_ptr<std::vector<TData>> m_buffer;
    // 求值缓存: todo
};

// 快捷构造Array
template<typename TIterator>
auto makeArray(TIterator beg, TIterator end)
{
    using TData = typename std::iterator_traits<TIterator>::value_type;
    using RawData = std::remove_cvref_t<TData>;
    return Array<RawData>(beg, end);
}

} // namespace MetaNN