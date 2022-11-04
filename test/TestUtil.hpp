#pragma once

#include <iostream>
#include <iomanip>
#include <string>
#include <source_location> // since C++20
#include <algorithm>
#include <iterator>
#include <utility>
#include <tuple>

// parsing first argument: -d to show details
inline bool parseDetailFlag(int argc, char const *argv[])
{
    return argc >= 2 && std::string(argv[1]) == "-d";
}

// 通用输出工具，对于需要输出但未定义operator<<的自定义类型则需要进行特化，或者定义operator<<，默认行为是调用operator<<
template<typename T>
struct PrintObj
{
    PrintObj(const T& val) : m_val(val) {}
    const T& m_val;
    void print(std::ostream& os) const
    {
        os << m_val;
    }
};

template<typename T>
std::ostream& operator<<(std::ostream& os, const PrintObj<T>& obj)
{
    obj.print(os);
    return os;
}

template<typename T1, typename T2>
struct PrintObj<std::pair<T1, T2>>
{
    PrintObj(const std::pair<T1, T2>& val) : m_pair(val) {}
    const std::pair<T1, T2>& m_pair;
    void operator()(std::ostream& os) const
    {
        os << "(" << m_pair.first << ", " << m_pair.second << ")";
    }
};

// 用于判等的函数对象，方便自定义类型特化以区分于operator==
template<typename T1, typename T2>
struct ObjEqual
{
    bool operator()(const T1& val1, const T2& val2) const
    {
        return val1 == val2;
    }
};

// manipulator for printing first N elements of a sequence
template<typename Iterator>
class PrintSequenceElements
{
    friend std::ostream& operator<<(std::ostream& os, const PrintSequenceElements& p)
    {
        int count = 0;
        auto iter = p.begin;
        for (; iter != p.end && count < p.num; ++count, ++iter)
        {
            os << *iter << " ";
        }
        if (iter != p.end)
        {
            os << "...";
        }
        return os;
    }
public:
    PrintSequenceElements(const Iterator& _begin, const Iterator& _end, std::size_t _num) : begin(_begin), end(_end), num(_num)
    {
    }
private:
    std::size_t num;
    const Iterator begin;
    const Iterator end;
};

template<typename Container>
PrintSequenceElements<typename Container::const_iterator> printContainerElememts(const Container& c, std::size_t num)
{
    return PrintSequenceElements<typename Container::const_iterator>(c.begin(), c.end(), num);
}

template<typename T>
PrintSequenceElements<T*> printArrayElements(T* arr, std::size_t size, std::size_t num)
{
    return PrintSequenceElements<T*>(arr, arr+size, num);
}

// test utilities
class TestUtil
{
public:
    TestUtil(bool _show, const std::string& _target, int _lineNumberWidth = 4, int _maxSequenceLength = 20, std::ostream& _os = std::cout)
        : groupPassedCount(0)
        , groupTotalCount(0)
        , passedCount(0)
        , totalCount(0)
        , lineNumberWidth(_lineNumberWidth)
        , maxSequenceLength(_maxSequenceLength)
        , showDetails(_show)
        , target(_target)
        , os(_os)
    {
        os.clear();
    }

    // 必须用于测试一个组之前
    void setTestGroup(const std::string& group)
    {
        passedCount += groupPassedCount;
        totalCount += groupTotalCount;
        groupPassedCount = 0;
        groupTotalCount = 0;
        curGroup = group;
        if (showDetails)
        {
            os << "Test of " << curGroup << ":\n";
        }
    }

    // 用于测试一个组的最后
    void showGroupResult()
    {
        os << std::boolalpha << std::dec;
        os << "Test result of " << std::setfill('_') << std::left << std::setw(30) << curGroup << ": ";
        os << std::right << std::setfill(' ');
        os << std::setw(3) << groupPassedCount << "/" << std::setw(3) << std::left << groupTotalCount << " passed";
        os << (groupPassedCount == groupTotalCount ? "\n" : " --------------------------> failed\n");
        if (showDetails)
        {
            os << "\n";
        }
        passedCount += groupPassedCount;
        totalCount += groupTotalCount;
        groupPassedCount = 0;
        groupTotalCount = 0;
    }

    void showFinalResult()
    {
        os << std::boolalpha << std::dec;
        os << "Test results of " << target << ": ";
        os << std::setw(4) << std::right << passedCount << "/" << std::setw(4) << std::left << totalCount << " passed";
        os << (passedCount == totalCount ? " ========================= success\n" : " ========================= failed\n");
    }

    template<typename T1, typename T2, typename Equal = ObjEqual<T1, T2>>
    void assertEqual(const T1& t1, const T2& t2, const Equal& eq = ObjEqual<T1, T2>(), const std::source_location& loc = std::source_location::current())
    {
        bool res = eq(t1, t2);
        groupPassedCount += (res ? 1 : 0);
        groupTotalCount++;
        if (!res && showDetails)
        {
            os << std::boolalpha << std::dec << std::right << std::setfill(' ');
            os << loc.file_name() << ":" << std::setw(lineNumberWidth) << loc.line() << ": "
                << "assertEqual: " << "left value( " << PrintObj<T1>(t1) << " ), right value( " << PrintObj<T2>(t2) << " )\n";
        }
    }

    template<typename T1, typename T2, typename Equal = ObjEqual<T1, T2>>
    void assertNotEqual(const T1& t1, const T2& t2, const Equal& eq = ObjEqual<T1, T2>(), const std::source_location& loc = std::source_location::current())
    {
        bool res = !eq(t1, t2);
        groupPassedCount += (res ? 1 : 0);
        groupTotalCount++;
        if (!res && showDetails)
        {
            os << std::boolalpha << std::dec << std::right << std::setfill(' ');
            os << loc.file_name() << ":" << std::setw(lineNumberWidth) << loc.line() << ": "
                << "assertEqual: " << "left value( " << PrintObj<T1>(t1) << " ), right value( " << PrintObj<T2>(t2) << " )\n";
        }
    }

    template<typename Container1, typename Container2>
    void assertSequenceEqual(const Container1& c1, const Container2& c2, const std::source_location& loc = std::source_location::current())
    {
        bool res = std::equal(c1.begin(), c1.end(), c2.begin());
        groupPassedCount += (res ? 1 : 0);
        groupTotalCount++;
        if (!res && showDetails)
        {
            os << std::boolalpha << std::dec << std::right << std::setfill(' ');
            os << loc.file_name() << ":" << std::setw(lineNumberWidth) << loc.line() << ": "
                << "assertSequenceEqual: "
                << "\n\tleft sequence: " << printContainerElememts(c1, maxSequenceLength)
                << "\n\tright sequence: " << printContainerElememts(c2, maxSequenceLength) << "\n";
        }
    }

    template<typename T1, typename T2>
    void assertArrayEqual(const T1* arr1, const T2* arr2, std::size_t size, const std::source_location& loc = std::source_location::current())
    {
        bool res = std::equal(arr1, arr1 + size, arr2);
        groupPassedCount += (res ? 1 : 0);
        groupTotalCount++;
        if (!res && showDetails)
        {
            os << std::boolalpha << std::dec << std::right << std::setfill(' ');
            os << loc.file_name() << ":" << std::setw(lineNumberWidth) << loc.line() << ": "
                << "assertArrayEqual: "
                << "\n\tleft array: " << printArrayElements(arr1, size, maxSequenceLength)
                << "\n\tright array: " << printArrayElements(arr2, size, maxSequenceLength) << "\n";
        }
    }

    // more generic version of assert sequence/array equal
    template<typename ForwardIterator1, typename ForwardIterator2>
    void assertRangeEqual(ForwardIterator1 b1, ForwardIterator1 e1, ForwardIterator2 b2, const std::source_location& loc = std::source_location::current())
    {
        bool res = std::equal(b1, e1, b2);
        groupPassedCount += (res ? 1 : 0);
        groupTotalCount++;
        if (!res && showDetails)
        {
            os << std::boolalpha << std::dec << std::right << std::setfill(' ');
            os << loc.file_name() << ":" << std::setw(lineNumberWidth) << loc.line() << ": "
                << "assertRangeEqual: "
                << "\n\tleft range: " << PrintSequenceElements(b1, e1, maxSequenceLength)
                << "\n\tright range: " << PrintSequenceElements(b2, std::next(b2, std::distance(b1, e1)), maxSequenceLength)  << "\n";
        }
    }
    template<typename ForwardIterator1, typename ForwardIterator2>
    void assertRangeEqual(ForwardIterator1 b1, ForwardIterator1 e1, ForwardIterator2 b2, ForwardIterator2 e2, const std::source_location& loc = std::source_location::current())
    {
        bool res = std::distance(b1, e1) == std::distance(b2, e2) && std::equal(b1, e1, b2);
        groupPassedCount += (res ? 1 : 0);
        groupTotalCount++;
        if (!res && showDetails)
        {
            os << std::boolalpha << std::dec << std::right << std::setfill(' ');
            os << loc.file_name() << ":" << std::setw(lineNumberWidth) << loc.line() << ": "
                << "assertRangeEqual: "
                << "\n\tleft range: " << PrintSequenceElements(b1, e1, maxSequenceLength)
                << "\n\tright range: " << PrintSequenceElements(b2, e2, maxSequenceLength)  << "\n";
        }
    }
    // assert a sequence is sorted
    template<typename InputIterator, typename Compare = std::less<typename std::iterator_traits<InputIterator>::value_type>>
    void assertSorted(InputIterator b, InputIterator e, const Compare& cmp = Compare(), const std::source_location& loc = std::source_location::current())
    {
        bool res = std::is_sorted(b, e, cmp);
        groupPassedCount += (res ? 1 : 0);
        groupTotalCount++;
        if (!res && showDetails)
        {
            os << std::boolalpha << std::dec << std::right << std::setfill(' ');
            os << loc.file_name() << ":" << std::setw(lineNumberWidth) << loc.line() << ": "
                << "assertSorted: "
                << "\n\tsequence: " << PrintSequenceElements(b, e, maxSequenceLength) << "\n";
        }
    }
    // assert two set is equal, do not consider order of elements.
    template<typename Container1, typename Container2>
    void assertSetEqual(const Container1& c1, const Container2& c2, const std::source_location& loc = std::source_location::current())
    {
        bool res = (std::size(c1) == std::size(c2) && std::is_permutation(c1.begin(), c1.end(), c2.begin()));
        groupPassedCount += (res ? 1 : 0);
        groupTotalCount++;
        if (!res && showDetails)
        {
            os << std::boolalpha << std::dec << std::right << std::setfill(' ');
            os << loc.file_name() << ":" << std::setw(lineNumberWidth) << loc.line() << ": "
                << "assertSetEqual: "
                << "\n\tleft set: " << printContainerElememts(c1, maxSequenceLength)
                << "\n\tright set: " << printContainerElememts(c2, maxSequenceLength) << "\n";
        }
    }
    template<typename ForwardIterator1, typename ForwardIterator2>
    void assertSetEqual(ForwardIterator1 b1, ForwardIterator1 e1, ForwardIterator2 b2, ForwardIterator2 e2, const std::source_location& loc = std::source_location::current())
    {
        bool res = (std::distance(b1, e1) == std::distance(b2, e2) && std::is_permutation(b1, e1, b2));
        groupPassedCount += (res ? 1 : 0);
        groupTotalCount++;
        if (!res && showDetails)
        {
            os << std::boolalpha << std::dec << std::right << std::setfill(' ');
            os << loc.file_name() << ":" << std::setw(lineNumberWidth) << loc.line() << ": "
                << "assertSetEqual: "
                << "\n\tleft set: " << PrintSequenceElements(b1, e1, maxSequenceLength)
                << "\n\tright set: " << PrintSequenceElements(b2, e2, maxSequenceLength)  << "\n";
        }
    }
private:
    int groupPassedCount;
    int groupTotalCount;
    int passedCount;
    int totalCount;
    int lineNumberWidth; // output width of line number
    int maxSequenceLength; // max output length of a sequence
    bool showDetails;
    std::string target;
    std::string curGroup;
    std::ostream& os;
};
