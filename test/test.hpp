#pragma once

#include <data/scalar.hpp>
#include <data/matrix/matrix.hpp>
#include <data/matrix/trivial_matrix.hpp>
#include <data/matrix/zero_matrix.hpp>
#include <data/matrix/one_hot_vector.hpp>
#include <data/batch/batch.hpp>
#include <data/batch/array.hpp>
#include <data/batch/duplicate.hpp>

#include "TestUtil.hpp"

inline TestUtil& getMetaNNTestUtil(bool showDetails = true)
{
    static TestUtil util(showDetails, "MetaNN");
    return util;
}

// 针对MetaNN类型特化判等与输出
// scalar
template<MetaNN::ScalarC T>
struct PrintObj<T>
{
    PrintObj(const T& val) : m_val(val) {}
    const T& m_val;
    void print(std::ostream& os) const
    {
        os << "Scalar: " << m_val.value();
    }
};

// matrix
template<MetaNN::MatrixC T>
struct PrintObj<T>
{
    PrintObj(const T& mat) : m_mat(mat) {}
    const T& m_mat;
    void print(std::ostream& os) const
    {
        os << "matrix: " << m_mat.rowNum() << "*" << m_mat.colNum() << "\n";
        for (std::size_t i = 0; i < m_mat.rowNum(); ++i)
        {
            os << "\t[";
            for (std::size_t j = 0; j < m_mat.colNum(); ++j)
            {
                os << std::setw(3) << m_mat(i, j) << ",";
            }
            os << "]\n";
        }
    }
};

// batch scalar
template<MetaNN::BatchScalarC T>
struct PrintObj<T>
{
    PrintObj(const T& batch) : m_batch(batch) {}
    const T& m_batch;
    void print(std::ostream& os) const
    {
        os << "batch scalar: [";
        for (std::size_t i = 0; i < m_batch.batchNum(); i++)
        {
            os << PrintObj<std::decay_t<decltype(m_batch[i])>>(m_batch[i]) << ", ";
        }
    }
};

// batch matrix
template<MetaNN::BatchMatrixC T>
struct PrintObj<T>
{
    PrintObj(const T& batch) : m_batch(batch) {}
    const T& m_batch;
    void print(std::ostream& os) const
    {
        os << "batch matrix: \n";
        for (std::size_t i = 0; i < m_batch.batchNum(); i++)
        {
            os << "[" << i << "]: ";
            os << PrintObj<std::decay_t<decltype(m_batch[i])>>(m_batch[i]);
        }
    }
};

// 判等
template<MetaNN::ScalarC T1, MetaNN::ScalarC T2>
struct ObjEqual<T1, T2>
{
    bool operator()(const T1& val1, const T2& val2) const
    {
        return val1.value() == val2.value();
    }
};

template<MetaNN::MatrixC T1, MetaNN::MatrixC T2>
struct ObjEqual<T1, T2>
{
    bool operator()(const T1& mat1, const T2& mat2) const
    {
        if (mat1.rowNum() != mat2.rowNum() || mat1.colNum() != mat2.colNum())
        {
            return false;
        }
        for (std::size_t i = 0; i < mat1.rowNum(); ++i)
        {
            for (std::size_t j = 0; j < mat1.colNum(); ++j)
            {
                if (mat1(i, j) != mat2(i, j))
                {
                    return false;
                }
            }
        }
        return true;
    }
};

template<MetaNN::BatchScalarC T1, MetaNN::BatchScalarC T2>
struct ObjEqual<T1, T2>
{
    bool operator()(const T1& batch1, const T2& batch2) const
    {
        if (batch1.batchNum() != batch2.batchNum())
        {
            return false;
        }
        for (std::size_t i = 0; i < batch1.batchNum(); ++i)
        {
            if (!ObjEqual<std::decay_t<decltype(batch1[i])>, std::decay_t<decltype(batch2[i])>>()(batch1[i], batch2[i]))
            {
                return false;
            }
        }
        return true;
    }
};

template<MetaNN::BatchMatrixC T1, MetaNN::BatchMatrixC T2>
struct ObjEqual<T1, T2>
{
    bool operator()(const T1& batch1, const T2& batch2) const
    {
        if (batch1.batchNum() != batch2.batchNum())
        {
            return false;
        }
        for (std::size_t i = 0; i < batch1.batchNum(); ++i)
        {
            if (!ObjEqual<std::decay_t<decltype(batch1[i])>, std::decay_t<decltype(batch2[i])>>()(batch1[i], batch2[i]))
            {
                return false;
            }
        }
        return true;
    }
};

// 初始化一个矩阵
template<typename TElem>
inline void iota(MetaNN::Matrix<TElem>& mat)
{
    int count = 0;
    for (std::size_t i = 0; i < mat.rowNum(); ++i)
    {
        for (std::size_t j = 0; j < mat.colNum(); ++j)
        {
            mat.setValue(i, j, count++);
        }
    }
}

// 初始化矩阵列表
template<typename TElem>
inline void iota(MetaNN::Batch<TElem, MetaNN::DeviceTags::CPU, MetaNN::CategoryTags::Matrix>& batch)
{
    int count = 0;
    for (std::size_t i = 0; i < batch.batchNum(); i++)
    {
        for (std::size_t j = 0; j < batch.rowNum(); j++)
        {
            for (std::size_t k = 0; k < batch.colNum(); k++)
            {
                batch.setValue(i, j, k, count++);
            }
        }
    }
}

// 测试函数声明
void test_facility(TestUtil& util = getMetaNNTestUtil());

void test_data(TestUtil& util = getMetaNNTestUtil());

void test_operator(TestUtil& util = getMetaNNTestUtil());

void test_policy(TestUtil& util = getMetaNNTestUtil());

void test_param_initializer(TestUtil& util = getMetaNNTestUtil());

void test_layer(TestUtil& util = getMetaNNTestUtil());

void test_evaluation(TestUtil& util = getMetaNNTestUtil());
