#include <data/tags.hpp>
#include <data/traits.hpp>
#include <data/allocator.hpp>
#include <data/lower_access.hpp>
#include <data/scalar.hpp>
#include <data/matrix/matrix.hpp>
#include <data/matrix/trivial_matrix.hpp>
#include <data/matrix/zero_matrix.hpp>
#include <data/matrix/one_hot_vector.hpp>
#include <data/batch/batch.hpp>
#include <data/batch/array.hpp>
#include <data/batch/duplicate.hpp>

#include "test.hpp"

using namespace MetaNN;

// scalar
static_assert(ScalarC<Scalar<double>>);
static_assert(ScalarC<Scalar<double, DeviceTags::CPU>>);
// matrix
static_assert(MatrixC<Matrix<double, DeviceTags::CPU>>);
static_assert(MatrixC<Matrix<double>>);
static_assert(MatrixC<TrivialMatrix<double, DeviceTags::CPU, Scalar<int, DeviceTags::CPU>>>);
static_assert(MatrixC<TrivialMatrix<double>>);
static_assert(MatrixC<ZeroMatrix<double, DeviceTags::CPU>>);
static_assert(MatrixC<ZeroMatrix<double>>);
static_assert(MatrixC<OneHotVector<double, DeviceTags::CPU>>);
static_assert(MatrixC<OneHotVector<double>>);
static_assert(LowerAccessC<Matrix<double>>);
static_assert(!LowerAccessC<OneHotVector<double>>);
// batch scalar, batch matrix
static_assert(BatchScalarC<Batch<double, DeviceTags::CPU, CategoryTags::Scalar>>);
static_assert(BatchMatrixC<Batch<double, DeviceTags::CPU, CategoryTags::Matrix>>);
static_assert(BatchScalarC<CpuBatchScalar<int>>);
static_assert(BatchMatrixC<CpuBatchMatix<int>>);
static_assert(LowerAccessC<Batch<double, DeviceTags::CPU, CategoryTags::Scalar>>);
static_assert(LowerAccessC<Batch<double, DeviceTags::CPU, CategoryTags::Matrix>>);
static_assert(BatchScalarC<Array<Scalar<double>>>);
static_assert(BatchMatrixC<Array<Matrix<double>>>);
static_assert(BatchScalarC<Duplicate<Scalar<double>>>);
static_assert(BatchMatrixC<Duplicate<Matrix<double>>>);
// PrincipalDataType
static_assert(std::same_as<Scalar<double>, PrincipalDataType<CategoryTags::Scalar, double, DeviceTags::CPU>>);
static_assert(std::same_as<Matrix<double>, PrincipalDataType<CategoryTags::Matrix, double, DeviceTags::CPU>>);
static_assert(std::same_as<Batch<double, DeviceTags::CPU, CategoryTags::Scalar>, PrincipalDataType<CategoryTags::BatchScalar, double, DeviceTags::CPU>>);
static_assert(std::same_as<Batch<double, DeviceTags::CPU, CategoryTags::Matrix>, PrincipalDataType<CategoryTags::BatchMatrix, double, DeviceTags::CPU>>);

void test_scalar(TestUtil& util);
void test_matrix(TestUtil& util);
void test_batch_scalar(TestUtil& util);
void test_batch_matrix(TestUtil& util);
void test_trivial_matrix(TestUtil& util);
void test_zero_matrix(TestUtil& util);
void test_one_hot_vector(TestUtil& util);
void test_array(TestUtil& util);
void test_duplicate(TestUtil& util);

void test_data(TestUtil& util)
{
    test_scalar(util);
    test_matrix(util);
    test_batch_scalar(util);
    test_batch_matrix(util);
    test_trivial_matrix(util);
    test_zero_matrix(util);
    test_one_hot_vector(util);
    test_array(util);
    test_duplicate(util);
}

void test_scalar(TestUtil& util)
{
    util.setTestGroup("data.scalar");
    {
        Scalar<double> s(1.1);
        util.assertEqual(s.value(), 1.1);
        s.value() = 10;
        util.assertEqual(s.value(), 10);
    }
    {
        const Scalar<double> s(1.2);
        util.assertEqual(s.value(), 1.2);
    }
    util.showGroupResult();
}

void test_matrix(TestUtil& util)
{
    util.setTestGroup("data.matrix");
    {
        // rowNum, colNum
        Matrix<double> mat(2, 3);
        iota(mat);
        util.assertEqual(mat.rowNum(), 2);
        util.assertEqual(mat.colNum(), 3);
        mat.setValue(1, 1, 10.5);
        util.assertEqual(mat(1, 1), 10.5);
        util.assertEqual(mat.availableForWrite(), true);
        // availableForWrite
        Matrix<double> mat2(mat);
        util.assertEqual(mat2(1, 1), 10.5);
        util.assertEqual(mat.availableForWrite(), false);
        util.assertEqual(mat2.availableForWrite(), false);
        // lower access
        auto acc = lowerAccess(mat);
        util.assertEqual(acc.rowLen(), 3);
        util.assertEqual(*(acc.rawMemory() + 1 * 3 + 1), 10.5);
    }
    // subMatrix
    {
        Matrix<double> mat(10, 10);
        iota(mat);
        Matrix<double> mat2(10, 10);
        iota(mat2);
        auto sub1 = mat.subMatrix(3, 8, 4, 10);
        auto sub2 = mat2.subMatrix(3, 8, 4, 10);
        util.assertEqual(sub1, sub2);
        sub1.setValue(1, 1, -1);
        sub2.setValue(1, 1, -1);
        util.assertEqual(sub1, sub2);
    }
    util.showGroupResult();
}

void test_batch_scalar(TestUtil& util)
{
    util.setTestGroup("data.batch_scalar");
    {
        using BatchScalarDouble = Batch<double, DeviceTags::CPU, CategoryTags::Scalar>;

        BatchScalarDouble s(10);
        util.assertEqual(s.batchNum(), 10);
        util.assertEqual(s.availableForWrite(), true);
        for (std::size_t i = 0; i < s.batchNum(); i++)
        {
            s.setValue(i, i);
        }
        util.assertEqual(s[1], 1);
        util.assertEqual(s[8], 8);
        // lower access
        {
            auto acc = lowerAccess(s);
            util.assertEqual(*(acc.rawMemory() + 3), 3);
            *(acc.mutableRawMemory() + 3) = 10;
            util.assertEqual(*(acc.rawMemory() + 3), 10);
        }
    }
    
    util.showGroupResult();
}

void test_batch_matrix(TestUtil& util)
{
    util.setTestGroup("data.batch_matrix");
    {
        using BatchMatrixDouble = Batch<double, DeviceTags::CPU, CategoryTags::Matrix>;
        BatchMatrixDouble batch1(10, 2, 3);
        BatchMatrixDouble batch2(10, 2, 3);
        util.assertEqual(batch1.rowNum(), batch2.rowNum());
        util.assertEqual(batch1.colNum(), batch2.colNum());
        util.assertEqual(batch1.batchNum(), batch2.batchNum());
        util.assertEqual(batch1.rowNum(), 2);
        util.assertEqual(batch1.colNum(), 3);
        util.assertEqual(batch1.batchNum(), 10);
        iota(batch1);
        iota(batch2);
        util.assertEqual(batch1, batch2);
        util.assertEqual(batch1.availableForWrite(), true);
        batch1.setValue(0, 0, 0, 10.2);
        util.assertEqual(batch1[0](0, 0), 10.2);
        Matrix<double> mat(2, 3);
        iota(mat);
        util.assertEqual(mat, batch2[0]);
        // subBatchMatrix
        auto sub1 = batch1.subBatchMatrix(0, 2, 1, 3);
        auto sub2 = batch2.subBatchMatrix(0, 2, 1, 3);
        util.assertEqual(sub1, sub2);
        // lower access
        auto acc = lowerAccess(sub1);
        *(acc.mutableRawMemory()) = -1;
        util.assertEqual(batch1[0](0,1), -1);
        util.assertEqual(*acc.rawMemory(), -1);
        util.assertEqual(acc.rowLen(), 3);
        util.assertEqual(acc.rawMatrixSize(), 6);
    }
    util.showGroupResult();
}

void test_trivial_matrix(TestUtil& util)
{
    util.setTestGroup("data.trivial_matrix");
    {
        TrivialMatrix<double> mat(10, 10, 9.9);
        util.assertEqual(mat.rowNum(), 10);
        util.assertEqual(mat.colNum(), 10);
        util.assertEqual(mat.elementValue().value(), 9.9);
    }
    // makeTrivialMatrix
    {
        auto mat = makeTrivialMatrix<double, DeviceTags::CPU>(10, 10, 9.9);
        util.assertEqual(mat.rowNum(), 10);
        util.assertEqual(mat.colNum(), 10);
        util.assertEqual(mat.elementValue().value(), 9.9);
    }
    {
        Scalar<double> s(3.3);
        auto mat = makeTrivialMatrix<double, DeviceTags::CPU>(10, 10, s);
        util.assertEqual(mat.rowNum(), 10);
        util.assertEqual(mat.colNum(), 10);
        util.assertEqual(mat.elementValue().value(), 3.3);
    }
    util.showGroupResult();
}

void test_zero_matrix(TestUtil& util)
{
    util.setTestGroup("data.zero_matrix");
    {
        ZeroMatrix<double> mat(10, 10);
        util.assertEqual(mat.rowNum(), 10);
        util.assertEqual(mat.colNum(), 10);
    }
    util.showGroupResult();
}

void test_one_hot_vector(TestUtil& util)
{
    util.setTestGroup("data.one_hot_vector");
    {
        OneHotVector<double> mat(10, 3);
        util.assertEqual(mat.rowNum(), 1);
        util.assertEqual(mat.colNum(), 10);
        util.assertEqual(mat.hotPos(), 3);
    }
    util.showGroupResult();
}

void test_array(TestUtil& util)
{
    util.setTestGroup("data.array");
    // as batch scalar
    {
        Array<Scalar<double>> arr1;
        arr1.push_back(1.0);
        arr1.push_back(2.0);
        arr1.emplace_back(3.0);
        util.assertEqual(arr1.batchNum(), 3);
        util.assertEqual(arr1.size(), 3);
        util.assertEqual(arr1.availableForWrite(), true);
        Array<Scalar<double>> arr2(arr1.begin(), arr1.end());
        util.assertEqual(arr1, arr2);
        util.assertEqual(arr1[0].value(), 1.0);
        util.assertEqual(arr1[2].value(), 3.0);
        arr1[0] = Scalar<double>(9.9);
        util.assertEqual(arr1[0].value(), 9.9);
        util.assertEqual((*arr1.begin()).value(), 9.9);
        arr1.clear();
        util.assertEqual(arr1.empty(), true);
    }
    // as batch matrix
    {
        Array<Matrix<double>> arr1(4, 5);
        Matrix<double> mat(4, 5);
        iota(mat);
        arr1.push_back(mat);
        arr1.emplace_back(mat);
        util.assertEqual(arr1.size(), 2);
        util.assertEqual(arr1.rowNum(), 4);
        util.assertEqual(arr1.colNum(), 5);
        util.assertEqual(arr1.batchNum(), 2);
        util.assertEqual(arr1.availableForWrite(), true);
        Array<Matrix<double>> arr2(arr1.begin(), arr1.end());
        util.assertEqual(arr1, arr2);
        util.assertEqual(arr1[0], mat);
        mat.setValue(1, 1, -1);
        arr1[0] = mat;
        util.assertEqual(arr1[0], mat);
        arr1.clear();
        util.assertEqual(arr1.empty(), true);
    }
    // makeArray
    {
        Array<Matrix<double>> arr1(4, 5);
        Matrix<double> mat(4, 5);
        iota(mat);
        arr1.push_back(mat);
        arr1.emplace_back(mat);

        auto arr2 = makeArray(arr1.begin(), arr1.end());
        util.assertEqual(arr2.size(), 2);
        util.assertEqual(arr2[0], mat);
    }
    util.showGroupResult();
}

void test_duplicate(TestUtil& util)
{
    util.setTestGroup("data.duplicate");
    // dupliate of scalar
    {
        Duplicate<Scalar<double>> dup(Scalar<double>(9.9), 10);
        util.assertEqual(dup.batchNum(), 10);
        util.assertEqual(dup.element().value(), 9.9);
    }
    // dupliate of matrix
    {
        Matrix<double> mat(4, 5);
        iota(mat);
        Duplicate<Matrix<double>> dup(mat, 10);
        util.assertEqual(dup.rowNum(), 4);
        util.assertEqual(dup.colNum(), 5);
        util.assertEqual(dup.batchNum(), 10);
        util.assertEqual(dup.element(), mat);
    }
    // makeDupliate
    {
        Matrix<double> mat(4, 5);
        iota(mat);
        auto dup = makeDuplicate(10, mat);
        util.assertEqual(dup.rowNum(), 4);
        util.assertEqual(dup.colNum(), 5);
        util.assertEqual(dup.batchNum(), 10);
        util.assertEqual(dup.element(), mat);
    }
    util.showGroupResult();
}
