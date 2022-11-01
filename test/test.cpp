// MetaNN headers
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
#include <operator/tags.hpp>
#include <operator/traits.hpp>
#include <operator/organizer.hpp>
#include <operator/operators.hpp>
#include <operator/sigmoid.hpp>
#include <operator/add.hpp>
#include <operator/transpose.hpp>
#include <operator/collapse.hpp>
#include <operator/abs.hpp>
#include <operator/sign.hpp>
#include <operator/tanh.hpp>
#include <operator/softmax.hpp>
#include <operator/subtract.hpp>
#include <operator/element_mul.hpp>
#include <operator/divide.hpp>
#include <operator/dot.hpp>
#include <operator/negative_log_likelihood.hpp>
#include <operator/softmax_derivation.hpp>
#include <operator/sigmoid_derivation.hpp>
#include <operator/tanh_derivation.hpp>
#include <operator/negative_log_likelihood_derivation.hpp>
#include <operator/interpolation.hpp>
// standard library headers
#include <iostream>

using namespace MetaNN;

void test_scalar()
{
    static_assert(ScalarC<Scalar<double>>);
    static_assert(ScalarC<Scalar<double, DeviceTags::CPU>>);
}

void test_matrix()
{
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
}

void test_batch()
{
    // Batch
    static_assert(BatchScalarC<Batch<double, DeviceTags::CPU, CategoryTags::Scalar>>);
    static_assert(BatchMatrixC<Batch<double, DeviceTags::CPU, CategoryTags::Matrix>>);
    // alias
    static_assert(BatchScalarC<CpuBatchScalar<int>>);
    static_assert(BatchMatrixC<CpuBatchMatix<int>>);

    static_assert(LowerAccessC<Batch<double, DeviceTags::CPU, CategoryTags::Scalar>>);
    static_assert(LowerAccessC<Batch<double, DeviceTags::CPU, CategoryTags::Matrix>>);

    // Array
    static_assert(BatchScalarC<Array<Scalar<double>>>);
    static_assert(BatchMatrixC<Array<Matrix<double>>>);
    // Duplicate
    static_assert(BatchScalarC<Duplicate<Scalar<double>>>);
    static_assert(BatchMatrixC<Duplicate<Matrix<double>>>);
}

void test_OpCategory()
{
    static_assert(std::same_as<OpCateCal<Matrix<double>, ZeroMatrix<double>, OneHotVector<double>>, CategoryTags::Matrix>);
}

int main()
{
    return 0;
}
