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
#include <data/matrix/matrix.hpp>
#include <data/matrix/zero_matrix.hpp>
#include <data/matrix/one_hot_vector.hpp>

#include "test.hpp"

using namespace MetaNN;


void test_OpCategory()
{
    static_assert(std::same_as<OpCateCal<Matrix<double>, ZeroMatrix<double>, OneHotVector<double>>, CategoryTags::Matrix>);
}

void test_operator(TestUtil& util)
{
    util.setTestGroup("operator");
    {

    }
    util.showGroupResult();
}
