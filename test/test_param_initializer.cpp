#include <param_initializer/init_policy.hpp>
#include <param_initializer/param_initializer.hpp>
#include <param_initializer/constant_filler.hpp>
#include <param_initializer/gaussian_filler.hpp>
#include <param_initializer/uniform_filler.hpp>
#include <param_initializer/var_scale_filler.hpp>

#include "test.hpp"

using namespace MetaNN;


void test_param_initializer(TestUtil& util)
{
    util.setTestGroup("parameter initializer");
    {

    }
    util.showGroupResult();
}
