#include <facility/var_type_dict.hpp>
#include <facility/data_copy.hpp>
#include <string>
#include <tuple>

#include "test.hpp"

using namespace MetaNN;
using namespace std::literals;

void test_facility_var_type_dict(TestUtil& util);
void test_facility_data_copy(TestUtil& util);

void test_facility(TestUtil& util)
{
    test_facility_var_type_dict(util);
    test_facility_data_copy(util);
}

using Params = VarTypeDict<struct A, struct B, struct C>;
template<typename T>
auto foo(const T& t)
{
    auto a = t.template get<A>();
    const auto& b = t.template get<B>();
    auto& c = t.template get<C>();
    return std::tuple{a, b, c};
}

void test_facility_var_type_dict(TestUtil& util)
{
    util.setTestGroup("facilit.var_type_dict");
    {
        auto res = foo(Params::create().set<A>(1u).set<B>(2.1).set<C>("hello"s));
        util.assertEqual(std::get<0>(res), 1u);
        util.assertEqual(std::get<1>(res), 2.1);
        util.assertEqual(std::get<2>(res), "hello"s);
    }
    util.showGroupResult();
}

void test_facility_data_copy(TestUtil& util)
{
    util.setTestGroup("facility.data_copy");
    {
        Matrix<double> mat1;
        iota(mat1);
        util.assertEqual(mat1.availableForWrite(), true);
        Matrix<double> mat2;
        dataCopy(mat1, mat2);
        util.assertEqual(mat1, mat2);
        util.assertEqual(mat1.availableForWrite(), true);
    }
    util.showGroupResult();
}
