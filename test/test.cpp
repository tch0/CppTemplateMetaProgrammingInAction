#include "test.hpp"

int main(int argc, char const *argv[])
{
    bool showDetails = parseDetailFlag(argc, argv);
    TestUtil& util = getMetaNNTestUtil(showDetails);
    test_facility();
    test_data();
    // test_operator();
    // test_policy();
    // test_param_initializer();
    // test_layer();
    // test_evaluation();
    util.showFinalResult();
    return 0;
}
