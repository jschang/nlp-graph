#define BOOST_LOG_DYN_LINK

#include <util/resource_pool.h>
#include <nlpgraph.h>
#include <boost/test/unit_test.hpp>
#include "../nlpgraph_tests.h"

#if RUN_TEST_ALL == 1 || RUN_TEST_UTIL_RESOURCE_POOL == 1

BOOST_AUTO_TEST_SUITE( nlpgraph_util_resourcepool )

BOOST_AUTO_TEST_CASE( do_stuff )
{
    NLPGraph::Util::ResourcePool<std::string*> stringPool;
    BOOST_CHECK(true);
}

BOOST_AUTO_TEST_SUITE_END()

#endif