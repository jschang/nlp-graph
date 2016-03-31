//
//  nlpgraph_tests.h
//  NLPGraph
//
//  Created by Jonathan Schang on 6/21/15.
//  Copyright (c) 2015 Jonathan Schang. All rights reserved.
//

#ifndef NLPGraph_nlpgraph_tests_h
#define NLPGraph_nlpgraph_tests_h

#define NLPGRAPH_TEST_DB_SCHEMA           "nlpgraph_tests"
#define NLPGRAPH_TEST_DB_CONN_STRING      "dbname=schang"

#ifndef RUN_TEST_ALL
// overrides all flags below
// will be passed in during automated builds
#define RUN_TEST_ALL                     0
#endif

#define RUN_TEST_CALC_KOHONEN_SOM        0
#define RUN_TEST_CALC_LEVENSTEIN_DAMERAU 0
#define RUN_TEST_CALC_SMITH_WATERMAN     1

#define RUN_TEST_NEURAL_NETWORK          0

#define RUN_TEST_DAO_POSTGRES_MODEL      0

#define RUN_TEST_UTIL_OPENCL             0
#define RUN_TEST_UTIL_RESOURCE_POOL      0

#endif
