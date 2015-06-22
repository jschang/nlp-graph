#include <nlpgraph.h>
#include <boost/test/unit_test.hpp>
#include "../../nlpgraph_tests.h"

using namespace NLPGraph::Dao;
using namespace NLPGraph::Dto;
using namespace NLPGraph::Util;

struct Fixture {
    ResourcePoolPtr<pqxx::connection*> connPool;
    ModelPtr model;
    Fixture() {
        BOOST_TEST_MESSAGE("Fixture setup");
        connPool.reset(new ResourcePool<pqxx::connection*>());
        connPool->addResource(new pqxx::connection("dbname=schang"));
        connPool->addResource(new pqxx::connection("dbname=schang"));
        connPool->addResource(new pqxx::connection("dbname=schang"));
        connPool->addResource(new pqxx::connection("dbname=schang"));
        connPool->addResource(new pqxx::connection("dbname=schang"));
        model.reset((Model*)new ModelPostgres(connPool,NLPGRAPH_TEST_SCHEMA));
    }
    ~Fixture() {
        BOOST_TEST_MESSAGE("Fixture teardown");
    }
};

BOOST_FIXTURE_TEST_SUITE( nlpgraph_dao_model_postgres, Fixture ) 

BOOST_AUTO_TEST_CASE( test_create_destroy )
{       
    if(model->isCreated()) {
        model->destroy();
    }
    BOOST_CHECK(!model->isCreated());
    model->create();
    BOOST_CHECK(model->isCreated());
    model->destroy();
}

BOOST_AUTO_TEST_CASE( test_new_stuff )
{       
    if(model->isCreated())
        model->destroy();
    model->create();
    model->prepare();
    
    InputChannelPtr ic(model->newInputChannel());
    
    SymbolPtr s = model->newSymbol(ic);
    model->addSymbolMember(s,model->newSymbol(ic));
    model->addSymbolMember(s,model->newSymbol(ic));
    
    RecollectionPtr r(model->newRecollection(s));
    model->addRecollectionException(r, RecExceptOpsEnum::RecExceptOpInsert, 0, s->getMember(0));
    model->addRecollectionException(r, RecExceptOpsEnum::RecExceptOpInsert, 0, SymbolPtr(nullptr));
    
    pqxx::connection* conn = connPool->obtain();
    pqxx::work w(*conn);
    pqxx::result res;
    try {
        res = w.parameterized("select * from input_channel where id = $1")(ic->getId()).exec();
        BOOST_CHECK_EQUAL(1,res.size());
        res = w.parameterized("select * from symbol where id = $1")(s->getId()).exec();
        BOOST_CHECK_EQUAL(1,res.size());
        res = w.parameterized("select * from symbol_member where parent_symbol_id = $1")(s->getId()).exec();
        BOOST_CHECK_EQUAL(2,res.size());
        res = w.parameterized("select * from symbol where id = $1")(s->getMember(0)->getId()).exec();
        BOOST_CHECK_EQUAL(1,res.size());
        res = w.parameterized("select * from symbol where id = $1")(s->getMember(1)->getId()).exec();
        BOOST_CHECK_EQUAL(1,res.size());
        res = w.parameterized("select * from recollection where id = $1")(r->getId()).exec();
        BOOST_CHECK_EQUAL(1,res.size());
        res = w.parameterized("select * from recollection_exception where recollection_id = $1")(r->getId()).exec();
        BOOST_CHECK_EQUAL(2,res.size());
        w.commit();
    } catch(...) {
        w.abort();
    }
    connPool->release(conn);
}

BOOST_AUTO_TEST_SUITE_END()