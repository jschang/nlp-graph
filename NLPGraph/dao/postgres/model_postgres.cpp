//
//  model_postgres.cpp
//  NLPGraph
//
//  Created by Jonathan Schang on 6/17/15.
//  Copyright (c) 2015 Jonathan Schang. All rights reserved.
//

#define BOOST_LOG_DYN_LINK
#include "model_postgres.h"
#include <cstdio>
#include <pqxx/pqxx>

#define NEW_INPUT_CHANNEL "new_input_channel"
#define NEW_SYMBOL "new_symbol"
#define NEW_SYMBOL_MEMBER "new_symbol_member"
#define GET_SYMBOL_MEMBER_COUNT "get_symbol_count"
#define NEW_RECOLLECTION "new_recollection"
#define NEW_RECOLLECTION_EXCEPTION "new_recollection_exception"

#define LOG(sev) BOOST_LOG_SEV(logger,sev) << __PRETTY_FUNCTION__ << " line:" << __LINE__ << " "

namespace NLPGraph {
namespace Dao {

NLPGraph::Util::LoggerType logger((boost::log::keywords::channel="NLPGraph::Dao::ModelPostgres"));

using namespace pqxx;
using namespace NLPGraph::Util;
using namespace NLPGraph::Dto;
using namespace boost::log;

std::string modelTables[5] = {
        "input_channel",
        "symbol",
        "symbol_member",
        "recollection",
        "recollection_exception"
    };
std::string createSql[5] = {
// input_channel
        "CREATE TABLE input_channel"
        "("
        "  id bigserial NOT NULL,"
        "  create_dtime timestamp without time zone DEFAULT now(),"
        "  CONSTRAINT input_channel_pkey PRIMARY KEY (id)"
        ")"
        "WITH ("
        "  OIDS=FALSE"
        ")",
// symbol
        "CREATE TABLE symbol"
        "("
        "  id bigserial NOT NULL,"
        "  input_channel_id bigint,"
        "  create_dtime timestamp without time zone DEFAULT now(),"
        "  CONSTRAINT symbol_pkey PRIMARY KEY (id),"
        "  CONSTRAINT symbol_input_channel_id_fk FOREIGN KEY (input_channel_id)"
        "      REFERENCES input_channel (id) MATCH SIMPLE"
        "      ON UPDATE NO ACTION ON DELETE NO ACTION"
        ")"
        "WITH ("
        "  OIDS=FALSE"
        ")",
// symbol_member
        "CREATE TABLE symbol_member"
        "("
        "  symbol_id bigint,"
        "  index integer NOT NULL,"
        "  parent_symbol_id bigint NOT NULL,"
        "  create_dtime timestamp without time zone DEFAULT now(),"
        "  CONSTRAINT symbol_member_pkey PRIMARY KEY (parent_symbol_id, index),"
        "  CONSTRAINT symbol_member_parent_symbol_id_fkey FOREIGN KEY (parent_symbol_id)"
        "      REFERENCES symbol (id) MATCH SIMPLE"
        "      ON UPDATE NO ACTION ON DELETE NO ACTION,"
        "  CONSTRAINT symbol_member_symbol_id_fkey FOREIGN KEY (symbol_id)"
        "      REFERENCES symbol (id) MATCH SIMPLE"
        "      ON UPDATE NO ACTION ON DELETE NO ACTION"
        ")"
        "WITH ("
        "  OIDS=FALSE"
        ")",
// recollection
        "CREATE TABLE recollection"
        "("
        "  id bigserial NOT NULL,"
        "  symbol_id bigint,"
        "  input_channel_id bigint,"
        "  create_dtime timestamp without time zone,"
        "  CONSTRAINT recollection_pkey PRIMARY KEY (id),"
        "  CONSTRAINT recollection_input_channel_id_fkey FOREIGN KEY (input_channel_id)"
        "      REFERENCES input_channel (id) MATCH SIMPLE"
        "      ON UPDATE NO ACTION ON DELETE NO ACTION,"
        "  CONSTRAINT recollection_symbol_id_fkey FOREIGN KEY (symbol_id)"
        "      REFERENCES symbol (id) MATCH SIMPLE"
        "      ON UPDATE NO ACTION ON DELETE NO ACTION"
        ")"
        "WITH ("
        "  OIDS=FALSE"
        ")",
// recollection_exception
        "CREATE TABLE recollection_exception"
        "("
        "  id bigserial NOT NULL,"
        "  recollection_id bigint NOT NULL,"
        "  operation_id smallint NOT NULL,"
        "  member_index integer NOT NULL,"
        "  symbol_id bigint,"
        "  CONSTRAINT recollection_exception_pkey PRIMARY KEY (id),"
        "  CONSTRAINT recollection_exception_symbol_id_fkey FOREIGN KEY (symbol_id)"
        "      REFERENCES symbol (id) MATCH SIMPLE"
        "      ON UPDATE NO ACTION ON DELETE NO ACTION"
        ")"
        "WITH ("
        "  OIDS=FALSE"
        ")"
    };
const int modelTableCount = 5;

ModelPostgres::ModelPostgres(ResourcePoolPtr<connection*> pool, std::string schema) {
    m_dbPool = pool;
    m_schema = schema;
}

ModelPostgres::~ModelPostgres() {
}

void ModelPostgres::prepare() {
    std::unique_ptr<std::vector<connection*>> pool(m_dbPool->obtainAll());
    try {
        std::vector<connection*>::iterator conn;
        for(conn = pool->begin(); conn!=pool->end(); conn++) {
            work t(**conn,"set schema");
            try {
                t.exec("set schema '"+m_schema+"'");
                t.commit();
            } catch(...) {
                t.abort();
            }
            (*conn)->prepare(NEW_INPUT_CHANNEL,"insert into input_channel (create_dtime) values (now()) returning id");
            (*conn)->prepare(NEW_SYMBOL,"insert into symbol (input_channel_id) values ($1) returning id");
            (*conn)->prepare(NEW_SYMBOL_MEMBER,"insert into symbol_member (parent_symbol_id, index, symbol_id) values ($1,$2,$3)");
            (*conn)->prepare(GET_SYMBOL_MEMBER_COUNT,"select count(1) count from symbol_member where parent_symbol_id = $1");
            (*conn)->prepare(NEW_RECOLLECTION,"insert into recollection (symbol_id,input_channel_id) values ($1,$2) returning id");
            (*conn)->prepare(NEW_RECOLLECTION_EXCEPTION,"insert into recollection_exception (recollection_id, symbol_id, member_index, operation_id) values ($1,$2,$3,$4) returning id");
        }
        m_dbPool->releaseAll();
    } catch(pqxx::pqxx_exception& e) {
        m_dbPool->releaseAll();
        LOG(severity_level::error) << e.base().what();
        ModelExceptionType except;
        except.msg = std::string(e.base().what());
        throw except;
    }
}

bool ModelPostgres::isCreated() {
    bool ret = true;
    connection *conn = m_dbPool->obtain();
    work t(*conn,"test model creation transaction");
    try {
        t.exec("set schema '" + m_schema + "'");
        for(int i=0; i<modelTableCount; i++) {
            t.exec("select 1 from " + std::string(modelTables[i])); 
        }
        t.commit();
        m_dbPool->release(conn);
    } catch(pqxx::pqxx_exception& e) {
        t.abort();m_dbPool->release(conn);
        ret = false;
    }
    return ret;
}

bool ModelPostgres::create() {
    bool ret = false;
    connection *conn = m_dbPool->obtain();
    work t(*conn,std::string("create model transaction"));
    try {
        t.exec("create schema " + m_schema + "");
        t.exec("set schema '" + m_schema + "'");
        for(int i=0; i<modelTableCount; i++) {
            std::string str = std::string(createSql[i]);
            result r = t.exec(str); 
        }
        ret = true;
        t.commit();
        m_dbPool->release(conn);
    } catch(pqxx::pqxx_exception& e) {
        t.abort();m_dbPool->release(conn);
        LOG(severity_level::error) << e.base().what();
        ModelExceptionType except;
        except.msg = std::string(e.base().what());
        throw except;
    }
    return ret;
}
bool ModelPostgres::destroy() {
    bool ret = false;
    connection *conn = m_dbPool->obtain();
    work t(*conn,"destroy model transaction");
    try {
        for(int i=0; i<modelTableCount; i++) {
            t.parameterized("drop table $1")(modelTables[i]); 
        }
        t.exec("drop schema " + m_schema + " cascade");
        ret = true;
        t.commit();
        m_dbPool->release(conn);
    } catch(pqxx::pqxx_exception& e) {
        t.abort();m_dbPool->release(conn);
        LOG(severity_level::error) << e.base().what();
        ModelExceptionType except;
        except.msg = std::string(e.base().what());
        throw except;
    }
    return ret;
}
bool ModelPostgres::reset() {
    bool ret = false;
    connection *conn = m_dbPool->obtain();
    work t(*conn,"reset model transaction");
    try {
        for(int i=0; i<modelTableCount; i++) {
            t.parameterized("delete from table $1")(modelTables[i]).exec();
        }
        ret = true;
        t.commit();
        m_dbPool->release(conn);
    } catch(pqxx::pqxx_exception& e) {
        m_dbPool->release(conn);t.abort();
        LOG(severity_level::error) << e.base().what();
        ModelExceptionType except;
        except.msg = std::string(e.base().what());
        throw except;
    }
    return ret;
}

InputChannelPtr ModelPostgres::newInputChannel() {
    InputChannelPtr ret = nullptr;
    connection *conn = m_dbPool->obtain();
    work w(*conn);
    try {
        auto r = w.prepared(NEW_INPUT_CHANNEL).exec();
        uint64_t id; 
        r[0]["id"].to(id);
        ret = InputChannelPtr(new InputChannel(id));
        w.commit();
        m_dbPool->release(conn);
    } catch(pqxx::pqxx_exception& e) {
        m_dbPool->release(conn);w.abort();
        LOG(severity_level::normal) << e.base().what();
        ModelExceptionType except;
        except.msg = std::string(e.base().what());
        throw except;
    }
    return ret;
}

SymbolPtr ModelPostgres::newSymbol(InputChannelPtr channel) {
    SymbolPtr ret = nullptr;
    connection *conn = m_dbPool->obtain();
    work w(*conn);
    try {
        auto r = w.prepared(NEW_SYMBOL)(channel->getId()).exec();
        uint64_t id; 
        r[0]["id"].to(id);
        ret = SymbolPtr(new Symbol(id,channel));
        w.commit();
        m_dbPool->release(conn);
    } catch(pqxx::pqxx_exception& e) {
        m_dbPool->release(conn);w.abort();
        LOG(severity_level::normal) << e.base().what();
        ModelExceptionType except;
        except.msg = std::string(e.base().what());
        throw except;
    }
    return ret;
}

long ModelPostgres::getSymbolMemberCount(SymbolPtr symbol) {
    long ret = 0;
    connection *conn = m_dbPool->obtain();
    work w(*conn);
    try {
        auto r = w.prepared(GET_SYMBOL_MEMBER_COUNT)(symbol->getId()).exec();
        r[0]["count"].to(ret);
        w.commit();
        m_dbPool->release(conn);
    } catch(pqxx::pqxx_exception& e) {
        m_dbPool->release(conn);w.abort();
        LOG(severity_level::normal) << e.base().what();
        ModelExceptionType except;
        except.msg = std::string(e.base().what());
        throw except;
    }
    return ret;
}

void ModelPostgres::addSymbolMember(SymbolPtr parentSymbol, SymbolPtr memberSymbol) {
    connection *conn = m_dbPool->obtain();
    work w(*conn);
    try {
        auto i = w.prepared(NEW_SYMBOL_MEMBER) (parentSymbol->getId()) (getSymbolMemberCount(parentSymbol));
        if(memberSymbol.get()==nullptr) {
            i();
        } else {
            i(memberSymbol->getId());
        }
        auto r = i.exec();
        parentSymbol->addMember(memberSymbol);
        w.commit();
        m_dbPool->release(conn);
    } catch(pqxx::pqxx_exception& e) {
        m_dbPool->release(conn);w.abort();
        LOG(severity_level::normal) << e.base().what();
        ModelExceptionType except;
        except.msg = std::string(e.base().what());
        throw except;
    }
}

RecollectionPtr ModelPostgres::newRecollection(Dto::SymbolPtr symbol) {
    RecollectionPtr ret = 0;
    uint64_t id = 0;
    connection *conn = m_dbPool->obtain();
    work w(*conn);
    try {
        auto r = w.prepared(NEW_RECOLLECTION)
            (symbol->getId())
            (symbol->getInputChannel()->getId()).exec();
        r[0]["id"].to(id);
        w.commit();
        ret = RecollectionPtr(new Recollection(id,symbol));
        m_dbPool->release(conn);
    } catch(pqxx::pqxx_exception& e) {
        m_dbPool->release(conn);w.abort();
        LOG(severity_level::normal) << e.base().what();
        ModelExceptionType except;
        except.msg = std::string(e.base().what());
        throw except;
    }
    return ret;
}

void ModelPostgres::addRecollectionException(Dto::RecollectionPtr recollection, RecExceptOpsEnum operationId, int symbolIdx, Dto::SymbolPtr symbol) {
    RecollectionExceptionPtr ret;
    uint64_t id = 0;
    connection *conn = m_dbPool->obtain();
    work w(*conn);
    try {
        auto i = w.prepared(NEW_RECOLLECTION_EXCEPTION);
        i(recollection->getId());
        if(symbol.get()!=nullptr) { 
            i(symbol->getId()); 
        } else { 
            i(); 
        }
        i(symbolIdx)((long)operationId);
        auto r = i.exec();
        r[0]["id"].to(id);
        w.commit();
        ret = RecollectionExceptionPtr(new RecollectionException(id,recollection,operationId,symbolIdx,symbol));
        recollection->addException(RecollectionExceptionPtr(ret));
        m_dbPool->release(conn);
    } catch(pqxx::pqxx_exception& e) {
        m_dbPool->release(conn);w.abort();
        LOG(severity_level::normal) << e.base().what();
        ModelExceptionType except;
        except.msg = std::string(e.base().what());
        throw except;
    }
}

}}
