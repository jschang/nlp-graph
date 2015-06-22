//
//  nlpgraph.h
//  NLPGraph
//
//  Created by Jonathan Schang on 6/16/15.
//  Copyright (c) 2015 Jonathan Schang. All rights reserved.
//

#ifndef __NLPGraph__nlpgraph__
#define __NLPGraph__nlpgraph__

#define BOOST_LOG_DYN_LINK 1

#include <boost/log/common.hpp>
#include <boost/multiprecision/cpp_int.hpp>
#include <boost/container/vector.hpp>
#include <boost/thread.hpp> 
#include <boost/shared_ptr.hpp>
#include <pqxx/pqxx>
#include <map>
#include <vector>

namespace NLPGraph {
namespace Dao {
    class Model;
    typedef boost::shared_ptr<Model> ModelPtr;
}
namespace Dto {
    class Recollection;
    typedef boost::shared_ptr<Recollection> RecollectionPtr;
    class Symbol;
    typedef boost::shared_ptr<Symbol> SymbolPtr;
    class InputChannel;
    typedef boost::shared_ptr<InputChannel> InputChannelPtr;
    class RecollectionException;
    typedef boost::shared_ptr<RecollectionException> RecollectionExceptionPtr;
}};

#include "context.h"

#include "util/logger.h"
#include "util/resource_pool.h"

#include "dto/input_channel.h"
#include "dto/symbol.h"
#include "dto/recollection.h"
#include "dto/recollection_exception.h"

#include "dao/model_exception.h"
#include "dao/symbol_provider.h"
#include "dao/model.h"
#include "dao/postgres/symbol_provider_postgres.h"
#include "dao/postgres/model_postgres.h"

#endif /* defined(__NLPGraph__nlpgraph__) */
