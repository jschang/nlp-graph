//
//  recollection.h
//  NLPGraph
//
//  Created by Jonathan Schang on 6/19/15.
//  Copyright (c) 2015 Jonathan Schang. All rights reserved.
//

#ifndef __NLPGraph__recollection__
#define __NLPGraph__recollection__

#include "../nlpgraph.h"

namespace NLPGraph {
namespace Dto {

class Recollection {
private:
    uint64_t m_id;
    SymbolPtr m_symbol;
    std::vector<RecollectionExceptionPtr> m_exceptions;
public:
    Recollection(uint64_t id, SymbolPtr symbol) 
            : m_id(0), m_symbol(nullptr) {
        m_id = id;
        m_symbol = symbol;
    }
    virtual ~Recollection() {}
    virtual uint64_t getId() { return m_id; }
    virtual SymbolPtr getSymbol() { return m_symbol; }
    /**
     * Should only ever be called by the Model
     */
    virtual void addException(RecollectionExceptionPtr exception) {
        m_exceptions.push_back(exception);
    }
};

}}

#endif /* defined(__NLPGraph__recollection__) */
