//
//  model.h
//  NLPGraph
//
//  Created by Jonathan Schang on 6/17/15.
//  Copyright (c) 2015 Jonathan Schang. All rights reserved.
//

#ifndef __NLPGraph__model__
#define __NLPGraph__model__

#include "model_exception.h"
#include "../dto/symbol.h"
#include "../dto/recollection.h"
#include "../dto/recollection_exception.h"
#include "../dto/input_channel.h"

namespace NLPGraph {
namespace Dao {

class Model {
public:
    virtual ~Model() {}
public:
    virtual bool isCreated() { return true; }
    virtual bool create() { return false; }
    virtual bool destroy() { return false; }
    virtual bool reset() { return false; }
    virtual void prepare() {}
    
    virtual Dto::InputChannelPtr newInputChannel() { 
        return nullptr;
    }
    
    virtual Dto::SymbolPtr newSymbol(Dto::InputChannelPtr channel) { 
        return nullptr; 
    }
    virtual void addSymbolMember(Dto::SymbolPtr parentSymbol, Dto::SymbolPtr memberSymbol) {
    }
    virtual long getSymbolMemberCount(Dto::SymbolPtr symbol) { 
        return 0L; 
    }
    virtual void toBinary(Dto::SymbolPtr s, uint64_t** ret) {
        unsigned long count = s->memberCount();
        (*ret) = new uint64_t[count];
        for(unsigned long i=0; i<count; i++) {
            (*ret)[i] = s->getMember(i)->getId();
        }
    }
    
    virtual Dto::RecollectionPtr newRecollection(Dto::SymbolPtr symbol) { 
        return nullptr;
    }
    virtual void addRecollectionException(Dto::RecollectionPtr recollection, Dto::RecExceptOpsEnum operationId, int symbolIdx, Dto::SymbolPtr symbol) {
    }
};

}}

#endif /* defined(__NLPGraph__model__) */
