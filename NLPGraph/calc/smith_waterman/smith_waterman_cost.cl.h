const char *kSmithWatermanCostMatrixOpenCLSource = BOOST_COMPUTE_STRINGIZE_SOURCE(

__kernel void calc_smith_waterman_cost_matrix(
        uint flags,
        __global char  *logOut,
        uint logLength,
        uint refWidth,
        uint opWidth,
        uint candCount,
        uint uniqueCount,
        __constant ulong *reference,
        __global ulong *candidates,
        __global long *matrices,
        __global long *costMatrix,
        __global ulong *distsAndOps,
        __global ulong *uniques
) {
    smith_waterman_type self;
    memset(&self,0,sizeof(smith_waterman_type));
    self.flags = flags;
    self.logOut = logOut;
    self.logLength = logLength;
    self.refWidth = refWidth;
    self.opWidth = opWidth;
    self.candCount = candCount;
    self.uniqueCount = uniqueCount;
    self.reference = reference;
    self.candidates = candidates;
    self.matrices = matrices;
    self.costMatrix = costMatrix;
    self.distsAndOps = distsAndOps;
    self.uniques = uniques;
    self.globalOffset = get_global_id(0);
    
    costMatrix_getCoordsFor(&self,self.globalOffset,self.costMatrixCoords);
    printf("offset %lu cost matrix coords: %lu, %lu\n",self.globalOffset,self.costMatrixCoords[0],self.costMatrixCoords[1]);
}

);