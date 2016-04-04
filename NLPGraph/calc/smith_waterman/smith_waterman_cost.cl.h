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
        __global ulong *costMatrix,
        __global ulong *distsAndOps,
        __global ulong *uniques
) {
    smith_waterman_type self;
    memset(&self,0,sizeof(smith_waterman_type));
    self.flags = flags;
    self.logOut = logOut;
    self.logLength = logLength;
    self.opWidth = opWidth;
    self.candCount = candCount;
    self.uniqueCount = uniqueCount;
    self.candidates = candidates;
    self.costMatrix = costMatrix;
    self.distAndOps = distAndOps;
    self.uniques = uniques;
    
    costMatrix_getCoordsFor(self,get_global_id(0),self.costMatrixCoords);
}

);