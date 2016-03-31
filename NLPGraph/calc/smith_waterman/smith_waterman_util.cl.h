const char *kSmithWatermanUtilOpenCLSource = BOOST_COMPUTE_STRINGIZE_SOURCE(

typedef struct smith_waterman {
    uint flags;
    __global char  *logOut;
    uint logLength;
    uint refWidth;
    uint opWidth;
    uint candCount;
    uint uniqueCount;
    __constant ulong *reference;
    __global ulong *candidates;
    __global ulong *costMatrix;
    __global ulong *distsAndOps;
    __global ulong *uniques
} smith_waterman_type;

uint costMatrix_indexOf(smith_waterman_type *self, ulong id) {
    uint startSearch = self->uniqueCount/2;
    
}

);