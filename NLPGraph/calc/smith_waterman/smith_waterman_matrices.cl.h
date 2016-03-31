const char *kSmithWatermanCreateMatricesOpenCLSource = BOOST_COMPUTE_STRINGIZE_SOURCE(

__kernel void calc_smith_waterman_matrices(
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
}

);
