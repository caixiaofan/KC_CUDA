#include "kc_gpu.h"
#include <iostream>
#include <vector>
#include <omp.h>
#include <algorithm>
#include <assert.h>
#include <thrust/sort.h>
#include <future>
#include "mapfile.hpp"

using namespace std;
const int BLOCKSIZE = 32;
#define FULL_MASK 0xffffffff
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }

const uint32_t NUM_THREADS_THIS_COMPUTER = omp_get_num_procs();
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess){
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

template <typename T> __device__ void inline swap_test_device1(T& a, T& b)
{
    T c(a); a=b; b=c;
}
// 这个init函数输出的内容其实有误，不过init之后其他函数的GPU运行时间会相对正常 //
void initGPU(const Bint_t edge_num, const uint32_t N)
{
    cudaDeviceProp deviceProp;
    gpuErrchk( cudaGetDeviceProperties(&deviceProp, 0) );
    size_t free_t, total_t, used_t;
    cudaMemGetInfo(&free_t,&total_t);
    used_t = total_t - free_t;
    uint64_t device_memory = deviceProp.totalGlobalMem;
    // nbr, fa, nbr_tc, nbr_mask
    uint64_t nbr_size = 1ll * edge_num * (sizeof(Edge_t) + sizeof(Bint_t) * 3 + sizeof(uint32_t) * 3);
    // nbr_u, deg
    nbr_size += 1ll * (N+1) * (sizeof(Bint_t) + sizeof(uint32_t));
    // 在这里假定数据不会超过显存容量
    // 计算所需的显存
#ifdef DEBUG_DF
    cout << "Global memory:"
         << device_memory / 1024 / 1024
         << "MB; Require: "
         << nbr_size / 1024 / 1024
         << " MB" << endl
         << "Already used memory: " << ((uint64_t)used_t / 1024 / 1024) << "MB." << endl
         << "Free memory: " << ((uint64_t)free_t / 1024 / 1024) << "MB." << endl;
#endif
}

__device__ unsigned long long dev_sum;
__device__ unsigned long long dev_nowEdge;
__device__ uint32_t dev_nowNode;
__device__ uint32_t dev_maxDeg;

// 此函数还有大量优化空间 //
__device__ void intersection_with_mask(const uint32_t *nbr, Bint_t li, Bint_t ri, Bint_t ln, Bint_t rn, 
                                       uint32_t *nbr_tc_double, uint32_t *nbr_mask_double, const uint32_t st,
                                       const uint32_t k)
{
    __shared__ uint32_t lblock[BLOCKSIZE];
    __shared__ uint32_t rblock[BLOCKSIZE];
    __shared__ uint32_t lmask[BLOCKSIZE];
    __shared__ uint32_t rmask[BLOCKSIZE];
    __shared__ int lpos[BLOCKSIZE];
    __shared__ int rpos[BLOCKSIZE];
    // 在此处统计 //
    
    const uint32_t *lbases = nbr + li;
    const uint32_t *rbases = nbr + ri;
    
    uint32_t i = 0, j = 0, flag = 0;
    uint32_t lsize = 0, rsize = 0;
    int p1 = 0;
    
    while (i < ln && j < rn) {

        lsize = min(uint32_t(ln - i), BLOCKSIZE);
        rsize = min(uint32_t(rn - j), BLOCKSIZE);
        
        lblock[threadIdx.x] = rblock[threadIdx.x] = uint32_t(-1);
        lmask[threadIdx.x] = rmask[threadIdx.x] = 0;
        lpos[threadIdx.x] = threadIdx.x + 1;
        rpos[threadIdx.x] = -threadIdx.x - 1;

        if(i + threadIdx.x < ln) {
            lblock[threadIdx.x] = lbases[i + threadIdx.x];
            lmask[threadIdx.x] = nbr_mask_double[li + i + threadIdx.x];
        }
        if(j + threadIdx.x < rn) {
            rblock[threadIdx.x] = rbases[j + threadIdx.x];
            rmask[threadIdx.x] = nbr_mask_double[ri + j + threadIdx.x];
        }        

        __threadfence_block();
        // 在此处读取完毕 //
        uint32_t llast = lblock[lsize - 1];
        uint32_t rlast = rblock[rsize - 1];
        
        if(lblock[threadIdx.x] > rblock[threadIdx.x])
        {
            swap_test_device1(lblock[threadIdx.x], rblock[threadIdx.x]);
            swap_test_device1(lpos[threadIdx.x], rpos[threadIdx.x]);
            swap_test_device1(lmask[threadIdx.x], rmask[threadIdx.x]);
        }
        __threadfence_block();
        for(int k = BLOCKSIZE - 1; k>=1; k=(k+1)/2-1)
        {
            if(threadIdx.x + (k + 1) / 2 < BLOCKSIZE & rblock[threadIdx.x] > lblock[threadIdx.x + (k + 1) / 2])
            {
                swap_test_device1(lblock[threadIdx.x + (k + 1) / 2], rblock[threadIdx.x]);
                swap_test_device1(lpos[threadIdx.x + (k + 1) / 2], rpos[threadIdx.x]);
                swap_test_device1(lmask[threadIdx.x + (k + 1) / 2], rmask[threadIdx.x]);
            }
            __threadfence_block();
        }
        
        flag = (lblock[threadIdx.x] == rblock[threadIdx.x]) & lmask[threadIdx.x] >= st & rmask[threadIdx.x] >= st;
        flag *= 2 - (lmask[threadIdx.x] == st) - (rmask[threadIdx.x] == st);
        p1 = rpos[threadIdx.x];
        if(flag == 0)
        {
            flag = (threadIdx.x > 0) & (lblock[threadIdx.x] == rblock[threadIdx.x - 1]) & (lmask[threadIdx.x] >= st) & (rmask[threadIdx.x - 1] >= st);
            flag *= 2 - (lmask[threadIdx.x] == st) - (rmask[threadIdx.x - 1] == st);
            p1 = rpos[threadIdx.x - 1];
        }
        // 此处的atomicSub 可以想办法先写到缓存，之后批量来做 //
        if(flag > 0)
        {
            uint32_t tc_1, tc_2;
            if(p1 > 0)
            {
                tc_1 =  i + p1 - 1;
                tc_2 =  j - lpos[threadIdx.x] - 1;
            }
            else
            {
                tc_1 =  i + lpos[threadIdx.x] - 1;
                tc_2 =  j - p1 - 1;
            }
            atomicSub(nbr_tc_double + tc_1 + li, flag);
            atomicSub(nbr_tc_double + tc_2 + ri, flag);
        }
        
        if(llast >= rlast)
        {
            j += rsize;
            
        }
        if(llast <= rlast)
        {
            i += lsize;
        }
    }
}
// 重点需要优化的函数 //
__global__ void __delete_edge(const uint32_t *nbr_direct_start, const uint32_t *nbr_direct, const Bint_t edge_num, const uint32_t *nbr, const Bint_t *nbr_u, 
                              uint32_t *nbr_tc_double, uint32_t *nbr_mask_double, const uint32_t *nbr_mask, const uint32_t st, const uint32_t k)
{
    __shared__ Bint_t EdgeI;
    __shared__ Bint_t EdgeEnd;
    if(threadIdx.x == 0)
    {
        EdgeI = EdgeEnd = 0;
    }
    while(true){
        // 动态调度 //
        
        if(threadIdx.x == 0){
            if(++EdgeI >= EdgeEnd){
                EdgeI = atomicAdd(&dev_nowEdge, 1);
                EdgeEnd = min(edge_num, EdgeI + 1);
            }
        }
        const Bint_t i = __shfl_sync(FULL_MASK, EdgeI, 0);
        if(i >= edge_num) break;
        if(nbr_mask[i] != st) 
            continue;
        const uint32_t u = nbr_direct_start[i];
        const uint32_t v = nbr_direct[i];
        Bint_t li = nbr_u[u];
        Bint_t ln = nbr_u[u+1] - li;
        Bint_t ri = nbr_u[v];
        Bint_t rn = nbr_u[v+1] - ri;
        intersection_with_mask(nbr, li, ri, ln, rn, nbr_tc_double, nbr_mask_double, st, k);
    }
}

__device__ void intersection_direct(const uint32_t *bases, uint32_t *tc, const Bint_t line_pos, const Bint_t li, const Bint_t ri, const Bint_t ln, const Bint_t rn)
{
    __shared__ uint32_t lblock[BLOCKSIZE];
    __shared__ uint32_t rblock[BLOCKSIZE];
    __shared__ int lpos[BLOCKSIZE];
    __shared__ int rpos[BLOCKSIZE];
    __shared__ uint32_t lstatus[BLOCKSIZE];
    __shared__ uint32_t rstatus[BLOCKSIZE];
    __shared__ uint32_t my_sum;

    uint32_t lsize = 0, rsize = 0;
    uint32_t i = 0, j = 0, sum = 0;
    int p1 = 0;
    const uint32_t *lbases = bases + li;
    const uint32_t *rbases = bases + ri;
    if(threadIdx.x == 0)
        my_sum = 0;
    lstatus[threadIdx.x] = 0;
    rstatus[threadIdx.x] = 0;
    while (i < ln && j < rn) {

        lsize = min(uint32_t(ln - i), BLOCKSIZE);
        rsize = min(uint32_t(rn - j), BLOCKSIZE);
        
        // 初始化成最大值-1 //
        lblock[threadIdx.x] = rblock[threadIdx.x] = uint32_t(-1);
        lpos[threadIdx.x] = threadIdx.x + 1;
        rpos[threadIdx.x] = -threadIdx.x - 1;

        if(i + threadIdx.x < ln) {
            lblock[threadIdx.x] = lbases[i + threadIdx.x];
        }
        if(j + threadIdx.x < rn) {
            rblock[threadIdx.x] = rbases[j + threadIdx.x];
        }
        __threadfence_block();
        
        uint32_t llast = lblock[lsize - 1];
        uint32_t rlast = rblock[rsize - 1];
        
        if(lblock[threadIdx.x] > rblock[threadIdx.x])
        {
            swap_test_device1(lblock[threadIdx.x], rblock[threadIdx.x]);
            swap_test_device1(lpos[threadIdx.x], rpos[threadIdx.x]);
        }
        __threadfence_block();
        for(int k = BLOCKSIZE - 1; k>=1; k=(k+1)/2-1)
        {
            if(threadIdx.x + (k + 1) / 2 < BLOCKSIZE & rblock[threadIdx.x] > lblock[threadIdx.x + (k + 1) / 2])
            {
                swap_test_device1(lblock[threadIdx.x + (k + 1) / 2], rblock[threadIdx.x]);
                swap_test_device1(lpos[threadIdx.x + (k + 1) / 2], rpos[threadIdx.x]);
            }
            __threadfence_block();
        }
        
        uint32_t flag = (lblock[threadIdx.x] == rblock[threadIdx.x] & lblock[threadIdx.x] != uint32_t(-1));
        p1 = rpos[threadIdx.x];
        
        if(!flag)
        {
            flag = threadIdx.x > 0 & lblock[threadIdx.x] == rblock[threadIdx.x - 1] & lblock[threadIdx.x] != uint32_t(-1);
            p1 = rpos[threadIdx.x - 1];
        }
        if(flag)
        {
            uint32_t tc_1, tc_2;
            if(p1 > 0)
            {
                tc_1 = i + p1 - 1;
                tc_2 = j - lpos[threadIdx.x] - 1;
            }
            else
            {
                tc_1 = i + lpos[threadIdx.x] - 1;
                tc_2 = j - p1 - 1;
            }
            
            atomicOr(&lstatus[(i/BLOCKSIZE)&(BLOCKSIZE-1)], 1 << (tc_1 - i));
            atomicOr(&rstatus[(j/BLOCKSIZE)&(BLOCKSIZE-1)], 1 << (tc_2 - j));
        }
        sum += flag;
        if(llast >= rlast) 
        {
            j += rsize;
            if(rsize == BLOCKSIZE & ((j / BLOCKSIZE) & (BLOCKSIZE-1)) == 0)
            {
                uint32_t j_b = j - BLOCKSIZE * BLOCKSIZE;
                for(uint32_t k = 0; k < BLOCKSIZE; k ++)
                {
                    if((rstatus[k] >> threadIdx.x) & 1)
                    {
                        atomicAdd(tc + ri + j_b + k * BLOCKSIZE + threadIdx.x, 2);
                    }
                }
                rstatus[threadIdx.x] = 0;
            }
        }

        if(llast <= rlast)
        {
            i += lsize;
            if(lsize == BLOCKSIZE & ((i / BLOCKSIZE) & (BLOCKSIZE-1)) == 0)
            {
                uint32_t i_b = i - BLOCKSIZE * BLOCKSIZE;
                
                for(uint32_t k = 0; k < BLOCKSIZE; k ++)
                {
                    if((lstatus[k] >> threadIdx.x) & 1)
                    {
                        atomicAdd(tc + li + i_b + k * BLOCKSIZE + threadIdx.x, 2);
                    }
                }
                lstatus[threadIdx.x] = 0;
            }
        }

    }
    
    atomicAdd(&my_sum, sum);
    if(threadIdx.x == 0)
    {
        atomicAdd(tc+line_pos, my_sum << 1);
    }

    uint32_t j_b = j / BLOCKSIZE / BLOCKSIZE * BLOCKSIZE;
    uint32_t k_max = j / BLOCKSIZE - j_b + 1;
    j_b *= BLOCKSIZE;
    for(uint32_t k = 0; k < k_max; k++)
    {
        if((rstatus[k] >> threadIdx.x) & 1)
        {
            atomicAdd(tc + ri + j_b + k * BLOCKSIZE + threadIdx.x, 2);
        }
    }

    uint32_t i_b = i / BLOCKSIZE / BLOCKSIZE * BLOCKSIZE;
    uint32_t k_max1 = i / BLOCKSIZE - i_b + 1;
    i_b *= BLOCKSIZE;
    for(uint32_t k = 0; k < k_max1; k++)
    {
        if((lstatus[k] >> threadIdx.x) & 1)
        {
            atomicAdd(tc + li + i_b + k * BLOCKSIZE + threadIdx.x, 2);
        }
    }
}

__device__ void intersection_direct_with_tc_limit(const uint32_t *bases, uint32_t *tc, const Bint_t line_pos, const Bint_t li, const Bint_t ri, const Bint_t ln, const Bint_t rn,
                                                  const int tc_limit)
{
    __shared__ uint32_t lblock[BLOCKSIZE];
    __shared__ uint32_t rblock[BLOCKSIZE];
    __shared__ int lpos[BLOCKSIZE];
    __shared__ int rpos[BLOCKSIZE];
    __shared__ uint32_t lstatus[BLOCKSIZE];
    __shared__ uint32_t rstatus[BLOCKSIZE];
    __shared__ uint32_t my_sum;

    uint32_t lsize = 0, rsize = 0;
    uint32_t i = 0, j = 0, break_flag = 0;
    int p1 = 0;
    const uint32_t *lbases = bases + li;
    const uint32_t *rbases = bases + ri;
    if(threadIdx.x == 0)
        my_sum = 0;
    lstatus[threadIdx.x] = 0;
    rstatus[threadIdx.x] = 0;

    while (i < ln && j < rn) {

        lsize = min(uint32_t(ln - i), BLOCKSIZE);
        rsize = min(uint32_t(rn - j), BLOCKSIZE);
        
        __threadfence_block();
        int now = 0;
        if(threadIdx.x == 0)
        {
            now = min(ln-i, rn-j) + my_sum;
        }
        now = __shfl_sync(FULL_MASK, now, 0);
        if(now < tc_limit)
        {
            break_flag = 1;
            break;
        }
        // 初始化成最大值-1 //
        lblock[threadIdx.x] = rblock[threadIdx.x] = uint32_t(-1);
        lpos[threadIdx.x] = threadIdx.x + 1;
        rpos[threadIdx.x] = -threadIdx.x - 1;

        if(i + threadIdx.x < ln) {
            lblock[threadIdx.x] = lbases[i + threadIdx.x];
        }
        if(j + threadIdx.x < rn) {
            rblock[threadIdx.x] = rbases[j + threadIdx.x];
        }
        __threadfence_block();
        
        uint32_t llast = lblock[lsize - 1];
        uint32_t rlast = rblock[rsize - 1];
        
        if(lblock[threadIdx.x] > rblock[threadIdx.x])
        {
            swap_test_device1(lblock[threadIdx.x], rblock[threadIdx.x]);
            swap_test_device1(lpos[threadIdx.x], rpos[threadIdx.x]);
        }
        __threadfence_block();
        for(int k = BLOCKSIZE - 1; k>=1; k=(k+1)/2-1)
        {
            if(threadIdx.x + (k + 1) / 2 < BLOCKSIZE & rblock[threadIdx.x] > lblock[threadIdx.x + (k + 1) / 2])
            {
                swap_test_device1(lblock[threadIdx.x + (k + 1) / 2], rblock[threadIdx.x]);
                swap_test_device1(lpos[threadIdx.x + (k + 1) / 2], rpos[threadIdx.x]);
            }
            __threadfence_block();
        }
        
        uint32_t flag = (lblock[threadIdx.x] == rblock[threadIdx.x] & lblock[threadIdx.x] != uint32_t(-1));
        p1 = rpos[threadIdx.x];
        
        if(!flag)
        {
            flag = threadIdx.x > 0 & lblock[threadIdx.x] == rblock[threadIdx.x - 1] & lblock[threadIdx.x] != uint32_t(-1);
            p1 = rpos[threadIdx.x - 1];
        }
        if(flag)
        {
            uint32_t tc_1, tc_2;
            if(p1 > 0)
            {
                tc_1 = i + p1 - 1;
                tc_2 = j - lpos[threadIdx.x] - 1;
            }
            else
            {
                tc_1 = i + lpos[threadIdx.x] - 1;
                tc_2 = j - p1 - 1;
            }
            atomicOr(&lstatus[(i/BLOCKSIZE)&(BLOCKSIZE-1)], 1 << (tc_1 - i));
            atomicOr(&rstatus[(j/BLOCKSIZE)&(BLOCKSIZE-1)], 1 << (tc_2 - j));
        }
        flag = __ballot_sync(FULL_MASK, flag);
        if(threadIdx.x == 0)
        {
            my_sum += __popc(flag);
        }
        if(llast >= rlast)
        {
            j += rsize;
            if(rsize == BLOCKSIZE & ((j / BLOCKSIZE) & (BLOCKSIZE-1)) == 0)
            {
                uint32_t j_b = j - BLOCKSIZE * BLOCKSIZE;
                for(uint32_t k = 0; k < BLOCKSIZE; k ++)
                {
                    if((rstatus[k] >> threadIdx.x) & 1)
                    {
                        atomicAdd(tc + ri + j_b + k * BLOCKSIZE + threadIdx.x, 2);
                    }
                }
                rstatus[threadIdx.x] = 0;
            }
            
        }
        if(llast <= rlast)
        {
            i += lsize;
            if(lsize == BLOCKSIZE & ((i / BLOCKSIZE) & (BLOCKSIZE-1)) == 0)
            {
                uint32_t i_b = i - BLOCKSIZE * BLOCKSIZE;
                
                for(uint32_t k = 0; k < BLOCKSIZE; k ++)
                {
                    if((lstatus[k] >> threadIdx.x) & 1)
                    {
                        atomicAdd(tc + li + i_b + k * BLOCKSIZE + threadIdx.x, 2);
                    }
                }
                lstatus[threadIdx.x] = 0;
            }
        }
    }
    if(threadIdx.x == 0 && break_flag == 0)
    {
        atomicAdd(tc+line_pos, my_sum << 1);
    }
    if(break_flag == 0)
    {
        uint32_t j_b = j / BLOCKSIZE / BLOCKSIZE * BLOCKSIZE;
        uint32_t k_max = j / BLOCKSIZE - j_b + 1;
        j_b *= BLOCKSIZE;
        for(uint32_t k = 0; k < k_max; k++)
        {
            if((rstatus[k] >> threadIdx.x) & 1)
            {
                atomicAdd(tc + ri + j_b + k * BLOCKSIZE + threadIdx.x, 2);
            }
        }

        uint32_t i_b = i / BLOCKSIZE / BLOCKSIZE * BLOCKSIZE;
        uint32_t k_max1 = i / BLOCKSIZE - i_b + 1;
        i_b *= BLOCKSIZE;
        for(uint32_t k = 0; k < k_max1; k++)
        {
            if((lstatus[k] >> threadIdx.x) & 1)
            {
                atomicAdd(tc + li + i_b + k * BLOCKSIZE + threadIdx.x, 2);
            }
        }
    }
}

__global__ void __tricount_direct(const uint32_t *nbr_direct, const uint32_t N, const Bint_t *nbr_u_direct, uint32_t *nbr_tc)
{
    __shared__ uint32_t NodeI;
    __shared__ uint32_t NodeEnd;
    if(threadIdx.x == 0)
    {
        NodeI = NodeEnd = 0;
    }
    while(true){
        // 动态调度 //
        
        if(threadIdx.x == 0){
            if(++NodeI >= NodeEnd){
                NodeI = atomicAdd(&dev_nowNode, 1);
                NodeEnd = min(N, NodeI + 1);
            }
        }
        const uint32_t u = __shfl_sync(FULL_MASK, NodeI, 0);
        if(u >= N) break;
        const Bint_t r = nbr_u_direct[u+1];
        for(Bint_t j = nbr_u_direct[u]; j < r; j++)
        {
            const uint32_t v = nbr_direct[j];
            Bint_t li = nbr_u_direct[u];
            Bint_t ln = nbr_u_direct[u+1] - li;
            Bint_t ri = nbr_u_direct[v];
            Bint_t rn = nbr_u_direct[v+1] - ri;
            intersection_direct(nbr_direct, nbr_tc, j, li, ri, ln, rn);
        }
    }
}

__global__ void __tricount_direct_with_tc_limit(const uint32_t *nbr_direct, const uint32_t N, const Bint_t *nbr_u_direct, uint32_t *nbr_tc,
                                                uint32_t *deg_rev, const uint32_t tc_limit)
{
    __shared__ uint32_t NodeI;
    __shared__ uint32_t NodeEnd;
    int limit = 0;
    if(threadIdx.x == 0)
    {
        NodeI = NodeEnd = 0;
    }
    while(true){
        // 动态调度 //
        
        if(threadIdx.x == 0){
            if(++NodeI >= NodeEnd){
                NodeI = atomicAdd(&dev_nowNode, 1);
                NodeEnd = min(N, NodeI + 1);
            }
        }
        const uint32_t u = __shfl_sync(FULL_MASK, NodeI, 0);
        if(u >= N) break;
        const Bint_t r = nbr_u_direct[u+1];
        for(Bint_t j = nbr_u_direct[u]; j < r; j++)
        {
            const uint32_t v = nbr_direct[j];
            Bint_t li = nbr_u_direct[u];
            Bint_t ln = nbr_u_direct[u+1] - li;
            Bint_t ri = nbr_u_direct[v];
            Bint_t rn = nbr_u_direct[v+1] - ri;
            
            if(threadIdx.x == 0)
            {
                uint32_t p = deg_rev[u];
                limit = tc_limit - p;
            }
            limit = __shfl_sync(FULL_MASK, limit - nbr_tc[j] / 2, 0);
            intersection_direct_with_tc_limit(nbr_direct, nbr_tc, j, li, ri, ln, rn, limit);
            // 在树中删除节点 v, pos[j] + 1 //
            if(threadIdx.x == 0)
            {
                atomicSub(&deg_rev[v], 1);
            }
           
        }
    }
}

void tricount_direct(const Bint_t *nbr_u_direct, const uint32_t *nbr_direct, const uint32_t N, const Bint_t *nbr_u, const uint32_t *nbr, 
                           uint32_t *nbr_tc, const Bint_t *left_son, const Bint_t *right_son, uint32_t *nbr_tc_double, uint32_t tc_limit)
{
    int numBlocks = 4096;
    uint32_t *dev_nbr_direct, *dev_nbr_tc;
    Bint_t *dev_nbr_u_direct;
    uint32_t *dev_deg_rev;
    // nbr_u[N]刚刚好就是当前的edgenum //
    Bint_t edge_num = nbr_u_direct[N];

    memset(nbr_tc, 0, size_nbr_tc);
    gpuErrchk(cudaMalloc((void**)&dev_nbr_tc, size_nbr_tc));
    gpuErrchk(cudaMalloc((void**)&dev_nbr_u_direct, size_nbr_u_direct));
    gpuErrchk(cudaMalloc((void**)&dev_nbr_direct, size_nbr_direct));

    gpuErrchk(cudaMemcpy(dev_nbr_tc, nbr_tc, size_nbr_tc, cudaMemcpyHostToDevice) );
    gpuErrchk(cudaMemcpy(dev_nbr_u_direct, nbr_u_direct, size_nbr_u_direct, cudaMemcpyHostToDevice) );
    gpuErrchk(cudaMemcpy(dev_nbr_direct, nbr_direct, size_nbr_direct, cudaMemcpyHostToDevice) );

    uint64_t tmp = 0;
    gpuErrchk(cudaMemcpyToSymbol(dev_nowNode, &tmp, sizeof(uint32_t)) );
    // nbr_u_small 记录N个树状数组的起始位置 //
    uint32_t *deg_rev;
    if(tc_limit != 0)
    {
        deg_rev = new uint32_t[N]();
#pragma omp parallel for
        for(uint32_t i = 0; i < N; i++)
        {
            deg_rev[i] = nbr_u[i+1] + nbr_u_direct[i] - nbr_u[i] - nbr_u_direct[i + 1];
        }
        uint64_t size_deg_rev = size_deg;
        gpuErrchk(cudaMalloc((void**)&dev_deg_rev, size_deg_rev));
        gpuErrchk(cudaMemcpy(dev_deg_rev, deg_rev, size_deg_rev, cudaMemcpyHostToDevice) );
    }
    // copy data to device /// 
    if(tc_limit == 0)
    {
        __tricount_direct<<<numBlocks, BLOCKSIZE>>>(dev_nbr_direct, N, dev_nbr_u_direct, dev_nbr_tc);
    }
    else
    {
        numBlocks = 4096;
        __tricount_direct_with_tc_limit<<<numBlocks, BLOCKSIZE>>>(dev_nbr_direct, N, dev_nbr_u_direct, dev_nbr_tc, dev_deg_rev, tc_limit);
    }
    gpuErrchk(cudaPeekAtLastError() );
    gpuErrchk(cudaDeviceSynchronize() );
    gpuErrchk(cudaMemcpy(nbr_tc, dev_nbr_tc, size_nbr_tc, cudaMemcpyDeviceToHost) );

    cudaFree(dev_nbr_tc);
    cudaFree(dev_nbr_u_direct);
    cudaFree(dev_nbr_direct);
    // 删除三角形所用到的数组 //
    if(tc_limit != 0)
    {
        delete deg_rev;
        cudaFree(dev_deg_rev);
    }
    if(tc_limit == 0)
    {
#pragma omp parallel for num_threads(NUM_THREADS_THIS_COMPUTER)
        for(Bint_t i = 0; i < edge_num; ++i)
        {
            nbr_tc_double[left_son[i]] = nbr_tc[i];
            nbr_tc_double[right_son[i]] = nbr_tc[i];
        }
    }
}

__global__ void __delete_node(const uint32_t __restrict__ *nbr, const Bint_t __restrict__ *nbr_u, uint32_t *mask, uint32_t *deg, const uint32_t N, const uint32_t k)
{
    __shared__ uint32_t NodeI;
    __shared__ uint32_t NodeEnd;
    uint32_t sum = 0;
    if(threadIdx.x == 0)
    {
        NodeI = NodeEnd = 0;
    }
    while(true){
        // 动态调度 //
        
        if(threadIdx.x == 0){
            if(++NodeI >= NodeEnd){
                NodeI = atomicAdd(&dev_nowNode, 8);
                NodeEnd = min(N, NodeI + 8);
            }
        }
        const uint32_t u = __shfl_sync(FULL_MASK, NodeI, 0);
        if(u >= N) break;
        if(mask[u]) continue;
        if(deg[u] >= k) continue;
        
        const Bint_t r = nbr_u[u+1];
        for(Bint_t j = nbr_u[u]; j < r; j+=BLOCKSIZE)
        {
            if(j + threadIdx.x < r)
            {
                const uint32_t v = nbr[j + threadIdx.x];
                if(mask[v] == 0)
                    atomicSub(&deg[v], 1);
            }
        }
        sum++;
        if(threadIdx.x == 0)
        {
            mask[u] = 1;
        }
    }
    if(threadIdx.x == 0)
        atomicAdd(&dev_sum, sum);
}

__global__ void __cout_deg(const uint32_t __restrict__ *nbr, const Bint_t __restrict__ *nbr_u, const uint32_t *mask, uint32_t *deg, const uint32_t N)
{
    __shared__ uint32_t NodeI;
    __shared__ uint32_t NodeEnd;
    if(threadIdx.x == 0)
    {
        NodeI = NodeEnd = 0;
    }
    while(true){
        // 动态调度 //
        if(threadIdx.x == 0){
            if(++NodeI >= NodeEnd){
                NodeI = atomicAdd(&dev_nowNode, 8);
                NodeEnd = min(N, NodeI + 8);
            }
        }
        const uint32_t u = __shfl_sync(FULL_MASK, NodeI, 0);
        if(u >= N) break;
        if(mask[u]) continue;
        
        const Bint_t r = nbr_u[u+1];
        for(Bint_t j = nbr_u[u]; j < r; j+=BLOCKSIZE)
        {
            if(j + threadIdx.x < r)
            {
                const uint32_t v = nbr[j + threadIdx.x];
                if(mask[v] == 0)
                    atomicAdd(&deg[v], 1);
            }
        }
    }
}

__global__ void __get_max_deg(const uint32_t N, const uint32_t *deg)
{
    uint32_t blockSize = blockDim.x * gridDim.x;
    uint32_t tid = threadIdx.x + blockIdx.x * blockDim.x;
    __shared__ uint32_t my_max;
    uint32_t num = 0;
    if(threadIdx.x == 0)
        my_max = 0;
    for(Bint_t i = tid; i < N; i += blockSize){
        num = max(deg[tid], num);
    }
    atomicMax(&my_max, num);
    if(threadIdx.x == 0)
    {
        atomicMax(&dev_maxDeg, my_max);
    }
}

__global__ void __mask_node(const uint32_t N, const uint32_t *deg, uint32_t *mask, const uint32_t k)
{
    uint32_t blockSize = blockDim.x * gridDim.x;
    uint32_t tid = threadIdx.x + blockIdx.x * blockDim.x;
    uint32_t sum = 0;
    __shared__ uint32_t my_sum;
    if(threadIdx.x == 0)
        my_sum = 0;
    for(Bint_t i = tid; i < N; i += blockSize){
        if(mask[i] == 0 & deg[i] < k)
        {
            mask[i] = 1;
            sum++;
        }
    }
    if(sum != 0)
        atomicAdd(&my_sum, sum);
    if(threadIdx.x == 0 & my_sum != 0)
    {
        atomicAdd(&dev_sum, my_sum);
    }
}

uint32_t kmax_core_gpu(const uint32_t N, const Bint_t *nbr_u, const uint32_t *nbr, const uint32_t *deg_origin,
                       uint32_t *&mask, uint32_t Launch_start)
{
    double avg_deg = nbr_u[N]*1.0 / N;
    uint32_t k = 0;
    uint32_t k_step;
    uint32_t node_delete = 0, node_delete_backup = 0;
    uint32_t numBlocks = 4096;
    
    Bint_t edge_num = nbr_u[N] / 2;
    uint32_t *deg = new uint32_t[N];
    uint32_t *dev_deg, *dev_nbr, *dev_mask;
    Bint_t *dev_nbr_u;
    const uint64_t tmp = 0;

    gpuErrchk(cudaMalloc((void**)&dev_deg, size_deg));
    gpuErrchk(cudaMalloc((void**)&dev_nbr, size_nbr));
    gpuErrchk(cudaMalloc((void**)&dev_mask, size_mask));
    gpuErrchk(cudaMalloc((void**)&dev_nbr_u, size_nbr_u));

    gpuErrchk(cudaMemcpy(dev_deg, deg_origin, size_deg, cudaMemcpyHostToDevice) );
    gpuErrchk(cudaMemcpy(dev_nbr, nbr, size_nbr, cudaMemcpyHostToDevice) );
    gpuErrchk(cudaMemcpy(dev_mask, mask, size_mask, cudaMemcpyHostToDevice) );
    gpuErrchk(cudaMemcpy(dev_nbr_u, nbr_u, size_nbr_u, cudaMemcpyHostToDevice) );
    if(Launch_start != uint32_t(-1))
    {
        k = Launch_start - 1;
        k_step = 1;
    }
    else
    {
        gpuErrchk(cudaMemcpyToSymbol(dev_maxDeg, &tmp, sizeof(uint32_t)) );
        __get_max_deg<<<numBlocks, BLOCKSIZE>>>(N, dev_deg);
        uint32_t max_deg = 0;
        gpuErrchk(cudaMemcpyFromSymbol(&max_deg, dev_maxDeg, sizeof(uint32_t)) );
        k_step = max(uint32_t(K_CORE_STEP_MIDDLE * 4), uint32_t(max_deg / avg_deg / 5));
    }
    
    uint64_t flag = 0;
    uint32_t first_time = 0;
    while(node_delete < N)
    {
        if(k_step > 1)
        {
            node_delete_backup = node_delete;
            gpuErrchk(cudaMemcpy(deg, dev_deg, size_deg, cudaMemcpyDeviceToHost) );
        }
        gpuErrchk(cudaMemcpy(mask, dev_mask, size_mask, cudaMemcpyDeviceToHost) );
        k += k_step;
        while(true)
        {
            if(first_time == 0 && ((Launch_start != uint32_t(-1) and Launch_start > K_CORE_STEP_LARGE) or k_step > K_CORE_STEP_LARGE))
            {
                gpuErrchk(cudaMemcpyToSymbol(dev_sum, &tmp, sizeof(uint64_t)) );
                __mask_node<<<numBlocks, BLOCKSIZE>>>(N, dev_deg, dev_mask, k);
                gpuErrchk(cudaPeekAtLastError() );
                gpuErrchk(cudaDeviceSynchronize() );
                gpuErrchk(cudaMemcpyFromSymbol(&flag, dev_sum, sizeof(uint64_t)) );
                node_delete += flag;
                if(flag == 0)
                    break;
                gpuErrchk(cudaMemset(dev_deg, 0, size_deg));
                
                gpuErrchk(cudaMemcpyToSymbol(dev_nowNode, &tmp, sizeof(uint32_t)) );
                __cout_deg<<<numBlocks, BLOCKSIZE>>>(dev_nbr, dev_nbr_u, dev_mask, dev_deg, N);
                gpuErrchk(cudaPeekAtLastError() );
                gpuErrchk(cudaDeviceSynchronize() );
                first_time = 1;
            }
            
            
            gpuErrchk(cudaMemcpyToSymbol(dev_nowNode, &tmp, sizeof(uint32_t)) );
            gpuErrchk(cudaMemcpyToSymbol(dev_sum, &tmp, sizeof(uint64_t)) );
            // 开始删除度数不满足的节点 //
            __delete_node<<<numBlocks, BLOCKSIZE>>>(dev_nbr, dev_nbr_u, dev_mask, dev_deg, N, k);
            gpuErrchk(cudaPeekAtLastError() );
            gpuErrchk(cudaDeviceSynchronize() );
            gpuErrchk(cudaMemcpyFromSymbol(&flag, dev_sum, sizeof(uint64_t)) );
            node_delete += flag;
            if(flag == 0)
                break;
        }
        if(node_delete == N && k_step > 1)
        {
            k -= k_step;
            if (k_step > K_CORE_STEP_MIDDLE)
                k_step = K_CORE_STEP_MIDDLE;
            else
                k_step = 1;
            node_delete = node_delete_backup;
            gpuErrchk(cudaMemcpy(dev_deg, deg, size_deg, cudaMemcpyHostToDevice) );
            gpuErrchk(cudaMemcpy(dev_mask, mask, size_mask, cudaMemcpyHostToDevice) );
            // 此处将CPU复制回GPU //
        }
        // 大步只走一次 //
        if(k_step > K_CORE_STEP_LARGE)
            k_step = K_CORE_STEP_MIDDLE * 4;
        if(Launch_start != uint32_t(-1))
            break;
    }
    if(Launch_start == uint32_t(-1))
        k--;
    gpuErrchk(cudaPeekAtLastError() );
    gpuErrchk(cudaDeviceSynchronize() );
    if(Launch_start != uint32_t(-1))
    {
        gpuErrchk(cudaMemcpy(mask, dev_mask, size_deg, cudaMemcpyDeviceToHost) );
    }
    cudaFree(dev_deg);
    cudaFree(dev_nbr);
    cudaFree(dev_mask);
    cudaFree(dev_nbr_u);
    delete deg;
    return k==0?0:k-1;
}
// 这个是三角形的数量是break出来的才会进入，初始化的nbr_mask是全都没有标记过的 //
__global__ void __mask_edge(const uint32_t edge_num, const uint32_t *nbr_tc, uint32_t *nbr_mask, const uint32_t st, const uint32_t k)
{
    uint32_t blockSize = blockDim.x * gridDim.x;
    uint32_t tid = threadIdx.x + blockIdx.x * blockDim.x;
    uint32_t sum = 0;
    __shared__ uint32_t my_sum;
    if(threadIdx.x == 0)
        my_sum = 0;
    for(Bint_t i = tid; i < edge_num; i += blockSize){
        uint32_t flag = (nbr_tc[i] < k);
        nbr_mask[i] = flag * (st + 1) - 1;
        sum += flag;
    }
    atomicAdd(&my_sum, sum);
    if(threadIdx.x == 0)
    {
        atomicAdd(&dev_sum, my_sum);
    }
}

__global__ void __mask_edge_with_link(const uint32_t edge_num, const uint32_t *nbr_tc, const uint32_t *nbr_tc_double, uint32_t *nbr_mask,
                                      uint32_t *nbr_mask_double, const uint32_t *left_son, const uint32_t *right_son, const uint32_t st, const uint32_t k)
{
    uint32_t blockSize = blockDim.x * gridDim.x;
    uint32_t tid = threadIdx.x + blockIdx.x * blockDim.x;
    uint32_t sum = 0;
    __shared__ uint32_t my_sum;
    if(threadIdx.x == 0)
        my_sum = 0;
    for(Bint_t i = tid; i < edge_num; i += blockSize){
        if(nbr_mask[i] == uint32_t(-1))
        {
            uint32_t ls = left_son[i];
            uint32_t rs = right_son[i];
            uint32_t flag = (nbr_tc_double[ls] + nbr_tc_double[rs] - nbr_tc[i] < k);
            sum += flag;
            if(flag)
            {
                nbr_mask_double[ls] = st;
                nbr_mask_double[rs] = st;
                nbr_mask[i] = st;
            }
        }
    }
    atomicAdd(&my_sum, sum);
    if(threadIdx.x == 0)
    {
        atomicAdd(&dev_sum, my_sum);
    }
}

// 试一试只使用有向边来统计三角形 //
// 判断ktruss是否合法，now_delete是当前新增的删除的边
void ktruss(const uint32_t N, Bint_t &now_delete, uint32_t &edge_delete, uint32_t accurate_tc,
            const Bint_t *nbr_u_direct, const uint32_t *nbr_direct, const uint32_t *nbr, 
            const Bint_t *nbr_u, uint32_t *nbr_tc, uint32_t *nbr_mask, uint32_t *nbr_tc_double, uint32_t *nbr_mask_double,
            const Bint_t *left_son, const Bint_t *right_son, uint32_t &st, const uint32_t k)
{
    int numBlocks = 4096;
    Bint_t edge_num = nbr_u[N] >> 1;
    
    Bint_t *dev_nbr_u;
    uint32_t *dev_nbr_tc_double, *dev_nbr_mask_double, *dev_nbr, *dev_nbr_direct, *dev_nbr_direct_start;
    uint32_t *dev_nbr_mask, *dev_nbr_tc;
    Bint_t *dev_left_son, *dev_right_son;
    // 构造起点 //
    uint32_t *nbr_direct_start; 
    // 这一步完全可以使用GPU来做 //
    // 只有真正要进行删边的时候再动手 //
    uint64_t first_time = 0;    
    const uint64_t tmp = 0;

    gpuErrchk(cudaMalloc((void**)&dev_nbr_tc, size_nbr_tc));
    gpuErrchk(cudaMalloc((void**)&dev_nbr_mask, size_nbr_mask));
    gpuErrchk(cudaMemcpy(dev_nbr_tc, nbr_tc, size_nbr_tc, cudaMemcpyHostToDevice));

    if(accurate_tc != 0)
    {
        gpuErrchk(cudaMalloc((void**)&dev_left_son, size_left_son));
        gpuErrchk(cudaMalloc((void**)&dev_right_son, size_right_son));
        gpuErrchk(cudaMalloc((void**)&dev_nbr_tc_double, size_nbr_tc_double));
        gpuErrchk(cudaMalloc((void**)&dev_nbr_mask_double, size_nbr_mask_double));

        gpuErrchk(cudaMemcpy(dev_left_son, left_son, size_left_son, cudaMemcpyHostToDevice));
        gpuErrchk(cudaMemcpy(dev_right_son, right_son, size_right_son, cudaMemcpyHostToDevice));
        gpuErrchk(cudaMemcpy(dev_nbr_mask, nbr_mask, size_nbr_mask, cudaMemcpyHostToDevice));
        gpuErrchk(cudaMemcpy(dev_nbr_tc_double, nbr_tc_double, size_nbr_tc_double, cudaMemcpyHostToDevice) );
        gpuErrchk(cudaMemcpy(dev_nbr_mask_double, nbr_mask_double, size_nbr_mask_double, cudaMemcpyHostToDevice));
    }
    while(now_delete < edge_num)
    {
        gpuErrchk(cudaMemcpyToSymbol(dev_sum, &tmp, sizeof(uint64_t)) );
        if(accurate_tc == 0)
        {
            __mask_edge<<<numBlocks, BLOCKSIZE>>>(edge_num, dev_nbr_tc, dev_nbr_mask, st, k);
        }
        else
        {
            __mask_edge_with_link<<<numBlocks, BLOCKSIZE>>>(edge_num, dev_nbr_tc, dev_nbr_tc_double, dev_nbr_mask, dev_nbr_mask_double, dev_left_son, dev_right_son, st, k);
            
        }
        gpuErrchk(cudaMemcpyFromSymbol(&edge_delete, dev_sum, sizeof(Bint_t)) );
        if((edge_delete * 3) >= edge_num or (now_delete + edge_delete) * 2 > edge_num or accurate_tc == 0)
        {
            break;
        }
        if(edge_delete == 0)
            break;
        // 需要的时候再生成，节约时间 //
        if(first_time == 0)
        {
            nbr_direct_start = new uint32_t[edge_num];
            // 基本不花时间, 但是同样可以使用GPU //
#pragma omp parallel for num_threads(NUM_THREADS_THIS_COMPUTER)
            for(uint32_t i = 0; i < N; ++i)
            {
                for(uint32_t j = nbr_u_direct[i]; j < nbr_u_direct[i+1]; ++j)
                {
                    nbr_direct_start[j] = i;
                }
            }
            gpuErrchk(cudaMalloc((void**)&dev_nbr_direct, size_nbr_direct));
            gpuErrchk(cudaMalloc((void**)&dev_nbr_u, size_nbr_u));
            gpuErrchk(cudaMalloc((void**)&dev_nbr, size_nbr));
            
            gpuErrchk(cudaMalloc((void**)&dev_nbr_direct_start, size_nbr_direct_start));
        
            gpuErrchk(cudaMemcpy(dev_nbr_u, nbr_u, size_nbr_u, cudaMemcpyHostToDevice) );
            gpuErrchk(cudaMemcpy(dev_nbr_direct, nbr_direct, size_nbr_direct, cudaMemcpyHostToDevice));
            gpuErrchk(cudaMemcpy(dev_nbr, nbr, size_nbr, cudaMemcpyHostToDevice) );
 
            gpuErrchk(cudaMemcpy(dev_nbr_direct_start, nbr_direct_start, size_nbr_direct_start, cudaMemcpyHostToDevice));
            first_time = 1;
        }
        
        gpuErrchk(cudaMemcpyToSymbol(dev_nowEdge, &tmp, sizeof(uint64_t)) );
        // 删除nbr_mask[i] == st的边 //
        __delete_edge<<<numBlocks, BLOCKSIZE>>>(dev_nbr_direct_start, dev_nbr_direct, edge_num, dev_nbr, dev_nbr_u, dev_nbr_tc_double, dev_nbr_mask_double, dev_nbr_mask, st, k);
        st ++;
        gpuErrchk(cudaPeekAtLastError() );
        gpuErrchk(cudaDeviceSynchronize() );
        now_delete += edge_delete;
        edge_delete = 0;
    }

    gpuErrchk(cudaPeekAtLastError() );
    gpuErrchk(cudaDeviceSynchronize() );
    gpuErrchk(cudaMemcpy(nbr_mask, dev_nbr_mask, size_nbr_mask, cudaMemcpyDeviceToHost) );
    if(accurate_tc != 0)
    {
        gpuErrchk(cudaMemcpy(nbr_mask_double, dev_nbr_mask_double, size_nbr_mask_double, cudaMemcpyDeviceToHost) );
    }
    if(first_time != 0)
    {
        gpuErrchk(cudaMemcpy(nbr_tc_double, dev_nbr_tc_double, size_nbr_tc_double, cudaMemcpyDeviceToHost) );
        cudaFree(dev_nbr);
        cudaFree(dev_nbr_u);
        cudaFree(dev_nbr_direct);
        cudaFree(dev_nbr_direct_start);
        delete nbr_direct_start;
    }
    cudaFree(dev_nbr_mask);
    cudaFree(dev_nbr_tc);
    if(accurate_tc != 0)
    {
        cudaFree(dev_left_son);
        cudaFree(dev_right_son);
        cudaFree(dev_nbr_tc_double);
        cudaFree(dev_nbr_mask_double);
    }
}

// 重新构图 【注意这个函数需要查看多线程是否会变快】
void reconstruct_Graph(Bint_t *&nbr_u_direct, uint32_t *&nbr_direct, Bint_t &edge_num, const uint32_t N, 
                     Bint_t *&nbr_u, uint32_t *&nbr, Bint_t *&fa, Bint_t *&left_son, Bint_t *&right_son,
                     uint32_t *&nbr_mask, uint32_t *&nbr_mask_double, uint32_t *&nbr_tc, uint32_t *&nbr_tc_double, 
                     const uint32_t tc_need, uint32_t st)
{
    uint32_t new_edge_num = 0;
    uint32_t *deg = new uint32_t[N]();
    uint32_t *deg_direct = new uint32_t[N]();
    if(tc_need == 0)
    {
        for(uint32_t i = 0; i < N; i++)
        {
            Bint_t rn = nbr_u_direct[i+1];
            uint32_t tmp = new_edge_num;
            uint32_t &u = i;
            for(uint32_t j = nbr_u_direct[i]; j < rn; ++j)
            {
                uint32_t &v = nbr_direct[j];
                // 对于没有被删除的边 //
                if(nbr_mask[j] > st)
                {
                    nbr_direct[new_edge_num] = v;
                    ++new_edge_num;
                }
            }
            deg[u] += new_edge_num - tmp;
            deg_direct[u] += new_edge_num - tmp;
        }
    }
    else
    {
        for(uint32_t i = 0; i < N; i++)
        {
            Bint_t rn = nbr_u_direct[i+1];
            uint32_t tmp = new_edge_num;
            uint32_t &u = i;
            for(uint32_t j = nbr_u_direct[i]; j < rn; ++j)
            {
                uint32_t &v = nbr_direct[j];
                // 对于没有被删除的边 //
                if(nbr_mask[j] > st)
                {
                    nbr_direct[new_edge_num] = v;
                    nbr_tc[new_edge_num] = nbr_tc_double[left_son[j]] + nbr_tc_double[right_son[j]] - nbr_tc[j];
                    ++new_edge_num;
                }
            }
            deg[u] += new_edge_num - tmp;
            deg_direct[u] += new_edge_num - tmp;
        }
    }
    edge_num = new_edge_num;
#pragma omp parallel for num_threads(NUM_THREADS_THIS_COMPUTER)
    for(Bint_t i = 0; i < edge_num; i++)
    {
#pragma omp atomic
        deg[nbr_direct[i]]++;
    }
    nbr_u[0] = 0; nbr_u_direct[0] = 0;
    for(uint32_t i = 1; i <= N; ++i)
    {
        nbr_u_direct[i] = nbr_u_direct[i-1] + deg_direct[i-1];
        nbr_u[i] = nbr_u[i-1] + deg[i-1];
    }
    
    build_graph_from_direct(nbr_direct, nbr_u_direct, N, nbr, nbr_u, fa, left_son, right_son);
    if(tc_need)
    {
#pragma omp parallel for num_threads(NUM_THREADS_THIS_COMPUTER)
        for(Bint_t i = 0; i < edge_num; ++i)
        {
            nbr_tc_double[left_son[i]] = nbr_tc[i];
            nbr_tc_double[right_son[i]] = nbr_tc[i];
        }
    }

    memset(nbr_mask, -1, sizeof(uint32_t) * edge_num);
    memset(nbr_mask_double, -1, sizeof(uint32_t) * edge_num * 2);
    delete deg_direct, deg;
}

void kmax_truss_GPU(uint32_t &k_max, Bint_t &k_max_num, Bint_t ans, uint32_t N, uint32_t k, uint32_t k_step,
                        Bint_t *&nbr_u_direct, uint32_t *&nbr_direct, Bint_t *&nbr_u, uint32_t *&nbr, Bint_t *&fa, Bint_t *&left_son, Bint_t *&right_son, 
                        uint32_t *&nbr_mask, uint32_t *&nbr_mask_double, uint32_t *&nbr_tc, uint32_t *&nbr_tc_double, uint32_t &accurate_tc)
{
    const uint32_t k_init = k + (k_step) * 2;
    // 这些备份和还原的事情以后想一想能不能就直接在GPU中做了 //
    Bint_t edge_num = nbr_u_direct[N];
    uint32_t *nbr_tc_backup, *nbr_mask_backup, *nbr_backup, *nbr_direct_backup, *nbr_tc_double_backup, *nbr_mask_double_backup;
    Bint_t *nbr_u_backup, *nbr_u_direct_backup;
    nbr_tc_backup = new uint32_t[edge_num];
    nbr_tc_double_backup = new uint32_t[edge_num << 1];
    nbr_mask_backup = new uint32_t[edge_num];
    nbr_mask_double_backup = new uint32_t[edge_num << 1];
    nbr_backup = new uint32_t[edge_num << 1];
    nbr_u_backup = new Bint_t[N + 1];
    nbr_u_direct_backup = new Bint_t[N + 1];
    nbr_direct_backup = new uint32_t[edge_num];
    // GPU only backup, 未来考虑利用显存进行accurate_tc备份 //
    Bint_t *left_son_backup, *right_son_backup;
    left_son_backup = new Bint_t[edge_num];
    right_son_backup = new Bint_t[edge_num];
    Bint_t now_delete_backup = 0, edge_num_backup = 0;
    uint32_t st_backup = 0;    // T = 0表示三角形是不满足的，胡乱数出来的三角形 //
    Bint_t now_delete = 0;
    uint32_t reconstruct_flag = 0;
    uint32_t st = 1;
    uint32_t accurate_tc_backup = 1;
    uint32_t have_full_graph = 0, have_full_graph_backup = 0;
    // 使用大小步策略, 快速计算k_max //
    while(true)
    {
        k += (k_step << 1);
        // backup, 需要备份的还挺多的
        // 重新构图编写完成 //
        if(now_delete != edge_num && now_delete * 2 >= edge_num)
        {
            reconstruct_Graph(nbr_u_direct, nbr_direct, edge_num, N, nbr_u, nbr, fa, left_son, right_son, nbr_mask, nbr_mask_double, nbr_tc, nbr_tc_double, 1, st);
            now_delete = 0; have_full_graph = 1;
        }
        if(k_step > 1 && reconstruct_flag == 0)
        {
            // 在k为下届以内的情况下，其实是不需要备份的，因为必定不会全部删完 //
            if(!(k <= k_init && ans != 0))
            {
                now_delete_backup = now_delete;
                edge_num_backup = edge_num;
                accurate_tc_backup = accurate_tc;
                st_backup = st;
                have_full_graph_backup = have_full_graph;
                auto fu_nbr_tc_backup = std::async(memcpy, nbr_tc_backup, nbr_tc, size_nbr_tc);
                auto fu_nbr_tc_double_backup = std::async(memcpy, nbr_tc_double_backup, nbr_tc_double, size_nbr_tc_double);
                auto fu_nbr_mask_backup = std::async(memcpy, nbr_mask_backup, nbr_mask, size_nbr_mask);
                auto fu_mask_double_backup = std::async(memcpy, nbr_mask_double_backup, nbr_mask_double, size_nbr_mask_double);          
                fu_nbr_tc_backup.get();
                fu_nbr_tc_double_backup.get();
                fu_nbr_mask_backup.get();
                fu_mask_double_backup.get();
            }
        }
        uint32_t edge_delete = 0;
        // 在删的过程中，发现删除了某一轮删除了超级多的边，则重新构图数三角形 //
        if(now_delete < edge_num)
        {
            ktruss(N, now_delete, edge_delete, accurate_tc, nbr_u_direct, nbr_direct, nbr, nbr_u, nbr_tc, nbr_mask, nbr_tc_double, nbr_mask_double, left_son, right_son, st, k);
        }
        // cout << k << " " << now_delete << " " << edge_delete << " " << edge_num << endl;
        if((edge_delete != 0 or  accurate_tc == 0) && now_delete + edge_delete < edge_num)
        {
            if(reconstruct_flag == 0 && k_step > 1)
            {
                if(!(k <= k_init && ans != 0))
                {
                    reconstruct_flag = 1;
                    auto fu_nbr_direct_backup = std::async(memcpy, nbr_direct_backup, nbr_direct, size_nbr_direct);
                    auto fu_nbr_u_direct_backup = std::async(memcpy, nbr_u_direct_backup, nbr_u_direct, size_nbr_u_direct);
                    auto fu_nbr_backup = std::async(memcpy, nbr_backup, nbr, size_nbr);
                    auto fu_nbr_u_backup = std::async(memcpy, nbr_u_backup, nbr_u, size_nbr_u);
                    auto fu_left_son_backup = std::async(memcpy, left_son_backup, left_son, size_left_son);
                    auto fu_right_son_backup = std::async(memcpy, right_son_backup, right_son, size_right_son);
                    fu_nbr_direct_backup.get();
                    fu_nbr_u_direct_backup.get();
                    fu_nbr_backup.get();
                    fu_nbr_u_backup.get();
                    fu_left_son_backup.get();
                    fu_right_son_backup.get();
                }
            }
            reconstruct_Graph(nbr_u_direct, nbr_direct, edge_num, N, nbr_u, nbr, fa, left_son, right_son, nbr_mask, nbr_mask_double, nbr_tc, nbr_tc_double, 0, st + 1);
            now_delete = 0; edge_delete = 0; have_full_graph = 1;
            tricount_direct(nbr_u_direct, nbr_direct, N, nbr_u, nbr, nbr_tc, left_son, right_son, nbr_tc_double);
            k-=(k_step << 1);
            accurate_tc = 1;
            continue;
        }
        now_delete += edge_delete;
        // 判断一个这个k是否是k_max
        if(edge_num - now_delete > 0)
        {
            k_max = (k>>1)+2;
            k_max_num = edge_num - now_delete;
            if(ans == k_max_num && k == k_init)
                break;
        }
        if(now_delete == edge_num)
        {
            // 大小步策略，如果步长过大则会缩短步长, 如果步长为1则不需要backup//
            if(k_step == 1)
            {
                break;
            }
            else
            {
                k -= (k_step << 1);
                // 如果大步超了，则大步缩短为原来的一半 //
                if(k_step > K_MIDDLE)
                    k_step /= 2;
                else
                    k_step = 1;  
            }
            // backup 恢复 //
            if(reconstruct_flag == 1)
            {
                swap(nbr_direct, nbr_direct_backup);
                swap(nbr_u_direct, nbr_u_direct_backup);
                swap(nbr, nbr_backup);
                swap(nbr_u, nbr_u_backup);
                swap(left_son, left_son_backup);
                swap(right_son, right_son_backup);
            }
            swap(nbr_mask, nbr_mask_backup);
            swap(nbr_tc, nbr_tc_backup);
            swap(nbr_tc_double, nbr_tc_double_backup);
            swap(nbr_mask_double, nbr_mask_double_backup);
            now_delete = now_delete_backup;
            edge_num = edge_num_backup;
            st = st_backup;
            accurate_tc = accurate_tc_backup;
            have_full_graph = have_full_graph_backup;
            if(have_full_graph == 0)
            {
                have_full_graph = 1;
                build_graph_from_direct(nbr_direct, nbr_u_direct, N, nbr, nbr_u, fa, left_son, right_son);
            }
            if(accurate_tc == 0)
            {
                // 如果是大步，则三角形不要真正的计数，提高效率 //
                if(k_step > K_MIDDLE)
                {
                    tricount_direct(nbr_u_direct, nbr_direct, N, nbr_u, nbr, nbr_tc, left_son, right_son, nbr_tc_double, k_step);
                    accurate_tc = 0;
                }
                else
                {
                    tricount_direct(nbr_u_direct, nbr_direct, N, nbr_u, nbr, nbr_tc, left_son, right_son, nbr_tc_double);
                    accurate_tc = 1;
                }
            }
        }
        else
        {
            // 大步结束, 大步只走一次 //
            if(k_step > K_MIDDLE)
                k_step = K_MIDDLE;
        }
        reconstruct_flag = 0;
    }
    delete nbr_tc_backup, nbr_backup, nbr_direct_backup, nbr_u_backup, nbr_mask_backup;
    delete nbr_u_direct_backup, nbr_tc_double_backup, nbr_mask_double_backup;
    delete left_son_backup, right_son_backup;
}

__global__ void __count_deg_with_mask(const uint32_t *nbr, const Bint_t *nbr_u, uint32_t N, const uint32_t *node_delete, uint32_t *deg_with_mask)
{
    __shared__ uint32_t NodeI;
    __shared__ uint32_t NodeEnd;
    __shared__ uint32_t my_sum;
    uint32_t sum = 0;
    if(threadIdx.x == 0)
    {
        NodeI = NodeEnd = 0;
        my_sum = 0;
    }
    uint32_t u, v;
    while(true){
        // 动态调度 //
        if(threadIdx.x == 0){
            if(++NodeI >= NodeEnd){
                NodeI = atomicAdd(&dev_nowNode, 8);
                NodeEnd = min(N, NodeI + 8);
            }
        }
        u = __shfl_sync(FULL_MASK, NodeI, 0);
        if(u >= N) break;
        if(node_delete[u]) continue;
        sum = 0;
        // TODO 此处可以利用有序的性质减少一半的访问 //
        const Bint_t r = nbr_u[u+1];
        for(Bint_t j = nbr_u[u]; j < r; j+=BLOCKSIZE)
        {
            if(j + threadIdx.x < r)
            {
                v = nbr[j + threadIdx.x];
                sum += (node_delete[v] == 0);
            }
        }
        atomicAdd(&my_sum, sum);
        if(threadIdx.x == 0)
        {
            atomicAdd(&deg_with_mask[u], my_sum);
            my_sum = 0;
        }
    }
}

__global__ void __count_deg_direct_with_mask(const uint32_t *nbr, const Bint_t *nbr_u, uint32_t N, const uint32_t *node_delete, const uint32_t *deg_with_mask,
                                             uint32_t *dev_deg_direct_with_mask)
{
    __shared__ uint32_t NodeI;
    __shared__ uint32_t NodeEnd;
    __shared__ uint32_t my_sum;
    uint32_t sum = 0;
    if(threadIdx.x == 0)
    {
        NodeI = NodeEnd = 0;
        my_sum = 0;
    }
    uint32_t u, v;
    uint32_t deg_u, deg_v;
    while(true){
        // 动态调度 //
        if(threadIdx.x == 0){
            if(++NodeI >= NodeEnd){
                NodeI = atomicAdd(&dev_nowNode, 8);
                NodeEnd = min(N, NodeI + 8);
            }
        }
        u = __shfl_sync(FULL_MASK, NodeI, 0);
        if(u >= N) break;
        if(node_delete[u]) continue;
        deg_u = deg_with_mask[u];
        sum = 0;
        // TODO 此处可以利用有序的性质减少一半的访问 //
        const Bint_t r = nbr_u[u+1];
        for(Bint_t j = nbr_u[u]; j < r; j+=BLOCKSIZE)
        {
            if(j + threadIdx.x < r)
            {
                v = nbr[j + threadIdx.x];
                deg_v = deg_with_mask[v];
                if((node_delete[v] == 0 & ((deg_u < deg_v) | (deg_u == deg_v & u > v))))
                {
                    sum ++;
                }
            }
        }
        atomicAdd(&my_sum, sum);
        if(threadIdx.x == 0)
        {
            atomicAdd(&dev_deg_direct_with_mask[u], my_sum);
            my_sum = 0;
        }
    }
}

// 此函数如果输入过大需要对输入进行切割 //
void count_deg_with_mask(const uint32_t *&nbr, const Bint_t *&nbr_u, uint32_t N, uint32_t *&node_delete, uint32_t *&deg_with_mask,
                         uint32_t *&to, Link_t *&li, uint32_t &new_N, uint32_t *&deg, uint32_t *&deg_direct)
{
    uint32_t numBlocks = 4096;
    Bint_t edge_num = nbr_u[N] / 2;
    uint32_t *deg_direct_with_mask = new uint32_t[N]();
    uint32_t *dev_nbr, *dev_node_delete, *dev_deg_with_mask;
    Bint_t *dev_nbr_u;
    uint32_t *dev_deg_direct_with_mask;
    
    uint64_t size_deg_with_mask = size_deg;
    uint64_t size_node_delete = size_deg;
    uint64_t size_deg_direct_with_mask = size_deg;
    // 申请显存空间 //
    gpuErrchk(cudaMalloc((void**)&dev_nbr_u, size_nbr_u));
    gpuErrchk(cudaMalloc((void**)&dev_nbr, size_nbr));
    gpuErrchk(cudaMalloc((void**)&dev_node_delete, size_node_delete));
    gpuErrchk(cudaMalloc((void**)&dev_deg_with_mask, size_deg_with_mask));
    gpuErrchk(cudaMalloc((void**)&dev_deg_direct_with_mask, size_deg_direct_with_mask));
    // 如果输入过大记得切块 //
    gpuErrchk(cudaMemcpy(dev_nbr_u, nbr_u, size_nbr_u, cudaMemcpyHostToDevice) );
    gpuErrchk(cudaMemcpy(dev_nbr, nbr, size_nbr, cudaMemcpyHostToDevice) );
    gpuErrchk(cudaMemcpy(dev_node_delete, node_delete, size_node_delete, cudaMemcpyHostToDevice) );
    gpuErrchk(cudaMemcpy(dev_deg_with_mask, deg_with_mask, size_deg_with_mask, cudaMemcpyHostToDevice) );
    gpuErrchk(cudaMemcpy(dev_deg_direct_with_mask, deg_direct_with_mask, size_deg_direct_with_mask, cudaMemcpyHostToDevice) );

    // 此处将dev_node初始化一下 //
    uint64_t tmp = 0;
    gpuErrchk(cudaMemcpyToSymbol(dev_nowNode, &tmp, sizeof(uint32_t)) );
    
    // copy data to device /// 
    __count_deg_with_mask<<<numBlocks, BLOCKSIZE>>>(dev_nbr, dev_nbr_u, N, dev_node_delete, dev_deg_with_mask);
    gpuErrchk(cudaPeekAtLastError() );
    gpuErrchk(cudaDeviceSynchronize() );
    gpuErrchk(cudaMemcpy(deg_with_mask, dev_deg_with_mask, size_deg_with_mask, cudaMemcpyDeviceToHost) );

    // 统计一下单向边的度数 //
    
    tmp = 0;
    gpuErrchk(cudaMemcpyToSymbol(dev_nowNode, &tmp, sizeof(uint32_t)) );
    __count_deg_direct_with_mask<<<numBlocks, BLOCKSIZE>>>(dev_nbr, dev_nbr_u, N, dev_node_delete, dev_deg_with_mask, dev_deg_direct_with_mask);
    gpuErrchk(cudaDeviceSynchronize() );
    gpuErrchk(cudaMemcpy(deg_direct_with_mask, dev_deg_direct_with_mask, size_deg_direct_with_mask, cudaMemcpyDeviceToHost) );
    
    // 使用度数进行排序，这个部分基本不花时间 //
    for(uint32_t i = 0; i < N; ++i)
    {
        if(node_delete[i] == 0)
        {
            li[new_N].x = i;
            li[new_N].d = deg_with_mask[i];
            li[new_N].t = new_N;
            new_N ++;
        }
    }
    sort(li, li+new_N, cmp_link);
    deg = new uint32_t[new_N]();
    deg_direct = new uint32_t[new_N]();
#pragma omp parallel for num_threads(NUM_THREADS_THIS_COMPUTER)
    for(uint32_t i = 0; i < new_N; ++i)
    {
        to[li[i].x] = i;
        deg[i] = deg_with_mask[li[i].x];
        deg_direct[i] = deg_direct_with_mask[li[i].x];
    }

    cudaFree(dev_nbr);
    cudaFree(dev_nbr_u);
    cudaFree(dev_node_delete);
    cudaFree(dev_deg_with_mask);
    cudaFree(dev_deg_direct_with_mask);
}

__global__ void __build_nbr_direct(const uint32_t *nbr, const Bint_t *nbr_u, uint32_t N, const uint32_t *to, const uint32_t *node_delete,
                                   uint32_t *nbr_direct, const Bint_t *nbr_u_direct, const uint32_t *deg)
{
    __shared__ uint32_t NodeI;
    __shared__ uint32_t NodeEnd;
    __shared__ unsigned long long now_pos;
    if(threadIdx.x == 0)
    {
        NodeI = NodeEnd = 0;
    }
    uint32_t u, v;
    uint32_t to_u, to_v;
    while(true){
        // 动态调度 //
        if(threadIdx.x == 0){
            if(++NodeI >= NodeEnd){
                NodeI = atomicAdd(&dev_nowNode, 8);
                NodeEnd = min(N, NodeI + 8);
            }
        }
        u = __shfl_sync(FULL_MASK, NodeI, 0);
        if(u >= N) break;
        if(node_delete[u]) continue;
        to_u = to[u];
        now_pos = nbr_u_direct[to_u];
        const Bint_t r = nbr_u[u+1];
        for(Bint_t j = nbr_u[u]; j < r; j+=BLOCKSIZE)
        {
            if(j + threadIdx.x < r)
            {
                v = nbr[j + threadIdx.x];
                to_v = to[v];
                
                if(node_delete[v] == 0 && to_v > to_u)
                {
                    unsigned long long pos = atomicAdd(&now_pos, 1);
                    nbr_direct[pos] = to_v;
                }
            }
        }
    }
}


// 注意此处的edge_num的定义是新的还是旧的, 并且注意如果输入过大，记得对数组切片处理// 【数组大小检查完成】
void build_nbr_direct(const uint32_t *&nbr, const Bint_t *&nbr_u, uint32_t N, uint32_t *&node_delete, uint32_t *&to,
                      uint32_t *&deg, Bint_t *&nbr_u_direct, uint32_t *&nbr_direct, uint32_t new_N)
{
    uint32_t numBlocks = 4096;
    Bint_t edge_num = nbr_u[N] / 2;
    Bint_t new_edge_num = nbr_u_direct[new_N];
    uint32_t *dev_nbr, *dev_node_delete, *dev_deg, *dev_to;
    uint32_t *dev_nbr_direct;
    Bint_t *dev_nbr_u, *dev_nbr_u_direct;
    
    uint64_t new_size_deg = 1ll * sizeof(uint32_t) * new_N;
    uint64_t new_size_nbr_u_direct = 1ll * sizeof(Bint_t) * (new_N + 1);
    uint64_t new_size_nbr_direct = 1ll * sizeof(uint32_t) * new_edge_num;

    uint64_t size_node_delete = size_deg;
    uint64_t size_to = size_deg;
    // 老size
    gpuErrchk(cudaMalloc((void**)&dev_nbr_u, size_nbr_u));
    gpuErrchk(cudaMalloc((void**)&dev_nbr, size_nbr));
    gpuErrchk(cudaMalloc((void**)&dev_node_delete, size_node_delete));
    gpuErrchk(cudaMalloc((void**)&dev_to, size_to));
    // 以下的是子图的size，所以比较小 //
    gpuErrchk(cudaMalloc((void**)&dev_deg, new_size_deg));
    gpuErrchk(cudaMalloc((void**)&dev_nbr_u_direct, new_size_nbr_u_direct));
    gpuErrchk(cudaMalloc((void**)&dev_nbr_direct, new_size_nbr_direct));

    gpuErrchk(cudaMemcpy(dev_nbr_u, nbr_u, size_nbr_u, cudaMemcpyHostToDevice) );
    gpuErrchk(cudaMemcpy(dev_nbr, nbr, size_nbr, cudaMemcpyHostToDevice) );
    gpuErrchk(cudaMemcpy(dev_node_delete, node_delete, size_node_delete, cudaMemcpyHostToDevice) );
    gpuErrchk(cudaMemcpy(dev_to, to, size_to, cudaMemcpyHostToDevice) );

    gpuErrchk(cudaMemcpy(dev_deg, deg, new_size_deg, cudaMemcpyHostToDevice) );
    gpuErrchk(cudaMemcpy(dev_nbr_u_direct, nbr_u_direct, new_size_nbr_u_direct, cudaMemcpyHostToDevice) );
    gpuErrchk(cudaMemcpy(dev_nbr_direct, nbr_direct, new_size_nbr_direct, cudaMemcpyHostToDevice) );

    uint64_t tmp = 0;
    gpuErrchk(cudaMemcpyToSymbol(dev_nowNode, &tmp, sizeof(uint32_t)) );
    __build_nbr_direct<<<numBlocks, BLOCKSIZE>>>(dev_nbr, dev_nbr_u, N, dev_to, dev_node_delete, dev_nbr_direct, dev_nbr_u_direct, dev_deg);
    
    gpuErrchk(cudaPeekAtLastError() );
    gpuErrchk(cudaDeviceSynchronize() );

    gpuErrchk(cudaMemcpy(nbr_direct, dev_nbr_direct, new_size_nbr_direct, cudaMemcpyDeviceToHost) );
    // 不得不使用CPU进行排序，因为GPU排序速度并不是很快 //
#pragma omp parallel for num_threads(NUM_THREADS_THIS_COMPUTER)
    for(uint32_t i = 0; i < new_N; ++i)
    {
        if(nbr_u_direct[i+1] - nbr_u_direct[i] > 1)
            sort(nbr_direct + nbr_u_direct[i], nbr_direct + nbr_u_direct[i+1]);
    }
    cudaFree(dev_nbr);
    cudaFree(dev_node_delete);
    cudaFree(dev_deg);
    cudaFree(dev_to);
    cudaFree(dev_nbr_direct);
    cudaFree(dev_nbr_u);
    cudaFree(dev_nbr_u_direct);
}

// 有向图转为无向图 //
__global__ void __build_graph_from_direct(const uint32_t *nbr_direct, const Bint_t *nbr_u_direct, uint32_t N, 
                                          uint32_t *nbr, Bint_t *nbr_u, uint32_t *from)
{
    __shared__ uint32_t NodeI;
    __shared__ uint32_t NodeEnd;
    if(threadIdx.x == 0)
    {
        NodeI = NodeEnd = 0;
    }
    uint32_t u, v;
    uint32_t pos;
    while(true){
        // 动态调度 //
        if(threadIdx.x == 0){
            if(++NodeI >= NodeEnd){
                NodeI = atomicAdd(&dev_nowNode, 8);
                NodeEnd = min(N, NodeI + 8);
            }
        }
        u = __shfl_sync(FULL_MASK, NodeI, 0);
        if(u >= N) break;

        const Bint_t r = nbr_u_direct[u+1];
        for(Bint_t j = nbr_u_direct[u]; j < r; j+=BLOCKSIZE)
        {
            if(j + threadIdx.x < r)
            {
                v = nbr_direct[j + threadIdx.x];
                pos = atomicAdd(&nbr_u[u], 1);
                nbr[pos] = v;
                from[pos] = j + threadIdx.x;
                pos = atomicAdd(&nbr_u[v], 1);
                nbr[pos] = u;
                from[pos] = j + threadIdx.x;
            }
        }
    }
}

// 此函数讲一个有向图构图成为无向图, 【数组大小检查完成】
void build_graph_from_direct(uint32_t *&nbr_direct, Bint_t *&nbr_u_direct, uint32_t N, uint32_t *&nbr, Bint_t *&nbr_u,
                             Bint_t *&fa, Bint_t *&left_son, Bint_t *&right_son)
{
    uint32_t numBlocks = 4096;
    Bint_t edge_num =  nbr_u_direct[N];
    uint32_t *dev_nbr_direct, *dev_nbr;
    uint32_t *dev_fa;
    Bint_t *dev_nbr_u_direct, *dev_nbr_u;
    
    gpuErrchk(cudaMalloc((void**)&dev_nbr_direct, size_nbr_direct));
    gpuErrchk(cudaMalloc((void**)&dev_nbr, size_nbr));
    gpuErrchk(cudaMalloc((void**)&dev_nbr_u_direct, size_nbr_u_direct));
    gpuErrchk(cudaMalloc((void**)&dev_nbr_u, size_nbr_u));
    gpuErrchk(cudaMalloc((void**)&dev_fa, size_fa));
    
    
    gpuErrchk(cudaMemcpy(dev_nbr_direct, nbr_direct, size_nbr_direct, cudaMemcpyHostToDevice) );
    gpuErrchk(cudaMemcpy(dev_nbr_u_direct, nbr_u_direct, size_nbr_u_direct, cudaMemcpyHostToDevice) );
    gpuErrchk(cudaMemcpy(dev_nbr_u, nbr_u, size_nbr_u, cudaMemcpyHostToDevice) );
    uint64_t tmp = 0;
    gpuErrchk(cudaMemcpyToSymbol(dev_nowNode, &tmp, sizeof(uint32_t)) );
    __build_graph_from_direct<<<numBlocks, BLOCKSIZE>>>(dev_nbr_direct, dev_nbr_u_direct, N, dev_nbr, dev_nbr_u, dev_fa);
    gpuErrchk(cudaPeekAtLastError() );
    gpuErrchk(cudaDeviceSynchronize() );

    gpuErrchk(cudaMemcpy(fa, dev_fa, size_fa, cudaMemcpyDeviceToHost) );
    gpuErrchk(cudaMemcpy(nbr, dev_nbr, size_nbr, cudaMemcpyDeviceToHost) );
#pragma omp parallel for num_threads(NUM_THREADS_THIS_COMPUTER)
    for(uint32_t i = 0; i < N; ++i)
    {
        if(nbr_u[i+1] - nbr_u[i] > 1)
            thrust::sort_by_key(nbr + nbr_u[i], nbr + nbr_u[i+1], fa + nbr_u[i]);
    }
#pragma omp parallel for num_threads(NUM_THREADS_THIS_COMPUTER)
    for(uint32_t i = 0; i < N; ++i)
    {
        for(Bint_t j = nbr_u[i]; j < nbr_u[i+1]; j++)
        {
            if(i < nbr[j])
                left_son[fa[j]] = j;
            else
                right_son[fa[j]] = j;
        }
    }
    
    cudaFree(dev_nbr_direct);
    cudaFree(dev_nbr);
    cudaFree(dev_nbr_u_direct);
    cudaFree(dev_nbr_u);
    cudaFree(dev_fa);
}

void test_cuda()
{
    // kmax-truss
}
