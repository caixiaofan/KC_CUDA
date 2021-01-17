#include <algorithm>
#include <cstdio>
#include <omp.h>
#include <iostream>
#include <cstring>
#include <sys/time.h>
#include <cassert>
#include <future>
#include "kc_gpu.h"
#include "mapfile.hpp"
using namespace std;
#ifdef DEBUG_DF
TimeInterval allTime;
TimeInterval preProcessTime;
TimeInterval tmpTime;
#endif

const uint32_t NUM_THREADS_THIS_COMPUTER = omp_get_num_procs();

bool cmp(Edge_t &a, Edge_t &b)
{
    if(a.u==b.u)
        return a.v < b.v;
    return a.u < b.u;
}

bool cmp_link(Link_t &a, Link_t &b)
{
    return (a.d == b.d && a.x > b.x) || a.d < b.d;
}

bool cmp_nbr(Nbr_t &a, Nbr_t &b)
{
    return a.v < b.v;
}

// 用于构图的初始化，节约内存，并且可以在k-core中直接使用节约时间 //
void init_graph(uint32_t *&edgeList, Bint_t &edge_num, uint32_t &N, uint64_t *&bp, uint32_t bp_lenth, Bint_t *&nbr_u, uint32_t *&nbr, uint32_t *&deg)
{
    nbr = new uint32_t[edge_num << 1];
    nbr_u = new Bint_t[N + 1];
    // 构建邻接表 //
    nbr_u[0] = 0;
    for(uint32_t i = 1; i <= N; ++i)
    {
        nbr_u[i] = nbr_u[i-1] + deg[i-1];
    }
    uint64_t hex = 0;
    uint64_t *bpos = new uint64_t[bp_lenth];
    bpos[0] = 0;
    for(uint32_t i = 1; i < bp_lenth; i++)
    {
        bpos[i] = bpos[i-1] + bp[(i-1)*3+2]-bp[(i-1)*3+1];
    }
#pragma omp parallel for num_threads(NUM_THREADS_THIS_COMPUTER)
    for(uint32_t i = 0; i < bp_lenth; i++)
    {
        uint32_t t_id = bp[i*3];
        uint32_t ln = bp[i*3+1];
        uint32_t rn = bp[i*3+2];
        memcpy(nbr + bpos[i], edgeList + ln, 1ll * sizeof(uint32_t)*(rn-ln));
    }
    delete bpos;

}

// 构图
uint32_t construct_Graph(uint32_t &N, const Bint_t *nbr_origin_u, const uint32_t *nbr_origin, const uint32_t *deg_origin, 
                         Bint_t *&nbr_u, uint32_t *&nbr, uint32_t *&nbr_mask, Bint_t *&nbr_u_direct, uint32_t *&nbr_direct,
                         Bint_t *&fa, Bint_t *&left_son, Bint_t *&right_son, uint32_t Launch_start)
{
    uint32_t k_min = 0;
    // 判断哪些节点不满足要求需要被删除 //
    uint32_t *node_delete = new uint32_t[N]();
    k_min = kmax_core_gpu(N, nbr_origin_u, nbr_origin, deg_origin, node_delete, Launch_start);
#ifdef DEBUG_DF
    tmpTime.print("k-max-core time cost");
    tmpTime.check();
    cout << "k_max core or last time k_max truss: " << k_min + 1 << endl;
#endif 
    uint32_t *deg_with_mask = new uint32_t[N]();
    // 将图变成edgeList //
    uint32_t new_N = 0;
    uint32_t *to = new uint32_t[N]();
    Link_t *li = new Link_t[N];
    // 申请邻接表索引所需空间 //
    uint32_t *deg, *deg_direct;
    count_deg_with_mask(nbr_origin, nbr_origin_u, N, node_delete, deg_with_mask, to, li, new_N, deg, deg_direct);
    // 获取新的节点数量，以及重新定序，CPU开销很小 //
    nbr_u = new Bint_t[new_N + 1];
    nbr_u_direct = new Bint_t[new_N + 1];
    // 计算新的有向边的degree //
    // 基本不消耗时间的CPU前缀和 //
    nbr_u[0] = 0; nbr_u_direct[0] = 0;
    for(uint32_t i = 1; i <= new_N; ++i)
    {
        nbr_u[i] = nbr_u[i-1] + deg[i-1];
        nbr_u_direct[i] = nbr_u_direct[i-1] + deg_direct[i-1];
    }
    // 申请邻接表索引所需空间 //
    Bint_t edge_num = nbr_u_direct[new_N];
    nbr_direct = new uint32_t[edge_num];
    build_nbr_direct(nbr_origin, nbr_origin_u, N, node_delete, to, deg, nbr_u_direct, nbr_direct, new_N);
    // 构建邻接表和fa数组
    nbr = new uint32_t[edge_num << 1];
    fa = new Bint_t[edge_num << 1];
    // 儿子的数量, 构图没有问题
    left_son = new Bint_t[edge_num];
    right_son = new Bint_t[edge_num];
#pragma omp parallel for num_threads(NUM_THREADS_THIS_COMPUTER)
    for(uint32_t i = 0;i < edge_num; i++)
    {
        left_son[i] = right_son[i] = i;
    }
    nbr_mask = new uint32_t[edge_num];
    memset(nbr_mask, -1, 1ll * sizeof(uint32_t) * edge_num);
    N = new_N;
    delete to, li;
    delete deg_direct;
    delete deg, deg_with_mask;
#ifdef DEBUG_DF
    cout << "N is: " << N << endl;
    cout << "edge_num is:"  << edge_num << endl;
#endif
    return k_min;
}


uint32_t kmax_truss(const uint32_t N_origin, const Bint_t *nbr_origin_u, const uint32_t *nbr_origin, const uint32_t *deg_origin, uint32_t Launch_start, Bint_t &ans)
{
    uint32_t *nbr_tc_double, *nbr, *nbr_direct, *nbr_tc;
    Bint_t *nbr_u, *nbr_u_direct, *left_son, *right_son, *fa;
    uint32_t *nbr_mask, *nbr_mask_double;
    uint32_t k_min = 0;
    uint32_t N = N_origin;
    k_min = construct_Graph(N, nbr_origin_u, nbr_origin, deg_origin, nbr_u, nbr, nbr_mask, nbr_u_direct, nbr_direct, fa, left_son, right_son, Launch_start);
    Bint_t edge_num = nbr_u_direct[N];
    nbr_mask_double = new uint32_t[edge_num << 1];
    memset(nbr_mask_double, -1, 1ll * sizeof(uint32_t) * (edge_num << 1));
#ifdef DEBUG_DF
    tmpTime.print("construct Graph Time Cost");
    preProcessTime.print("Total prepocessing Time Cost");
    tmpTime.check();
#endif  
    uint32_t k = 0, k_max = 0;
    Bint_t k_max_num = 0, now_delete = 0;
    uint32_t st = 1;
    // 二分所需参数 //
    uint32_t k_step = K_MIDDLE;
    uint32_t accurate_tc = 1;
    // 弹射起步 //
    if(Launch_start != uint32_t(-1) && k_min != 0)
    {
        // 如果kmax-core 已经得到一个很接近的k，则弹射起步，> 100的数据基本是准的 //
        if(k_min >= 100)
        {
            k_step = 1;
        }
        // k < 100因为有可能预测不准， 否则步长设置为10~20，第一轮k即使不准，也可以快速定位下一轮的答案，也可以快速找到答案 //
        else
        {
            k_step = K_MIDDLE - 2;
        }
        k = (k_min << 1) - (k_step * 2);
    }
    // 大步 //
    if(Launch_start == uint32_t(-1) && k_min != 0)
    {
        // 综合多个数据集考量，kmax_truss和(k_min - 1) / 2.1有一定的相关性，在大数据上 //
        k_step = max(k_step, uint32_t((k_min - 1) / 2.1));
        // 大步在150以上相对要准确，但是150以内容易偏大，即使在此不/2，在之后的处理中如果大步超过了也会缩短步长//
        if(k_step < 200)
            k_step = max(uint32_t(K_MIDDLE), uint32_t(k_step / 2));
    }
    // 统计三角形数量，可以利用更快速的方法统计三角形数量, 弹射起步 //
    // 对于每条边所统计的三角形的数量 //
    nbr_tc = new uint32_t[edge_num]();
    nbr_tc_double = new uint32_t[edge_num << 1];
    if(Launch_start==uint32_t(-1))
    {
        tricount_direct(nbr_u_direct, nbr_direct, N, nbr_u, nbr, nbr_tc, left_son, right_son, nbr_tc_double, k_step);
        accurate_tc = 0;
    }
    else
    {
        tricount_direct(nbr_u_direct, nbr_direct, N, nbr_u, nbr, nbr_tc, left_son, right_son, nbr_tc_double, k_min);
        accurate_tc = 0;
    }

#ifdef DEBUG_DF
    tmpTime.print("Triangle count Time Cost");
    tmpTime.check();
#endif
    
    kmax_truss_GPU(k_max, k_max_num, ans, N, k, k_step, nbr_u_direct, nbr_direct, nbr_u, nbr, fa, left_son, right_son, 
                   nbr_mask, nbr_mask_double, nbr_tc, nbr_tc_double, accurate_tc);

    delete nbr_tc, nbr, nbr_direct, nbr_u, nbr_u_direct, nbr_mask, left_son, right_son;
    delete nbr_mask_double, nbr_tc_double;
    delete fa;
    
#ifdef DEBUG_DF
    tmpTime.print("K-truss count Time Cost");
    tmpTime.check();
#endif
    ans = k_max_num;
    return k_max;
}
int main(int argc, char *argv[])
{
    if (argc != 3 || strcmp(argv[1], "-f") != 0) {
        cerr << "Usage: -f [data_file_path]" << endl;
        exit(1);
    }
    auto futureFunction = std::async(initGPU, 1, 1);
    const char *filepath = argv[2];
#ifdef DEBUG_DF
    cout << filepath << endl;
#endif
    // 将文件映射到内存
    const string command_rd = "vmtouch -qt " + string(filepath);
    MapFile mapfile(filepath);
    const uint64_t file_lenth = mapfile.getLen();
    char *tsvdata = (char *) mapfile.getAddr();
    uint64_t len_edgeList = min(uint64_t(1800000000), max(file_lenth / 4, uint64_t(3000000)));
    uint32_t* edgeList = new uint32_t[len_edgeList];
    Bint_t edge_num = 0;
    uint32_t N = 0;
    uint64_t *pos_list = new uint64_t[NUM_THREADS_THIS_COMPUTER]();
    for(uint32_t i = 0; i < NUM_THREADS_THIS_COMPUTER; i++)
        pos_list[i] = len_edgeList / NUM_THREADS_THIS_COMPUTER * i;
    // openmp 获取当前线程id //
    // openmp 设置线程数量 //
    uint32_t *nbr_origin, *deg_origin;
    Bint_t *nbr_origin_u;
    deg_origin = new uint32_t[min(uint64_t(60000000), max(file_lenth / 4, uint64_t(3000000)))]();
    // 此处利用更好的并行 //
    uint64_t file_read_step = max(file_lenth / NUM_THREADS_THIS_COMPUTER / 5, uint64_t(1024));
    int s = system(command_rd.c_str());
    assert(s == 0);
    // bp save begin pos and end pos //
    uint64_t *bp = new uint64_t[(file_lenth / file_read_step + 10) * 3];
#pragma omp parallel for reduction(max: N) schedule(dynamic, 1) num_threads(NUM_THREADS_THIS_COMPUTER)
    for(uint64_t i = 0; i < file_lenth; i += file_read_step)
    {
        uint32_t id = i / file_read_step;
        uint64_t j = i;
        uint64_t r = min(i + file_read_step, file_lenth);
        uint32_t x, y;
        const uint64_t thread_id = omp_get_thread_num();
        uint64_t pos = pos_list[thread_id];
        bp[id * 3] = thread_id;
        bp[id * 3 + 1] = pos;
        while(j < file_lenth && j!=0 && tsvdata[j]!='\n')j++;
        while(j < file_lenth && !(tsvdata[j]>='0'))j++;
        while(r < file_lenth && r!=0 && tsvdata[r]!='\n')r++;
        assert(j == file_lenth || (tsvdata[j] >='0' and tsvdata[j] <= '9'));
        assert(r == file_lenth || tsvdata[r] == '\n');
        uint32_t last_val = uint32_t(-1);
        uint32_t last_val_num = 0;
        while(j < r)
        {
            x = y = 0;
            while(tsvdata[j] >= '0')
            {
                x = (x << 3) + (x << 1) + tsvdata[j] - '0';
                ++j;
            }
            while(!(tsvdata[j] >= '0'))
            {
                ++j;
            }
            while(tsvdata[j] >= '0')
            {
                y = (y << 3) + (y << 1) + tsvdata[j] - '0';
                ++j;
            }
            while(j < r && tsvdata[j]!='\n')
                ++j;
            while(j < r && !(tsvdata[j] >= '0'))
                ++j;
            
            if(y != last_val)
            {
                if(last_val != uint32_t(-1))
                {
#pragma omp atomic
                    deg_origin[last_val]+=last_val_num;
                }
                last_val = y;
                last_val_num = 1;
                N = max(N, y);
            }
            else
                last_val_num ++;
            edgeList[pos] = x;
            pos++;
        }
        if(last_val != uint32_t(-1))
        {
#pragma omp atomic
            deg_origin[last_val]+=last_val_num;
        }
        bp[id * 3 + 2] = pos;
        pos_list[thread_id] = pos;
    }
    N++;
    mapfile.release();
    uint32_t bp_lenth = file_lenth / file_read_step + (file_lenth % file_read_step > 0 ? 1 : 0);
    for(uint32_t i = 0; i < bp_lenth; i++)
        edge_num += bp[i*3+2] - bp[i*3+1];
    edge_num /= 2;
#ifdef DEBUG_DF
    cout << "Edge Num: " << edge_num << endl;
    cout << "N : " << N << endl;
#endif
#ifdef DEBUG_DF
    tmpTime.print("Reading file Time Cost");
    preProcessTime.check();
    tmpTime.check();
#endif
    init_graph(edgeList, edge_num, N, bp, bp_lenth, nbr_origin_u, nbr_origin, deg_origin);
    delete edgeList;
    delete bp;
#ifdef DEBUG_DF
    tmpTime.print("Build graph to adjList cost");
    tmpTime.check();
#endif 
    futureFunction.get();
    initGPU(edge_num, N);
#ifdef DEBUG_DF
    tmpTime.print("init GPU Time Cost");
    tmpTime.check();
#endif 
    Bint_t ans = 0;
    uint32_t k_max = kmax_truss(N, nbr_origin_u, nbr_origin, deg_origin, uint32_t(-1), ans);
#ifdef DEBUG_DF
    cout << "kmax_guess = " << k_max << ", Edges in kmax-truss = " << ans <<".\n";
#endif
    k_max = kmax_truss(N, nbr_origin_u, nbr_origin, deg_origin, k_max == 0? k_max : k_max - 1, ans);
    cout << "kmax = " << k_max << ", Edges in kmax-truss = " << ans <<".\n";
#ifdef DEBUG_DF
    allTime.print("All time cost");
#endif
    delete nbr_origin, deg_origin, nbr_origin_u;
    return 0;
}
