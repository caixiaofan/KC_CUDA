#include <algorithm>
#include <cstdio>
#include <omp.h>
#include <iostream>
#include <cstring>
#include <vector>
#include <queue>
#include <cassert>
#include "mapfile.hpp"
//#include "tc_count.cpp"
using namespace std;
// 根据数据量进行修改，如果边的数量比较少的话，可以用uint32_t节约空间
// typedef uint64_t Bint_t;
#ifdef DEBUG_DF

TimeInterval allTime;
TimeInterval preProcessTime;
TimeInterval tmpTime;
#endif

static const uint32_t NUM_THREADS_THIS_COMPUTER = omp_get_num_procs();

// size must be 8
struct Edge_t {
    uint32_t u;
    uint32_t v;
    Edge_t(){};
    Edge_t(const uint32_t &x, const uint32_t &y): u(x), v(y) {};
} __attribute__ ((aligned (4)));

struct Nbr_t {
    uint32_t v;
    Bint_t fa;
    Nbr_t(){};
    Nbr_t(uint32_t nex, Bint_t f): v(nex), fa(f) {};
} __attribute__ ((aligned (8)));

struct Link_t {
    // 初始节点，目标节点，到达节点 //
    uint32_t x, t, d;
} __attribute__ ((aligned (4)));

bool cmp(Edge_t &a, Edge_t &b)
{
    if(a.u==b.u)
        return a.v < b.v;
    return a.u < b.u;
}

bool cmp_nbr(Nbr_t &a, Nbr_t &b)
{
    return a.v < b.v;
}

bool cmp_link(Link_t &a, Link_t &b)
{
    return (a.d == b.d && a.x > b.x) || a.d < b.d;
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
        //cout << bpos[i] << endl;
    }
#pragma omp parallel for
    for(uint32_t i = 0; i < bp_lenth; i++)
    {
        uint32_t t_id = bp[i*3];
        uint32_t ln = bp[i*3+1];
        uint32_t rn = bp[i*3+2];
        memcpy(nbr + bpos[i], edgeList + ln, sizeof(uint32_t)*(rn-ln));
    }
    delete edgeList;
    delete bpos;
    delete bp;
}

// 一个三角形只会被统计一遍 //
uint32_t count_tc(const uint32_t *nbr, const Bint_t* nbr_u, Edge_t e, uint32_t *&nbr_tc, const int &tc_limit)
{
    uint32_t &u = e.u, &v = e.v;
    Bint_t i = nbr_u[u];
    Bint_t j = nbr_u[v];
    uint32_t ans = 0;
    while(i < nbr_u[u + 1] && j < nbr_u[v + 1])
    {
        if(*(nbr + i) == *(nbr + j))
        {
            ans ++;
#pragma omp atomic
            (*(nbr_tc + i)) += 2;
#pragma omp atomic
            (*(nbr_tc + j)) += 2;
        }
        uint32_t tmp = (*(nbr + i) >= *(nbr + j));
        i += (*(nbr + i) <= *(nbr + j));
        j += tmp;
        if(tc_limit!=0 && int(min(nbr_u[u+1]-i,nbr_u[v+1]-j)+ans) + int(NUM_THREADS_THIS_COMPUTER - 1) < tc_limit)
        {
            break;
        }
    }
    return ans;
}

void delete_edge(uint32_t* nbr, Bint_t *fa,Bint_t* nbr_u,
                 uint32_t* nbr_tc, uint32_t* nbr_mask, Edge_t e, uint32_t &st, uint32_t &k)
{
    uint32_t &u = e.u, &v = e.v;
    Bint_t i = nbr_u[u];
    Bint_t j = nbr_u[v];
    while(i < nbr_u[u + 1] && j < nbr_u[v + 1])
    {
        Bint_t &i_fa = *(fa + i);
        Bint_t &j_fa = *(fa + j);
        uint32_t flag = (*(nbr + i) == *(nbr + j)) && nbr_mask[i_fa] >= st && nbr_mask[j_fa] >= st;
        uint32_t num = 2 - (nbr_mask[i_fa] == st) - (nbr_mask[j_fa] == st);
        // 这个位置进行串行化即可 //
        if(flag && num > 0)
        {
#pragma omp atomic
            nbr_tc[i_fa] -= num;
#pragma omp atomic
            nbr_tc[j_fa] -= num;
            if(nbr_mask[j_fa] == -1 and nbr_tc[j_fa] < k)
            {
                nbr_mask[j_fa] = st + 1;
            }
            if(nbr_mask[i_fa] == -1 and nbr_tc[i_fa] < k)
            {
                nbr_mask[i_fa] = st + 1;
            }
        }
        uint32_t tmp = (*(nbr + i) >= *(nbr + j));
        i += (*(nbr + i) <= *(nbr + j));
        j += tmp;
    }
}

void delete_note(int i, uint32_t *&mask, int &node_delete, const Bint_t *&nbr_u, const uint32_t *&nbr, uint32_t *&deg, int &k)
{
    node_delete++;
    Bint_t rn = nbr_u[i+1];
    uint32_t tmp;
    for(int j = nbr_u[i]; j < rn; j++)
    {
#pragma omp atomic capture
        tmp = --deg[nbr[j]];
        if(tmp == k-1)
        {
#pragma omp atomic capture
            tmp = ++mask[nbr[j]];
            if(tmp == 1)
                delete_note(nbr[j], mask, node_delete, nbr_u, nbr, deg, k);
        }
    }
}

void count_node(int i, uint32_t *&mask, const Bint_t *&nbr_u, const uint32_t *&nbr, uint32_t *&deg)
{
    uint32_t ans = 0;
    Bint_t rn = nbr_u[i+1];
    for(int j = nbr_u[i]; j < rn; j++)
    {
        if(mask[nbr[j]] == 0)
            ans++;
    }
    deg[i] = ans;
}

uint32_t kmax_core(const uint32_t N, const Bint_t *nbr_u, const uint32_t *nbr, const uint32_t *deg_origin, uint32_t *&mask, uint32_t Launch_start)
{
    int k = 0;
    int k_max = 0, k_step = 40;
    int node_delete = 0, node_delete_backup = 0;
    uint32_t *deg = new uint32_t[N];
    uint32_t *mask_backup = new uint32_t[N];
    uint32_t *deg_backup = new uint32_t[N];
    memcpy(deg, deg_origin, N * sizeof(uint32_t));
    if(Launch_start != uint32_t(-1))
    {
        k = Launch_start;
        k_step = 1;
    }
    // 这里也可以使用reconstruct的方式重新构图 //
    while(node_delete < N)
    {
        // cout << k << " " << node_delete << " "<< N << endl;
        // backup
        if(k_step > 1)
        {
            node_delete_backup = node_delete;
            memcpy(deg_backup, deg, N * sizeof(uint32_t));
        }
        memcpy(mask_backup, mask, N * sizeof(uint32_t));
        if(Launch_start == uint32_t(-1))
            k += k_step;
        while(true)
        {
            uint32_t flag = 0;
#pragma omp parallel for reduction(+:node_delete) schedule(dynamic, 1024) 
            for(int i = 0;i < N;i++)
            {   
                if(mask[i] == 0 && deg[i] < k)
                {
                    mask[i] = 1;
                    flag = 1;
                    node_delete++;
                    Bint_t rn = nbr_u[i+1];
                    for(int j = nbr_u[i]; j < rn; j++)
#pragma omp atomic
                        deg[nbr[j]] --;
                }
            }
            if(flag == 0)
                break;
        }
        if(node_delete == N and k_step > 1)
        {
            k-=k_step;k_step = 1;
            node_delete = node_delete_backup;
            memcpy(deg, deg_backup, N * sizeof(uint32_t));
            memcpy(mask, mask_backup, N * sizeof(uint32_t));
        }
        // 如果是使用输入的Launch_start 作为k，则不需要计算kmax-core //
        if(Launch_start != uint32_t(-1))
            break;
    }
    if(Launch_start == uint32_t(-1))
        k--;
    delete deg_backup;
    if(Launch_start == uint32_t(-1))
        memcpy(mask, mask_backup, N * sizeof(uint32_t));
    delete mask_backup;
    delete deg;
    return k==0? 0: k-1;
}

// 构图
uint32_t construct_Graph(uint32_t &N, const Bint_t *nbr_origin_u, const uint32_t *nbr_origin, const uint32_t *deg_origin, 
                     Bint_t *&nbr_u, uint32_t *&nbr, Bint_t *&fa, uint32_t *&nbr_mask,
                     Bint_t *&nbr_u_direct, uint32_t *&nbr_direct, uint32_t Launch_start)
{
    uint32_t k_min = 0;
    // 判断哪些节点不满足要求需要被删除 //
    uint32_t *node_delete = new uint32_t[N]();
    k_min = kmax_core(N, nbr_origin_u, nbr_origin, deg_origin, node_delete, Launch_start);

#ifdef DEBUG_DF
    tmpTime.print("k-max-core time cost");
    tmpTime.check();
    cout << "k_max core or last time k_max truss: " << k_min + 1 << endl;
#endif  
    // 获取新的节点数量 //
    uint32_t new_N = 0;
    uint32_t *to = new uint32_t[N]();
    Link_t *li = new Link_t[N];
    for(uint32_t i = 0; i < N; ++i)
    {
        to[i] = i;
        if(node_delete[i] == 0)
        {
            li[new_N].x = i;
            li[new_N].d = deg_origin[i];
            li[new_N].t = new_N;
            new_N ++;
        }
    }
    sort(li, li+new_N, cmp_link);
#pragma omp parallel for
    for(uint32_t i = 0; i < new_N; ++i)
    {
        to[li[i].x] = i;
    }
    // new_N = N;
    // 申请邻接表索引所需空间 //
    uint32_t *deg = new uint32_t[new_N]();
    uint32_t *deg_direct = new uint32_t[new_N]();
    
    nbr_u = new Bint_t[new_N + 1];
    nbr_u_direct = new Bint_t[new_N + 1];
#pragma omp parallel for
    for(uint32_t i = 0; i < N; ++i)
    {
        uint32_t &u = i;
        if(node_delete[u] != 0)
            continue;
        Bint_t rn = nbr_origin_u[i+1];
        for(Bint_t j = nbr_origin_u[i]; j < rn; ++j)
        {
            const uint32_t &v = nbr_origin[j];
            if(deg_origin[u] > deg_origin[v] || (deg_origin[u] == deg_origin[v] and u < v))
                continue;
            if(node_delete[v]!=0)
                continue;
            uint32_t ru = to[u], rv = to[v];
#pragma omp atomic
            ++deg[to[u]];
#pragma omp atomic
            ++deg[to[v]];
#pragma omp atomic
            ++deg_direct[to[u]];
        }
    }
    nbr_u[0] = 0; nbr_u_direct[0] = 0;
    for(uint32_t i = 1; i <= new_N; ++i)
    {
        nbr_u[i] = nbr_u[i-1] + deg[i-1];
        nbr_u_direct[i] = nbr_u_direct[i-1] + deg_direct[i-1];
    }
    // 申请邻接表索引所需空间 //
    Bint_t edge_num = nbr_u_direct[new_N];
    Nbr_t *nbr_t = new Nbr_t[edge_num << 1];
    nbr_direct = new uint32_t[edge_num];
#pragma omp parallel for
    for(uint32_t i = 0; i < N; ++i)
    {
        uint32_t &u = i;
        if(node_delete[u] != 0)
            continue;
        Bint_t rn = nbr_origin_u[i+1];
        for(Bint_t j = nbr_origin_u[i]; j < rn; ++j)
        {
            const uint32_t &v = nbr_origin[j];
            if(deg_origin[u] > deg_origin[v] || (deg_origin[u] == deg_origin[v] and u < v))
                continue;
            if(node_delete[v]!=0)
                continue;
            nbr_direct[nbr_u_direct[to[u]]++] = to[v];
        }
    }
    
    nbr_u_direct[0] = 0;
    for(uint32_t i = 1; i <= new_N; ++i)
    {
        nbr_u_direct[i] = nbr_u_direct[i-1] + deg_direct[i-1];
    }
#pragma omp parallel for
    for(uint32_t i = 0; i < new_N; ++i)
    {
        sort(nbr_direct + nbr_u_direct[i], nbr_direct + nbr_u_direct[i+1]);
    }
    for(uint32_t i = 0; i < new_N; ++i)
    {
        uint32_t &u = i;
        Bint_t rn = nbr_u_direct[i+1];
        for(Bint_t j = nbr_u_direct[i]; j < rn; ++j)
        {
            const uint32_t &v = nbr_direct[j];
            nbr_t[nbr_u[u]++] = Nbr_t(v, j);
            nbr_t[nbr_u[v]++] = Nbr_t(u, j);
        }
    }
    
    // 再次构建邻接表 //
    nbr_u[0] = 0;
    for(uint32_t i = 1; i <= new_N; ++i)
    {
        nbr_u[i] = nbr_u[i-1] + deg[i-1];
    }
    // 对于每条边所有的标记 //
    nbr_mask = new uint32_t[edge_num];
    memset(nbr_mask, -1, sizeof(uint32_t) * edge_num);
    // 输入是没有重边的，所以sort就好了 //
#pragma omp parallel for
    for(uint32_t i = 0; i < new_N; ++i)
    {
        sort(nbr_t + nbr_u[i], nbr_t + nbr_u[i+1], cmp_nbr);
    }
    // 构建邻接表和fa数组
    nbr = new uint32_t[edge_num << 1];
    fa = new Bint_t[edge_num << 1];
#pragma omp parallel for
    for(Bint_t i = 0; i < (edge_num << 1); ++i)
    {
        nbr[i] = nbr_t[i].v;
        fa[i] = nbr_t[i].fa;    
    }
    N = new_N;
    delete nbr_t;
    delete deg_direct;
    delete deg;
#ifdef DEBUG_DF
    cout << "N is: " << N << endl;
    cout << "edge_num is:"  << edge_num << endl;
#endif
    return k_min;
}
// 重新构图
void reconstruct_Graph(Bint_t *&nbr_u_direct, uint32_t *&nbr_direct, Bint_t &edge_num, const uint32_t N, 
                     Bint_t *&nbr_u, uint32_t *&nbr, Bint_t *&fa, uint32_t *&nbr_mask, uint32_t *&nbr_tc, uint32_t st)
{
    uint32_t new_edge_num = 0;
    uint32_t *deg = new uint32_t[N]();
    uint32_t *deg_direct = new uint32_t[N]();
    for(uint32_t i = 0; i < N; i++)
    {
        Bint_t rn = nbr_u_direct[i+1];
        for(uint32_t j = nbr_u_direct[i]; j < rn; ++j)
        {
            // 对于没有被删除的边 //
            if(nbr_mask[j] == uint32_t(-1) || nbr_mask[j] > st)
            {
                Edge_t e = Edge_t(i, nbr_direct[j]);
                ++deg[e.u];
                ++deg[e.v];
                ++deg_direct[e.u];
                nbr_direct[new_edge_num] = e.v;
                nbr_tc[new_edge_num] = nbr_tc[j];
                nbr_mask[new_edge_num] = nbr_mask[j];
                ++new_edge_num;
            }
        }
    }
    nbr_u[0] = 0;
    for(uint32_t i = 1; i <= N; ++i)
    {
        nbr_u_direct[i] = nbr_u_direct[i-1] + deg_direct[i-1];
        nbr_u[i] = nbr_u[i-1] + deg[i-1];
    }
    delete deg_direct;
    edge_num = new_edge_num;
    if(edge_num == 0)
    {
        delete deg;
        return;
    }
    Nbr_t *nbr_t = new Nbr_t[edge_num << 1];
    // 构建邻接表 //
    for(uint32_t i = 0; i < N; i++)
    {
        Bint_t rn = nbr_u_direct[i + 1];
        for(uint32_t j = nbr_u_direct[i]; j < rn; ++j)
        {
            Edge_t e = Edge_t(i, nbr_direct[j]);
            nbr_t[nbr_u[e.u]++] = Nbr_t(e.v, j);
            nbr_t[nbr_u[e.v]++] = Nbr_t(e.u, j);
        }
    }
    nbr_u[0] = 0;
    for(uint32_t i = 1; i <= N; ++i)
    {
        nbr_u[i] = nbr_u[i-1] + deg[i-1];
    }
#pragma omp parallel for
    for(uint32_t i = 0; i < N; ++i)
    {
        sort(nbr_t + nbr_u[i], nbr_t + nbr_u[i+1], cmp_nbr);
    }
#pragma omp parallel for
    for(Bint_t i = 0; i < (edge_num << 1); ++i)
    {
        nbr[i] = nbr_t[i].v;
        fa[i] = nbr_t[i].fa;    
    }
    delete deg;
}

// 三角形计数 //
void triangle_count(Bint_t edge_num, uint32_t N, const Bint_t *nbr_u_direct, const uint32_t *nbr_direct, uint32_t *nbr_tc, uint32_t tc_limit = 0)
{
    memset(nbr_tc, 0, sizeof(uint32_t) * edge_num);
#pragma omp parallel for schedule(dynamic, 1)
    for(uint32_t i = 0; i < N; ++i)
    {
        const Bint_t rn = nbr_u_direct[i+1];
        for(uint32_t j = nbr_u_direct[i]; j < rn; j++)
        {
             uint32_t val = count_tc(nbr_direct, nbr_u_direct, Edge_t(i, nbr_direct[j]), nbr_tc, int(tc_limit) - nbr_tc[j] / 2) << 1;
 #pragma omp atomic
             nbr_tc[j] += val;
        }
    }
}

// 注意Bint_t传递的不是引用 //
uint32_t kmax_truss(const uint32_t N_origin, const Bint_t *nbr_origin_u, const uint32_t *nbr_origin, const uint32_t *deg_origin, uint32_t Launch_start, Bint_t &ans)
{
    uint32_t *nbr_tc, *nbr, *nbr_direct;
    Bint_t *nbr_u, *fa, *nbr_u_direct;
    uint32_t *nbr_mask;
    uint32_t k_min = 0;
    uint32_t N = N_origin;
    k_min = construct_Graph(N, nbr_origin_u, nbr_origin, deg_origin, nbr_u, nbr, fa, nbr_mask, nbr_u_direct, nbr_direct, Launch_start);
    Bint_t edge_num = nbr_u_direct[N];
#ifdef DEBUG_DF
    tmpTime.print("Data preprocessing Time Cost");
    preProcessTime.print("Total prepocessing Time Cost");
    tmpTime.check();
    preProcessTime.check();
#endif
    uint32_t k = 0, k_max = 0;
    Bint_t k_max_num = 0, now_delete = 0;
    uint32_t st = 1;
    // 二分所需参数 //
    uint32_t k_step = 20;
    uint32_t *nbr_tc_backup, *nbr_mask_backup, *nbr_backup, *nbr_direct_backup;
    Bint_t *fa_backup, *nbr_u_backup, *nbr_u_direct_backup;
    nbr_tc_backup = new uint32_t[edge_num];
    nbr_mask_backup = new uint32_t[edge_num];
    nbr_backup = new uint32_t[edge_num << 1];
    fa_backup = new Bint_t[edge_num << 1];
    nbr_u_backup = new Bint_t[N + 1];
    nbr_u_direct_backup = new Bint_t[N + 1];
    nbr_direct_backup = new uint32_t[edge_num];
    uint32_t now_delete_backup = 0, edge_num_backup = 0;
    uint32_t st_backup = 0;
    // T = 0表示三角形是不满足的，胡乱数出来的三角形 //
    uint32_t accurate_tc = 1;
    // 弹射起步 //
    if(Launch_start != uint32_t(-1) && k_min != 0)
    {
        if(k_min >= 100)
        {
            k = (k_min << 1) - 2;
            k_step = 1;
        }
        else
        {
            k = (k_min << 1) - 40;
            k_step = 20;
        }
    }
    // 大步 //
    if(Launch_start == uint32_t(-1) && k_min != 0)
    {
        k_step = max(k_step, uint32_t(k_min / 2.3));
        //k_step = 1366;
    }
    // 统计三角形数量 //
    nbr_tc = new uint32_t[edge_num];
    if(Launch_start==uint32_t(-1))
    {
        // 此处注意，如果kmax-core是一个非常密集的子图，就别加限制了 //
        if(k_step <= 100)
            triangle_count(edge_num, N, nbr_u_direct, nbr_direct, nbr_tc);
        else
        {
            triangle_count(edge_num, N, nbr_u_direct, nbr_direct, nbr_tc, k_step);
            accurate_tc = 0;
        }
    }
    else
    {
        if(k_min >= 30)
        {
            triangle_count(edge_num, N, nbr_u_direct, nbr_direct, nbr_tc, k_min);
            accurate_tc = 0;
        }
        else
        {
            triangle_count(edge_num, N, nbr_u_direct, nbr_direct, nbr_tc);
        }
    }

#ifdef DEBUG_DF
    tmpTime.print("Triangle count Time Cost");
    tmpTime.check();
#endif
    // 使用大小步策略, 快速计算k_max //
    bool reconstruct_flag = 0;
    uint32_t accurate_tc_backup = 0;
    while(now_delete < edge_num)
    {
        k += (k_step << 1);
        // backup, 需要备份的还挺多的
        // 此处是否可以重新构图呢，如果失败了
        if(k_step > 1 && reconstruct_flag == 0)
        {
            memcpy(nbr_tc_backup, nbr_tc, edge_num * sizeof(uint32_t));
            memcpy(nbr_mask_backup, nbr_mask, edge_num * sizeof(uint32_t));
            now_delete_backup = now_delete;
            edge_num_backup = edge_num;
            st_backup = st;
            accurate_tc_backup = accurate_tc;
        }
        // cout << now_delete << " " << k << " " << edge_num << endl;
        uint32_t edge_delete = 0;
#pragma omp parallel for reduction(+:edge_delete) schedule(dynamic, 8)
        for(uint32_t i = 0; i < N; ++i)
        {
            const Bint_t rn = nbr_u_direct[i + 1];
            for(Bint_t j = nbr_u_direct[i]; j < rn ;++j)
            {
                if(nbr_mask[j] == -1 && nbr_tc[j] < k)
                {
                    nbr_mask[j] = st;
                    edge_delete++;
                }
            }
        }
        // cout << k << " " << now_delete << " " << edge_num << " " << edge_delete << endl;
        // 如果三角形是假的数法，并且全部都被删除掉了, 则不能删边，因为三角形数量是不对的，怎么删呢~//
        // TimeInterval temp;
        while(now_delete < edge_num && edge_delete + now_delete < edge_num)
        {
            if(((edge_delete * 3) >= edge_num or  accurate_tc == 0))
                break;
            edge_delete = 0;
            bool flag = 0;
            // 线程的数量开成的CPU的超线程数量就可以了， 这个num_thread应该是不用加的//
#pragma omp parallel for reduction(+:now_delete) reduction(+:edge_num) schedule(dynamic, 8)
            for(uint32_t i = 0; i < N; ++i)
            {
                const Bint_t rn = nbr_u_direct[i + 1];
                for(Bint_t j = nbr_u_direct[i]; j < rn; ++j)
                {
                    if(nbr_mask[j] == st)
                    {
                        delete_edge(nbr, fa, nbr_u, nbr_tc, nbr_mask, Edge_t(i, nbr_direct[j]), st, k);
                        now_delete += 1;
                        flag = 1;
                    }
                }
            }
            if(flag == 0)
                break;
            if((now_delete << 1) >= edge_num && now_delete < edge_num)
            {
                if(reconstruct_flag == 0 and k_step > 1)
                {
                    memcpy(nbr_direct_backup, nbr_direct, edge_num * sizeof(uint32_t));
                    memcpy(nbr_u_direct_backup, nbr_u_direct, (N + 1) * sizeof(Bint_t));
                    memcpy(nbr_backup, nbr, (edge_num << 1) * sizeof(uint32_t));
                    memcpy(fa_backup, fa, (edge_num << 1) * sizeof(Bint_t));
                    memcpy(nbr_u_backup, nbr_u, (N+1) * sizeof(Bint_t));
                    reconstruct_flag = 1;
                }
                reconstruct_Graph(nbr_u_direct, nbr_direct, edge_num, N, nbr_u, nbr, fa, nbr_mask, nbr_tc, st);
                now_delete = 0;
            }
            st++;
            
        #pragma omp parallel for reduction(+:edge_delete) schedule(dynamic, 8)
            for(uint32_t i = 0; i < N; ++i)
            {
                const Bint_t rn = nbr_u_direct[i + 1];
                for(Bint_t j = nbr_u_direct[i]; j < rn; ++j)
                {
                    if(nbr_mask[j] == st)
                    {
                        edge_delete ++;
                    }
                }
            }
            if(((edge_delete * 3) >= edge_num or  accurate_tc == 0) && now_delete + edge_delete < edge_num)
            {
                break;
            }
        }
        //temp.print("k truss time cost");
        if(((edge_delete * 3) >= edge_num or  accurate_tc == 0) && now_delete + edge_delete < edge_num)
        {
            if(reconstruct_flag == 0 and k_step > 1)
            {
                memcpy(nbr_direct_backup, nbr_direct, edge_num * sizeof(uint32_t));
                memcpy(nbr_u_direct_backup, nbr_u_direct, (N + 1) * sizeof(Bint_t));
                memcpy(nbr_backup, nbr, (edge_num << 1) * sizeof(uint32_t));
                memcpy(fa_backup, fa, (edge_num << 1) * sizeof(Bint_t));
                memcpy(nbr_u_backup, nbr_u, (N+1) * sizeof(Bint_t));
                reconstruct_flag = 1;
            }
            
            bool tmp_flag = 0;
            if(edge_delete *3 >= edge_num)
                tmp_flag = 1;
            
            reconstruct_Graph(nbr_u_direct, nbr_direct, edge_num, N, nbr_u, nbr, fa, nbr_mask, nbr_tc, st + 1);
            // 这一步理论上是有一点用的 //
            
            if(tmp_flag)
            {
                triangle_count(edge_num, N, nbr_u_direct, nbr_direct, nbr_tc, k>>1);
                //cout << "::" << edge_delete << " " << (k >> 1) << endl;
                accurate_tc = 0;
            }
            else
            {
                triangle_count(edge_num, N, nbr_u_direct, nbr_direct, nbr_tc);
                accurate_tc = 1;
            }
            now_delete = 0;
            k-=(k_step << 1);
            continue;
        }
        // 如果刚好所有边都在此轮被删除，将edge_delete叠加到当前已经删除的边上，不需要真的将其删除，只需要知道数量即可 //
        now_delete += edge_delete;
        // 判断一个这个k是否是k_max
        if(edge_num - now_delete > 0 && now_delete <= edge_num)
        {
            //cout << "k = " << (k>>1) + 2 << ", edge count = " << edge_num - now_delete << endl;
            k_max = (k>>1)+2;
            k_max_num = edge_num - now_delete;
            if(edge_num - now_delete == ans and k == k_min << 1)
                break;
        }
        if(now_delete >= edge_num)
        {
            // 大小步策略，如果步长过大则会缩短步长, 如果步长为1则不需要backup//
            if(k_step == 1)
            {
                break;
            }
            else
            {
                k -= (k_step << 1);
                if(k_step > 20)
                    k_step = 20;
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
                swap(fa, fa_backup);
            }
            swap(nbr_mask, nbr_mask_backup);
            swap(nbr_tc, nbr_tc_backup);
            now_delete = now_delete_backup;
            edge_num = edge_num_backup;
            st = st_backup;
            accurate_tc = accurate_tc_backup;
            // TODO ！！！ 如果三角形数量不准确，则重新计算三角形 //
            if(accurate_tc == 0)
            {
                triangle_count(edge_num, N, nbr_u_direct, nbr_direct, nbr_tc);
                accurate_tc = 1;
            }
        }
        // 大步结束 //
        if(k_step > 20)
            k_step = 20;
        reconstruct_flag = 0;
    }
    delete nbr_tc, nbr, nbr_direct, nbr_u, fa, nbr_u_direct, nbr_mask;
    delete nbr_tc_backup, nbr_backup, nbr_direct_backup, nbr_u_backup, nbr_mask_backup;
    delete fa_backup, nbr_u_direct_backup;
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
    const char *filepath = argv[2];
#ifdef DEBUG_DF
    cout << filepath << endl;
#endif
    // 将文件映射到内存
    const string command_rd = "vmtouch -qt " + string(filepath);
    MapFile mapfile(filepath);
    const uint64_t file_lenth = mapfile.getLen();
    char *tsvdata = (char *) mapfile.getAddr();
    uint64_t len_edgeList = min(uint64_t(1800000000), max(file_lenth / 10, uint64_t(3000000)));
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
    deg_origin = new uint32_t[min(uint64_t(60000000), max(file_lenth / 10, uint64_t(3000000)))]();
    // 此处利用更好的并行 //
    uint64_t file_read_step = max(file_lenth / NUM_THREADS_THIS_COMPUTER / 5, uint64_t(1024));
    int s = system(command_rd.c_str());
    assert(s == 0);
    // bp save begin pos and end pos //
    uint64_t *bp = new uint64_t[1024];
#pragma omp parallel for reduction(max: N) schedule(dynamic, 1)
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
#ifdef DEBUG_DF
    tmpTime.print("Build graph to adjList cost");
    tmpTime.check();
#endif 

    Bint_t ans = 0;
    uint32_t k_max = kmax_truss(N, nbr_origin_u, nbr_origin, deg_origin, uint32_t(-1), ans);
#ifdef DEBUG_DF
    cout << "kmax_guess = " << k_max << ", Edges in kmax-truss = " << ans <<".\n";
#endif 
    k_max = kmax_truss(N, nbr_origin_u, nbr_origin, deg_origin, k_max == 0? 2 : k_max - 1, ans);
    cout << "kmax = " << k_max << ", Edges in kmax-truss = " << ans <<".\n";
    
#ifdef DEBUG_DF
    allTime.print("All time cost");
#endif
    return 0;
}