#ifndef KC_GPU_H
#define KC_GPU_H

#include <cstdint>
#include <unistd.h>
typedef uint32_t Bint_t;
#define size_deg 1ll * sizeof(uint32_t) * N
#define size_nbr 1ll * sizeof(uint32_t) * edge_num * 2
#define size_mask 1ll * sizeof(uint32_t) * N
#define size_nbr_u 1ll * sizeof(Bint_t) * (N + 1)
#define size_nbr_tc 1ll * sizeof(uint32_t) * edge_num
#define size_nbr_tc_double 1ll * sizeof(uint32_t) * edge_num * 2
#define size_nbr_direct 1ll * sizeof(uint32_t) * edge_num
#define size_nbr_u_direct 1ll * sizeof(Bint_t) * (N + 1)
#define size_nbr_mask 1ll * sizeof(uint32_t) * edge_num
#define size_nbr_mask_double 1ll * sizeof(uint32_t) * edge_num * 2
#define size_nbr_direct_start 1ll * sizeof(uint32_t) * edge_num
#define size_fa 1ll * sizeof(uint32_t) * edge_num * 2
#define size_left_son 1ll * sizeof(Bint_t) * edge_num
#define size_right_son 1ll * sizeof(Bint_t) * edge_num 
#define K_MIDDLE 19
#define K_CORE_STEP_LARGE 500
#define K_CORE_STEP_MIDDLE 25

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

bool cmp_nbr(Nbr_t &a, Nbr_t &b);
bool cmp_link(Link_t &a, Link_t &b);

void initGPU(const Bint_t edge_num, const uint32_t N);

void tricount_direct(const Bint_t *nbr_u_direct, const uint32_t *nbr_direct, const uint32_t N, const Bint_t *nbr_u, const uint32_t *nbr, 
                           uint32_t *nbr_tc, const Bint_t *left_son, const Bint_t *right_son, uint32_t *nbr_tc_double, uint32_t tc_limit = 0);

uint32_t kmax_core_gpu(const uint32_t N, const Bint_t *nbr_u, const uint32_t *nbr, const uint32_t *deg_origin, uint32_t *&mask, uint32_t Launch_start);

void kmax_truss_GPU(uint32_t &k_max, Bint_t &k_max_num, Bint_t ans, uint32_t N, uint32_t k, uint32_t k_step,
                    Bint_t *&nbr_u_direct, uint32_t *&nbr_direct, Bint_t *&nbr_u, uint32_t *&nbr, Bint_t *&fa, Bint_t *&left_son, Bint_t *&right_son, 
                    uint32_t *&nbr_mask, uint32_t *&nbr_mask_double, uint32_t *&nbr_tc, uint32_t *&nbr_tc_double, uint32_t &accurate_tc);

void reconstruct_Graph(Bint_t *&nbr_u_direct, uint32_t *&nbr_direct, Bint_t &edge_num, const uint32_t N, 
                     Bint_t *&nbr_u, uint32_t *&nbr, Bint_t *&fa, Bint_t *&left_son, Bint_t *&right_son,
                     uint32_t *&nbr_mask, uint32_t *&nbr_tc, uint32_t *&nbr_tc_double, uint32_t st);

void count_deg_with_mask(const uint32_t *&nbr, const Bint_t *&nbr_u, uint32_t N, uint32_t *&node_delete, uint32_t *&deg_with_mask,
                         uint32_t *&to, Link_t *&li, uint32_t &new_N, uint32_t *&deg, uint32_t *&deg_direct);
void build_nbr_direct(const uint32_t *&nbr, const Bint_t *&nbr_u, uint32_t N, uint32_t *&node_delete, uint32_t *&to,
                      uint32_t *&deg, Bint_t *&nbr_u_direct, uint32_t *&nbr_direct, uint32_t new_N);
void build_graph_from_direct(uint32_t *&nbr_direct, Bint_t *&nbr_u_direct, uint32_t N, uint32_t *&nbr, Bint_t *&nbr_u,
                             Bint_t *&fa, Bint_t *&left_son, Bint_t *&right_son);
void test_cuda();
#endif