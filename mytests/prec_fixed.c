/* Precision of src/fixed/fixed.c against exact 128-bit and 64-bit
 * references: q_mul over edge values and random pairs, op1 to op3 over random
 * pairs and edges. Build and run from the repository root:
 *
 *   cc -O2 -I. -o prec_fixed mytests/prec_fixed.c src/fixed/fixed.c
 *   ./prec_fixed [N]       N random pairs per test, default 10^8
 */
#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include "src/fixed/fixed.h"

static uint64_t rs = 0x9E3779B97F4A7C15ull;
static uint64_t rnd(void) /* splitmix64 */
{
    uint64_t z = (rs += 0x9E3779B97F4A7C15ull);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
}
/* Random int32 with a uniformly random bit length (so small and large
 * magnitudes both occur) and random sign. */
static int32_t rnd_scaled(void)
{
    uint64_t r = rnd();
    int bits = (int) (r % 33); /* 0..32 */
    uint32_t v = bits == 32 ? (uint32_t) (r >> 32)
                            : (uint32_t) (r >> 32) & ((1u << bits) - 1);
    return (r >> 8) & 1 ? (int32_t) (0u - v) : (int32_t) v;
}

/* ------------------------- q_mul ------------------------- */
typedef struct {
    uint64_t n, inrange, ovf, ovf_bad, ties, ties_up, ties_bad, bad_round;
    __int128 maxerr;
    int32_t ma, mb; /* max |q*2^20 - a*b| among in-range */
} qst;

static void q_check(qst *s, int32_t a, int32_t b)
{
    __int128 p = (__int128) a * b; /* exact, units 2^-40 */
    /* exact reference: floor((p + 2^19) / 2^20), done with __int128 floor div
     */
    __int128 t = p + ((__int128) 1 << 19);
    __int128 fl = t >= 0 ? t / (1 << 20) : -((-t + (1 << 20) - 1) / (1 << 20));
    int32_t q = q_mul(a, b);
    s->n++;
    if (fl >= INT32_MIN && fl <= INT32_MAX) {
        s->inrange++;
        if (q != (int32_t) fl)
            s->bad_round++;
        __int128 d = (__int128) q * (1 << 20) - p; /* error in units 2^-40 */
        __int128 ad = d < 0 ? -d : d;
        if (ad > s->maxerr) {
            s->maxerr = ad;
            s->ma = a;
            s->mb = b;
        }
        if ((p & 0xFFFFF) == 0x80000) {
            s->ties++;
            if (d == ((__int128) 1 << 19))
                s->ties_up++;
            else
                s->ties_bad++;
        }
    } else {
        s->ovf++;
        if ((uint32_t) q != (uint32_t) (uint64_t) fl)
            s->ovf_bad++;
    }
}

static void q_report(const char *name, const qst *s)
{
    printf("%-22s n=%" PRIu64 " in-range=%" PRIu64 " overflow(wrap)=%" PRIu64
           " (%.4f%%) wrap!=low32: %" PRIu64 "\n",
           name, s->n, s->inrange, s->ovf, 100.0 * s->ovf / s->n, s->ovf_bad);
    printf("%-22s != floor(p/2^20+1/2): %" PRIu64
           "; max |err| = %.6f ulp(2^-20) at a=0x%08X (%.9f) b=0x%08X (%.9f)\n",
           "", s->bad_round, (double) s->maxerr / (1 << 20), (uint32_t) s->ma,
           s->ma / 1048576.0, (uint32_t) s->mb, s->mb / 1048576.0);
    printf("%-22s exact ties: %" PRIu64 ", rounded toward +inf: %" PRIu64
           ", otherwise: %" PRIu64 "\n",
           "", s->ties, s->ties_up, s->ties_bad);
}

/* ------------------------- op1/op2/op3 ------------------------- */
/* Independent pass-count reference: the loop ends after pass k iff the carry
 * word c_k is zero. c_k is nonzero iff some carry chain is >= k long, so count
 * passes by simulating carries in 64-bit (separate code, same semantics). */
static int ref_passes(uint32_t a, uint32_t b)
{
    uint64_t x = a, y = b;
    int k = 0;
    while (y & 0xFFFFFFFFu) {
        uint64_t c = ((x & y) << 1) & 0xFFFFFFFFu;
        x = (x ^ y) & 0xFFFFFFFFu;
        y = c;
        k++;
    }
    return k;
}

typedef struct {
    uint64_t n, sum_bad, pass_bad, d_bad, avg_bad;
    uint64_t hist[34];
    int maxp;
    uint32_t pa, pb;
} ost;

static void o_check(ost *s, uint32_t a, uint32_t b)
{
    int n = -1;
    uint32_t r = op1(a, b, &n);
    s->n++;
    if (r != (uint32_t) ((uint64_t) a + b))
        s->sum_bad++;
    int rp = ref_passes(a, b);
    if (n != rp)
        s->pass_bad++;
    if (n >= 0 && n <= 33)
        s->hist[n]++;
    if (n > s->maxp) {
        s->maxp = n;
        s->pa = a;
        s->pb = b;
    }
    if (op2(a, b) != (uint32_t) ((uint64_t) a - b))
        s->d_bad++;
    if (op3(a, b) != (uint32_t) (((uint64_t) a + b) >> 1))
        s->avg_bad++;
}

int main(int argc, char **argv)
{
    uint64_t N = argc > 1 ? strtoull(argv[1], 0, 10) : 100000000ull;

    /* q_mul edge set: all pairs from a list of edge values */
    static const int32_t E[] = {INT32_MIN,   INT32_MIN + 1,
                                INT32_MAX,   INT32_MAX - 1,
                                0,           1,
                                -1,          2,
                                -2,          3,
                                -3,          524288,
                                -524288,     524287,
                                524289,      1048576,
                                -1048576,    1048575,
                                1048577,     -1048575,
                                1 << 30,     -(1 << 30),
                                46340 << 10, 0x5A827,
                                -0x5A827,    0x16A09E66,
                                -0x16A09E66, 0xB504F,
                                0xB504F3,    0x7FFF,
                                0x8000,      0xFFFF,
                                0x10000,     -0x10000,
                                0x7FFFFFF,   0x8000000};
    int NE = sizeof E / sizeof E[0];
    qst qe = {0}, qu = {0}, qs = {0};
    for (int i = 0; i < NE; i++)
        for (int j = 0; j < NE; j++)
            q_check(&qe, E[i], E[j]);
    for (uint64_t i = 0; i < N; i++) {
        uint64_t r = rnd();
        q_check(&qu, (int32_t) r, (int32_t) (r >> 32));
    }
    for (uint64_t i = 0; i < N; i++)
        q_check(&qs, rnd_scaled(), rnd_scaled());
    printf("== q_mul (Q11.20) ==\n");
    q_report("edges (all pairs)", &qe);
    q_report("uniform int32 pairs", &qu);
    q_report("log-uniform magnitude", &qs);
    printf(
        "examples: q_mul(3,524288)=%d q_mul(-3,524288)=%d q_mul(1,524288)=%d "
        "q_mul(-1,524288)=%d\n",
        q_mul(3, 524288), q_mul(-3, 524288), q_mul(1, 524288),
        q_mul(-1, 524288));
    printf(
        "          q_mul(INT32_MIN,INT32_MIN)=0x%08X "
        "q_mul(INT32_MIN,-1048576)=0x%08X q_mul(INT32_MAX,INT32_MAX)=0x%08X\n",
        (uint32_t) q_mul(INT32_MIN, INT32_MIN),
        (uint32_t) q_mul(INT32_MIN, -1048576),
        (uint32_t) q_mul(INT32_MAX, INT32_MAX));

    /* op1/op2/op3 */
    static const uint32_t OE[] = {
        0,          1,          2,          3,          0x7FFFFFFF,
        0x80000000, 0x80000001, 0xFFFFFFFF, 0xFFFFFFFE, 0xFFFF,
        0x10000,    0x55555555, 0xAAAAAAAA, 0x7FFFFFFE, 0x40000000};
    int NO = sizeof OE / sizeof OE[0];
    ost oe = {0}, ou = {0};
    for (int i = 0; i < NO; i++)
        for (int j = 0; j < NO; j++)
            o_check(&oe, OE[i], OE[j]);
    for (uint64_t i = 0; i < N; i++) {
        uint64_t r = rnd();
        o_check(&ou, (uint32_t) r, (uint32_t) (r >> 32));
    }
    /* exhaustive over b for the worst a, and the full-chain family */
    ost ox = {0};
    for (uint32_t k = 0; k < 32; k++)
        o_check(&ox, 0xFFFFFFFFu, 1u << k), o_check(&ox, 1u << k, 0xFFFFFFFFu);
    printf("== op1/op2/op3 ==\n");
    const ost *const L[] = {&oe, &ou, &ox};
    const char *nm[] = {"edges (all pairs)", "uniform pairs",
                        "0xFFFFFFFF x 2^k"};
    for (int t = 0; t < 3; t++) {
        const ost *s = L[t];
        printf("%-20s n=%" PRIu64 " op1 sum wrong=%" PRIu64
               " passes wrong=%" PRIu64 " op2 wrong=%" PRIu64
               " op3 wrong=%" PRIu64 " max passes=%d at a=0x%08X b=0x%08X\n",
               nm[t], s->n, s->sum_bad, s->pass_bad, s->d_bad, s->avg_bad,
               s->maxp, s->pa, s->pb);
    }
    printf("pass-count histogram (uniform pairs):\n");
    for (int k = 0; k <= 33; k++)
        if (ou.hist[k])
            printf("  %2d: %" PRIu64 " (%.6f%%)\n", k, ou.hist[k],
                   100.0 * ou.hist[k] / ou.n);
    double mean = 0;
    for (int k = 0; k <= 33; k++)
        mean += (double) k * ou.hist[k];
    printf("  mean passes = %.4f\n", mean / ou.n);
    int n;
    op1(0xFFFFFFFF, 1, &n);
    printf("op1(0xFFFFFFFF,1) passes=%d; ", n);
    op1(0xFFFF, 1, &n);
    printf("op1(0xFFFF,1) passes=%d; ", n);
    op1(1, 0xFFFFFFFF, &n);
    printf("op1(1,0xFFFFFFFF) passes=%d\n", n);
    return 0;
}
