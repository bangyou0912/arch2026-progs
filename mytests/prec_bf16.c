/* Precision of src/bf16/bf16.c against a double-based reference, over all
 * 2^32 inputs of from_f32, all 2^32 operand pairs of mul and all 65,536
 * inputs of sqrt. Build and run from the repository root:
 *
 *   cc -O2 -I. -o prec_bf16 mytests/prec_bf16.c src/bf16/bf16.c -lm -lpthread
 *   ./prec_bf16 [f32|mul|sqrt|all]
 */
#include <math.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "src/bf16/bf16.h"

#define NTH 10

static double bf2d(uint16_t h)
{
    uint32_t u = (uint32_t) h << 16;
    float f;
    memcpy(&f, &u, 4);
    return (double) f;
}
static double f2d(uint32_t u)
{
    float f;
    memcpy(&f, &u, 4);
    return (double) f;
}

/* Correct RNE rounding of a finite double to bf16 (with gradual underflow,
 * overflow to Inf). mode 0 = RNE, 1 = RTZ. Independent of the bit trick. */
static uint16_t d2bf(double x, int mode)
{
    uint16_t s = signbit(x) ? 0x8000 : 0;
    double a = fabs(x);
    if (a == 0)
        return s;
    int e;
    frexp(a, &e); /* a in [2^(e-1), 2^e) */
    int E = e - 1;
    if (E < -126)
        E = -126;
    double q = ldexp(1.0, E - 7); /* quantum */
    double n = a / q;             /* exact (power-of-two scaling) */
    double r = (mode == 0 ? nearbyint(n) : trunc(n)) * q;
    if (r >= ldexp(1.0, 128))
        return mode == 0 ? (s | 0x7F80) : (s | 0x7F7F);
    float f = (float) r; /* exact: r has <= 8 significant bits */
    uint32_t u;
    memcpy(&u, &f, 4);
    return s | (uint16_t) (u >> 16);
}

/* ---------------- bf16_from_f32 ---------------- */
typedef struct {
    uint64_t finite, mism, nan_in, nan_to_inf, nan_mism, ovf_inf;
    double maxrel_norm;
    uint32_t arg_norm;
    double maxrel_sub;
    uint32_t arg_sub;
    double maxrel_nin;
    uint32_t arg_nin;
    uint32_t first_mism;
} f32st;

/* cppcheck-suppress constParameterCallback */
static void *f32_worker(void *p)
{
    long t = (long) p;
    f32st *st = calloc(1, sizeof *st);
    if (!st)
        return NULL;
    uint64_t lo = ((uint64_t) 1 << 32) * t / NTH,
             hi = ((uint64_t) 1 << 32) * (t + 1) / NTH;
    for (uint64_t i = lo; i < hi; i++) {
        uint32_t u = (uint32_t) i;
        uint16_t r = bf16_from_f32(u);
        if (((u >> 23) & 0xFF) == 0xFF) {
            if (u & 0x7FFFFF) {
                st->nan_in++;
                if ((r & 0x7FFF) == 0x7F80)
                    st->nan_to_inf++;
                if (r != (uint16_t) (u >> 16))
                    st->nan_mism++;
            } else if (r != (uint16_t) (u >> 16))
                st->nan_mism++;
            continue;
        }
        st->finite++;
        double x = f2d(u);
        uint16_t ref = d2bf(x, 0);
        if (r != ref) {
            if (!st->mism)
                st->first_mism = u;
            st->mism++;
        }
        if ((r & 0x7FFF) == 0x7F80) {
            st->ovf_inf++;
            continue;
        }
        if (x == 0)
            continue;
        double rel = fabs(bf2d(r) - x) / fabs(x);
        if (((u >> 23) & 0xFF) && rel > st->maxrel_nin) {
            st->maxrel_nin = rel;
            st->arg_nin = u;
        }
        int normal_res = (r & 0x7F80) != 0;
        if (normal_res) {
            if (rel > st->maxrel_norm) {
                st->maxrel_norm = rel;
                st->arg_norm = u;
            }
        } else {
            if (rel > st->maxrel_sub) {
                st->maxrel_sub = rel;
                st->arg_sub = u;
            }
        }
    }
    return st;
}

static void run_f32(void)
{
    pthread_t th[NTH];
    for (long t = 0; t < NTH; t++)
        pthread_create(&th[t], 0, f32_worker, (void *) t);
    f32st T = {0};
    for (int t = 0; t < NTH; t++) {
        f32st *s;
        pthread_join(th[t], (void **) &s);
        if (!s) {
            fprintf(stderr, "out of memory\n");
            exit(1);
        }
        if (s->mism && !T.mism)
            T.first_mism = s->first_mism;
        T.finite += s->finite;
        T.mism += s->mism;
        T.nan_in += s->nan_in;
        T.nan_to_inf += s->nan_to_inf;
        T.nan_mism += s->nan_mism;
        T.ovf_inf += s->ovf_inf;
        if (s->maxrel_norm > T.maxrel_norm) {
            T.maxrel_norm = s->maxrel_norm;
            T.arg_norm = s->arg_norm;
        }
        if (s->maxrel_nin > T.maxrel_nin) {
            T.maxrel_nin = s->maxrel_nin;
            T.arg_nin = s->arg_nin;
        }
        if (s->maxrel_sub > T.maxrel_sub) {
            T.maxrel_sub = s->maxrel_sub;
            T.arg_sub = s->arg_sub;
        }
        free(s);
    }
    printf("== bf16_from_f32 (all 2^32 patterns) ==\n");
    printf("finite inputs           : %llu\n", (unsigned long long) T.finite);
    printf("mismatch vs RNE ref     : %llu (first 0x%08X)\n",
           (unsigned long long) T.mism, T.first_mism);
    printf("finite -> Inf (overflow): %llu\n", (unsigned long long) T.ovf_inf);
    printf(
        "NaN inputs              : %llu, became Inf: %llu, Inf/NaN not top16: "
        "%llu\n",
        (unsigned long long) T.nan_in, (unsigned long long) T.nan_to_inf,
        (unsigned long long) T.nan_mism);
    printf(
        "max rel err (normal res): %.10g = 2^%.6f at 0x%08X (%.9g -> 0x%04X = "
        "%.9g)\n",
        T.maxrel_norm, log2(T.maxrel_norm), T.arg_norm, f2d(T.arg_norm),
        bf16_from_f32(T.arg_norm), bf2d(bf16_from_f32(T.arg_norm)));
    printf(
        "max rel err (normal f32 input, finite res): %.10g = 2^%.6f at 0x%08X "
        "(%.9g -> 0x%04X = %.9g)\n",
        T.maxrel_nin, log2(T.maxrel_nin), T.arg_nin, f2d(T.arg_nin),
        bf16_from_f32(T.arg_nin), bf2d(bf16_from_f32(T.arg_nin)));
    printf(
        "max rel err (subnormal/zero res): %.10g at 0x%08X (%.9g -> 0x%04X = "
        "%.9g)\n",
        T.maxrel_sub, T.arg_sub, f2d(T.arg_sub), bf16_from_f32(T.arg_sub),
        bf2d(bf16_from_f32(T.arg_sub)));
}

/* ---------------- bf16_mul ---------------- */
typedef struct {
    double v;
    uint16_t a, b;
} mx;
static void upd(mx *m, double v, uint16_t a, uint16_t b)
{
    if (v > m->v) {
        m->v = v;
        m->a = a;
        m->b = b;
    }
}

typedef struct {
    uint64_t pairs; /* finite nonzero x finite nonzero */
    uint64_t n_norm, norm_ne_rne, norm_ne_rtz, norm_eq_rne;
    mx ulp_norm, rel_norm;
    uint64_t n_sub, sub_ne_rne, sub_ne_rtz;
    mx rel_sub, ulp_sub; /* ulp_sub in units of 2^-133 */
    uint64_t n_zero, zero_ref_nonzero;
    mx rel_zero_nz; /* zero result though RNE ref nonzero */
    uint64_t n_inf, inf_ref_finite;
    uint64_t n_ref_sub;  /* RNE reference is subnormal */
    uint64_t ref_sub_eq; /* ... and contract matches it */
} mulst;

/* cppcheck-suppress constParameterCallback */
static void *mul_worker(void *p)
{
    long t = (long) p;
    mulst *st = calloc(1, sizeof *st);
    if (!st)
        return NULL;
    const double ulp_sub_unit = ldexp(1.0, -133);
    for (uint32_t a = t; a < 65536; a += NTH) {
        int ea = (a >> 7) & 0xFF;
        if (ea == 0xFF || (a & 0x7FFF) == 0)
            continue;
        double da = bf2d(a);
        for (uint32_t b = 0; b < 65536; b++) {
            int eb = (b >> 7) & 0xFF;
            if (eb == 0xFF || (b & 0x7FFF) == 0)
                continue;
            st->pairs++;
            uint16_t r = bf16_mul(a, b);
            double ex = da * bf2d(b); /* exact: 8x8-bit significands */
            uint16_t rne = d2bf(ex, 0), rtz = d2bf(ex, 1);
            double dr = bf2d(r);
            int er = (r >> 7) & 0xFF;
            if ((rne & 0x7F80) == 0 && (rne & 0x7F)) {
                st->n_ref_sub++;
                if (r == rne)
                    st->ref_sub_eq++;
            }
            if (er == 0xFF) {
                st->n_inf++;
                if ((rne & 0x7F80) != 0x7F80)
                    st->inf_ref_finite++;
            } else if (er) {
                st->n_norm++;
                if (r != rne)
                    st->norm_ne_rne++;
                else
                    st->norm_eq_rne++;
                if (r != rtz)
                    st->norm_ne_rtz++;
                double ulp = ldexp(1.0, er - 127 - 7);
                upd(&st->ulp_norm, fabs(ex - dr) / ulp, a, b);
                upd(&st->rel_norm, fabs(ex - dr) / fabs(ex), a, b);
            } else if (r & 0x7F) {
                st->n_sub++;
                if (r != rne)
                    st->sub_ne_rne++;
                if (r != rtz)
                    st->sub_ne_rtz++;
                upd(&st->rel_sub, fabs(ex - dr) / fabs(ex), a, b);
                upd(&st->ulp_sub, fabs(ex - dr) / ulp_sub_unit, a, b);
            } else {
                st->n_zero++;
                if (rne & 0x7FFF) {
                    st->zero_ref_nonzero++;
                    upd(&st->rel_zero_nz, fabs(ex - dr) / fabs(ex), a, b);
                }
            }
        }
    }
    return st;
}

static void pr_mx(const char *name, mx m, int ulp)
{
    double ex = bf2d(m.a) * bf2d(m.b);
    uint16_t r = bf16_mul(m.a, m.b);
    printf(
        "%-26s: %.10g%s  a=0x%04X (%.9g) b=0x%04X (%.9g) exact=%.12g got "
        "0x%04X (%.12g) RNE 0x%04X\n",
        name, m.v, ulp ? " ulp" : "", m.a, bf2d(m.a), m.b, bf2d(m.b), ex, r,
        bf2d(r), d2bf(ex, 0));
}

static void run_mul(void)
{
    pthread_t th[NTH];
    for (long t = 0; t < NTH; t++)
        pthread_create(&th[t], 0, mul_worker, (void *) t);
    mulst T = {0};
#define ADD(f) T.f += s->f
#define MX(f)           \
    if (s->f.v > T.f.v) \
    T.f = s->f
    for (int t = 0; t < NTH; t++) {
        mulst *s;
        pthread_join(th[t], (void **) &s);
        if (!s) {
            fprintf(stderr, "out of memory\n");
            exit(1);
        }
        ADD(pairs);
        ADD(n_norm);
        ADD(norm_ne_rne);
        ADD(norm_ne_rtz);
        ADD(norm_eq_rne);
        ADD(n_sub);
        ADD(sub_ne_rne);
        ADD(sub_ne_rtz);
        ADD(n_zero);
        ADD(zero_ref_nonzero);
        ADD(n_inf);
        ADD(inf_ref_finite);
        ADD(n_ref_sub);
        ADD(ref_sub_eq);
        MX(ulp_norm);
        MX(rel_norm);
        MX(rel_sub);
        MX(ulp_sub);
        MX(rel_zero_nz);
        free(s);
    }
    /* Special operands: check against the contract text, all pairs with a or b
     * special/zero. */
    uint64_t spec = 0, spec_bad = 0;
    for (uint32_t a = 0; a < 65536; a++)
        for (uint32_t b = 0; b < 65536; b++) {
            int ea = (a >> 7) & 0xFF, eb = (b >> 7) & 0xFF;
            int za = !(a & 0x7FFF), zb = !(b & 0x7FFF);
            if (!(ea == 0xFF || eb == 0xFF || za || zb))
                continue;
            spec++;
            uint16_t s = (a ^ b) & 0x8000, want;
            if (ea == 0xFF)
                want = (a & 0x7F) ? a : zb ? 0x7FC0 : (s | 0x7F80);
            else if (eb == 0xFF)
                want = (b & 0x7F) ? b : za ? 0x7FC0 : (s | 0x7F80);
            else
                want = s;
            if (bf16_mul(a, b) != want)
                spec_bad++;
        }
    printf("== bf16_mul (all 2^32 pairs) ==\n");
    printf(
        "special/zero-operand pairs: %llu, differ from contract text: %llu\n",
        (unsigned long long) spec, (unsigned long long) spec_bad);
    printf("finite nonzero pairs      : %llu\n", (unsigned long long) T.pairs);
    printf(
        "normal results            : %llu; == RTZ: %llu mismatches; == RNE: "
        "%llu (%.4f%%), != RNE: %llu (%.4f%%)\n",
        (unsigned long long) T.n_norm, (unsigned long long) T.norm_ne_rtz,
        (unsigned long long) T.norm_eq_rne, 100.0 * T.norm_eq_rne / T.n_norm,
        (unsigned long long) T.norm_ne_rne, 100.0 * T.norm_ne_rne / T.n_norm);
    pr_mx("max ulp err (normal)", T.ulp_norm, 1);
    pr_mx("max rel err (normal)", T.rel_norm, 0);
    printf("  (2^-7 = %.10g, log2(max rel) = %.6f)\n", ldexp(1, -7),
           log2(T.rel_norm.v));
    printf("subnormal results         : %llu; != RNE %llu, != RTZ %llu\n",
           (unsigned long long) T.n_sub, (unsigned long long) T.sub_ne_rne,
           (unsigned long long) T.sub_ne_rtz);
    pr_mx("max rel err (subnormal)", T.rel_sub, 0);
    pr_mx("max abs err (subn, 2^-133u)", T.ulp_sub, 1);
    printf("zero results (nonzero ops): %llu; of which RNE ref nonzero: %llu\n",
           (unsigned long long) T.n_zero,
           (unsigned long long) T.zero_ref_nonzero);
    if (T.zero_ref_nonzero)
        pr_mx("  (flushed, rel err)", T.rel_zero_nz, 0);
    printf(
        "RNE ref subnormal         : %llu, contract matches RNE there: %llu "
        "(%.4f%%)\n",
        (unsigned long long) T.n_ref_sub, (unsigned long long) T.ref_sub_eq,
        100.0 * T.ref_sub_eq / T.n_ref_sub);
    printf("Inf results               : %llu; of which RNE ref finite: %llu\n",
           (unsigned long long) T.n_inf, (unsigned long long) T.inf_ref_finite);
}

/* ---------------- bf16_sqrt ---------------- */
static void run_sqrt(void)
{
    uint64_t n = 0, eq = 0, eq_rtz = 0, below = 0, above = 0, spec_bad = 0;
    double maxrel = 0, maxulp = 0, maxrel_rne = 0;
    uint16_t arel = 0, aulp = 0;
    int ulp_hist[8] = {0};
    for (uint32_t a = 0; a < 65536; a++) {
        int e = (a >> 7) & 0xFF;
        uint16_t r = bf16_sqrt(a);
        if (!(a >= 0x0080 && a <= 0x7F7F)) {
            uint16_t want;
            if (e == 0xFF)
                want = (a & 0x7F) ? a : (a & 0x8000) ? 0x7FC0 : a;
            else if (!(a & 0x7FFF))
                want = 0;
            else if (a & 0x8000)
                want = 0x7FC0;
            else
                want = 0; /* denormal */
            if (r != want)
                spec_bad++;
            continue;
        }
        n++;
        double x = bf2d(a), s = sqrt(x), dr = bf2d(r);
        uint16_t rne = d2bf(s, 0), rtz = d2bf(s, 1);
        if (r == rne)
            eq++;
        if (r == rtz)
            eq_rtz++;
        if (dr < s)
            below++;
        else if (dr > s)
            above++;
        double rel = fabs(dr - s) / s;
        double ulp = ldexp(1.0, ((r >> 7) & 0xFF) - 127 - 7);
        double ue = fabs(dr - s) / ulp;
        int ui = (int) (fabs((double) (int) r - (double) (int) rne));
        if (ui > 7)
            ui = 7;
        ulp_hist[ui]++;
        double relr = fabs(bf2d(rne) - s) / s;
        if (relr > maxrel_rne)
            maxrel_rne = relr;
        if (rel > maxrel) {
            maxrel = rel;
            arel = a;
        }
        if (ue > maxulp) {
            maxulp = ue;
            aulp = a;
        }
    }
    printf("== bf16_sqrt (all 65536 inputs) ==\n");
    printf("non-positive-normal inputs differing from contract text: %llu\n",
           (unsigned long long) spec_bad);
    printf(
        "positive normals: %llu; == RNE sqrt: %llu (%.4f%%); == RTZ sqrt: %llu "
        "(%.4f%%)\n",
        (unsigned long long) n, (unsigned long long) eq, 100.0 * eq / n,
        (unsigned long long) eq_rtz, 100.0 * eq_rtz / n);
    printf("result below exact: %llu, above: %llu, exact: %llu\n",
           (unsigned long long) below, (unsigned long long) above,
           (unsigned long long) (n - below - above));
    printf("|r - RNE| in bf16 steps: 0:%d 1:%d 2:%d 3:%d >=4:%d\n", ulp_hist[0],
           ulp_hist[1], ulp_hist[2], ulp_hist[3],
           ulp_hist[4] + ulp_hist[5] + ulp_hist[6] + ulp_hist[7]);
    printf(
        "max rel err: %.10g (%.6f%%) at a=0x%04X (%.9g): got 0x%04X (%.9g), "
        "sqrt=%.12g, RNE 0x%04X\n",
        maxrel, 100 * maxrel, arel, bf2d(arel), bf16_sqrt(arel),
        bf2d(bf16_sqrt(arel)), sqrt(bf2d(arel)), d2bf(sqrt(bf2d(arel)), 0));
    printf(
        "max ulp err: %.6f ulp at a=0x%04X (%.9g): got 0x%04X (%.9g), "
        "sqrt=%.12g, RNE 0x%04X\n",
        maxulp, aulp, bf2d(aulp), bf16_sqrt(aulp), bf2d(bf16_sqrt(aulp)),
        sqrt(bf2d(aulp)), d2bf(sqrt(bf2d(aulp)), 0));
    printf(
        "(for comparison, max rel err of correctly rounded sqrt: %.10g = "
        "%.6f%%)\n",
        maxrel_rne, 100 * maxrel_rne);
}

int main(int argc, char **argv)
{
    const char *w = argc > 1 ? argv[1] : "all";
    if (!strcmp(w, "f32") || !strcmp(w, "all"))
        run_f32();
    if (!strcmp(w, "mul") || !strcmp(w, "all"))
        run_mul();
    if (!strcmp(w, "sqrt") || !strcmp(w, "all"))
        run_sqrt();
    return 0;
}
