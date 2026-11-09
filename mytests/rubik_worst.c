/* Worst-case instruction counts of rubik_solve and rubik_solve_asm on
 * rv32emu over the states in rubik_states.h. Build and run with
 * `make -f mytests/Makefile` from the repository root.
 */
#include <stddef.h>
#include <stdint.h>
#include "print.h"
#include "rubik_states.h"
#include "src/rubik/rubik.h"

extern uint64_t get_instret(void);

static uint64_t measure(int (*solve)(const char *, char *, size_t),
                        const char *s,
                        int *n)
{
    char out[96];
    uint64_t t0 = get_instret();
    *n = solve(s, out, sizeof out);
    uint64_t t1 = get_instret();
    uint64_t t2 = get_instret();
    return (t1 - t0) - (get_instret() - t2);
}

int main(void)
{
    uint64_t worst_c[12] = {0}, worst_a[12] = {0};
    const char *ws_c[12] = {0}, *ws_a[12] = {0};
    int bad = 0;
    for (size_t i = 0; i < sizeof worst_cases / sizeof worst_cases[0]; i++) {
        const char *s = worst_cases[i].state;
        int d = worst_cases[i].distance, nc, na;
        uint64_t c = measure(rubik_solve, s, &nc);
        uint64_t a = measure(rubik_solve_asm, s, &na);
        if (nc != d || na != d) {
            bm_printf("MISMATCH %s: distance %d, C %d, asm %d\n", s, d, nc, na);
            bad = 1;
        }
        if (c > worst_c[d]) {
            worst_c[d] = c;
            ws_c[d] = s;
        }
        if (a > worst_a[d]) {
            worst_a[d] = a;
            ws_a[d] = s;
        }
    }
    for (int d = 6; d < 12; d++)
        bm_printf("@@worst d=%d c=%llu (%s) asm=%llu (%s)\n", d, worst_c[d],
                  ws_c[d], worst_a[d], ws_a[d]);
    return bad;
}
