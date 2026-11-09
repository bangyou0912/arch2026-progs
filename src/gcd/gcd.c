#include "gcd.h"

/* Count trailing zeros of a nonzero value by a binary search over the bit
 * position. RV32I has no ctz instruction, and __builtin_ctz may lower to a
 * libgcc routine, which the bare-metal build does not link.
 */
static unsigned ctz32(uint32_t x)
{
    unsigned n = 0;
    if (!(x & 0xFFFFu)) {
        n += 16;
        x >>= 16;
    }
    if (!(x & 0xFFu)) {
        n += 8;
        x >>= 8;
    }
    if (!(x & 0xFu)) {
        n += 4;
        x >>= 4;
    }
    if (!(x & 0x3u)) {
        n += 2;
        x >>= 2;
    }
    return n + !(x & 1u);
}

/* Binary GCD (Stein's algorithm):
 *   gcd(2a, 2b) = 2 gcd(a, b)
 *   gcd(a, 2b)  = gcd(a, b)        when a is odd
 *   gcd(a, b)   = gcd(a, b - a)    when a and b are odd and a <= b
 * b - a of two odd values is even, so every pass removes at least one bit.
 */
uint32_t gcd_u32(uint32_t a, uint32_t b)
{
    if (a == 0)
        return b;
    if (b == 0)
        return a;

    unsigned k = ctz32(a | b);
    a >>= ctz32(a);
    do {
        b >>= ctz32(b);
        if (a > b) {
            uint32_t t = a;
            a = b;
            b = t;
        }
        b -= a;
    } while (b != 0);
    return a << k;
}
