#include "fixed.h"

/* Addition from logic: the carry word is folded back in until no carry is
 * left. Every pass moves each carry one bit up and a carry out of bit 31 is
 * dropped, so the loop ends after at most 32 passes.
 */
uint32_t op1(uint32_t a, uint32_t b, int *n)
{
    int passes = 0;
    while (b) {
        uint32_t carry = (a & b) << 1;
        a ^= b;
        b = carry;
        passes++;
    }
    *n = passes;
    return a;
}

uint32_t op2(uint32_t a, uint32_t b)
{
    return a - b;
}

/* a + b = 2 (a & b) + (a ^ b): the common bits count twice and the differing
 * bits once, so halving the second term alone gives the floor of the mean
 * without forming the 33-bit sum.
 */
uint32_t op3(uint32_t a, uint32_t b)
{
    return (a & b) + ((a ^ b) >> 1);
}

/* 16 x 16 -> 32 bit product by shift and add, ending as soon as the
 * multiplier runs out of set bits. Both operands are below 2^16, so the
 * shifted multiplicand never leaves the 32-bit word.
 */
static uint32_t mul16(uint32_t x, uint32_t y)
{
    uint32_t r = 0;
    while (y) {
        if (y & 1)
            r += x;
        x <<= 1;
        y >>= 1;
    }
    return r;
}

/* Q11.20 product. RV32I has no multiply and the bare-metal build links no
 * 64-bit multiply routine, so the 64-bit magnitude is assembled from four
 * 16-bit partial products, then the sign, the rounding constant 2^19 and the
 * shift by 20 are applied. The arithmetic shift of the signed product keeps
 * ties rounding toward plus infinity for negative values as well.
 */
int32_t q_mul(int32_t a, int32_t b)
{
    uint32_t ma = a < 0 ? 0u - (uint32_t) a : (uint32_t) a;
    uint32_t mb = b < 0 ? 0u - (uint32_t) b : (uint32_t) b;
    uint32_t al = ma & 0xFFFFu, ah = ma >> 16;
    uint32_t bl = mb & 0xFFFFu, bh = mb >> 16;

    uint64_t mid = (uint64_t) mul16(al, bh) + mul16(ah, bl);
    uint64_t mag = (uint64_t) mul16(al, bl) + (mid << 16) +
                   ((uint64_t) mul16(ah, bh) << 32);
    if ((a < 0) != (b < 0))
        mag = 0u - mag;

    int64_t p = (int64_t) mag + ((int64_t) 1 << (Q - 1));
    return (int32_t) (p >> Q);
}
