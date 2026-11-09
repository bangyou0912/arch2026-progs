#include "bf16.h"

#define BF16_EXP_BIAS 127
#define BF16_INF 0x7F80u
#define BF16_NAN 0x7FC0u

/* Round to nearest, ties to even: adding 0x7FFF carries into bit 16 exactly
 * when the low half is above one half, and the extra bit 16 of the input
 * turns a tie into a carry only when the kept part is odd.
 */
uint16_t bf16_from_f32(uint32_t f32bits)
{
    if (((f32bits >> 23) & 0xFF) == 0xFF)
        return (uint16_t) (f32bits >> 16);
    f32bits += ((f32bits >> 16) & 1) + 0x7FFF;
    return (uint16_t) (f32bits >> 16);
}

uint16_t bf16_mul(uint16_t a, uint16_t b)
{
    uint32_t sign = (uint32_t) ((a ^ b) & 0x8000);
    int32_t exp_a = (a >> 7) & 0xFF;
    int32_t exp_b = (b >> 7) & 0xFF;
    uint32_t mant_a = a & 0x7F;
    uint32_t mant_b = b & 0x7F;

    if (exp_a == 0xFF) {
        if (mant_a)
            return a;
        if (!exp_b && !mant_b)
            return BF16_NAN;
        return (uint16_t) (sign | BF16_INF);
    }
    if (exp_b == 0xFF) {
        if (mant_b)
            return b;
        if (!exp_a && !mant_a)
            return BF16_NAN;
        return (uint16_t) (sign | BF16_INF);
    }
    if ((!exp_a && !mant_a) || (!exp_b && !mant_b))
        return (uint16_t) sign;

    /* Normalize a denormal: shift the leading one up to bit 7. Starting the
     * exponent at 1 and lowering it per shift folds in the adjustment.
     */
    if (!exp_a) {
        exp_a = 1;
        while (!(mant_a & 0x80)) {
            mant_a <<= 1;
            exp_a--;
        }
    } else {
        mant_a |= 0x80;
    }
    if (!exp_b) {
        exp_b = 1;
        while (!(mant_b & 0x80)) {
            mant_b <<= 1;
            exp_b--;
        }
    } else {
        mant_b |= 0x80;
    }

    uint32_t mant = mant_a * mant_b;
    int32_t exp = exp_a + exp_b - BF16_EXP_BIAS;
    if (mant & 0x8000) {
        mant = (mant >> 8) & 0x7F;
        exp++;
    } else {
        mant = (mant >> 7) & 0x7F;
    }

    if (exp >= 0xFF)
        return (uint16_t) (sign | BF16_INF);
    if (exp <= 0) {
        if (exp < -6)
            return (uint16_t) sign;
        mant >>= 1 - exp;
        exp = 0;
    }
    return (uint16_t) (sign | ((uint32_t) exp << 7) | mant);
}

uint16_t bf16_sqrt(uint16_t a)
{
    uint32_t sign = a >> 15;
    int32_t exp = (a >> 7) & 0xFF;
    uint32_t mant = a & 0x7F;

    if (exp == 0xFF) {
        if (mant)
            return a;
        return sign ? BF16_NAN : a;
    }
    if (!exp && !mant)
        return 0;
    if (sign)
        return BF16_NAN;
    if (!exp)
        return 0;

    /* sqrt(2^e m) = 2^(e/2) sqrt(m) once e is even. */
    int32_t e = exp - BF16_EXP_BIAS;
    int32_t new_exp;
    uint32_t m = 0x80 | mant;
    if (e & 1) {
        m <<= 1;
        new_exp = ((e - 1) >> 1) + BF16_EXP_BIAS;
    } else {
        new_exp = (e >> 1) + BF16_EXP_BIAS;
    }

    /* Largest mid with (mid * mid) / 128 <= m. */
    uint32_t low = 90, high = 256, result = 128;
    while (low <= high) {
        uint32_t mid = (low + high) >> 1;
        if ((mid * mid) / 128 <= m) {
            result = mid;
            low = mid + 1;
        } else {
            high = mid - 1;
        }
    }

    if (result >= 256) {
        result >>= 1;
        new_exp++;
    } else {
        while (result < 128 && new_exp > 1) {
            result <<= 1;
            new_exp--;
        }
    }

    if (new_exp >= 0xFF)
        return BF16_INF;
    if (new_exp <= 0)
        return 0;
    return (uint16_t) (((uint32_t) new_exp << 7) | (result & 0x7F));
}
