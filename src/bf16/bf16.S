# BFloat16 routines: hand-written RV32I.
#
# uint16_t bf16_from_f32_asm(uint32_t f32bits)   a0 -> a0
# uint16_t bf16_mul_asm(uint16_t a, uint16_t b)  a0, a1 -> a0
# uint16_t bf16_sqrt_asm(uint16_t a)             a0 -> a0
#
# RV32I and Zicsr only: no mul, div, rem, no floating point, no compressed
# instructions. Tests: tests/tbf16.c.
#
# The uint16_t arguments arrive zero-extended, as the calling convention
# widens a narrow unsigned scalar to the full register.

    .text

# bf16_from_f32: Inf and NaN keep their top half; everything else adds 0x7FFF
# plus bit 16 (round to nearest, ties to even) and keeps the top half.
# The exponent test compares f32bits << 1 against 0xFF000000, which drops the
# sign without a mask.
    .globl  bf16_from_f32_asm
    .type   bf16_from_f32_asm, @function
    .align  2
bf16_from_f32_asm:
    slli    t0, a0, 1
    lui     t1, 0xFF000
    bgeu    t0, t1, f32_top         # exponent field 0xFF
    srli    t0, a0, 16
    andi    t0, t0, 1               # bit 16, the lowest kept bit
    lui     t1, 0x8                 # 0x8000
    add     a0, a0, t0
    add     a0, a0, t1
    addi    a0, a0, -1              # + 0x7FFF + bit 16
f32_top:
    srli    a0, a0, 16
    ret
    .size   bf16_from_f32_asm, .-bf16_from_f32_asm

# bf16_mul: same contract as the C version (bf16.h).
#
# The common case, two normal operands, runs straight through: two unsigned
# range tests (exponent - 1 < 254) send zeros, denormals, Inf and NaN to the
# side paths, which keep the order the contract prescribes (a examined before
# b). Both significands then have bit 7 set, so the 8 x 8 product starts from
# ma << 7 and adds ma << i for each set bit i of the low seven bits of mb,
# unrolled; a zero fraction skips them all. Renormalization picks the shift
# 7 or 8 from bit 15 without a branch, and one more range test on the result
# exponent leaves only overflow and underflow to a side path.
#
# Register use:
#   t6       sign of the product, at bit 15
#   t0, t1   exponents of a and b, then t0 the result exponent
#   t2, t3   significands of a and b
#   a2       product of the significands, then the result fraction
#   t4, t5   scratch
    .globl  bf16_mul_asm
    .type   bf16_mul_asm, @function
    .align  2
bf16_mul_asm:
    xor     t6, a0, a1
    srli    t6, t6, 15
    slli    t6, t6, 15              # sign
    srli    t0, a0, 7
    andi    t0, t0, 0xFF            # exp_a
    srli    t1, a1, 7
    andi    t1, t1, 0xFF            # exp_b
    andi    t2, a0, 0x7F            # mant_a
    andi    t3, a1, 0x7F            # mant_b
    addi    t4, t0, -1
    sltiu   t4, t4, 0xFE
    beqz    t4, mul_a_odd           # exp_a is 0 or 0xFF
    ori     t2, t2, 0x80
mul_a_ok:
    addi    t4, t1, -1
    sltiu   t4, t4, 0xFE
    beqz    t4, mul_b_odd           # exp_b is 0 or 0xFF
    ori     t3, t3, 0x80
mul_b_ok:
    # Significand product, 2^14 <= ma mb < 2^16.
    slli    a2, t2, 7               # bit 7 of mb is always set
    andi    t4, t3, 0x7F
    beqz    t4, mul_prod
    andi    t4, t3, 1
    beqz    t4, 1f
    add     a2, a2, t2
1:  andi    t4, t3, 2
    beqz    t4, 1f
    slli    t5, t2, 1
    add     a2, a2, t5
1:  andi    t4, t3, 4
    beqz    t4, 1f
    slli    t5, t2, 2
    add     a2, a2, t5
1:  andi    t4, t3, 8
    beqz    t4, 1f
    slli    t5, t2, 3
    add     a2, a2, t5
1:  andi    t4, t3, 16
    beqz    t4, 1f
    slli    t5, t2, 4
    add     a2, a2, t5
1:  andi    t4, t3, 32
    beqz    t4, 1f
    slli    t5, t2, 5
    add     a2, a2, t5
1:  andi    t4, t3, 64
    beqz    t4, mul_prod
    slli    t5, t2, 6
    add     a2, a2, t5
mul_prod:
    add     t0, t0, t1
    addi    t0, t0, -127            # exp_a + exp_b - bias
    srli    t4, a2, 15              # 1 when the product reached bit 15
    add     t0, t0, t4
    addi    t4, t4, 7
    srl     a2, a2, t4
    andi    a2, a2, 0x7F            # truncated fraction
    addi    t4, t0, -1
    sltiu   t4, t4, 0xFE
    beqz    t4, mul_e_odd           # overflow or underflow
    slli    t0, t0, 7
    or      a0, t6, t0
    or      a0, a0, a2
    ret

mul_e_odd:
    bgtz    t0, mul_inf             # exp >= 0xFF
    li      t4, -6
    blt     t0, t4, mul_zero        # too small even for a denormal
    li      t4, 1
    sub     t4, t4, t0
    srl     a2, a2, t4              # fraction >> (1 - exp)
    or      a0, t6, a2
    ret

    # exp_a is 0 or 0xFF.
mul_a_odd:
    bnez    t0, mul_a_special
    beqz    t2, mul_a_zero
    li      t0, 1                   # denormal: normalize, exponent from 1
1:  slli    t2, t2, 1
    addi    t0, t0, -1
    andi    t4, t2, 0x80
    beqz    t4, 1b
    j       mul_a_ok

    # exp_b is 0 or 0xFF; a is finite and nonzero here.
mul_b_odd:
    bnez    t1, mul_b_special
    beqz    t3, mul_zero
    li      t1, 1
1:  slli    t3, t3, 1
    addi    t1, t1, -1
    andi    t4, t3, 0x80
    beqz    t4, 1b
    j       mul_b_ok

    # a is Inf or NaN.
mul_a_special:
    bnez    t2, mul_ret_a           # NaN a
    slli    t4, a1, 17
    bnez    t4, mul_inf             # b is not a zero
    j       mul_nan

    # a is a zero: only an Inf or NaN b changes the zero result.
mul_a_zero:
    li      t4, 0xFF
    bne     t1, t4, mul_zero
    bnez    t3, mul_ret_b           # NaN b
mul_nan:
    lui     a0, 0x8
    addi    a0, a0, -64             # 0x7FC0
    ret

    # b is Inf or NaN and a is finite and nonzero.
mul_b_special:
    bnez    t3, mul_ret_b
mul_inf:
    lui     a0, 0x8
    addi    a0, a0, -128            # 0x7F80
    or      a0, a0, t6
    ret
mul_zero:
    mv      a0, t6
mul_ret_a:
    ret
mul_ret_b:
    mv      a0, a1
    ret
    .size   bf16_mul_asm, .-bf16_mul_asm

# bf16_sqrt: same contract as the C version (bf16.h).
#
# The search of the C version finds the largest mid with
# floor(mid^2 / 128) <= m, that is mid^2 < 128 (m + 1), so the result is
# isqrt(128 m + 127). With 128 <= m <= 510 this lies in [128, 255]: the
# result never needs renormalizing and the exponent never leaves the normal
# range. The exponent (e / 2 rounded down, plus the bias) is
# (exp + 127) >> 1, and m is doubled exactly when exp is even (e odd).
#
# isqrt is the restoring digit-by-digit method for one bit of the root per
# step. N >= 2^14 makes the first step known: the remainder starts at
# N - 2^14 and the partial root at 2^14. The remaining seven steps are
# unrolled with their constants 2^12, 2^10, ..., 2^0 as immediates where they
# fit, so there is no loop counter and no shifting of the step constant.
# With the leading one of the root in bit 7, the packed result is
# ((exp' - 1) << 7) + root, the root's leading one supplying the missing 1
# of the exponent.
#
# Register use:
#   a1   exponent, then the result exponent
#   a2   m, then N, then the remainder
#   a3   partial root
#   a4   trial subtrahend
#   t0   constant 0x7F80, then 2^12
    .globl  bf16_sqrt_asm
    .type   bf16_sqrt_asm, @function
    .align  2
bf16_sqrt_asm:
    lui     t0, 0x8
    addi    t0, t0, -128            # 0x7F80
    srli    t1, a0, 15
    bnez    t1, sq_neg
    bgeu    a0, t0, sq_ret          # +Inf and NaN unchanged
    sltiu   t1, a0, 0x80
    bnez    t1, sq_zero             # +0 and denormals
    srli    a1, a0, 7               # exponent, 1 .. 254
    andi    a2, a0, 0x7F
    ori     a2, a2, 0x80            # m
    andi    t1, a1, 1
    xori    t1, t1, 1
    sll     a2, a2, t1              # doubled when exp is even
    addi    a1, a1, 127
    srli    a1, a1, 1               # result exponent
    slli    a2, a2, 7
    addi    a2, a2, 127             # N = 128 m + 127

    lui     a3, 0x4                 # root 2^14 after the first step
    sub     a2, a2, a3              # remainder
    lui     t0, 0x1                 # 2^12
    add     a4, a3, t0
    srli    a3, a3, 1
    bltu    a2, a4, 1f
    sub     a2, a2, a4
    add     a3, a3, t0
1:  addi    a4, a3, 1024
    srli    a3, a3, 1
    bltu    a2, a4, 1f
    sub     a2, a2, a4
    addi    a3, a3, 1024
1:  addi    a4, a3, 256
    srli    a3, a3, 1
    bltu    a2, a4, 1f
    sub     a2, a2, a4
    addi    a3, a3, 256
1:  addi    a4, a3, 64
    srli    a3, a3, 1
    bltu    a2, a4, 1f
    sub     a2, a2, a4
    addi    a3, a3, 64
1:  addi    a4, a3, 16
    srli    a3, a3, 1
    bltu    a2, a4, 1f
    sub     a2, a2, a4
    addi    a3, a3, 16
1:  addi    a4, a3, 4
    srli    a3, a3, 1
    bltu    a2, a4, 1f
    sub     a2, a2, a4
    addi    a3, a3, 4
1:  addi    a4, a3, 1
    srli    a3, a3, 1
    bltu    a2, a4, 1f
    addi    a3, a3, 1               # last step: the remainder is not needed
1:  addi    a1, a1, -1
    slli    a1, a1, 7
    add     a0, a1, a3
    ret

    # Negative: -0 gives +0, a NaN stays, anything else is the quiet NaN.
sq_neg:
    slli    t1, a0, 17
    srli    t1, t1, 17              # magnitude
    beqz    t1, sq_zero
    bltu    t0, t1, sq_ret          # NaN
    addi    a0, t0, 64              # 0x7FC0
    ret
sq_zero:
    li      a0, 0
sq_ret:
    ret
    .size   bf16_sqrt_asm, .-bf16_sqrt_asm
