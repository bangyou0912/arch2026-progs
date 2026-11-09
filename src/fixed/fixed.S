# op1, op2, op3 and q_mul: hand-written RV32I.
#
# uint32_t op1_asm(uint32_t a, uint32_t b, int *n)   a0, a1, a2 -> a0
# uint32_t op2_asm(uint32_t a, uint32_t b)           a0, a1 -> a0
# uint32_t op3_asm(uint32_t a, uint32_t b)           a0, a1 -> a0
# int32_t  q_mul_asm(int32_t a, int32_t b)           a0, a1 -> a0
#
# RV32I and Zicsr only: no mul, div, rem, no floating point, no compressed
# instructions. q_mul forms a 64-bit product in a register pair. Tests:
# tests/tfixed.c.

    .text

# op1: the bitwise adder. Each pass is carry = (a & b) << 1, a ^= b,
# b = carry, until b is zero; the pass count goes to *n.
#
# The loop is unrolled four times. A pass costs four instructions (and, xor,
# shift, test) instead of five, since the counter is bumped once per four
# passes; an early exit adds the passes of the partial round.
    .globl  op1_asm
    .type   op1_asm, @function
    .align  2
op1_asm:
    li      t1, 0                   # passes
    beqz    a1, op1_store
op1_loop:
    and     t0, a0, a1
    xor     a0, a0, a1
    slli    a1, t0, 1
    beqz    a1, op1_exit1
    and     t0, a0, a1
    xor     a0, a0, a1
    slli    a1, t0, 1
    beqz    a1, op1_exit2
    and     t0, a0, a1
    xor     a0, a0, a1
    slli    a1, t0, 1
    beqz    a1, op1_exit3
    and     t0, a0, a1
    xor     a0, a0, a1
    slli    a1, t0, 1
    addi    t1, t1, 4
    bnez    a1, op1_loop
op1_store:
    sw      t1, 0(a2)
    ret
op1_exit1:
    addi    t1, t1, 1
    sw      t1, 0(a2)
    ret
op1_exit2:
    addi    t1, t1, 2
    sw      t1, 0(a2)
    ret
op1_exit3:
    addi    t1, t1, 3
    sw      t1, 0(a2)
    ret
    .size   op1_asm, .-op1_asm

# op2: a - b modulo 2^32.
    .globl  op2_asm
    .type   op2_asm, @function
    .align  2
op2_asm:
    sub     a0, a0, a1
    ret
    .size   op2_asm, .-op2_asm

# op3: floor((a + b) / 2) = (a & b) + ((a ^ b) >> 1), no 33-bit sum needed.
    .globl  op3_asm
    .type   op3_asm, @function
    .align  2
op3_asm:
    and     t0, a0, a1
    xor     a0, a0, a1
    srli    a0, a0, 1
    add     a0, a0, t0
    ret
    .size   op3_asm, .-op3_asm

# q_mul: Q11.20 product, (a * b + 2^19) >> 20 on the exact signed 64-bit
# product, low 32 bits kept.
#
# The magnitudes are split into 16-bit halves, x = xh:xl (the larger) and
# y = yh:yl (the smaller), and
#     |a b| = xl yl + (xh yl + xl yh) 2^16 + xh yh 2^32.
# One shift-and-add loop per half of y forms both partial products that share
# it, so each multiplier bit is tested once for two products. A partial
# product of 16-bit values fits a word, and a loop ends as soon as its
# multiplier half has no set bit left; taking y as the smaller magnitude makes
# that happen as early as possible.
#
# The sign and the rounding are folded together: for a negative product,
# -m + 2^19 = ~m + (2^19 + 1), so the 64-bit magnitude is complemented with the
# sign mask and the constant grows by one, with no branch.
#
# Register use:
#   t6       a ^ b, the sign of the product in bit 31
#   a0, a1   |a| and |b|, swapped so that a1 is the smaller
#   a2, a3   xl and xh (shifted by the second loop)
#   a4, a5   yl and yh (consumed by the loops)
#   t0, t1   xl and xh shifted by the first loop
#   a6, a7   xl yl and xh yl
#   t3, t4   xl yh and xh yh
#   t2, t5   scratch
    .globl  q_mul_asm
    .type   q_mul_asm, @function
    .align  2
q_mul_asm:
    xor     t6, a0, a1
    srai    t2, a0, 31
    xor     a0, a0, t2
    sub     a0, a0, t2              # |a|
    srai    t2, a1, 31
    xor     a1, a1, t2
    sub     a1, a1, t2              # |b|
    bgeu    a0, a1, qm_split
    mv      t2, a0
    mv      a0, a1
    mv      a1, t2
qm_split:
    slli    a2, a0, 16
    srli    a2, a2, 16              # xl
    srli    a3, a0, 16              # xh
    slli    a4, a1, 16
    srli    a4, a4, 16              # yl
    srli    a5, a1, 16              # yh

    # xl yl and xh yl
    mv      t0, a2
    mv      t1, a3
    li      a6, 0
    li      a7, 0
    beqz    a4, qm_high
qm_low_loop:
    andi    t2, a4, 1
    beqz    t2, qm_low_skip
    add     a6, a6, t0
    add     a7, a7, t1
qm_low_skip:
    slli    t0, t0, 1
    slli    t1, t1, 1
    srli    a4, a4, 1
    bnez    a4, qm_low_loop

    # xl yh and xh yh; xl and xh are not needed afterward
qm_high:
    li      t3, 0
    li      t4, 0
    beqz    a5, qm_sum
qm_high_loop:
    andi    t2, a5, 1
    beqz    t2, qm_high_skip
    add     t3, t3, a2
    add     t4, t4, a3
qm_high_skip:
    slli    a2, a2, 1
    slli    a3, a3, 1
    srli    a5, a5, 1
    bnez    a5, qm_high_loop

    # mid = xh yl + xl yh, 33 bits: carry in t5
qm_sum:
    add     a7, a7, t3
    sltu    t5, a7, t3
    # lo = xl yl + (mid << 16), hi = xh yh + (mid >> 16) + carries
    slli    t2, a7, 16
    add     a6, a6, t2
    sltu    t2, a6, t2
    srli    a7, a7, 16
    slli    t5, t5, 16
    add     t4, t4, a7
    add     t4, t4, t5
    add     t4, t4, t2              # magnitude in t4:a6

    # Apply the sign and add 2^19 (2^19 + 1 when negative).
    srai    t6, t6, 31              # 0 or -1
    xor     a6, a6, t6
    xor     t4, t4, t6
    lui     t2, 0x80                # 2^19
    sub     t2, t2, t6
    add     a6, a6, t2
    sltu    t2, a6, t2
    add     t4, t4, t2

    # Bits 20 to 51 of the rounded product.
    srli    a6, a6, 20
    slli    t4, t4, 12
    or      a0, a6, t4
    ret
    .size   q_mul_asm, .-q_mul_asm
