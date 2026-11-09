# gcd_u32_asm: binary GCD, hand-written RV32I.
#
# uint32_t gcd_u32_asm(uint32_t a, uint32_t b)   a in a0, b in a1, result in a0
#
# RV32I and Zicsr only: no mul, div, rem, no floating point, no compressed
# instructions. Tests: tests/tgcd.c.
#
# Stein's algorithm. Both operands are kept odd; each pass subtracts the
# smaller from the larger and strips the trailing zeros of the difference.
#
# Instead of swapping registers when the order flips, the loop exists in two
# mirrored copies: in loop A the odd value u lives in a0 and the other
# operand v in a1; loop B has the roles reversed. Crossing to the other copy
# costs nothing on one side (fall-through) and one jump on the other, so no
# moves are ever spent on a swap.
#
# Register use:
#   a0, a1  operands (u and v, roles depend on the loop copy)
#   a2      k, the number of common factors of two
#   t0      scratch for the low bit test
#   t1      a | b while counting k

    .text

    .globl  gcd_u32_asm
    .type   gcd_u32_asm, @function
    .align  2
gcd_u32_asm:
    beqz    a0, ret_b               # gcd(0, b) = b
    beqz    a1, ret_a               # gcd(a, 0) = a

    # k = ctz(a | b), one bit at a time; usually zero or one step.
    or      t1, a0, a1
    li      a2, 0
    andi    t0, t1, 1
    bnez    t0, strip_a0
count_k:
    srli    t1, t1, 1
    addi    a2, a2, 1
    andi    t0, t1, 1
    beqz    t0, count_k

    # Make a odd.
strip_a0:
    andi    t0, a0, 1
    bnez    t0, check_b
strip_a_loop:
    srli    a0, a0, 1
    andi    t0, a0, 1
    beqz    t0, strip_a_loop

    # b may still be even: strip it before the first comparison.
check_b:
    andi    t0, a1, 1
    bnez    t0, a_loop

    # ---- Loop A: odd u in a0, v in a1 (v even on entry to a_strip) ----
a_strip:
    srli    a1, a1, 1
    andi    t0, a1, 1
    beqz    t0, a_strip
a_loop:
    bltu    a1, a0, a_less          # v < u: the difference goes into a0
    sub     a1, a1, a0              # v - u, even
    bnez    a1, a_strip
    sll     a0, a0, a2              # v == u: the answer is u << k
    ret
a_less:
    sub     a0, a0, a1              # u - v, even and nonzero; v is now min
                                    # fall through into loop B

    # ---- Loop B: odd u in a1, v in a0 (v even on entry to b_strip) ----
b_strip:
    srli    a0, a0, 1
    andi    t0, a0, 1
    beqz    t0, b_strip
b_loop:
    bltu    a0, a1, b_less
    sub     a0, a0, a1
    bnez    a0, b_strip
    sll     a0, a1, a2
    ret
b_less:
    sub     a1, a1, a0
    j       a_strip

ret_b:
    mv      a0, a1
ret_a:
    ret
    .size   gcd_u32_asm, .-gcd_u32_asm
