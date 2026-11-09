# rubik_solve_asm: optimal 2x2x2 cube solver, hand-written RV32I.
#
# int rubik_solve_asm(const char *state, char *out, size_t cap)
#                                   a0 = state, a1 = out, a2 = cap; result in a0
#
# RV32I and Zicsr only: no mul, div, rem, no floating point, no compressed
# instructions, no heap, no recursion. Tests: tests/trubik.c.
#
# The algorithm is the one of rubik.c: IDA* over the two joint coordinates of
# Homework 1 (positions and orientations of cubies 0-2 and 3-5), pruned by
# their pattern databases, with every child that has at most five moves left
# looked up in the table of states within five moves instead of expanded.
# The tables are in rubik_tables.S.
#
# The search loop follows the Homework 1 assembly: the child coordinates stay
# in t0 and t1, the turn counter in t3 and the face row base in s5, and the
# parent's cursor is written to its frame only when the search descends, so a
# pruned child costs no memory traffic besides the two transition loads and
# the pattern database loads. All three turns of a face are produced by
# repeated quarter-turn lookups from the previous child.
#
# Register use in the search:
#   s0  transition table, face 0 row     s6  near table bucket starts
#   s1  pattern database of A            s7  pattern database of B
#   s2  current frame                    s8  near table entries
#   s3  depth                            s9  move buffer
#   s4  bound                            s10 moves left for the children
#   s5  row of the current face          s11 current face
#   a0  row stride (one face)            a1  face of the previous move
#   a3  current face * 3 - 1             t0, t1  child coordinates A, B
#   t3  turns made on the current face   t4, t5  scratch
# The near table lookup takes a4 = A, a5 = B and returns a6 = distance or -1,
# changing a7, t4 and t5 only.

    .equ    JOINT, 5670             # values of one joint coordinate
    .equ    STRIDE, 11340           # bytes of one face row of transitions
    .equ    NEAR_DEPTH, 5           # depth covered by the near table

    # A search frame: the node, its latest child on the current face, and
    # the cursor of its move loop.
    .equ    F_A, 0
    .equ    F_B, 2
    .equ    F_NA, 4
    .equ    F_NB, 6
    .equ    F_LEFT, 8
    .equ    F_LAST, 9
    .equ    F_FACE, 10
    .equ    F_TURN, 11
    .equ    F_ROW, 12
    .equ    FRAME, 16

    # Stack layout.
    .equ    SP_OUT, 52
    .equ    SP_CAP, 56
    .equ    SP_A, 60
    .equ    SP_B, 62
    .equ    SP_X, 64                # position of each cubie
    .equ    SP_T, 72                # orientation of each cubie
    .equ    SP_MOVES, 80            # up to 11 moves
    .equ    SP_FRAMES, 96           # up to 12 frames
    .equ    SP_SIZE, 288

    .section .rodata
move_letter:
    .byte   'R', 'R', 'R', 'B', 'B', 'B', 'D', 'D', 'D'
move_suffix:
    .byte   0, '2', '\'', 0, '2', '\'', 0, '2', '\''

    .text

# dst = joint rank of the cubies off .. off + 2, from the X and T arrays:
#   (((x0 * 6 + second) * 5 + third) * 27 + (t0 * 3 + t1) * 3 + t2
# with second = x1 - (x1 > x0) and third = x2 - (x2 > x0) - (x2 > x1).
# Every constant multiply is a shift and add.
    .macro  JOINT_RANK dst, off
    lbu     t0, SP_X + \off(sp)
    lbu     t1, SP_X + \off + 1(sp)
    lbu     t2, SP_X + \off + 2(sp)
    sltu    t3, t0, t2
    sltu    t4, t1, t2
    sub     t2, t2, t3
    sub     t2, t2, t4              # third
    sltu    t3, t0, t1
    sub     t1, t1, t3              # second
    slli    t3, t0, 1
    slli    t0, t0, 2
    add     t0, t0, t3              # x0 * 6
    add     t0, t0, t1
    slli    t3, t0, 2
    add     t0, t0, t3              # * 5
    add     t0, t0, t2              # position rank
    slli    t3, t0, 5
    slli    t4, t0, 2
    sub     t3, t3, t4
    sub     t3, t3, t0              # * 27
    lbu     t1, SP_T + \off(sp)
    lbu     t2, SP_T + \off + 1(sp)
    lbu     t4, SP_T + \off + 2(sp)
    slli    t5, t1, 1
    add     t1, t1, t5
    add     t1, t1, t2
    slli    t5, t1, 1
    add     t1, t1, t5
    add     t1, t1, t4              # orientation rank
    add     \dst, t3, t1
    .endm

    .globl  rubik_solve_asm
    .type   rubik_solve_asm, @function
    .align  2
rubik_solve_asm:
    addi    sp, sp, -SP_SIZE
    sw      ra, 0(sp)
    sw      s0, 4(sp)
    sw      s1, 8(sp)
    sw      s2, 12(sp)
    sw      s3, 16(sp)
    sw      s4, 20(sp)
    sw      s5, 24(sp)
    sw      s6, 28(sp)
    sw      s7, 32(sp)
    sw      s8, 36(sp)
    sw      s9, 40(sp)
    sw      s10, 44(sp)
    sw      s11, 48(sp)
    sw      a1, SP_OUT(sp)
    sw      a2, SP_CAP(sp)

    # Seven distinct cubie digits; X[cubie] = position. A NUL fails the
    # range test, so a short string is never read past its end.
    li      t0, 0
    li      t1, 0                   # cubies seen
    li      t6, 7
parse_cubie:
    add     t3, a0, t0
    lbu     t4, 0(t3)
    addi    t4, t4, -'1'
    bgeu    t4, t6, invalid
    srl     t5, t1, t4
    andi    t5, t5, 1
    bnez    t5, invalid
    li      t5, 1
    sll     t5, t5, t4
    or      t1, t1, t5
    add     t5, sp, t4
    sb      t0, SP_X(t5)
    addi    t0, t0, 1
    bne     t0, t6, parse_cubie

    # Seven orientation digits; T[cubie at the position] = orientation.
    li      t0, 0
    li      t1, 0                   # orientation sum
    li      t2, 3
parse_twist:
    add     t3, a0, t0
    lbu     t4, 7(t3)
    addi    t4, t4, -'1'
    bgeu    t4, t2, invalid
    add     t1, t1, t4
    lbu     t5, 0(t3)
    add     t5, sp, t5
    sb      t4, SP_T - '1'(t5)
    addi    t0, t0, 1
    bne     t0, t6, parse_twist
    lbu     t4, 14(a0)
    bnez    t4, invalid             # longer than 14 characters
1:  bltu    t1, t2, 2f              # sum modulo 3, sum <= 14
    addi    t1, t1, -3
    j       1b
2:  bnez    t1, invalid

    JOINT_RANK a4, 0
    JOINT_RANK a5, 3
    sh      a4, SP_A(sp)
    sh      a5, SP_B(sp)

    la      s0, rubik_asm_joint_next
    la      s1, rubik_asm_pdb
    li      t4, JOINT
    add     s7, s1, t4
    la      s6, rubik_asm_near_start
    la      s8, rubik_asm_near_entry
    addi    s9, sp, SP_MOVES
    li      a0, STRIDE

    # Within five moves: read the whole solution from the table.
    jal     near
    bltz    a6, root_far
    mv      t0, a4
    mv      t1, a5
    mv      t2, s9
    mv      s3, a6                  # length
    jal     a2, descend
    j       write

root_far:
    add     t4, s1, a4
    lbu     s4, 0(t4)
    add     t4, s7, a5
    lbu     t5, 0(t4)
    bgeu    s4, t5, 1f
    mv      s4, t5
1:  li      t4, NEAR_DEPTH + 1
    bgeu    s4, t4, bound_loop
    mv      s4, t4                  # bound = max(h, 6)

bound_loop:
    addi    s2, sp, SP_FRAMES
    lhu     t0, SP_A(sp)
    lhu     t1, SP_B(sp)
    sh      t0, F_A(s2)
    sh      t1, F_B(s2)
    addi    s10, s4, -1
    sb      s10, F_LEFT(s2)
    li      a1, 3                   # no previous move
    sb      a1, F_LAST(s2)
    li      s3, 0
    li      s11, 0
    li      t3, 0
    mv      s5, s0
    li      a3, -1

edge:
    slli    t4, t0, 1
    add     t4, s5, t4
    lhu     t0, 0(t4)               # A after one more quarter turn
    slli    t4, t1, 1
    add     t4, s5, t4
    lhu     t1, 0(t4)               # B
    addi    t3, t3, 1
    add     t4, s1, t0
    lbu     t5, 0(t4)
    bltu    s10, t5, reject
    add     t4, s7, t1
    lbu     t5, 0(t4)
    bltu    s10, t5, reject
    add     t4, s9, s3
    add     t5, a3, t3
    sb      t5, 0(t4)               # moves[depth] = face * 3 + turn - 1
    li      t4, NEAR_DEPTH
    bgeu    t4, s10, probe

    # Descend: save the parent's cursor, open the child's frame.
    sh      t0, F_NA(s2)
    sh      t1, F_NB(s2)
    sb      s11, F_FACE(s2)
    sb      t3, F_TURN(s2)
    sw      s5, F_ROW(s2)
    addi    s2, s2, FRAME
    addi    s3, s3, 1
    sh      t0, F_A(s2)
    sh      t1, F_B(s2)
    addi    s10, s10, -1
    sb      s10, F_LEFT(s2)
    mv      a1, s11
    sb      a1, F_LAST(s2)
    li      s11, 0
    li      t3, 0
    mv      s5, s0
    li      a3, -1
    bnez    a1, edge                # face 0 unless it was the last move
    j       next_face

reject:
    li      t4, 3
    bltu    t3, t4, edge
next_face:
    addi    s11, s11, 1
    li      t4, 3
    beq     s11, t4, pop
    add     s5, s5, a0
    addi    a3, a3, 3
    lhu     t0, F_A(s2)
    lhu     t1, F_B(s2)
    li      t3, 0
    beq     s11, a1, next_face      # never the face of the previous move
    j       edge

pop:
    addi    s3, s3, -1
    bltz    s3, next_bound
    addi    s2, s2, -FRAME
    lhu     t0, F_NA(s2)
    lhu     t1, F_NB(s2)
    lbu     s11, F_FACE(s2)
    lbu     t3, F_TURN(s2)
    lw      s5, F_ROW(s2)
    lbu     s10, F_LEFT(s2)
    lbu     a1, F_LAST(s2)
    slli    a3, s11, 1
    add     a3, a3, s11
    addi    a3, a3, -1
    li      t4, 3
    bltu    t3, t4, edge
    j       next_face

next_bound:
    addi    s4, s4, 1
    j       bound_loop

    # The child has at most five moves left: it is solvable in time exactly
    # when the table holds it with a distance that fits.
probe:
    mv      a4, t0
    mv      a5, t1
    jal     near
    bltz    a6, reject
    bltu    s10, a6, reject
    addi    s3, s3, 1
    add     t2, s9, s3              # the rest goes after moves[depth]
    add     s3, s3, a6              # length
    jal     a2, descend

    # Text: a letter per move, a suffix for half and inverse turns, single
    # spaces between moves, and the NUL: 2 n + suffixes bytes, or 1 for n = 0.
write:
    lw      a1, SP_OUT(sp)
    lw      a2, SP_CAP(sp)
    la      t5, move_letter
    la      t6, move_suffix
    li      t1, 1
    beqz    s3, 2f
    slli    t1, s3, 1
    li      t0, 0
1:  add     t4, s9, t0
    lbu     t4, 0(t4)
    add     t4, t6, t4
    lbu     t4, 0(t4)
    snez    t4, t4
    add     t1, t1, t4
    addi    t0, t0, 1
    bne     t0, s3, 1b
2:  bgeu    a2, t1, 3f
    li      a0, -2                  # does not fit
    beqz    a2, done
    sb      zero, 0(a1)
    j       done
3:  li      t0, 0
    li      t2, ' '
    beqz    s3, 6f
4:  add     t4, s9, t0
    lbu     t4, 0(t4)
    add     t3, t5, t4
    lbu     t3, 0(t3)
    sb      t3, 0(a1)
    addi    a1, a1, 1
    add     t3, t6, t4
    lbu     t3, 0(t3)
    beqz    t3, 5f
    sb      t3, 0(a1)
    addi    a1, a1, 1
5:  addi    t0, t0, 1
    beq     t0, s3, 6f
    sb      t2, 0(a1)
    addi    a1, a1, 1
    j       4b
6:  sb      zero, 0(a1)
    mv      a0, s3
    j       done

invalid:
    li      a0, -1
done:
    lw      ra, 0(sp)
    lw      s0, 4(sp)
    lw      s1, 8(sp)
    lw      s2, 12(sp)
    lw      s3, 16(sp)
    lw      s4, 20(sp)
    lw      s5, 24(sp)
    lw      s6, 28(sp)
    lw      s7, 32(sp)
    lw      s8, 36(sp)
    lw      s9, 40(sp)
    lw      s10, 44(sp)
    lw      s11, 48(sp)
    addi    sp, sp, SP_SIZE
    ret

# near: a6 = distance of (a4, a5) if within five moves, else -1. The bucket
# of A is sorted by B and holds two entries on average, so a linear scan
# that stops at the first larger key is enough. Changes a7, t4 and t5.
near:
    slli    a7, a4, 1
    add     a7, a7, s6
    lhu     t4, 0(a7)
    lhu     t5, 2(a7)
    slli    t4, t4, 1
    add     t4, t4, s8
    slli    t5, t5, 1
    add     t5, t5, s8
1:  bgeu    t4, t5, 2f
    lhu     a7, 0(t4)
    addi    t4, t4, 2
    srli    a6, a7, 3
    bltu    a6, a5, 1b
    bne     a6, a5, 2f
    andi    a6, a7, 7
    ret
2:  li      a6, -1
    ret

# descend (link in a2): from (t0, t1), a6 moves from solved, store at t2 the
# first move of each step that reaches a state one move closer. Changes
# t0 to t6, a3 to a7, ra, s10 and s11.
descend:
    mv      s10, a6
1:  beqz    s10, 5f
    addi    a3, s10, -1             # distance wanted next
    mv      t6, s0
    li      t3, 0                   # move
2:  mv      a4, t0
    mv      a5, t1
    li      s11, 3
3:  slli    t4, a4, 1
    add     t4, t6, t4
    lhu     a4, 0(t4)
    slli    t4, a5, 1
    add     t4, t6, t4
    lhu     a5, 0(t4)
    jal     near
    beq     a6, a3, 4f
    addi    t3, t3, 1
    addi    s11, s11, -1
    bnez    s11, 3b
    add     t6, t6, a0
    j       2b
4:  sb      t3, 0(t2)
    addi    t2, t2, 1
    mv      t0, a4
    mv      t1, a5
    addi    s10, s10, -1
    j       1b
5:  jr      a2
    .size   rubik_solve_asm, .-rubik_solve_asm
