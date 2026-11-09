#include "rubik.h"
#include "rubik_tables.h"

/* Optimal 2x2x2 solver, ported from Homework 1 and extended.
 *
 * Homework 1 searched with IDA* over three coordinates: the corner
 * permutation and two joint coordinates, each holding the positions and the
 * orientations of three cubies (A: cubies 0 to 2, B: cubies 3 to 5, written as
 * the digits 1 to 6). A joint coordinate is position_rank * 27 +
 * orientation_rank, 5,670 values, and one quarter-turn table serves both
 * groups. A and B together fix six cubies and, through the twist sum, the
 * seventh, so the pair is the whole state.
 *
 * This version keeps the joint coordinates, their shared transition table,
 * their pattern databases and the explicit search stack, and drops the
 * permutation coordinate in favor of a table of the 12,224 states within 5
 * moves of solved, keyed by (A, B). A child that has at most 5 moves left is
 * looked up instead of expanded: the deepest and widest levels of the search
 * tree become one table probe. The remaining moves are then read back from
 * the same table by a greedy descent. See rubik_tables.h for the layout.
 */

enum {
    CUBIES = 7,
    NEAR_DEPTH = RUBIK_NEAR_DEPTH,
    MAX_MOVES = 11,
};

/* Rank of the positions and orientations of cubies base .. base + 2. */
static unsigned joint_rank(const uint8_t *p, const uint8_t *o, unsigned base)
{
    unsigned x[3] = {0, 0, 0}, t[3] = {0, 0, 0};
    for (unsigned i = 0; i < CUBIES; i++) {
        unsigned k = p[i] - base;
        if (k < 3) {
            x[k] = i;
            t[k] = o[i];
        }
    }
    unsigned second = x[1] - (x[1] > x[0]);
    unsigned third = x[2] - (x[2] > x[0]) - (x[2] > x[1]);
    unsigned pos = (x[0] * 6 + second) * 5 + third;
    return pos * 27 + (t[0] * 3 + t[1]) * 3 + t[2];
}

/* Distance of the state (a, b) if it is within NEAR_DEPTH moves, else -1.
 * The entries of one value of a are sorted by b and there are few of them.
 */
static int near_distance(unsigned a, unsigned b)
{
    unsigned end = rubik_near_start[a + 1];
    for (unsigned i = rubik_near_start[a]; i < end; i++) {
        unsigned key = rubik_near_entry[i] >> 3;
        if (key >= b)
            return key == b ? (int) (rubik_near_entry[i] & 7) : -1;
    }
    return -1;
}

/* Append the moves from (a, b), which is d moves from solved, by stepping to
 * a neighbor one move closer each time; the table always has one.
 */
static void descend(unsigned a, unsigned b, int d, uint8_t *moves)
{
    while (d > 0) {
        unsigned move = 0, found = 0;
        for (unsigned face = 0; face < 3 && !found; face++) {
            const uint16_t *row = rubik_joint_next[face];
            unsigned na = a, nb = b;
            for (unsigned t = 0; t < 3; t++, move++) {
                na = row[na];
                nb = row[nb];
                if (near_distance(na, nb) == d - 1) {
                    *moves++ = (uint8_t) move;
                    a = na;
                    b = nb;
                    found = 1;
                    break;
                }
            }
        }
        d--;
    }
}

struct frame {
    uint16_t a, b;   /* the node */
    uint16_t na, nb; /* its latest child on the current face */
    uint8_t face, turn, last_face;
};

/* One IDA* iteration with the given bound (> NEAR_DEPTH). Returns the
 * solution length, or 0 when no path of that length exists.
 */
static int search(unsigned a, unsigned b, int bound, uint8_t *moves)
{
    struct frame st[MAX_MOVES];
    int depth = 0;
    st[0] = (struct frame) {
        (uint16_t) a, (uint16_t) b, (uint16_t) a, (uint16_t) b, 0, 0, 3};
    while (depth >= 0) {
        struct frame *f = &st[depth];
        if (f->face == 3) {
            depth--;
            continue;
        }
        if (f->face == f->last_face || f->turn == 3) {
            f->face++;
            f->turn = 0;
            f->na = f->a;
            f->nb = f->b;
            continue;
        }
        /* The next turn of this face starts from the previous child, so
         * both coordinates advance even when the child is pruned.
         */
        const uint16_t *row = rubik_joint_next[f->face];
        unsigned na = row[f->na], nb = row[f->nb];
        f->na = (uint16_t) na;
        f->nb = (uint16_t) nb;
        unsigned move = f->face * 3u + f->turn;
        f->turn++;

        int left = bound - depth - 1;
        if (rubik_pdb[0][na] > left || rubik_pdb[1][nb] > left)
            continue;
        moves[depth] = (uint8_t) move;
        if (left <= NEAR_DEPTH) {
            int d = near_distance(na, nb);
            if (d >= 0 && d <= left) {
                descend(na, nb, d, moves + depth + 1);
                return depth + 1 + d;
            }
            continue;
        }
        depth++;
        st[depth] = (struct frame) {
            (uint16_t) na, (uint16_t) nb, (uint16_t) na, (uint16_t) nb, 0, 0,
            f->face};
    }
    return 0;
}

/* Parse and check the 14-character state; returns 0 when it is valid. */
static int parse(const char *s, uint8_t *p, uint8_t *o)
{
    unsigned seen = 0, twist = 0;
    for (int i = 0; i < CUBIES; i++) {
        unsigned c = (unsigned) (unsigned char) s[i] - '1';
        if (c >= CUBIES || (seen >> c & 1))
            return -1;
        seen |= 1u << c;
        p[i] = (uint8_t) c;
    }
    for (int i = 0; i < CUBIES; i++) {
        unsigned t = (unsigned) (unsigned char) s[CUBIES + i] - '1';
        if (t >= 3)
            return -1;
        o[i] = (uint8_t) t;
        twist += t;
    }
    if (s[RUBIK_STATE_LEN] != '\0')
        return -1;
    while (twist >= 3)
        twist -= 3;
    return twist ? -1 : 0;
}

int rubik_solve(const char *state, char *out, size_t cap)
{
    uint8_t p[CUBIES], o[CUBIES];
    if (parse(state, p, o))
        return -1;
    unsigned a = joint_rank(p, o, 0), b = joint_rank(p, o, 3);

    uint8_t moves[MAX_MOVES];
    int n = near_distance(a, b);
    if (n >= 0) {
        descend(a, b, n, moves);
    } else {
        int h = rubik_pdb[0][a] > rubik_pdb[1][b] ? rubik_pdb[0][a]
                                                  : rubik_pdb[1][b];
        int bound = h > NEAR_DEPTH ? h : NEAR_DEPTH + 1;
        while (!(n = search(a, b, bound, moves)))
            bound++;
    }

    /* Move m is face m / 3 turned m % 3 + 1 quarter turns: its letter plus a
     * suffix for the half and the inverse turn. Moves are separated by
     * single spaces and followed by the NUL. Tables stand in for the
     * division, which RV32I lacks.
     */
    static const char letter[9] = {'R', 'R', 'R', 'B', 'B', 'B', 'D', 'D', 'D'};
    static const char suffix[9] = {0, '2', '\'', 0, '2', '\'', 0, '2', '\''};
    size_t need = 1;
    for (int i = 0; i < n; i++)
        need += (size_t) (suffix[moves[i]] ? 2 : 1) + (i > 0);
    if (cap < need) {
        if (cap)
            out[0] = '\0';
        return -2;
    }
    char *w = out;
    for (int i = 0; i < n; i++) {
        if (i)
            *w++ = ' ';
        *w++ = letter[moves[i]];
        if (suffix[moves[i]])
            *w++ = suffix[moves[i]];
    }
    *w = '\0';
    return n;
}
