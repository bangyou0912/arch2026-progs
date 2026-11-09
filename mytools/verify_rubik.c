/* Host check of src/rubik/rubik_tables.h against an independent cube model.
 *
 * Build and run from the repository root:
 *
 *   cc -O2 -o /tmp/verify_rubik mytools/verify_rubik.c && /tmp/verify_rubik
 *
 * An optional argument sets the sampling stride for distances 0..9 (default
 * 50; 1 runs the solver on every state). Distances 10 and 11 are always run in
 * full. It checks:
 *   - joint_next commutes with the cube model for every state and face;
 *   - pdb equals the BFS distance in the joint graph, is admissible for every
 *     state and consistent (neighbors differ by at most 1);
 *   - the near table holds exactly the states at distance <= 5, with their
 *     distances, sorted, with correct starts, and (A, B) is unique;
 *   - the IDA* of the solver finds optimal lengths, and a greedy completion
 *     through the near table yields a full path that solves the cube.
 */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../src/rubik/rubik_tables.h"

enum { CUBIES = 7, NSTATES = 5040 * 729, MAXD = 11 };

struct cube {
    uint8_t p[CUBIES], o[CUBIES];
};

static const uint8_t source[3][CUBIES] = {
    {1, 4, 2, 0, 3, 5, 6},
    {0, 1, 2, 4, 5, 6, 3},
    {0, 2, 5, 3, 1, 4, 6},
};
static const uint8_t twist[3][CUBIES] = {
    {1, 2, 0, 2, 1, 0, 0},
    {0, 0, 0, 1, 2, 1, 2},
    {0, 0, 0, 0, 0, 0, 0},
};

static int failures;
#define CHECK(cond, ...)                      \
    do {                                      \
        if (!(cond)) {                        \
            if (failures++ < 10)              \
                fprintf(stderr, __VA_ARGS__); \
        }                                     \
    } while (0)

static struct cube quarter(struct cube c, int f)
{
    struct cube n;
    for (int i = 0; i < CUBIES; i++) {
        n.p[i] = c.p[source[f][i]];
        n.o[i] = (uint8_t) ((c.o[source[f][i]] + twist[f][i]) % 3);
    }
    return n;
}

static struct cube move(struct cube c, int m)
{
    for (int t = 0; t <= m % 3; t++)
        c = quarter(c, m / 3);
    return c;
}

static uint32_t encode(const struct cube *c)
{
    uint32_t r = 0;
    for (int i = 0; i < CUBIES; i++) {
        int less = 0;
        for (int j = i + 1; j < CUBIES; j++)
            less += c->p[j] < c->p[i];
        r = r * (uint32_t) (CUBIES - i) + (uint32_t) less;
    }
    uint32_t o = 0;
    for (int i = 0; i < 6; i++)
        o = o * 3 + c->o[i];
    return r * 729 + o;
}

static struct cube decode(uint32_t idx)
{
    struct cube c;
    uint32_t o = idx % 729, r = idx / 729;
    int sum = 0;
    for (int i = 5; i >= 0; i--) {
        c.o[i] = (uint8_t) (o % 3);
        sum += c.o[i];
        o /= 3;
    }
    c.o[6] = (uint8_t) ((3 - sum % 3) % 3);
    int digits[CUBIES];
    for (int i = CUBIES - 1; i >= 0; i--) {
        digits[i] = (int) (r % (uint32_t) (CUBIES - i));
        r /= (uint32_t) (CUBIES - i);
    }
    int used[CUBIES] = {0};
    for (int i = 0; i < CUBIES; i++) {
        int k = digits[i];
        for (int v = 0; v < CUBIES; v++) {
            if (used[v])
                continue;
            if (k-- == 0) {
                c.p[i] = (uint8_t) v;
                used[v] = 1;
                break;
            }
        }
    }
    return c;
}

static int coord(const struct cube *c, int g)
{
    int x[3], oo[3];
    for (int k = 0; k < 3; k++) {
        for (int i = 0; i < CUBIES; i++) {
            if (c->p[i] == 3 * g + k) {
                x[k] = i;
                oo[k] = c->o[i];
            }
        }
    }
    int second = x[1] - (x[1] > x[0]);
    int third = x[2] - (x[2] > x[0]) - (x[2] > x[1]);
    return ((x[0] * 6 + second) * 5 + third) * 27 + (oo[0] * 3 + oo[1]) * 3 +
           oo[2];
}

static uint8_t *dist;

/* Returns the near distance of (a, b), or -1; counts one lookup. */
static long lookups;
static int near(int a, int b)
{
    lookups++;
    int lo = rubik_near_start[a], hi = rubik_near_start[a + 1];
    while (lo < hi) {
        int mid = (lo + hi) / 2;
        int key = rubik_near_entry[mid] >> 3;
        if (key == b)
            return rubik_near_entry[mid] & 7;
        if (key < b)
            lo = mid + 1;
        else
            hi = mid;
    }
    return -1;
}

static void check_tables(void)
{
    /* joint_next commutes with the projection. */
    static int edge[3][RUBIK_JOINT];
    for (int f = 0; f < 3; f++)
        for (int j = 0; j < RUBIK_JOINT; j++)
            edge[f][j] = -1;
    for (uint32_t s = 0; s < NSTATES; s++) {
        struct cube c = decode(s);
        CHECK(encode(&c) == s, "encode/decode mismatch at %u\n", s);
        for (int f = 0; f < 3; f++) {
            struct cube n = quarter(c, f);
            for (int g = 0; g < 2; g++) {
                int j = coord(&c, g), nj = coord(&n, g);
                CHECK(rubik_joint_next[f][j] == nj,
                      "joint_next[%d][%d]=%d, model %d\n", f, j,
                      rubik_joint_next[f][j], nj);
                edge[f][j] = nj;
            }
        }
    }
    printf("joint_next: commutes for all %d states x 3 faces\n", NSTATES);

    /* pdb equals the joint-graph BFS built from the model's edges. */
    for (int g = 0; g < 2; g++) {
        static int jd[RUBIK_JOINT], queue[RUBIK_JOINT];
        int goal = g ? RUBIK_GOAL_B : 0, head = 0, tail = 0;
        for (int j = 0; j < RUBIK_JOINT; j++)
            jd[j] = -1;
        jd[goal] = 0;
        queue[tail++] = goal;
        while (head < tail) {
            int j = queue[head++];
            for (int f = 0; f < 3; f++) {
                int c = j;
                for (int t = 0; t < 3; t++) {
                    c = edge[f][c];
                    if (jd[c] < 0) {
                        jd[c] = jd[j] + 1;
                        queue[tail++] = c;
                    }
                }
            }
        }
        int maxd = 0;
        for (int j = 0; j < RUBIK_JOINT; j++) {
            CHECK(jd[j] == rubik_pdb[g][j], "pdb[%d][%d]=%d, BFS %d\n", g, j,
                  rubik_pdb[g][j], jd[j]);
            if (jd[j] > maxd)
                maxd = jd[j];
        }
        printf("pdb[%c]: matches joint BFS, max %d\n", g ? 'B' : 'A', maxd);
    }

    /* Admissible and consistent over the full state space. */
    long inadmissible = 0, inconsistent = 0;
    for (uint32_t s = 0; s < NSTATES; s++) {
        struct cube c = decode(s);
        int a = coord(&c, 0), b = coord(&c, 1);
        if (rubik_pdb[0][a] > dist[s] || rubik_pdb[1][b] > dist[s])
            inadmissible++;
        for (int m = 0; m < 9; m++) {
            struct cube n = move(c, m);
            int na = coord(&n, 0), nb = coord(&n, 1);
            if (abs(rubik_pdb[0][na] - rubik_pdb[0][a]) > 1 ||
                abs(rubik_pdb[1][nb] - rubik_pdb[1][b]) > 1)
                inconsistent++;
        }
    }
    CHECK(!inadmissible && !inconsistent, "pdb: %ld inadmissible, %ld bad\n",
          inadmissible, inconsistent);
    printf("pdb: admissible and consistent on all states (%ld, %ld bad)\n",
           inadmissible, inconsistent);

    /* Near table: exactly the states at distance <= 5. */
    CHECK(
        rubik_near_start[0] == 0 && rubik_near_start[RUBIK_JOINT] == RUBIK_NEAR,
        "near_start ends wrong\n");
    for (int a = 0; a < RUBIK_JOINT; a++) {
        CHECK(rubik_near_start[a] <= rubik_near_start[a + 1],
              "near_start not monotone at %d\n", a);
        for (int k = rubik_near_start[a] + 1; k < rubik_near_start[a + 1]; k++)
            CHECK((rubik_near_entry[k - 1] >> 3) < (rubik_near_entry[k] >> 3),
                  "near_entry not strictly sorted at %d\n", k);
    }
    static uint8_t hit[RUBIK_NEAR];
    long near_states = 0;
    for (uint32_t s = 0; s < NSTATES; s++) {
        struct cube c = decode(s);
        int a = coord(&c, 0), b = coord(&c, 1);
        int lo = rubik_near_start[a], hi = rubik_near_start[a + 1], at = -1;
        for (int k = lo; k < hi; k++)
            if ((rubik_near_entry[k] >> 3) == b)
                at = k;
        if (dist[s] <= RUBIK_NEAR_DEPTH) {
            near_states++;
            CHECK(at >= 0, "state %u (d %d) missing from near\n", s, dist[s]);
            if (at >= 0) {
                CHECK(!hit[at], "near entry %d hit twice: (A,B) not unique\n",
                      at);
                hit[at] = 1;
                CHECK((rubik_near_entry[at] & 7) == dist[s],
                      "near distance wrong for %u\n", s);
            }
        } else {
            CHECK(at < 0, "state %u at distance %d is in near\n", s, dist[s]);
        }
    }
    for (int k = 0; k < RUBIK_NEAR; k++)
        CHECK(hit[k], "near entry %d matches no state\n", k);
    printf("near: %ld states at distance <= 5, table %d entries, exact\n",
           near_states, RUBIK_NEAR);

    /* (A, B) unique over all states: the seventh cubie follows. */
    long collisions = 0;
    {
        uint8_t *seen = calloc((size_t) RUBIK_JOINT * RUBIK_JOINT, 1);
        if (!seen) {
            fprintf(stderr, "out of memory\n");
            exit(1);
        }
        for (uint32_t s = 0; s < NSTATES; s++) {
            struct cube c = decode(s);
            size_t key = (size_t) coord(&c, 0) * RUBIK_JOINT + coord(&c, 1);
            collisions += seen[key];
            seen[key] = 1;
        }
        free(seen);
    }
    CHECK(!collisions, "(A, B) collisions: %ld\n", collisions);
    printf("(A, B): unique over all %d states\n", NSTATES);
}

/* The IDA* of the intended solver. Returns the solution length and fills
 * path[] (moves to the near state) and *near_len.
 */
static long edges;
static int solve(int a0, int b0, int *path, int *depth_out, int *near_dist)
{
    int d0 = near(a0, b0);
    if (d0 >= 0) {
        *depth_out = 0;
        *near_dist = d0;
        return d0;
    }
    int h0 = rubik_pdb[0][a0] > rubik_pdb[1][b0] ? rubik_pdb[0][a0]
                                                 : rubik_pdb[1][b0];
    struct frame {
        int a, b, last, m, ca;
    } st[MAXD + 1];
    for (int bound = h0 > 6 ? h0 : 6; bound <= MAXD; bound++) {
        int sp = 0;
        st[0] = (struct frame) {a0, b0, -1, 0, 0};
        while (sp >= 0) {
            struct frame *fr = &st[sp];
            if (fr->m == 9) {
                sp--;
                continue;
            }
            int f = fr->m / 3, t = fr->m % 3;
            if (f == fr->last) {
                fr->m += 3;
                continue;
            }
            /* Child by repeated quarter turns of the parent. */
            int ca = t == 0 ? fr->a : fr->ca, cb;
            static int cbs[MAXD + 1];
            if (t == 0)
                cbs[sp] = fr->b;
            ca = rubik_joint_next[f][ca];
            cb = cbs[sp] = rubik_joint_next[f][cbs[sp]];
            fr->ca = ca;
            int m = fr->m++;
            edges++;
            int left = bound - sp - 1;
            if (rubik_pdb[0][ca] > left || rubik_pdb[1][cb] > left)
                continue;
            path[sp] = m;
            if (left <= RUBIK_NEAR_DEPTH) {
                int d = near(ca, cb);
                if (d >= 0 && d <= left) {
                    *depth_out = sp + 1;
                    *near_dist = d;
                    return sp + 1 + d;
                }
                continue;
            }
            sp++;
            st[sp] = (struct frame) {ca, cb, f, 0, 0};
        }
    }
    return -1;
}

int main(int argc, char **argv)
{
    int stride = argc > 1 ? atoi(argv[1]) : 50;
    if (stride < 1)
        stride = 1;
    dist = malloc(NSTATES);
    uint32_t *queue = malloc(sizeof(uint32_t) * NSTATES);
    if (!dist || !queue) {
        fprintf(stderr, "out of memory\n");
        return 1;
    }
    memset(dist, 0xff, NSTATES);
    struct cube solved = {{0, 1, 2, 3, 4, 5, 6}, {0}};
    uint32_t head = 0, tail = 0, s0 = encode(&solved);
    dist[s0] = 0;
    queue[tail++] = s0;
    while (head < tail) {
        uint32_t s = queue[head++];
        struct cube c = decode(s);
        for (int m = 0; m < 9; m++) {
            struct cube n = move(c, m);
            uint32_t ns = encode(&n);
            if (dist[ns] == 0xff) {
                dist[ns] = (uint8_t) (dist[s] + 1);
                queue[tail++] = ns;
            }
        }
    }
    long count[MAXD + 2] = {0};
    for (uint32_t s = 0; s < NSTATES; s++)
        count[dist[s] <= MAXD ? dist[s] : MAXD + 1]++;
    printf("BFS: %u states reached;", tail);
    for (int d = 0; d <= MAXD; d++)
        printf(" %ld", count[d]);
    printf("\n");
    CHECK(tail == NSTATES && count[MAXD + 1] == 0, "BFS incomplete\n");
    free(queue);

    check_tables();

    long runs[MAXD + 1] = {0}, wrong = 0, bad_path = 0;
    long sum_e[MAXD + 1] = {0}, sum_l[MAXD + 1] = {0};
    long max_e[MAXD + 1] = {0}, max_l[MAXD + 1] = {0};
    uint32_t worst_e[MAXD + 1] = {0}, worst_l[MAXD + 1] = {0};
    for (uint32_t s = 0; s < NSTATES; s++) {
        int d = dist[s];
        if (d < 10 && s % (uint32_t) stride)
            continue;
        struct cube c = decode(s);
        int path[MAXD + 1], depth, nd;
        edges = lookups = 0;
        int len = solve(coord(&c, 0), coord(&c, 1), path, &depth, &nd);
        if (len != d)
            wrong++;
        /* Replay the path in the full model, then complete greedily. */
        struct cube x = c;
        for (int i = 0; i < depth; i++)
            x = move(x, path[i]);
        long saved = lookups;
        int steps = depth;
        for (int cur = nd; cur > 0; cur--) {
            int found = 0;
            for (int m = 0; m < 9 && !found; m++) {
                struct cube y = move(x, m);
                if (near(coord(&y, 0), coord(&y, 1)) == cur - 1) {
                    x = y;
                    found = 1;
                    steps++;
                }
            }
            if (!found)
                break;
        }
        lookups = saved;
        if (encode(&x) != s0 || steps != d)
            bad_path++;
        runs[d]++;
        sum_e[d] += edges;
        sum_l[d] += lookups;
        if (edges > max_e[d]) {
            max_e[d] = edges;
            worst_e[d] = s;
        }
        if (lookups > max_l[d]) {
            max_l[d] = lookups;
            worst_l[d] = s;
        }
    }
    CHECK(!wrong && !bad_path, "solver: %ld wrong lengths, %ld bad paths\n",
          wrong, bad_path);
    printf("solver: %ld wrong lengths, %ld bad paths (stride %d below d10)\n",
           wrong, bad_path, stride);
    printf(" d     runs   avg_edges  max_edges  avg_lookups  max_lookups\n");
    for (int d = 0; d <= MAXD; d++) {
        if (!runs[d])
            continue;
        printf("%2d %8ld %11.1f %10ld %12.1f %12ld\n", d, runs[d],
               (double) sum_e[d] / runs[d], max_e[d],
               (double) sum_l[d] / runs[d], max_l[d]);
    }
    for (int d = 10; d <= MAXD; d++) {
        const uint32_t w[2] = {worst_e[d], worst_l[d]};
        for (int k = 0; k < 2; k++) {
            struct cube c = decode(w[k]);
            printf("worst %s at d%d: \"", k ? "lookups" : "edges", d);
            for (int i = 0; i < CUBIES; i++)
                putchar('1' + c.p[i]);
            for (int i = 0; i < CUBIES; i++)
                putchar('1' + c.o[i]);
            printf("\"\n");
        }
    }
    free(dist);
    if (failures)
        printf("FAILED (%d)\n", failures);
    else
        printf("ALL CHECKS PASS\n");
    return failures != 0;
}
