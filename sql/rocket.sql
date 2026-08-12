-- ROCKET in pure SQL — the zero-build fallback, and the in-database statement of the spec.
--
-- PLAN.md Phase 4 asks for this first, before any C++: it establishes the semantics inside
-- DuckDB, it runs anywhere DuckDB runs with nothing to compile, and if it turned out to be
-- merely 5-10x slower than Python rather than 1000x, it would be worth stopping here.
-- `scripts/sql_rocket_check.py` measures that and checks this file against the Phase 1 oracle.
--
-- Everything below is normative in SPEC.md. Where this file and `duckdb_rocket/` disagree, the
-- Python is right and this is a bug.
--
-- ----------------------------------------------------------------------------------------
-- Representing u64 in a database with no unsigned wrapping arithmetic
-- ----------------------------------------------------------------------------------------
-- DuckDB's UBIGINT *traps* on overflow rather than wrapping, and its bitwise operators do not
-- cover HUGEINT, so neither of the obvious representations works directly. A u64 is therefore
-- carried as two 32-bit halves in BIGINT columns, `hi` and `lo`, each in [0, 2^32). Every
-- 32-bit intermediate fits a BIGINT with room to spare, `xor` is defined there, and the
-- wrapping is explicit -- which is the point, since silent wrapping is what we cannot have.

-- Split a 64-bit value carried as (hi, lo) -- helpers are written to take and return that pair.

CREATE OR REPLACE MACRO u64_xor_hi(ah, al, bh, bl) AS xor(ah::BIGINT, bh::BIGINT);
CREATE OR REPLACE MACRO u64_xor_lo(ah, al, bh, bl) AS xor(al::BIGINT, bl::BIGINT);

-- Logical right shift by n (0 < n < 64), returning the high half of the result.
CREATE OR REPLACE MACRO u64_shr_hi(h, l, n) AS
    CASE WHEN n >= 32 THEN 0 ELSE (h::BIGINT // (1::BIGINT << n)) END;

-- ...and the low half. For n >= 32 the result's low half comes entirely out of `hi`.
CREATE OR REPLACE MACRO u64_shr_lo(h, l, n) AS
    CASE
        WHEN n >= 32 THEN (h::BIGINT // (1::BIGINT << (n - 32)))
        WHEN n = 0   THEN l::BIGINT
        ELSE ((l::BIGINT // (1::BIGINT << n))
              + ((h::BIGINT % (1::BIGINT << n)) * (1::BIGINT << (32 - n)))) % 4294967296
    END;

-- Addition mod 2^64, carrying from the low half into the high half.
CREATE OR REPLACE MACRO u64_add_lo(ah, al, bh, bl) AS (al::BIGINT + bl::BIGINT) % 4294967296;
CREATE OR REPLACE MACRO u64_add_hi(ah, al, bh, bl) AS
    (ah::BIGINT + bh::BIGINT + ((al::BIGINT + bl::BIGINT) // 4294967296)) % 4294967296;

-- Multiplication mod 2^64, with a = ah*2^32 + al and b = bh*2^32 + bl:
--
--       al*bl  +  ((al*bh + ah*bl) mod 2^32) * 2^32
--
-- because every term containing ah*bh is a multiple of 2^64 and vanishes.
--
-- The partial products are computed in HUGEINT, not BIGINT. Two 32-bit factors reach just under
-- 2^64, and BIGINT tops out at 2^63-1 -- so `al * bh` in BIGINT silently wraps into a negative
-- number and every downstream value is quietly wrong. The failure surfaces far from its cause,
-- as an out-of-range cast several macros later.
CREATE OR REPLACE MACRO u64_mul_lo(ah, al, bh, bl) AS
    ((al::HUGEINT * bl::HUGEINT) % 4294967296)::BIGINT;
CREATE OR REPLACE MACRO u64_mul_hi(ah, al, bh, bl) AS
    ((((al::HUGEINT * bl::HUGEINT) // 4294967296)
      + (al::HUGEINT * bh::HUGEINT)
      + (ah::HUGEINT * bl::HUGEINT)) % 4294967296)::BIGINT;

-- SplitMix64's three constants, as (hi, lo) pairs. SPEC.md section 1.1.
CREATE OR REPLACE MACRO golden_gamma_hi() AS 2654435769;  -- 0x9E3779B9
CREATE OR REPLACE MACRO golden_gamma_lo() AS 2135587861;  -- 0x7F4A7C15
CREATE OR REPLACE MACRO mix_a_hi()        AS 3210233709;  -- 0xBF58476D
CREATE OR REPLACE MACRO mix_a_lo()        AS 484763065;   -- 0x1CE4E5B9
CREATE OR REPLACE MACRO mix_b_hi()        AS 2496678331;  -- 0x94D049BB
CREATE OR REPLACE MACRO mix_b_lo()        AS 321982955;   -- 0x133111EB

-- SPEC.md 1.2: mix(z) = ((z ^ (z>>30)) * MIX_A); ((z ^ (z>>27)) * MIX_B); z ^ (z>>31)
--
-- Written as one expression per output half. It is repetitive because SQL macros cannot hold
-- an intermediate, and inlining by hand is preferable to a temp table per round.
-- The CTE columns are named `m1h`/`m1l` rather than `h`/`l` because a CTE column sharing a name
-- with a macro parameter is a binder error inside a macro body, not a shadowing rule.
CREATE OR REPLACE MACRO mix_hi(zh, zl) AS (
    WITH s1 AS (SELECT xor(zh::BIGINT, u64_shr_hi(zh, zl, 30)) AS x1h,
                       xor(zl::BIGINT, u64_shr_lo(zh, zl, 30)) AS x1l),
         r1 AS (SELECT u64_mul_hi((SELECT x1h FROM s1), (SELECT x1l FROM s1),
                                  mix_a_hi(), mix_a_lo()) AS m1h,
                       u64_mul_lo((SELECT x1h FROM s1), (SELECT x1l FROM s1),
                                  mix_a_hi(), mix_a_lo()) AS m1l),
         s2 AS (SELECT xor((SELECT m1h FROM r1),
                           u64_shr_hi((SELECT m1h FROM r1), (SELECT m1l FROM r1), 27)) AS x2h,
                       xor((SELECT m1l FROM r1),
                           u64_shr_lo((SELECT m1h FROM r1), (SELECT m1l FROM r1), 27)) AS x2l),
         r2 AS (SELECT u64_mul_hi((SELECT x2h FROM s2), (SELECT x2l FROM s2),
                                  mix_b_hi(), mix_b_lo()) AS m2h,
                       u64_mul_lo((SELECT x2h FROM s2), (SELECT x2l FROM s2),
                                  mix_b_hi(), mix_b_lo()) AS m2l)
    SELECT xor((SELECT m2h FROM r2),
               u64_shr_hi((SELECT m2h FROM r2), (SELECT m2l FROM r2), 31))
);

CREATE OR REPLACE MACRO mix_lo(zh, zl) AS (
    WITH s1 AS (SELECT xor(zh::BIGINT, u64_shr_hi(zh, zl, 30)) AS x1h,
                       xor(zl::BIGINT, u64_shr_lo(zh, zl, 30)) AS x1l),
         r1 AS (SELECT u64_mul_hi((SELECT x1h FROM s1), (SELECT x1l FROM s1),
                                  mix_a_hi(), mix_a_lo()) AS m1h,
                       u64_mul_lo((SELECT x1h FROM s1), (SELECT x1l FROM s1),
                                  mix_a_hi(), mix_a_lo()) AS m1l),
         s2 AS (SELECT xor((SELECT m1h FROM r1),
                           u64_shr_hi((SELECT m1h FROM r1), (SELECT m1l FROM r1), 27)) AS x2h,
                       xor((SELECT m1l FROM r1),
                           u64_shr_lo((SELECT m1h FROM r1), (SELECT m1l FROM r1), 27)) AS x2l),
         r2 AS (SELECT u64_mul_hi((SELECT x2h FROM s2), (SELECT x2l FROM s2),
                                  mix_b_hi(), mix_b_lo()) AS m2h,
                       u64_mul_lo((SELECT x2h FROM s2), (SELECT x2l FROM s2),
                                  mix_b_hi(), mix_b_lo()) AS m2l)
    SELECT xor((SELECT m2l FROM r2),
               u64_shr_lo((SELECT m2h FROM r2), (SELECT m2l FROM r2), 31))
);

-- SPEC.md 1.4: the top 53 bits, scaled by 2^-53. Taking `hi` whole (32 bits) and the top 21
-- bits of `lo` is exactly `u >> 11`.
CREATE OR REPLACE MACRO u64_to_double(h, l) AS
    ((h::DOUBLE * 2097152.0) + (l::BIGINT // 2048)::DOUBLE) / 9007199254740992.0;

-- SPEC.md 2: kernel_seed(master, i) = mix(master + (i+1) * GOLDEN_GAMMA).
--
-- (i+1) * GOLDEN_GAMMA is computed with the same wrapping multiply as everything else, so a
-- large kernel index wraps identically to the Python reference rather than saturating.
CREATE OR REPLACE MACRO kernel_seed_hi(master_hi, master_lo, i) AS
    mix_hi(
        u64_add_hi(master_hi, master_lo,
                   u64_mul_hi(((i + 1)::BIGINT // 4294967296), ((i + 1)::BIGINT % 4294967296),
                              golden_gamma_hi(), golden_gamma_lo()),
                   u64_mul_lo(((i + 1)::BIGINT // 4294967296), ((i + 1)::BIGINT % 4294967296),
                              golden_gamma_hi(), golden_gamma_lo())),
        u64_add_lo(master_hi, master_lo,
                   u64_mul_hi(((i + 1)::BIGINT // 4294967296), ((i + 1)::BIGINT % 4294967296),
                              golden_gamma_hi(), golden_gamma_lo()),
                   u64_mul_lo(((i + 1)::BIGINT // 4294967296), ((i + 1)::BIGINT % 4294967296),
                              golden_gamma_hi(), golden_gamma_lo())));

CREATE OR REPLACE MACRO kernel_seed_lo(master_hi, master_lo, i) AS
    mix_lo(
        u64_add_hi(master_hi, master_lo,
                   u64_mul_hi(((i + 1)::BIGINT // 4294967296), ((i + 1)::BIGINT % 4294967296),
                              golden_gamma_hi(), golden_gamma_lo()),
                   u64_mul_lo(((i + 1)::BIGINT // 4294967296), ((i + 1)::BIGINT % 4294967296),
                              golden_gamma_hi(), golden_gamma_lo())),
        u64_add_lo(master_hi, master_lo,
                   u64_mul_hi(((i + 1)::BIGINT // 4294967296), ((i + 1)::BIGINT % 4294967296),
                              golden_gamma_hi(), golden_gamma_lo()),
                   u64_mul_lo(((i + 1)::BIGINT // 4294967296), ((i + 1)::BIGINT % 4294967296),
                              golden_gamma_hi(), golden_gamma_lo())));

-- ----------------------------------------------------------------------------------------
-- The draw stream
-- ----------------------------------------------------------------------------------------
-- SPEC.md 1.3 advances the state by GOLDEN_GAMMA per call and returns mix(state), so the state
-- after `step` calls is `seed + step * GOLDEN_GAMMA` in closed form. That is what makes the
-- stream expressible as a *table* -- draw `step` for kernel `i` is computable directly, with no
-- recursion and no ordering dependency, and DuckDB can produce the whole grid in parallel.
--
-- This is the same property SPEC.md 2 invokes to make kernels independently addressable; it
-- applies just as well within a kernel's own substream.
CREATE OR REPLACE MACRO draw(master_hi, master_lo, i, step) AS
    u64_to_double(
        mix_hi(u64_add_hi(kernel_seed_hi(master_hi, master_lo, i),
                          kernel_seed_lo(master_hi, master_lo, i),
                          u64_mul_hi((step::BIGINT // 4294967296), (step::BIGINT % 4294967296),
                                     golden_gamma_hi(), golden_gamma_lo()),
                          u64_mul_lo((step::BIGINT // 4294967296), (step::BIGINT % 4294967296),
                                     golden_gamma_hi(), golden_gamma_lo())),
               u64_add_lo(kernel_seed_hi(master_hi, master_lo, i),
                          kernel_seed_lo(master_hi, master_lo, i),
                          u64_mul_hi((step::BIGINT // 4294967296), (step::BIGINT % 4294967296),
                                     golden_gamma_hi(), golden_gamma_lo()),
                          u64_mul_lo((step::BIGINT // 4294967296), (step::BIGINT % 4294967296),
                                     golden_gamma_hi(), golden_gamma_lo()))),
        mix_lo(u64_add_hi(kernel_seed_hi(master_hi, master_lo, i),
                          kernel_seed_lo(master_hi, master_lo, i),
                          u64_mul_hi((step::BIGINT // 4294967296), (step::BIGINT % 4294967296),
                                     golden_gamma_hi(), golden_gamma_lo()),
                          u64_mul_lo((step::BIGINT // 4294967296), (step::BIGINT % 4294967296),
                                     golden_gamma_hi(), golden_gamma_lo())),
               u64_add_lo(kernel_seed_hi(master_hi, master_lo, i),
                          kernel_seed_lo(master_hi, master_lo, i),
                          u64_mul_hi((step::BIGINT // 4294967296), (step::BIGINT % 4294967296),
                                     golden_gamma_hi(), golden_gamma_lo()),
                          u64_mul_lo((step::BIGINT // 4294967296), (step::BIGINT % 4294967296),
                                     golden_gamma_hi(), golden_gamma_lo()))));

-- `step` is 1-based: SPEC.md 2 is explicit that a kernel's first draw is
-- mix(kernel_seed + GOLDEN_GAMMA), not mix(kernel_seed).
