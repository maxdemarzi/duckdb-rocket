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

-- ----------------------------------------------------------------------------------------
-- Kernel generation (SPEC.md 3)
-- ----------------------------------------------------------------------------------------
-- The draw order is normative: length, then `length` normals, then bias, then dilation, then
-- the padding coin. The awkward part in SQL is the normals, because SPEC.md 1.5's polar method
-- rejects roughly 21% of pairs and therefore consumes a data-dependent number of draws -- a
-- loop, in a language without one.
--
-- The way through is that the draw stream is addressable (see `draw` above): rather than
-- iterating, generate a fixed grid of draws per kernel, evaluate every candidate pair at once,
-- and use a running count of accepted pairs to work out both which pairs supplied the normals
-- AND what step the stream had reached afterwards -- which is what bias, dilation and padding
-- need in order to continue from the right place.
--
-- MAX_PAIRS is a safety margin, not a tuning knob. At ~78.5% acceptance, needing 11 normals
-- takes ~14 pairs; 80 makes exhaustion a ~1e-30 event. `rocket_kernels` asserts it anyway,
-- because the alternative to an assertion here is silently short weights.
CREATE OR REPLACE MACRO rocket_max_pairs() AS 80;

-- Every candidate pair for every kernel, with its acceptance decision.
-- Pair p occupies steps (2 + 2p) and (3 + 2p); step 1 is the length draw.
CREATE OR REPLACE MACRO rocket_pairs(master_hi, master_lo, first_kernel, n_kernels) AS TABLE
SELECT
    i AS kernel_id,
    p,
    2 * u - 1 AS u,
    2 * v - 1 AS v,
    (2 * u - 1) * (2 * u - 1) + (2 * v - 1) * (2 * v - 1) AS s,
    3 + 2 * p AS end_step
FROM (
    SELECT
        i, p,
        draw(master_hi, master_lo, first_kernel + i, 2 + 2 * p) AS u,
        draw(master_hi, master_lo, first_kernel + i, 3 + 2 * p) AS v
    FROM range(n_kernels) AS t(i), range(rocket_max_pairs()) AS q(p)
);

-- Kernel scalars: length, bias, dilation, padding -- plus the stream position they were drawn
-- from, which is what ties them to the normals that preceded them.
CREATE OR REPLACE MACRO rocket_kernel_params(master_hi, master_lo, first_kernel, n_kernels,
                                             n_timepoints) AS TABLE
WITH lengths AS (
    SELECT i AS kernel_id,
           [7, 9, 11][1 + floor(draw(master_hi, master_lo, first_kernel + i, 1) * 3)::INT]
               AS length
    FROM range(n_kernels) AS t(i)
),
accepted AS (
    -- Accepted pairs only, numbered in stream order. `s = 0` is astronomically unlikely but
    -- would divide by zero; `s >= 1` is the ordinary rejection outside the unit disc.
    SELECT kernel_id, p, u, s, end_step,
           row_number() OVER (PARTITION BY kernel_id ORDER BY p) AS nth
    FROM rocket_pairs(master_hi, master_lo, first_kernel, n_kernels)
    WHERE s > 0 AND s < 1
),
-- Where the stream stands once `length` normals have been produced.
cursor AS (
    SELECT a.kernel_id, l.length, a.end_step
    FROM accepted a JOIN lengths l USING (kernel_id)
    WHERE a.nth = l.length
)
SELECT
    c.kernel_id,
    c.length,
    -- SPEC.md 3: bias, then dilation, then the padding coin, in that order.
    -1 + 2 * draw(master_hi, master_lo, first_kernel + c.kernel_id, c.end_step + 1) AS bias,
    floor(pow(2.0,
              draw(master_hi, master_lo, first_kernel + c.kernel_id, c.end_step + 2)
              * log2((n_timepoints - 1)::DOUBLE / (c.length - 1)::DOUBLE)))::BIGINT AS dilation,
    CASE WHEN floor(draw(master_hi, master_lo, first_kernel + c.kernel_id, c.end_step + 3) * 2)
              = 1
         THEN ((c.length - 1) *
               floor(pow(2.0,
                         draw(master_hi, master_lo, first_kernel + c.kernel_id, c.end_step + 2)
                         * log2((n_timepoints - 1)::DOUBLE / (c.length - 1)::DOUBLE)))::BIGINT)
              // 2
         ELSE 0 END AS padding,
    c.end_step
FROM cursor c;

-- The weights: `length` normals, mean-centred (SPEC.md 3).
CREATE OR REPLACE MACRO rocket_kernels(master_hi, master_lo, first_kernel, n_kernels,
                                       n_timepoints) AS TABLE
WITH params AS (
    SELECT * FROM rocket_kernel_params(master_hi, master_lo, first_kernel, n_kernels,
                                       n_timepoints)
),
normals AS (
    SELECT a.kernel_id,
           a.nth - 1 AS j,
           a.u * sqrt(-2 * ln(a.s) / a.s) AS raw
    FROM (
        SELECT kernel_id, p, u, s,
               row_number() OVER (PARTITION BY kernel_id ORDER BY p) AS nth
        FROM rocket_pairs(master_hi, master_lo, first_kernel, n_kernels)
        WHERE s > 0 AND s < 1
    ) a
    JOIN params pm USING (kernel_id)
    WHERE a.nth <= pm.length
),
centred AS (
    -- Plain arithmetic mean over the kernel's own weights.
    SELECT kernel_id, j, raw - avg(raw) OVER (PARTITION BY kernel_id) AS weight
    FROM normals
)
SELECT c.kernel_id, c.j, c.weight, p.length, p.bias, p.dilation, p.padding
FROM centred c JOIN params p USING (kernel_id);

-- ----------------------------------------------------------------------------------------
-- The transform (SPEC.md 4)
-- ----------------------------------------------------------------------------------------
-- `series` is (series_id, t, value) in long form -- one row per timepoint. That is the shape a
-- join can work with; `sql_rocket_check.py` unnests a LIST column into it.
--
-- This is where the pure-SQL path stops being competitive, and visibly so: the join below is
-- series x kernels x output_positions x taps, which for the paper's configuration is billions
-- of rows. It is written for clarity and for its value as an executable statement of the spec,
-- not for throughput. That is exactly what PLAN.md Phase 4 predicted, and the measurement in
-- `sql_rocket_check.py` is what turns the prediction into a number.
CREATE OR REPLACE MACRO rocket_features(master_hi, master_lo, first_kernel, n_kernels,
                                        n_timepoints, series) AS TABLE
WITH kernels AS (
    SELECT * FROM rocket_kernels(master_hi, master_lo, first_kernel, n_kernels, n_timepoints)
),
shape AS (
    SELECT DISTINCT kernel_id, length, bias, dilation, padding,
           n_timepoints + 2 * padding - (length - 1) * dilation AS output_length
    FROM kernels
),
conv AS (
    SELECT s.series_id, sh.kernel_id, o.k,
           sh.bias + sum(kw.weight * x.value) AS value
    FROM shape sh
    CROSS JOIN (SELECT DISTINCT series_id FROM series) s
    CROSS JOIN LATERAL range(sh.output_length) AS o(k)
    JOIN kernels kw ON kw.kernel_id = sh.kernel_id
    -- A tap outside [0, n) lands on the zero padding and contributes nothing, so an inner
    -- join expresses the padding exactly (SPEC.md 4).
    JOIN series x ON x.series_id = s.series_id
                 AND x.t = o.k + kw.j * sh.dilation - sh.padding
    GROUP BY s.series_id, sh.kernel_id, o.k, sh.bias
)
SELECT series_id, kernel_id,
       max(value) AS max_feature,
       (count(*) FILTER (WHERE value > 0))::DOUBLE / count(*)::DOUBLE AS ppv_feature
FROM conv
GROUP BY series_id, kernel_id;
