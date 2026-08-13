#pragma once

// ROCKET kernel generation and transform, in C++.
//
// This is a direct port of `duckdb_rocket/rocket.py` and `duckdb_rocket/prng.py`, specified in
// SPEC.md. The Python is the reference: where the two disagree, this file is wrong. The golden
// vectors in `reference/golden/` are what actually adjudicates, via test/conformance.
//
// Deliberately header-only and free of DuckDB types, so the numerics can be compiled and tested
// on their own -- a conformance failure should never require guessing whether the bug is in the
// arithmetic or in the vector plumbing around it.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

namespace duckdb_rocket {

// SPEC.md 1.1. GOLDEN_GAMMA is the odd 64-bit approximation of 2^64 / phi.
static constexpr uint64_t GOLDEN_GAMMA = 0x9E3779B97F4A7C15ULL;
static constexpr uint64_t MIX_A = 0xBF58476D1CE4E5B9ULL;
static constexpr uint64_t MIX_B = 0x94D049BB133111EBULL;

// 2^-53: the top 53 bits of a u64 are exactly the binary64 mantissa width (SPEC.md 1.4).
static constexpr double TWO_POW_NEG_53 = 1.0 / static_cast<double>(1ULL << 53);

// ROCKET draws kernel length uniformly from this set. Fixed by the paper, not a tunable.
static constexpr int KERNEL_LENGTHS[3] = {7, 9, 11};

// Each kernel contributes exactly two features: global max and PPV.
static constexpr int FEATURES_PER_KERNEL = 2;

// SPEC.md 1.2. Unsigned arithmetic wraps by definition in C++, which is what this needs;
// the Python reference spells the same thing as an explicit mask.
inline uint64_t Mix(uint64_t z) {
	z = (z ^ (z >> 30)) * MIX_A;
	z = (z ^ (z >> 27)) * MIX_B;
	return z ^ (z >> 31);
}

// SPEC.md 2. Kernel `index` is a pure function of (master_seed, index) -- there is no
// sequential stream across kernels. That is what lets a group be generated without generating
// its predecessors, and kernels be fanned across threads with no shared state.
//
// The `index + 1` is required: with `index`, kernel 0 would collide with the unmixed state.
inline uint64_t KernelSeed(uint64_t master_seed, uint64_t index) {
	return Mix(master_seed + (index + 1) * GOLDEN_GAMMA);
}

// SPEC.md 1.3-1.5.
class SplitMix64 {
public:
	explicit SplitMix64(uint64_t seed) : state_(seed) {
	}

	uint64_t NextU64() {
		state_ += GOLDEN_GAMMA;
		return Mix(state_);
	}

	double NextDouble() {
		return static_cast<double>(NextU64() >> 11) * TWO_POW_NEG_53;
	}

	double NextUniform(double low, double high) {
		return low + (high - low) * NextDouble();
	}

	// Multiply-and-floor rather than rejection sampling: biased by at most n * 2^-53, which at
	// the n <= 3 used here is one part in 3e15, and in exchange the number of draws consumed is
	// fixed. A data-dependent draw count is an easy thing for a port to get subtly wrong.
	int64_t NextBelow(int64_t n) {
		return static_cast<int64_t>(NextDouble() * static_cast<double>(n));
	}

	// Marsaglia polar. Box-Muller is avoided so that no sin/cos is involved: neither glibc nor
	// the MSVC CRT guarantees correctly-rounded trig, so a trig-based transform can differ in
	// the last ulp between the oracle and this port -- exactly the drift the golden vectors
	// exist to catch, showing up as noise rather than as the real mismatch it would mask.
	//
	// The second normal of each accepted pair is DISCARDED (SPEC.md 1.5). Caching it would make
	// every call's result depend on the parity of the number of preceding calls -- hidden state
	// that each reimplementation would then have to match exactly.
	double NextNormal() {
		for (;;) {
			const double u = 2.0 * NextDouble() - 1.0;
			const double v = 2.0 * NextDouble() - 1.0;
			const double s = u * u + v * v;
			if (s > 0.0 && s < 1.0) {
				return u * std::sqrt(-2.0 * std::log(s) / s);
			}
		}
	}

private:
	uint64_t state_;
};

// One kernel, as SPEC.md 3 (and 7 for the multivariate case) defines it.
struct Kernel {
	int length;
	std::vector<double> weights; // already mean-centred, laid out channel-major
	double bias;
	int64_t dilation;
	int64_t padding;
	std::vector<int64_t> channels; // sorted ascending; {0} when univariate
};

// SPEC.md 7.3. Partial Fisher-Yates over an identity permutation, consuming exactly
// `n_selected` draws whatever comes up -- no rejection and no data-dependent draw count, for
// the same reason NextBelow does not reject: a port that consumes a different number of draws
// desynchronises the whole stream from that point on.
inline std::vector<int64_t> SelectChannels(SplitMix64 &rng, int64_t n_channels, int64_t n_selected) {
	std::vector<int64_t> perm(static_cast<size_t>(n_channels));
	for (int64_t c = 0; c < n_channels; c++) {
		perm[static_cast<size_t>(c)] = c;
	}
	for (int64_t k = 0; k < n_selected; k++) {
		const int64_t j = k + rng.NextBelow(n_channels - k);
		std::swap(perm[static_cast<size_t>(k)], perm[static_cast<size_t>(j)]);
	}
	std::vector<int64_t> chosen(perm.begin(), perm.begin() + static_cast<size_t>(n_selected));
	// Sorted: selection order carries no meaning, and a stable order means a channel's weight
	// block is found by position rather than by remembering the order the swaps produced.
	std::sort(chosen.begin(), chosen.end());
	return chosen;
}

// SPEC.md 3 and 7.2. The ORDER OF THESE DRAWS IS NORMATIVE -- reordering them, or inserting
// one, changes every kernel and invalidates every golden vector.
inline Kernel GenerateKernel(uint64_t master_seed, uint64_t index, int64_t n_timepoints, int64_t n_channels = 1) {
	SplitMix64 rng(KernelSeed(master_seed, index));

	Kernel kernel;
	kernel.length = KERNEL_LENGTHS[rng.NextBelow(3)];

	// SPEC.md 7.1: at one channel NO channel draw is made. The subset is forced there, so a
	// draw would spend randomness deciding something already decided -- and would shift every
	// later value in the stream, invalidating every committed golden vector for nothing.
	int64_t n_selected = 1;
	if (n_channels > 1) {
		const double upper = std::log2(static_cast<double>(n_channels) + 1.0);
		n_selected = static_cast<int64_t>(std::floor(std::pow(2.0, rng.NextUniform(0.0, upper))));
		if (n_selected < 1) {
			n_selected = 1;
		}
		if (n_selected > n_channels) {
			n_selected = n_channels;
		}
		kernel.channels = SelectChannels(rng, n_channels, n_selected);
	} else {
		kernel.channels.assign(1, 0);
	}

	// Weights are channel-major: all of the first selected channel's `length` values, then the
	// next. Mean-centring is PER CHANNEL (SPEC.md 7.4) -- each channel can carry its own
	// offset, and one global mean would only remove their average.
	kernel.weights.resize(static_cast<size_t>(kernel.length * n_selected));
	for (int64_t c = 0; c < n_selected; c++) {
		const size_t base = static_cast<size_t>(c * kernel.length);
		double sum = 0.0;
		for (int j = 0; j < kernel.length; j++) {
			kernel.weights[base + static_cast<size_t>(j)] = rng.NextNormal();
			sum += kernel.weights[base + static_cast<size_t>(j)];
		}
		// Plain arithmetic mean, summed in index order (SPEC.md 3).
		const double mean = sum / static_cast<double>(kernel.length);
		for (int j = 0; j < kernel.length; j++) {
			kernel.weights[base + static_cast<size_t>(j)] -= mean;
		}
	}

	kernel.bias = rng.NextUniform(-1.0, 1.0);

	// The bound guarantees (length - 1) * dilation <= n - 1, which is what makes
	// output_length >= 1 structural rather than a runtime check. dilation >= 1 always.
	const double limit = std::log2(static_cast<double>(n_timepoints - 1) / static_cast<double>(kernel.length - 1));
	kernel.dilation = static_cast<int64_t>(std::floor(std::pow(2.0, rng.NextUniform(0.0, limit))));

	kernel.padding = (rng.NextBelow(2) == 1) ? ((static_cast<int64_t>(kernel.length) - 1) * kernel.dilation) / 2 : 0;
	return kernel;
}

// SPEC.md 4. Writes this kernel's two features into out[0] (max) and out[1] (PPV).
//
// Accumulation order is normative-ish: the reference accumulates one tap at a time across all
// output positions, i.e. `j` is the OUTER loop. Floating-point addition is not associative, so
// a different order can differ in the last ulp -- which is why the conformance test uses a
// tight but non-zero tolerance rather than demanding bit-identical output.
inline void ApplyKernel(const double *series, int64_t n, const Kernel &kernel, double *out) {
	const int64_t output_length = n + 2 * kernel.padding - (static_cast<int64_t>(kernel.length) - 1) * kernel.dilation;

	std::vector<double> conv(static_cast<size_t>(output_length), kernel.bias);
	for (int j = 0; j < kernel.length; j++) {
		const double w = kernel.weights[static_cast<size_t>(j)];
		const int64_t offset = static_cast<int64_t>(j) * kernel.dilation - kernel.padding;
		for (int64_t k = 0; k < output_length; k++) {
			const int64_t idx = k + offset;
			// A tap outside [0, n) lands on the zero padding and contributes nothing.
			if (idx >= 0 && idx < n) {
				conv[static_cast<size_t>(k)] += w * series[idx];
			}
		}
	}

	double max_value = conv[0];
	int64_t positive = 0;
	for (int64_t k = 0; k < output_length; k++) {
		const double value = conv[static_cast<size_t>(k)];
		if (value > max_value) {
			max_value = value;
		}
		if (value > 0.0) { // strictly greater than zero
			positive++;
		}
	}

	out[0] = max_value;
	out[1] = static_cast<double>(positive) / static_cast<double>(output_length);
}

// SPEC.md 7.5. `series` is channel-major and contiguous: channel c's samples are at
// series[c * n .. c * n + n). The channels are summed INSIDE one convolution, which is why a
// kernel still yields exactly 2 features however many channels it reads.
inline void ApplyKernelMultivariate(const double *series, int64_t n, const Kernel &kernel, double *out) {
	const int64_t output_length = n + 2 * kernel.padding - (static_cast<int64_t>(kernel.length) - 1) * kernel.dilation;

	std::vector<double> conv(static_cast<size_t>(output_length), kernel.bias);
	// Channel-major, and within a channel one tap at a time across all output positions --
	// SPEC.md 7.5 fixes this order because floating-point addition is not associative.
	for (size_t ci = 0; ci < kernel.channels.size(); ci++) {
		const double *channel = series + kernel.channels[ci] * n;
		const size_t base = ci * static_cast<size_t>(kernel.length);
		for (int j = 0; j < kernel.length; j++) {
			const double w = kernel.weights[base + static_cast<size_t>(j)];
			const int64_t offset = static_cast<int64_t>(j) * kernel.dilation - kernel.padding;
			for (int64_t k = 0; k < output_length; k++) {
				const int64_t idx = k + offset;
				if (idx >= 0 && idx < n) {
					conv[static_cast<size_t>(k)] += w * channel[idx];
				}
			}
		}
	}

	double max_value = conv[0];
	int64_t positive = 0;
	for (int64_t k = 0; k < output_length; k++) {
		const double value = conv[static_cast<size_t>(k)];
		if (value > max_value) {
			max_value = value;
		}
		if (value > 0.0) {
			positive++;
		}
	}

	out[0] = max_value;
	out[1] = static_cast<double>(positive) / static_cast<double>(output_length);
}

// SPEC.md 5: features are INTERLEAVED -- column 2i is kernel i's max, 2i+1 its PPV. That is
// what makes a contiguous column range correspond to a whole set of kernels, which is what
// group extraction needs.
// `n_reference` is what the kernels are DRAWN against; `n` is the length of this particular
// series. They differ only for variable-length data (SPEC.md 8), where one bank must be shared
// across rows of differing length or column `j` stops meaning the same thing in every row.
inline std::vector<double> Transform(const double *series, int64_t n, int64_t n_reference, uint64_t master_seed,
                                     int64_t kernels_per_group, int64_t first_kernel) {
	std::vector<double> features(static_cast<size_t>(kernels_per_group * FEATURES_PER_KERNEL));
	for (int64_t i = 0; i < kernels_per_group; i++) {
		const Kernel kernel = GenerateKernel(master_seed, static_cast<uint64_t>(first_kernel + i), n_reference);
		ApplyKernel(series, n, kernel, &features[static_cast<size_t>(i * FEATURES_PER_KERNEL)]);
	}
	return features;
}

// The multivariate bank. `series` is channel-major, `n_channels * n` doubles.
inline std::vector<double> TransformMultivariate(const double *series, int64_t n_channels, int64_t n,
                                                 int64_t n_reference, uint64_t master_seed, int64_t kernels_per_group,
                                                 int64_t first_kernel) {
	std::vector<double> features(static_cast<size_t>(kernels_per_group * FEATURES_PER_KERNEL));
	for (int64_t i = 0; i < kernels_per_group; i++) {
		const Kernel kernel =
		    GenerateKernel(master_seed, static_cast<uint64_t>(first_kernel + i), n_reference, n_channels);
		ApplyKernelMultivariate(series, n, kernel, &features[static_cast<size_t>(i * FEATURES_PER_KERNEL)]);
	}
	return features;
}

} // namespace duckdb_rocket
