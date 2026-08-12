#define DUCKDB_EXTENSION_MAIN

#include "rocket_extension.hpp"
#include "rocket.hpp"

#include "duckdb.hpp"
#include "duckdb/common/exception.hpp"
#include "duckdb/common/vector_operations/vector_operations.hpp"
#include "duckdb/function/scalar_function.hpp"
#include "duckdb/main/extension/extension_loader.hpp"

namespace duckdb {

// The reference length a bank is drawn against (SPEC.md 8).
//
// Without the optional fifth argument this is the row's own length, which is correct only when
// every row is the same length. On ragged data that silently gives each row its OWN kernel
// bank -- weights and lengths would match, but dilation and padding are drawn against `n`, so
// one extra timepoint is enough to change them. The result is a well-formed feature matrix in
// which every row was measured with a different instrument, and column `j` means nothing.
//
// A series shorter than the reference is rejected rather than padded: padding would fabricate
// data and change what the features measure.
static int64_t ReferenceLength(DataChunk &args, idx_t count, idx_t row, int64_t n) {
	if (args.ColumnCount() < 5) {
		return n;
	}
	UnifiedVectorFormat format;
	args.data[4].ToUnifiedFormat(count, format);
	const auto idx = format.sel->get_index(row);
	if (!format.validity.RowIsValid(idx)) {
		return n;
	}
	const auto reference = UnifiedVectorFormat::GetData<int64_t>(format)[idx];
	if (reference < duckdb_rocket::KERNEL_LENGTHS[2]) {
		throw InvalidInputException(
		    "rocket_transform: n_reference %lld is shorter than the longest kernel (%d)",
		    static_cast<long long>(reference), duckdb_rocket::KERNEL_LENGTHS[2]);
	}
	if (n < reference) {
		throw InvalidInputException(
		    "rocket_transform: series has %lld timepoints but n_reference is %lld; a series "
		    "shorter than the reference is rejected rather than padded (SPEC.md 8.2). Draw the "
		    "bank against the shortest series in the dataset",
		    static_cast<long long>(n), static_cast<long long>(reference));
	}
	return reference;
}

// rocket_transform(series, kernels_per_group, seed, first_kernel[, n_reference]) -> DOUBLE[]
//
// The argument order deliberately mirrors the Python reference's
// `generate_kernels(seed, n_timepoints, kernels_per_group, first_kernel)` plus `transform`, so
// a conformance failure can be reproduced in the oracle without translating anything. Series
// length is not an argument because it is the length of the list passed in.
//
// `first_kernel` is what makes group extraction work: group g is global kernel indices
// [g*k, (g+1)*k), and because kernel i is a pure function of (seed, i), asking for group 7
// yields exactly the kernels a full 10,000-kernel generation would have put at those indices
// (SPEC.md 2, 6).
static void RocketTransformFunction(DataChunk &args, ExpressionState &state, Vector &result) {
	const auto count = args.size();

	UnifiedVectorFormat series_format;
	args.data[0].ToUnifiedFormat(count, series_format);
	const auto list_entries = UnifiedVectorFormat::GetData<list_entry_t>(series_format);

	auto &child_vector = ListVector::GetEntry(args.data[0]);
	const auto child_count = ListVector::GetListSize(args.data[0]);
	UnifiedVectorFormat child_format;
	child_vector.ToUnifiedFormat(child_count, child_format);
	const auto child_data = UnifiedVectorFormat::GetData<double>(child_format);

	UnifiedVectorFormat kernels_format, seed_format, first_format;
	args.data[1].ToUnifiedFormat(count, kernels_format);
	args.data[2].ToUnifiedFormat(count, seed_format);
	args.data[3].ToUnifiedFormat(count, first_format);
	const auto kernels_data = UnifiedVectorFormat::GetData<int64_t>(kernels_format);
	const auto seed_data = UnifiedVectorFormat::GetData<int64_t>(seed_format);
	const auto first_data = UnifiedVectorFormat::GetData<int64_t>(first_format);

	// Size the output child up front. Every row produces exactly
	// kernels_per_group * 2 features (SPEC.md 5), so this is not a guess.
	idx_t total_features = 0;
	for (idx_t row = 0; row < count; row++) {
		const auto k_idx = kernels_format.sel->get_index(row);
		if (!kernels_format.validity.RowIsValid(k_idx)) {
			continue;
		}
		total_features += static_cast<idx_t>(kernels_data[k_idx] *
		                                     duckdb_rocket::FEATURES_PER_KERNEL);
	}

	result.SetVectorType(VectorType::FLAT_VECTOR);
	ListVector::Reserve(result, total_features);
	ListVector::SetListSize(result, total_features);
	const auto result_entries = FlatVector::GetData<list_entry_t>(result);
	auto &result_child = ListVector::GetEntry(result);
	const auto result_data = FlatVector::GetData<double>(result_child);
	auto &result_validity = FlatVector::Validity(result);

	std::vector<double> series;
	idx_t offset = 0;

	for (idx_t row = 0; row < count; row++) {
		const auto s_idx = series_format.sel->get_index(row);
		const auto k_idx = kernels_format.sel->get_index(row);
		const auto seed_idx = seed_format.sel->get_index(row);
		const auto f_idx = first_format.sel->get_index(row);

		if (!series_format.validity.RowIsValid(s_idx) ||
		    !kernels_format.validity.RowIsValid(k_idx) ||
		    !seed_format.validity.RowIsValid(seed_idx) ||
		    !first_format.validity.RowIsValid(f_idx)) {
			result_validity.SetInvalid(row);
			result_entries[row] = list_entry_t(offset, 0);
			continue;
		}

		const auto kernels_per_group = kernels_data[k_idx];
		const auto first_kernel = first_data[f_idx];
		if (kernels_per_group <= 0) {
			throw InvalidInputException("rocket_transform: kernels_per_group must be positive");
		}
		if (first_kernel < 0) {
			throw InvalidInputException("rocket_transform: first_kernel must be non-negative");
		}

		// Gather the series into a contiguous buffer. The child vector may carry a selection
		// vector, so indexing it directly would silently read the wrong elements.
		const auto entry = list_entries[s_idx];
		series.clear();
		series.reserve(static_cast<size_t>(entry.length));
		for (idx_t j = 0; j < entry.length; j++) {
			const auto c_idx = child_format.sel->get_index(entry.offset + j);
			if (!child_format.validity.RowIsValid(c_idx)) {
				throw InvalidInputException("rocket_transform: series contains NULL values");
			}
			series.push_back(child_data[c_idx]);
		}

		const auto n = static_cast<int64_t>(series.size());
		// Dilation is drawn against series length, so a length below the longest kernel makes
		// the log2 bound undefined rather than merely awkward. Reject it rather than produce
		// features that are not comparable with anything.
		if (n < duckdb_rocket::KERNEL_LENGTHS[2]) {
			throw InvalidInputException(
			    "rocket_transform: series length %lld is shorter than the longest kernel (%d)",
			    static_cast<long long>(n), duckdb_rocket::KERNEL_LENGTHS[2]);
		}

		const auto n_reference = ReferenceLength(args, count, row, n);

		const auto features = duckdb_rocket::Transform(
		    series.data(), n, n_reference, static_cast<uint64_t>(seed_data[seed_idx]),
		    kernels_per_group, first_kernel);

		for (size_t j = 0; j < features.size(); j++) {
			result_data[offset + j] = features[j];
		}
		result_entries[row] = list_entry_t(offset, features.size());
		offset += features.size();
	}

	if (count == 1) {
		result.SetVectorType(VectorType::CONSTANT_VECTOR);
	}
}

// rocket_transform(series DOUBLE[][], kernels_per_group, seed, first_kernel) -> DOUBLE[]
//
// The multivariate overload (SPEC.md 7). The outer list is channels, the inner one timepoints,
// and every channel must be the same length -- a ragged series has no single `n` to draw
// dilations against.
//
// Still two features per kernel: a kernel sums its selected channels inside one convolution
// rather than producing a feature per channel.
static void RocketTransformMultivariateFunction(DataChunk &args, ExpressionState &state,
                                                Vector &result) {
	const auto count = args.size();

	UnifiedVectorFormat outer_format;
	args.data[0].ToUnifiedFormat(count, outer_format);
	const auto outer_entries = UnifiedVectorFormat::GetData<list_entry_t>(outer_format);

	auto &channel_vector = ListVector::GetEntry(args.data[0]);
	const auto channel_count = ListVector::GetListSize(args.data[0]);
	UnifiedVectorFormat channel_format;
	channel_vector.ToUnifiedFormat(channel_count, channel_format);
	const auto channel_entries = UnifiedVectorFormat::GetData<list_entry_t>(channel_format);

	auto &sample_vector = ListVector::GetEntry(channel_vector);
	const auto sample_count = ListVector::GetListSize(channel_vector);
	UnifiedVectorFormat sample_format;
	sample_vector.ToUnifiedFormat(sample_count, sample_format);
	const auto sample_data = UnifiedVectorFormat::GetData<double>(sample_format);

	UnifiedVectorFormat kernels_format, seed_format, first_format;
	args.data[1].ToUnifiedFormat(count, kernels_format);
	args.data[2].ToUnifiedFormat(count, seed_format);
	args.data[3].ToUnifiedFormat(count, first_format);
	const auto kernels_data = UnifiedVectorFormat::GetData<int64_t>(kernels_format);
	const auto seed_data = UnifiedVectorFormat::GetData<int64_t>(seed_format);
	const auto first_data = UnifiedVectorFormat::GetData<int64_t>(first_format);

	idx_t total_features = 0;
	for (idx_t row = 0; row < count; row++) {
		const auto k_idx = kernels_format.sel->get_index(row);
		if (!kernels_format.validity.RowIsValid(k_idx)) {
			continue;
		}
		total_features +=
		    static_cast<idx_t>(kernels_data[k_idx] * duckdb_rocket::FEATURES_PER_KERNEL);
	}

	result.SetVectorType(VectorType::FLAT_VECTOR);
	ListVector::Reserve(result, total_features);
	ListVector::SetListSize(result, total_features);
	const auto result_entries = FlatVector::GetData<list_entry_t>(result);
	auto &result_child = ListVector::GetEntry(result);
	const auto result_data = FlatVector::GetData<double>(result_child);
	auto &result_validity = FlatVector::Validity(result);

	std::vector<double> series; // channel-major, contiguous
	idx_t offset = 0;

	for (idx_t row = 0; row < count; row++) {
		const auto o_idx = outer_format.sel->get_index(row);
		const auto k_idx = kernels_format.sel->get_index(row);
		const auto seed_idx = seed_format.sel->get_index(row);
		const auto f_idx = first_format.sel->get_index(row);

		if (!outer_format.validity.RowIsValid(o_idx) ||
		    !kernels_format.validity.RowIsValid(k_idx) ||
		    !seed_format.validity.RowIsValid(seed_idx) ||
		    !first_format.validity.RowIsValid(f_idx)) {
			result_validity.SetInvalid(row);
			result_entries[row] = list_entry_t(offset, 0);
			continue;
		}

		const auto kernels_per_group = kernels_data[k_idx];
		const auto first_kernel = first_data[f_idx];
		if (kernels_per_group <= 0) {
			throw InvalidInputException("rocket_transform: kernels_per_group must be positive");
		}
		if (first_kernel < 0) {
			throw InvalidInputException("rocket_transform: first_kernel must be non-negative");
		}

		const auto outer = outer_entries[o_idx];
		const auto n_channels = static_cast<int64_t>(outer.length);
		if (n_channels < 1) {
			throw InvalidInputException("rocket_transform: series has no channels");
		}

		int64_t n = -1;
		series.clear();
		for (idx_t c = 0; c < outer.length; c++) {
			const auto c_idx = channel_format.sel->get_index(outer.offset + c);
			if (!channel_format.validity.RowIsValid(c_idx)) {
				throw InvalidInputException("rocket_transform: series contains a NULL channel");
			}
			const auto channel = channel_entries[c_idx];
			const auto this_n = static_cast<int64_t>(channel.length);
			if (n < 0) {
				n = this_n;
			} else if (this_n != n) {
				// Dilations are drawn against one series length; a ragged series has no single
				// length to draw against, so the kernels would not be well defined.
				throw InvalidInputException(
				    "rocket_transform: channel %llu has %lld timepoints but channel 0 has "
				    "%lld; every channel must be the same length",
				    static_cast<unsigned long long>(c), static_cast<long long>(this_n),
				    static_cast<long long>(n));
			}
			for (idx_t j = 0; j < channel.length; j++) {
				const auto s_idx = sample_format.sel->get_index(channel.offset + j);
				if (!sample_format.validity.RowIsValid(s_idx)) {
					throw InvalidInputException(
					    "rocket_transform: series contains NULL values");
				}
				series.push_back(sample_data[s_idx]);
			}
		}

		if (n < duckdb_rocket::KERNEL_LENGTHS[2]) {
			throw InvalidInputException(
			    "rocket_transform: series length %lld is shorter than the longest kernel (%d)",
			    static_cast<long long>(n), duckdb_rocket::KERNEL_LENGTHS[2]);
		}

		const auto n_reference = ReferenceLength(args, count, row, n);

		const auto features = duckdb_rocket::TransformMultivariate(
		    series.data(), n_channels, n, n_reference,
		    static_cast<uint64_t>(seed_data[seed_idx]), kernels_per_group, first_kernel);

		for (size_t j = 0; j < features.size(); j++) {
			result_data[offset + j] = features[j];
		}
		result_entries[row] = list_entry_t(offset, features.size());
		offset += features.size();
	}

	if (count == 1) {
		result.SetVectorType(VectorType::CONSTANT_VECTOR);
	}
}

static void LoadInternal(ExtensionLoader &loader) {
	// Two overloads on one name. Overload resolution is by argument type, so a DOUBLE[] series
	// takes the univariate path and DOUBLE[][] the multivariate one -- and SPEC.md 7.1
	// guarantees a single-channel DOUBLE[][] produces exactly what the DOUBLE[] form does.
	const auto uni = LogicalType::LIST(LogicalType::DOUBLE);
	const auto multi = LogicalType::LIST(LogicalType::LIST(LogicalType::DOUBLE));
	const auto out = LogicalType::LIST(LogicalType::DOUBLE);
	const auto i64 = LogicalType::BIGINT;

	ScalarFunctionSet rocket_transform("rocket_transform");
	rocket_transform.AddFunction(
	    ScalarFunction({uni, i64, i64, i64}, out, RocketTransformFunction));
	rocket_transform.AddFunction(
	    ScalarFunction({multi, i64, i64, i64}, out, RocketTransformMultivariateFunction));
	// The five-argument forms take an explicit reference length, which is what makes one bank
	// shared across rows of differing length (SPEC.md 8).
	rocket_transform.AddFunction(
	    ScalarFunction({uni, i64, i64, i64, i64}, out, RocketTransformFunction));
	rocket_transform.AddFunction(
	    ScalarFunction({multi, i64, i64, i64, i64}, out, RocketTransformMultivariateFunction));
	loader.RegisterFunction(rocket_transform);
}

void RocketExtension::Load(ExtensionLoader &loader) {
	LoadInternal(loader);
}

std::string RocketExtension::Name() {
	return "rocket";
}

std::string RocketExtension::Version() const {
#ifdef EXT_VERSION_ROCKET
	return EXT_VERSION_ROCKET;
#else
	return "";
#endif
}

} // namespace duckdb

extern "C" {

DUCKDB_CPP_EXTENSION_ENTRY(rocket, loader) {
	duckdb::LoadInternal(loader);
}
}
