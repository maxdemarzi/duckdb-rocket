#define DUCKDB_EXTENSION_MAIN

#include "rocket_extension.hpp"
#include "rocket.hpp"

#include "duckdb.hpp"
#include "duckdb/common/exception.hpp"
#include "duckdb/common/vector_operations/vector_operations.hpp"
#include "duckdb/function/scalar_function.hpp"
#include "duckdb/main/extension/extension_loader.hpp"

namespace duckdb {

// rocket_transform(series, kernels_per_group, seed, first_kernel) -> DOUBLE[]
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

		const auto features = duckdb_rocket::Transform(
		    series.data(), n, static_cast<uint64_t>(seed_data[seed_idx]), kernels_per_group,
		    first_kernel);

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
	ScalarFunction rocket_transform(
	    "rocket_transform",
	    {LogicalType::LIST(LogicalType::DOUBLE), LogicalType::BIGINT, LogicalType::BIGINT,
	     LogicalType::BIGINT},
	    LogicalType::LIST(LogicalType::DOUBLE), RocketTransformFunction);
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
