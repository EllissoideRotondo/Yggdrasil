#include "casadi_cxxwrap.hpp"

namespace casadi_cxxwrap
{

Sparsity sparsity_empty(const std::int64_t rows, const std::int64_t cols)
{
  return Sparsity(checked_nonnegative(rows, "rows"), checked_nonnegative(cols, "cols"));
}

Sparsity sparsity_dense(const std::int64_t rows, const std::int64_t cols)
{
  return Sparsity::dense(checked_nonnegative(rows, "rows"), checked_nonnegative(cols, "cols"));
}

Sparsity sparsity_ccs(
  const std::int64_t rows,
  const std::int64_t cols,
  jlcxx::ArrayRef<std::int64_t> colind,
  jlcxx::ArrayRef<std::int64_t> row,
  const bool order_rows)
{
  return Sparsity(
    checked_nonnegative(rows, "rows"),
    checked_nonnegative(cols, "cols"),
    to_casadi_int_vector(colind),
    to_casadi_int_vector(row),
    order_rows);
}

Sparsity sparsity_triplet(
  const std::int64_t rows,
  const std::int64_t cols,
  jlcxx::ArrayRef<std::int64_t> row,
  jlcxx::ArrayRef<std::int64_t> col)
{
  if(row.size() != col.size())
  {
    throw std::invalid_argument("row and col must have the same length");
  }
  return Sparsity::triplet(
    checked_nonnegative(rows, "rows"),
    checked_nonnegative(cols, "cols"),
    to_casadi_int_vector(row),
    to_casadi_int_vector(col));
}

Sparsity sparsity_rowcol(
  jlcxx::ArrayRef<std::int64_t> row,
  jlcxx::ArrayRef<std::int64_t> col,
  const std::int64_t rows,
  const std::int64_t cols)
{
  return Sparsity::rowcol(
    to_casadi_int_vector(row),
    to_casadi_int_vector(col),
    checked_nonnegative(rows, "rows"),
    checked_nonnegative(cols, "cols"));
}

Sparsity sparsity_nonzeros(
  const std::int64_t rows,
  const std::int64_t cols,
  jlcxx::ArrayRef<std::int64_t> nonzeros,
  const bool ind1)
{
  return Sparsity::nonzeros(
    checked_nonnegative(rows, "rows"),
    checked_nonnegative(cols, "cols"),
    to_casadi_int_vector(nonzeros),
    ind1);
}

Sparsity sparsity_compressed(jlcxx::ArrayRef<std::int64_t> values, const bool order_rows)
{
  return Sparsity::compressed(to_casadi_int_vector(values), order_rows);
}

Sparsity sparsity_permutation(jlcxx::ArrayRef<std::int64_t> permutation, const bool invert)
{
  return Sparsity::permutation(to_casadi_int_vector(permutation), invert);
}

std::int64_t sparsity_rows(const Sparsity& sp)
{
  return static_cast<std::int64_t>(sp.size1());
}

std::int64_t sparsity_cols(const Sparsity& sp)
{
  return static_cast<std::int64_t>(sp.size2());
}

std::int64_t sparsity_numel(const Sparsity& sp)
{
  return static_cast<std::int64_t>(sp.numel());
}

std::int64_t sparsity_nnz(const Sparsity& sp)
{
  return static_cast<std::int64_t>(sp.nnz());
}

std::vector<std::int64_t> sparsity_size(const Sparsity& sp)
{
  return {static_cast<std::int64_t>(sp.size1()), static_cast<std::int64_t>(sp.size2())};
}

std::int64_t sparsity_size_axis(const Sparsity& sp, const std::int64_t axis)
{
  if(axis != 0 && axis != 1)
  {
    throw std::out_of_range("axis must be 0 or 1");
  }
  return static_cast<std::int64_t>(sp.size(static_cast<casadi_int>(axis)));
}

std::vector<std::int64_t> sparsity_row(const Sparsity& sp)
{
  return from_casadi_int_vector(sp.get_row());
}

std::vector<std::int64_t> sparsity_col(const Sparsity& sp)
{
  return from_casadi_int_vector(sp.get_col());
}

std::vector<std::int64_t> sparsity_colind(const Sparsity& sp)
{
  return from_casadi_int_vector(sp.get_colind());
}

std::vector<std::int64_t> sparsity_compress(const Sparsity& sp, const bool canonical)
{
  return from_casadi_int_vector(sp.compress(canonical));
}

std::vector<std::int64_t> sparsity_permutation_vector(const Sparsity& sp, const bool invert)
{
  return from_casadi_int_vector(sp.permutation_vector(invert));
}

std::vector<std::int64_t> sparsity_find(const Sparsity& sp, const bool ind1)
{
  return from_casadi_int_vector(sp.find(ind1));
}

std::vector<std::int64_t> sparsity_lower(const Sparsity& sp)
{
  return from_casadi_int_vector(sp.get_lower());
}

std::vector<std::int64_t> sparsity_upper(const Sparsity& sp)
{
  return from_casadi_int_vector(sp.get_upper());
}

bool sparsity_has_nz(const Sparsity& sp, const std::int64_t row, const std::int64_t col)
{
  return sp.has_nz(checked_nonnegative(row, "row"), checked_nonnegative(col, "col"));
}

std::int64_t sparsity_get_nz(const Sparsity& sp, const std::int64_t row, const std::int64_t col)
{
  const casadi_int result = sp.get_nz(checked_nonnegative(row, "row"), checked_nonnegative(col, "col"));
  if(result < 0)
  {
    throw std::out_of_range("(row, col) is not in the sparsity pattern");
  }
  return static_cast<std::int64_t>(result);
}

std::vector<std::int64_t> sparsity_get_nz_vector(
  const Sparsity& sp,
  jlcxx::ArrayRef<std::int64_t> row,
  jlcxx::ArrayRef<std::int64_t> col)
{
  if(row.size() != col.size())
  {
    throw std::invalid_argument("row and col must have the same length");
  }
  return from_casadi_int_vector(sp.get_nz(to_casadi_int_vector(row), to_casadi_int_vector(col)));
}

std::vector<std::int64_t> sparsity_get_nz_indices(const Sparsity& sp, jlcxx::ArrayRef<std::int64_t> indices)
{
  std::vector<casadi_int> out = to_casadi_int_vector(indices);
  sp.get_nz(out);
  return from_casadi_int_vector(out);
}

std::int64_t sparsity_colind_at(const Sparsity& sp, const std::int64_t col)
{
  return static_cast<std::int64_t>(sp.colind(checked_nonnegative(col, "col")));
}

std::int64_t sparsity_row_at(const Sparsity& sp, const std::int64_t nonzero)
{
  return static_cast<std::int64_t>(sp.row(checked_nonnegative(nonzero, "nonzero")));
}

Sparsity sparsity_transpose(const Sparsity& sp)
{
  return sp.T();
}

SparsityMappingResult sparsity_transpose_with_mapping(const Sparsity& sp, const bool invert_mapping)
{
  std::vector<casadi_int> mapping;
  Sparsity value = sp.transpose(mapping, invert_mapping);
  return {value, from_casadi_int_vector(mapping)};
}

Sparsity sparsity_combine(const Sparsity& lhs, const Sparsity& rhs, const bool f0x_is_zero, const bool function0_is_zero)
{
  return lhs.combine(rhs, f0x_is_zero, function0_is_zero);
}

Sparsity sparsity_unite(const Sparsity& lhs, const Sparsity& rhs)
{
  return lhs.unite(rhs);
}

Sparsity sparsity_intersect(const Sparsity& lhs, const Sparsity& rhs)
{
  return lhs.intersect(rhs);
}

Sparsity sparsity_reshape(const Sparsity& sp, const std::int64_t rows, const std::int64_t cols)
{
  return Sparsity::reshape(sp, checked_nonnegative(rows, "rows"), checked_nonnegative(cols, "cols"));
}

Sparsity sparsity_sub(
  const Sparsity& sp,
  jlcxx::ArrayRef<std::int64_t> row,
  jlcxx::ArrayRef<std::int64_t> col,
  const bool ind1)
{
  std::vector<casadi_int> mapping;
  return sp.sub(to_casadi_int_vector(row), to_casadi_int_vector(col), mapping, ind1);
}

SparsityMappingResult sparsity_sub_with_mapping(
  const Sparsity& sp,
  jlcxx::ArrayRef<std::int64_t> row,
  jlcxx::ArrayRef<std::int64_t> col,
  const bool ind1)
{
  std::vector<casadi_int> mapping;
  Sparsity value = sp.sub(to_casadi_int_vector(row), to_casadi_int_vector(col), mapping, ind1);
  return {value, from_casadi_int_vector(mapping)};
}

SparsityMappingResult sparsity_sub_sparsity_with_mapping(
  const Sparsity& sp,
  jlcxx::ArrayRef<std::int64_t> row,
  const Sparsity& pattern,
  const bool ind1)
{
  std::vector<casadi_int> mapping;
  Sparsity value = sp.sub(to_casadi_int_vector(row), pattern, mapping, ind1);
  return {value, from_casadi_int_vector(mapping)};
}

Sparsity sparsity_pmult(
  const Sparsity& sp,
  jlcxx::ArrayRef<std::int64_t> permutation,
  const bool permute_rows,
  const bool permute_columns,
  const bool invert_permutation)
{
  return sp.pmult(to_casadi_int_vector(permutation), permute_rows, permute_columns, invert_permutation);
}

IntVectorPairResult sparsity_get_ccs(const Sparsity& sp)
{
  std::vector<casadi_int> first;
  std::vector<casadi_int> second;
  sp.get_ccs(first, second);
  return {from_casadi_int_vector(first), from_casadi_int_vector(second)};
}

IntVectorPairResult sparsity_get_crs(const Sparsity& sp)
{
  std::vector<casadi_int> first;
  std::vector<casadi_int> second;
  sp.get_crs(first, second);
  return {from_casadi_int_vector(first), from_casadi_int_vector(second)};
}

IntVectorPairResult sparsity_get_triplet(const Sparsity& sp)
{
  std::vector<casadi_int> first;
  std::vector<casadi_int> second;
  sp.get_triplet(first, second);
  return {from_casadi_int_vector(first), from_casadi_int_vector(second)};
}

SparsityMappingResult sparsity_get_diag(const Sparsity& sp)
{
  std::vector<casadi_int> mapping;
  Sparsity value = sp.get_diag(mapping);
  return {value, from_casadi_int_vector(mapping)};
}

Sparsity sparsity_enlarge(
  const Sparsity& sp,
  const std::int64_t rows,
  const std::int64_t cols,
  jlcxx::ArrayRef<std::int64_t> row,
  jlcxx::ArrayRef<std::int64_t> col,
  const bool ind1)
{
  Sparsity value = sp;
  value.enlarge(
    checked_nonnegative(rows, "rows"),
    checked_nonnegative(cols, "cols"),
    to_casadi_int_vector(row),
    to_casadi_int_vector(col),
    ind1);
  return value;
}

Sparsity sparsity_enlarge_rows(
  const Sparsity& sp,
  const std::int64_t rows,
  jlcxx::ArrayRef<std::int64_t> row,
  const bool ind1)
{
  Sparsity value = sp;
  value.enlargeRows(checked_nonnegative(rows, "rows"), to_casadi_int_vector(row), ind1);
  return value;
}

Sparsity sparsity_enlarge_columns(
  const Sparsity& sp,
  const std::int64_t cols,
  jlcxx::ArrayRef<std::int64_t> col,
  const bool ind1)
{
  Sparsity value = sp;
  value.enlargeColumns(checked_nonnegative(cols, "cols"), to_casadi_int_vector(col), ind1);
  return value;
}

SparsityMappingResult sparsity_make_dense(const Sparsity& sp)
{
  std::vector<casadi_int> mapping;
  Sparsity value = sp.makeDense(mapping);
  return {value, from_casadi_int_vector(mapping)};
}

SparsityMappingResult sparsity_erase_rows_cols(
  const Sparsity& sp,
  jlcxx::ArrayRef<std::int64_t> row,
  jlcxx::ArrayRef<std::int64_t> col,
  const bool ind1)
{
  Sparsity value = sp;
  std::vector<casadi_int> mapping = value.erase(to_casadi_int_vector(row), to_casadi_int_vector(col), ind1);
  return {value, from_casadi_int_vector(mapping)};
}

SparsityMappingResult sparsity_erase_elements(
  const Sparsity& sp,
  jlcxx::ArrayRef<std::int64_t> indices,
  const bool ind1)
{
  Sparsity value = sp;
  std::vector<casadi_int> mapping = value.erase(to_casadi_int_vector(indices), ind1);
  return {value, from_casadi_int_vector(mapping)};
}

Sparsity sparsity_append(const Sparsity& sp, const Sparsity& other)
{
  Sparsity value = sp;
  value.append(other);
  return value;
}

Sparsity sparsity_append_columns(const Sparsity& sp, const Sparsity& other)
{
  Sparsity value = sp;
  value.appendColumns(other);
  return value;
}

SparsityMappingResult sparsity_remove_duplicates(const Sparsity& sp, jlcxx::ArrayRef<std::int64_t> mapping)
{
  Sparsity value = sp;
  std::vector<casadi_int> out = to_casadi_int_vector(mapping);
  value.removeDuplicates(out);
  return {value, from_casadi_int_vector(out)};
}

SparsityLdlResult sparsity_ldl(const Sparsity& sp, const bool amd)
{
  std::vector<casadi_int> permutation;
  Sparsity lt = sp.ldl(permutation, amd);
  return {lt, from_casadi_int_vector(permutation)};
}

SparsityQrResult sparsity_qr_sparse(const Sparsity& sp, const bool amd)
{
  Sparsity v;
  Sparsity r;
  std::vector<casadi_int> prinv;
  std::vector<casadi_int> pc;
  sp.qr_sparse(v, r, prinv, pc, amd);
  return {v, r, from_casadi_int_vector(prinv), from_casadi_int_vector(pc)};
}

SparsitySccResult sparsity_scc(const Sparsity& sp)
{
  std::vector<casadi_int> index;
  std::vector<casadi_int> offset;
  const auto components = sp.scc(index, offset);
  return {static_cast<std::int64_t>(components), from_casadi_int_vector(index), from_casadi_int_vector(offset)};
}

SparsityBtfResult sparsity_btf(const Sparsity& sp)
{
  std::vector<casadi_int> rowperm;
  std::vector<casadi_int> colperm;
  std::vector<casadi_int> rowblock;
  std::vector<casadi_int> colblock;
  std::vector<casadi_int> coarse_rowblock;
  std::vector<casadi_int> coarse_colblock;
  const auto blocks = sp.btf(rowperm, colperm, rowblock, colblock, coarse_rowblock, coarse_colblock);
  return {
    static_cast<std::int64_t>(blocks),
    from_casadi_int_vector(rowperm),
    from_casadi_int_vector(colperm),
    from_casadi_int_vector(rowblock),
    from_casadi_int_vector(colblock),
    from_casadi_int_vector(coarse_rowblock),
    from_casadi_int_vector(coarse_colblock)};
}

std::string sparsity_export_code(const Sparsity& sp, const std::string& language, const GenericType& options)
{
  std::ostringstream out;
  sp.export_code(language, out, generic_as_dict(options, "sparsity export_code options"));
  return out.str();
}

DM dm_from_sparsity_values(const Sparsity& sp, jlcxx::ArrayRef<double> values)
{
  if(values.size() != static_cast<std::size_t>(sp.nnz()))
  {
    throw std::invalid_argument("number of values must match sparsity nnz");
  }
  return DM(sp, to_vector(values), true);
}

DM dm_triplet(
  const std::int64_t rows,
  const std::int64_t cols,
  jlcxx::ArrayRef<std::int64_t> row,
  jlcxx::ArrayRef<std::int64_t> col,
  jlcxx::ArrayRef<double> values)
{
  if(row.size() != col.size() || row.size() != values.size())
  {
    throw std::invalid_argument("row, col, and values must have the same length");
  }
  return DM::triplet(
    to_casadi_int_vector(row),
    to_casadi_int_vector(col),
    DM(to_vector(values)),
    checked_nonnegative(rows, "rows"),
    checked_nonnegative(cols, "cols"));
}

std::vector<double> dm_nonzeros(const DM& value)
{
  return value.get_nonzeros();
}

void register_sparsity_bindings(jlcxx::Module& mod)
{
  mod.method(raw_method("sparsity_mapping_value"), [](const SparsityMappingResult& result) { return result.value; });
  mod.method(raw_method("sparsity_mapping_mapping"), [](const SparsityMappingResult& result) { return result.mapping; });
  mod.method(raw_method("int_vector_pair_first"), [](const IntVectorPairResult& result) { return result.first; });
  mod.method(raw_method("int_vector_pair_second"), [](const IntVectorPairResult& result) { return result.second; });
  mod.method(raw_method("sparsity_ldl_lt"), [](const SparsityLdlResult& result) { return result.lt; });
  mod.method(raw_method("sparsity_ldl_permutation"), [](const SparsityLdlResult& result) { return result.permutation; });
  mod.method(raw_method("sparsity_qr_v"), [](const SparsityQrResult& result) { return result.v; });
  mod.method(raw_method("sparsity_qr_r"), [](const SparsityQrResult& result) { return result.r; });
  mod.method(raw_method("sparsity_qr_prinv"), [](const SparsityQrResult& result) { return result.prinv; });
  mod.method(raw_method("sparsity_qr_pc"), [](const SparsityQrResult& result) { return result.pc; });
  mod.method(raw_method("sparsity_scc_components"), [](const SparsitySccResult& result) { return result.components; });
  mod.method(raw_method("sparsity_scc_index"), [](const SparsitySccResult& result) { return result.index; });
  mod.method(raw_method("sparsity_scc_offset"), [](const SparsitySccResult& result) { return result.offset; });
  mod.method(raw_method("sparsity_btf_blocks"), [](const SparsityBtfResult& result) { return result.blocks; });
  mod.method(raw_method("sparsity_btf_rowperm"), [](const SparsityBtfResult& result) { return result.rowperm; });
  mod.method(raw_method("sparsity_btf_colperm"), [](const SparsityBtfResult& result) { return result.colperm; });
  mod.method(raw_method("sparsity_btf_rowblock"), [](const SparsityBtfResult& result) { return result.rowblock; });
  mod.method(raw_method("sparsity_btf_colblock"), [](const SparsityBtfResult& result) { return result.colblock; });
  mod.method(raw_method("sparsity_btf_coarse_rowblock"), [](const SparsityBtfResult& result) { return result.coarse_rowblock; });
  mod.method(raw_method("sparsity_btf_coarse_colblock"), [](const SparsityBtfResult& result) { return result.coarse_colblock; });

  mod.method(raw_method("sparsity_empty"), &sparsity_empty);
  mod.method(raw_method("sparsity_scalar"), [](const bool dense_scalar) { return Sparsity::scalar(dense_scalar); });
  mod.method(raw_method("sparsity_dense"), &sparsity_dense);
  mod.method(raw_method("sparsity_ccs"), &sparsity_ccs);
  mod.method(raw_method("sparsity_triplet"), &sparsity_triplet);
  mod.method(raw_method("sparsity_rowcol"), &sparsity_rowcol);
  mod.method(raw_method("sparsity_nonzeros"), &sparsity_nonzeros);
  mod.method(raw_method("sparsity_compressed"), &sparsity_compressed);
  mod.method(raw_method("sparsity_unit"), [](const std::int64_t n, const std::int64_t el) {
    return Sparsity::unit(checked_nonnegative(n, "n"), checked_nonnegative(el, "el"));
  });
  mod.method(raw_method("sparsity_upper"), [](const std::int64_t n) { return Sparsity::upper(checked_nonnegative(n, "n")); });
  mod.method(raw_method("sparsity_lower"), [](const std::int64_t n) { return Sparsity::lower(checked_nonnegative(n, "n")); });
  mod.method(raw_method("sparsity_diag"), [](const std::int64_t rows, const std::int64_t cols) {
    return Sparsity::diag(checked_nonnegative(rows, "rows"), checked_nonnegative(cols, "cols"));
  });
  mod.method(raw_method("sparsity_band"), [](const std::int64_t n, const std::int64_t p) {
    return Sparsity::band(checked_nonnegative(n, "n"), checked_casadi_int(p, "p"));
  });
  mod.method(raw_method("sparsity_banded"), [](const std::int64_t n, const std::int64_t p) {
    return Sparsity::banded(checked_nonnegative(n, "n"), checked_nonnegative(p, "p"));
  });
  mod.method(raw_method("sparsity_permutation"), &sparsity_permutation);
  mod.method(raw_method("sparsity_string"), &to_string<Sparsity>);
  mod.method(raw_method("sparsity_rows"), &sparsity_rows);
  mod.method(raw_method("sparsity_cols"), &sparsity_cols);
  mod.method(raw_method("sparsity_size"), &sparsity_size);
  mod.method(raw_method("sparsity_size_axis"), &sparsity_size_axis);
  mod.method(raw_method("sparsity_numel"), &sparsity_numel);
  mod.method(raw_method("sparsity_nnz"), &sparsity_nnz);
  mod.method(raw_method("sparsity_density"), [](const Sparsity& sp) { return sp.density(); });
  mod.method(raw_method("sparsity_nnz_upper"), [](const Sparsity& sp, const bool strictly) {
    return static_cast<std::int64_t>(sp.nnz_upper(strictly));
  });
  mod.method(raw_method("sparsity_nnz_lower"), [](const Sparsity& sp, const bool strictly) {
    return static_cast<std::int64_t>(sp.nnz_lower(strictly));
  });
  mod.method(raw_method("sparsity_nnz_diag"), [](const Sparsity& sp) { return static_cast<std::int64_t>(sp.nnz_diag()); });
  mod.method(raw_method("sparsity_bw_upper"), [](const Sparsity& sp) { return static_cast<std::int64_t>(sp.bw_upper()); });
  mod.method(raw_method("sparsity_bw_lower"), [](const Sparsity& sp) { return static_cast<std::int64_t>(sp.bw_lower()); });
  mod.method(raw_method("sparsity_dim"), [](const Sparsity& sp, const bool with_nz) { return sp.dim(with_nz); });
  mod.method(raw_method("sparsity_postfix_dim"), [](const Sparsity& sp) { return sp.postfix_dim(); });
  mod.method(raw_method("sparsity_repr_el"), [](const Sparsity& sp, const std::int64_t nonzero) {
    return sp.repr_el(checked_nonnegative(nonzero, "nonzero"));
  });
  mod.method(raw_method("sparsity_is_empty"), [](const Sparsity& sp) { return sp.is_empty(); });
  mod.method(raw_method("sparsity_is_empty"), [](const Sparsity& sp, const bool both) { return sp.is_empty(both); });
  mod.method(raw_method("sparsity_is_scalar"), [](const Sparsity& sp, const bool scalar_and_dense) {
    return sp.is_scalar(scalar_and_dense);
  });
  mod.method(raw_method("sparsity_is_dense"), [](const Sparsity& sp) { return sp.is_dense(); });
  mod.method(raw_method("sparsity_is_row"), [](const Sparsity& sp) { return sp.is_row(); });
  mod.method(raw_method("sparsity_is_column"), [](const Sparsity& sp) { return sp.is_column(); });
  mod.method(raw_method("sparsity_is_vector"), [](const Sparsity& sp) { return sp.is_vector(); });
  mod.method(raw_method("sparsity_is_diag"), [](const Sparsity& sp) { return sp.is_diag(); });
  mod.method(raw_method("sparsity_is_square"), [](const Sparsity& sp) { return sp.is_square(); });
  mod.method(raw_method("sparsity_is_symmetric"), [](const Sparsity& sp) { return sp.is_symmetric(); });
  mod.method(raw_method("sparsity_is_triu"), [](const Sparsity& sp, const bool strictly) { return sp.is_triu(strictly); });
  mod.method(raw_method("sparsity_is_tril"), [](const Sparsity& sp, const bool strictly) { return sp.is_tril(strictly); });
  mod.method(raw_method("sparsity_is_singular"), [](const Sparsity& sp) { return sp.is_singular(); });
  mod.method(raw_method("sparsity_is_permutation"), [](const Sparsity& sp) { return sp.is_permutation(); });
  mod.method(raw_method("sparsity_is_selection"), [](const Sparsity& sp, const bool allow_empty) {
    return sp.is_selection(allow_empty);
  });
  mod.method(raw_method("sparsity_is_orthonormal"), [](const Sparsity& sp, const bool allow_empty) {
    return sp.is_orthonormal(allow_empty);
  });
  mod.method(raw_method("sparsity_is_orthonormal_rows"), [](const Sparsity& sp, const bool allow_empty) {
    return sp.is_orthonormal_rows(allow_empty);
  });
  mod.method(raw_method("sparsity_is_orthonormal_columns"), [](const Sparsity& sp, const bool allow_empty) {
    return sp.is_orthonormal_columns(allow_empty);
  });
  mod.method(raw_method("sparsity_rows_sequential"), [](const Sparsity& sp, const bool strictly) {
    return sp.rowsSequential(strictly);
  });
  mod.method(raw_method("sparsity_row"), &sparsity_row);
  mod.method(raw_method("sparsity_col"), &sparsity_col);
  mod.method(raw_method("sparsity_colind"), &sparsity_colind);
  mod.method(raw_method("sparsity_colind_at"), &sparsity_colind_at);
  mod.method(raw_method("sparsity_row_at"), &sparsity_row_at);
  mod.method(raw_method("sparsity_compress"), &sparsity_compress);
  mod.method(raw_method("sparsity_permutation_vector"), &sparsity_permutation_vector);
  mod.method(raw_method("sparsity_find"), &sparsity_find);
  mod.method(raw_method("sparsity_lower_entries"), &sparsity_lower);
  mod.method(raw_method("sparsity_upper_entries"), &sparsity_upper);
  mod.method(raw_method("sparsity_has_nz"), &sparsity_has_nz);
  mod.method(raw_method("sparsity_get_nz"), &sparsity_get_nz);
  mod.method(raw_method("sparsity_get_nz_vector"), &sparsity_get_nz_vector);
  mod.method(raw_method("sparsity_get_nz_indices"), &sparsity_get_nz_indices);
  mod.method(raw_method("sparsity_get_ccs"), &sparsity_get_ccs);
  mod.method(raw_method("sparsity_get_crs"), &sparsity_get_crs);
  mod.method(raw_method("sparsity_get_triplet"), &sparsity_get_triplet);
  mod.method(raw_method("sparsity_get_diag"), &sparsity_get_diag);
  mod.method(raw_method("sparsity_eq"), [](const Sparsity& lhs, const Sparsity& rhs) { return lhs == rhs; });
  mod.method(raw_method("sparsity_is_equal"), [](const Sparsity& lhs, const Sparsity& rhs) { return lhs.is_equal(rhs); });
  mod.method(raw_method("sparsity_is_stacked"), [](const Sparsity& lhs, const Sparsity& rhs, const std::int64_t n) {
    return lhs.is_stacked(rhs, checked_nonnegative(n, "n"));
  });
  mod.method(raw_method("sparsity_is_transpose"), [](const Sparsity& lhs, const Sparsity& rhs) {
    return lhs.is_transpose(rhs);
  });
  mod.method(raw_method("sparsity_is_reshape"), [](const Sparsity& lhs, const Sparsity& rhs) {
    return lhs.is_reshape(rhs);
  });
  mod.method(raw_method("sparsity_is_subset"), [](const Sparsity& lhs, const Sparsity& rhs) {
    return lhs.is_subset(rhs);
  });
  mod.method(raw_method("sparsity_transpose"), &sparsity_transpose);
  mod.method(raw_method("sparsity_transpose_with_mapping"), &sparsity_transpose_with_mapping);
  mod.method(raw_method("sparsity_combine"), &sparsity_combine);
  mod.method(raw_method("sparsity_unite"), &sparsity_unite);
  mod.method(raw_method("sparsity_intersect"), &sparsity_intersect);
  mod.method(raw_method("sparsity_pattern_inverse"), [](const Sparsity& sp) { return sp.pattern_inverse(); });
  mod.method(raw_method("sparsity_sparsity_cast_mod"), [](const Sparsity& sp, const Sparsity& x, const Sparsity& y) {
    return sp.sparsity_cast_mod(x, y);
  });
  mod.method(raw_method("sparsity_sub"), &sparsity_sub);
  mod.method(raw_method("sparsity_sub_with_mapping"), &sparsity_sub_with_mapping);
  mod.method(raw_method("sparsity_sub_sparsity_with_mapping"), &sparsity_sub_sparsity_with_mapping);
  mod.method(raw_method("sparsity_horzcat"), [](jlcxx::ArrayRef<Sparsity> values) { return horzcat(to_vector(values)); });
  mod.method(raw_method("sparsity_vertcat"), [](jlcxx::ArrayRef<Sparsity> values) { return vertcat(to_vector(values)); });
  mod.method(raw_method("sparsity_diagcat"), [](jlcxx::ArrayRef<Sparsity> values) { return diagcat(to_vector(values)); });
  mod.method(raw_method("sparsity_mtimes"), [](const Sparsity& lhs, const Sparsity& rhs) { return Sparsity::mtimes(lhs, rhs); });
  mod.method(raw_method("sparsity_mtimes_many"), [](jlcxx::ArrayRef<Sparsity> values) { return mtimes(to_vector(values)); });
  mod.method(raw_method("sparsity_reshape"), &sparsity_reshape);
  mod.method(raw_method("sparsity_sparsity_cast"), [](const Sparsity& sp, const Sparsity& target) {
    return Sparsity::sparsity_cast(sp, target);
  });
  mod.method(raw_method("sparsity_kron"), [](const Sparsity& lhs, const Sparsity& rhs) { return Sparsity::kron(lhs, rhs); });
  mod.method(raw_method("sparsity_triu"), [](const Sparsity& sp, const bool include_diagonal) {
    return Sparsity::triu(sp, include_diagonal);
  });
  mod.method(raw_method("sparsity_tril"), [](const Sparsity& sp, const bool include_diagonal) {
    return Sparsity::tril(sp, include_diagonal);
  });
  mod.method(raw_method("sparsity_sum1"), [](const Sparsity& sp) { return Sparsity::sum1(sp); });
  mod.method(raw_method("sparsity_sum2"), [](const Sparsity& sp) { return Sparsity::sum2(sp); });
  mod.method(raw_method("sparsity_pmult"), &sparsity_pmult);
  mod.method(raw_method("sparsity_enlarge"), &sparsity_enlarge);
  mod.method(raw_method("sparsity_enlarge_rows"), &sparsity_enlarge_rows);
  mod.method(raw_method("sparsity_enlarge_columns"), &sparsity_enlarge_columns);
  mod.method(raw_method("sparsity_make_dense"), &sparsity_make_dense);
  mod.method(raw_method("sparsity_erase_rows_cols"), &sparsity_erase_rows_cols);
  mod.method(raw_method("sparsity_erase_elements"), &sparsity_erase_elements);
  mod.method(raw_method("sparsity_append"), &sparsity_append);
  mod.method(raw_method("sparsity_append_columns"), &sparsity_append_columns);
  mod.method(raw_method("sparsity_remove_duplicates"), &sparsity_remove_duplicates);
  mod.method(raw_method("sparsity_etree"), [](const Sparsity& sp, const bool ata) {
    return from_casadi_int_vector(sp.etree(ata));
  });
  mod.method(raw_method("sparsity_sprank"), [](const Sparsity& sp) {
    return static_cast<std::int64_t>(Sparsity::sprank(sp));
  });
  mod.method(raw_method("sparsity_norm_0_mul"), [](const Sparsity& lhs, const Sparsity& rhs) {
    return static_cast<std::int64_t>(Sparsity::norm_0_mul(lhs, rhs));
  });
  mod.method(raw_method("sparsity_ldl"), &sparsity_ldl);
  mod.method(raw_method("sparsity_qr_sparse"), &sparsity_qr_sparse);
  mod.method(raw_method("sparsity_scc"), &sparsity_scc);
  mod.method(raw_method("sparsity_btf"), &sparsity_btf);
  mod.method(raw_method("sparsity_amd"), [](const Sparsity& sp) { return from_casadi_int_vector(sp.amd()); });
  mod.method(raw_method("sparsity_largest_first"), [](const Sparsity& sp) {
    return from_casadi_int_vector(sp.largest_first());
  });
  mod.method(raw_method("sparsity_uni_coloring"), [](const Sparsity& sp, const Sparsity& at, const std::int64_t cutoff) {
    return sp.uni_coloring(at, checked_casadi_int(cutoff, "cutoff"));
  });
  mod.method(raw_method("sparsity_star_coloring"), [](const Sparsity& sp, const std::int64_t ordering, const std::int64_t cutoff) {
    return sp.star_coloring(checked_casadi_int(ordering, "ordering"), checked_casadi_int(cutoff, "cutoff"));
  });
  mod.method(raw_method("sparsity_star_coloring2"), [](const Sparsity& sp, const std::int64_t ordering, const std::int64_t cutoff) {
    return sp.star_coloring2(checked_casadi_int(ordering, "ordering"), checked_casadi_int(cutoff, "cutoff"));
  });
  mod.method(raw_method("sparsity_kkt"), [](const Sparsity& h, const Sparsity& j, const bool with_x_diag, const bool with_lam_g_diag) {
    return Sparsity::kkt(h, j, with_x_diag, with_lam_g_diag);
  });
  mod.method(raw_method("sparsity_info"), [](const Sparsity& sp) { return GenericType(sp.info()); });
  mod.method(raw_method("sparsity_serialize"), [](const Sparsity& sp) { return sp.serialize(); });
  mod.method(raw_method("sparsity_deserialize"), [](const std::string& value) { return Sparsity::deserialize(value); });
  mod.method(raw_method("sparsity_to_file"), [](const Sparsity& sp, const std::string& filename, const std::string& format) {
    sp.to_file(filename, format);
  });
  mod.method(raw_method("sparsity_from_file"), [](const std::string& filename, const std::string& format_hint) {
    return Sparsity::from_file(filename, format_hint);
  });
  mod.method(raw_method("sparsity_export_code"), &sparsity_export_code);

  mod.method(raw_method("dm_from_sparsity_values"), &dm_from_sparsity_values);
  mod.method(raw_method("dm_triplet"), &dm_triplet);
  mod.method(raw_method("dm_nonzeros"), &dm_nonzeros);
}

} // namespace casadi_cxxwrap
