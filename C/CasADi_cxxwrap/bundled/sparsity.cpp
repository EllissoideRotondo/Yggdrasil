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

bool sparsity_has_nz(const Sparsity& sp, const std::int64_t row, const std::int64_t col)
{
  return sp.has_nz(checked_index(row, "row"), checked_index(col, "col"));
}

std::int64_t sparsity_get_nz(const Sparsity& sp, const std::int64_t row, const std::int64_t col)
{
  return static_cast<std::int64_t>(sp.get_nz(checked_index(row, "row"), checked_index(col, "col")));
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
  mod.method(raw_method("sparsity_empty"), &sparsity_empty);
  mod.method(raw_method("sparsity_dense"), &sparsity_dense);
  mod.method(raw_method("sparsity_triplet"), &sparsity_triplet);
  mod.method(raw_method("sparsity_string"), &to_string<Sparsity>);
  mod.method(raw_method("sparsity_rows"), &sparsity_rows);
  mod.method(raw_method("sparsity_cols"), &sparsity_cols);
  mod.method(raw_method("sparsity_numel"), &sparsity_numel);
  mod.method(raw_method("sparsity_nnz"), &sparsity_nnz);
  mod.method(raw_method("sparsity_is_empty"), [](const Sparsity& sp) { return sp.is_empty(); });
  mod.method(raw_method("sparsity_is_dense"), [](const Sparsity& sp) { return sp.is_dense(); });
  mod.method(raw_method("sparsity_row"), &sparsity_row);
  mod.method(raw_method("sparsity_col"), &sparsity_col);
  mod.method(raw_method("sparsity_colind"), &sparsity_colind);
  mod.method(raw_method("sparsity_has_nz"), &sparsity_has_nz);
  mod.method(raw_method("sparsity_get_nz"), &sparsity_get_nz);
  mod.method(raw_method("sparsity_eq"), [](const Sparsity& lhs, const Sparsity& rhs) { return lhs == rhs; });

  mod.method(raw_method("dm_from_sparsity_values"), &dm_from_sparsity_values);
  mod.method(raw_method("dm_triplet"), &dm_triplet);
  mod.method(raw_method("dm_nonzeros"), &dm_nonzeros);
}

} // namespace casadi_cxxwrap
