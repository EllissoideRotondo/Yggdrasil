#include "casadi_cxxwrap.hpp"

#include <type_traits>

namespace casadi_cxxwrap
{

template<typename T>
T scalar(const double value)
{
  return T(value);
}

template<typename T>
T zeros(const std::int64_t rows, const std::int64_t cols)
{
  return T::zeros(checked_nonnegative(rows, "rows"), checked_nonnegative(cols, "cols"));
}

template<typename T>
T ones(const std::int64_t rows, const std::int64_t cols)
{
  return T::ones(checked_nonnegative(rows, "rows"), checked_nonnegative(cols, "cols"));
}

template<typename T>
T eye(const std::int64_t rows)
{
  return T::eye(checked_nonnegative(rows, "rows"));
}

template<typename T>
T dense_from_vector(jlcxx::ArrayRef<double> values, const std::int64_t rows, const std::int64_t cols)
{
  const auto r = checked_nonnegative(rows, "rows");
  const auto c = checked_nonnegative(cols, "cols");
  if(static_cast<std::int64_t>(values.size()) != rows * cols)
  {
    throw std::invalid_argument("number of values must match rows * cols");
  }
  return reshape(T(to_vector(values)), r, c);
}

template<typename T>
T element_at(const T& value, const std::int64_t row, const std::int64_t col)
{
  return T(value(checked_index(row, "row"), checked_index(col, "col")));
}

template<typename T>
std::int64_t rows(const T& value)
{
  return static_cast<std::int64_t>(value.size1());
}

template<typename T>
std::int64_t cols(const T& value)
{
  return static_cast<std::int64_t>(value.size2());
}

template<typename T>
std::int64_t nonzeros(const T& value)
{
  return static_cast<std::int64_t>(value.nnz());
}

template<typename T>
std::int64_t numel(const T& value)
{
  return static_cast<std::int64_t>(value.numel());
}

template<typename T>
std::vector<std::int64_t> matrix_size(const T& value)
{
  return {
    static_cast<std::int64_t>(value.size1()),
    static_cast<std::int64_t>(value.size2())};
}

template<typename T>
T hessian_value(const T& expression, const T& variable)
{
  T gradient;
  return hessian(expression, variable, gradient);
}

template<typename T>
T tangent_value(const T& expression, const T& variable, const GenericType& options)
{
  return tangent(expression, variable, generic_as_dict(options, "tangent options"));
}

template<typename T>
T jtimes_value(
  const T& expression,
  const T& variable,
  const T& seed,
  const bool transpose,
  const GenericType& options)
{
  return jtimes(expression, variable, seed, transpose, generic_as_dict(options, "jtimes options"));
}

template<typename T>
T hessian_gradient(const T& expression, const T& variable)
{
  T gradient;
  hessian(expression, variable, gradient);
  return gradient;
}

template<typename T>
std::vector<T> symbolic_variables(const T& expression)
{
  return symvar(expression);
}

template<typename T>
std::vector<bool> symbolic_dependencies(
  const T& expression,
  const T& variable,
  const std::int64_t order,
  const bool transpose)
{
  return which_depends(expression, variable, checked_nonnegative(order, "order"), transpose);
}

template<typename T>
bool symbolic_contains(jlcxx::ArrayRef<T> values, const T& needle)
{
  return contains(to_vector(values), needle);
}

template<typename T>
bool symbolic_contains_all(jlcxx::ArrayRef<T> values, jlcxx::ArrayRef<T> needles)
{
  return contains_all(to_vector(values), to_vector(needles));
}

template<typename T>
bool symbolic_contains_any(jlcxx::ArrayRef<T> values, jlcxx::ArrayRef<T> needles)
{
  return contains_any(to_vector(values), to_vector(needles));
}

template<typename T>
std::vector<T> symbolic_substitute_vector(
  jlcxx::ArrayRef<T> expressions,
  jlcxx::ArrayRef<T> variables,
  jlcxx::ArrayRef<T> replacements)
{
  return substitute(to_vector(expressions), to_vector(variables), to_vector(replacements));
}

template<typename T>
std::vector<T> symbolic_cse_vector(jlcxx::ArrayRef<T> expressions)
{
  return cse(to_vector(expressions));
}

template<typename T>
std::int64_t symbolic_node_count(const T& expression)
{
  return static_cast<std::int64_t>(n_nodes(expression));
}

template<typename T>
std::vector<T> matrix_horzsplit_offsets(const T& value, jlcxx::ArrayRef<std::int64_t> offsets)
{
  return horzsplit(value, to_casadi_int_vector(offsets));
}

template<typename T>
std::vector<T> matrix_vertsplit_offsets(const T& value, jlcxx::ArrayRef<std::int64_t> offsets)
{
  return vertsplit(value, to_casadi_int_vector(offsets));
}

template<typename T>
std::vector<T> matrix_diagsplit_offsets(
  const T& value,
  jlcxx::ArrayRef<std::int64_t> row_offsets,
  jlcxx::ArrayRef<std::int64_t> col_offsets)
{
  return diagsplit(value, to_casadi_int_vector(row_offsets), to_casadi_int_vector(col_offsets));
}

template<typename T>
T matrix_conditional(
  const T& index,
  jlcxx::ArrayRef<T> values,
  const T& default_value,
  const bool short_circuit)
{
  return conditional(index, to_vector(values), default_value, short_circuit);
}

MX mx_graph_substitute(const MX& expression, jlcxx::ArrayRef<MX> variables, jlcxx::ArrayRef<MX> replacements)
{
  return graph_substitute(expression, to_vector(variables), to_vector(replacements));
}

std::vector<MX> mx_graph_substitute_vector(
  jlcxx::ArrayRef<MX> expressions,
  jlcxx::ArrayRef<MX> variables,
  jlcxx::ArrayRef<MX> replacements)
{
  return graph_substitute(to_vector(expressions), to_vector(variables), to_vector(replacements));
}

std::vector<double> dm_full(const DM& value)
{
  std::vector<double> out;
  out.reserve(static_cast<std::size_t>(value.numel()));
  for(int col = 0; col != value.size2(); ++col)
  {
    for(int row = 0; row != value.size1(); ++row)
    {
      out.push_back(static_cast<double>(value(row, col)));
    }
  }
  return out;
}

template<typename T>
void register_matrix_serialization(jlcxx::Module& mod, const std::string& prefix)
{
  mod.method(raw_method(prefix, "serialize"), [](const T& value) { return value.serialize(); });
  mod.method(raw_method(prefix, "deserialize"), [](const std::string& value) { return T::deserialize(value); });
  mod.method(raw_method(prefix, "export_code"), [](const T& value, const std::string& language, const GenericType& options) {
    std::ostringstream out;
    value.export_code(language, out, generic_as_dict(options, "matrix export_code options"));
    return out.str();
  });
}

template<typename T>
void register_matrix_common(jlcxx::Module& mod, const std::string& prefix)
{
  mod.method(raw_method(prefix, "scalar"), &scalar<T>);
  mod.method(raw_method(prefix, "zeros"), &zeros<T>);
  mod.method(raw_method(prefix, "ones"), &ones<T>);
  mod.method(raw_method(prefix, "eye"), &eye<T>);
  mod.method(raw_method(prefix, "string"), &to_string<T>);
  mod.method(raw_method(prefix, "rows"), &rows<T>);
  mod.method(raw_method(prefix, "cols"), &cols<T>);
  mod.method(raw_method(prefix, "size"), &matrix_size<T>);
  mod.method(raw_method(prefix, "nnz"), &nonzeros<T>);
  mod.method(raw_method(prefix, "numel"), &numel<T>);
  mod.method(raw_method(prefix, "is_empty"), [](const T& value) { return value.is_empty(); });
  mod.method(raw_method(prefix, "is_scalar"), [](const T& value) { return value.is_scalar(); });
  mod.method(raw_method(prefix, "is_dense"), [](const T& value) { return value.is_dense(); });
  mod.method(raw_method(prefix, "is_vector"), [](const T& value) { return value.is_vector(); });
  mod.method(raw_method(prefix, "is_row"), [](const T& value) { return value.is_row(); });
  mod.method(raw_method(prefix, "is_column"), [](const T& value) { return value.is_column(); });
  mod.method(raw_method(prefix, "is_tril"), [](const T& value) { return value.is_tril(); });
  mod.method(raw_method(prefix, "is_triu"), [](const T& value) { return value.is_triu(); });
  mod.method(raw_method(prefix, "get"), &element_at<T>);
  mod.method(raw_method(prefix, "sparsity"), [](const T& value) { return value.get_sparsity(); });
  mod.method(raw_method(prefix, "is_regular"), [](const T& value) { return value.is_regular(); });
  mod.method(raw_method(prefix, "is_symbolic"), [](const T& value) { return value.is_symbolic(); });
  mod.method(raw_method(prefix, "is_valid_input"), [](const T& value) { return value.is_valid_input(); });
  mod.method(raw_method(prefix, "is_constant"), [](const T& value) { return value.is_constant(); });
  mod.method(raw_method(prefix, "is_zero"), [](const T& value) { return value.is_zero(); });
  mod.method(raw_method(prefix, "is_one"), [](const T& value) { return value.is_one(); });
  mod.method(raw_method(prefix, "is_minus_one"), [](const T& value) { return value.is_minus_one(); });
  mod.method(raw_method(prefix, "is_eye"), [](const T& value) { return value.is_eye(); });
  mod.method(raw_method(prefix, "op"), [](const T& value) { return static_cast<std::int64_t>(value.op()); });
  mod.method(raw_method(prefix, "is_op"), [](const T& value, const std::int64_t op) {
    return value.is_op(checked_index(op, "op"));
  });
  mod.method(raw_method(prefix, "info"), [](const T& value) {
    return GenericType(value.info());
  });

  mod.method(raw_method(prefix, "neg"), [](const T& value) { return -value; });
  mod.method(raw_method(prefix, "add"), [](const T& lhs, const T& rhs) { return lhs + rhs; });
  mod.method(raw_method(prefix, "sub"), [](const T& lhs, const T& rhs) { return lhs - rhs; });
  mod.method(raw_method(prefix, "mul"), [](const T& lhs, const T& rhs) { return lhs * rhs; });
  mod.method(raw_method(prefix, "div"), [](const T& lhs, const T& rhs) { return lhs / rhs; });
  mod.method(raw_method(prefix, "pow"), [](const T& lhs, const T& rhs) { return pow(lhs, rhs); });
  mod.method(raw_method(prefix, "mtimes"), [](const T& lhs, const T& rhs) { return mtimes(lhs, rhs); });
  mod.method(raw_method(prefix, "lt"), [](const T& lhs, const T& rhs) { return lhs < rhs; });
  mod.method(raw_method(prefix, "le"), [](const T& lhs, const T& rhs) { return lhs <= rhs; });
  mod.method(raw_method(prefix, "gt"), [](const T& lhs, const T& rhs) { return lhs > rhs; });
  mod.method(raw_method(prefix, "ge"), [](const T& lhs, const T& rhs) { return lhs >= rhs; });
  mod.method(raw_method(prefix, "eq"), [](const T& lhs, const T& rhs) { return lhs == rhs; });
  mod.method(raw_method(prefix, "ne"), [](const T& lhs, const T& rhs) { return lhs != rhs; });
  mod.method(raw_method(prefix, "logic_and"), [](const T& lhs, const T& rhs) { return logic_and(lhs, rhs); });
  mod.method(raw_method(prefix, "logic_or"), [](const T& lhs, const T& rhs) { return logic_or(lhs, rhs); });
  mod.method(raw_method(prefix, "logic_not"), [](const T& value) { return logic_not(value); });
  mod.method(raw_method(prefix, "is_equal"), [](const T& lhs, const T& rhs, const std::int64_t depth) {
    return is_equal(lhs, rhs, checked_nonnegative(depth, "depth"));
  });

  mod.method(raw_method(prefix, "transpose"), [](const T& value) { return transpose(value); });
  mod.method(raw_method(prefix, "reshape"), [](const T& value, const std::int64_t r, const std::int64_t c) {
    return reshape(value, checked_nonnegative(r, "rows"), checked_nonnegative(c, "cols"));
  });
  mod.method(raw_method(prefix, "repmat"), [](const T& value, const std::int64_t r, const std::int64_t c) {
    return repmat(value, checked_nonnegative(r, "row_repetitions"), checked_nonnegative(c, "col_repetitions"));
  });
  mod.method(raw_method(prefix, "vec"), [](const T& value) { return vec(value); });
  mod.method(raw_method(prefix, "vertcat"), [](jlcxx::ArrayRef<T> values) { return vertcat(to_vector(values)); });
  mod.method(raw_method(prefix, "horzcat"), [](jlcxx::ArrayRef<T> values) { return horzcat(to_vector(values)); });
  mod.method(raw_method(prefix, "diagcat"), [](jlcxx::ArrayRef<T> values) { return diagcat(to_vector(values)); });
  mod.method(raw_method(prefix, "mtimes_many"), [](jlcxx::ArrayRef<T> values) { return mtimes(to_vector(values)); });
  mod.method(raw_method(prefix, "horzsplit"), [](const T& value, const std::int64_t increment) {
    return horzsplit(value, checked_nonnegative(increment, "increment"));
  });
  mod.method(raw_method(prefix, "horzsplit_offsets"), &matrix_horzsplit_offsets<T>);
  mod.method(raw_method(prefix, "vertsplit"), [](const T& value, const std::int64_t increment) {
    return vertsplit(value, checked_nonnegative(increment, "increment"));
  });
  mod.method(raw_method(prefix, "vertsplit_offsets"), &matrix_vertsplit_offsets<T>);
  mod.method(raw_method(prefix, "diagsplit"), [](const T& value, const std::int64_t increment) {
    return diagsplit(value, checked_nonnegative(increment, "increment"));
  });
  mod.method(raw_method(prefix, "diagsplit_offsets"), &matrix_diagsplit_offsets<T>);
  mod.method(raw_method(prefix, "reshape_sparsity"), [](const T& value, const Sparsity& sp) {
    return reshape(value, sp);
  });
  mod.method(raw_method(prefix, "sparsity_cast"), [](const T& value, const Sparsity& sp) {
    return sparsity_cast(value, sp);
  });
  mod.method(raw_method(prefix, "kron"), [](const T& lhs, const T& rhs) { return kron(lhs, rhs); });
  mod.method(raw_method(prefix, "mac"), [](const T& x, const T& y, const T& z) { return mac(x, y, z); });
  mod.method(raw_method(prefix, "project"), [](const T& value, const Sparsity& sp, const bool intersect) {
    return project(value, sp, intersect);
  });
  mod.method(raw_method(prefix, "densify"), [](const T& value) { return densify(value); });
  mod.method(raw_method(prefix, "densify_value"), [](const T& value, const T& fill_value) {
    return densify(value, fill_value);
  });
  mod.method(raw_method(prefix, "triu"), [](const T& value, const bool include_diagonal) {
    return triu(value, include_diagonal);
  });
  mod.method(raw_method(prefix, "tril"), [](const T& value, const bool include_diagonal) {
    return tril(value, include_diagonal);
  });

  mod.method(raw_method(prefix, "sin"), [](const T& value) { return sin(value); });
  mod.method(raw_method(prefix, "cos"), [](const T& value) { return cos(value); });
  mod.method(raw_method(prefix, "tan"), [](const T& value) { return tan(value); });
  mod.method(raw_method(prefix, "asin"), [](const T& value) { return asin(value); });
  mod.method(raw_method(prefix, "acos"), [](const T& value) { return acos(value); });
  mod.method(raw_method(prefix, "atan"), [](const T& value) { return atan(value); });
  mod.method(raw_method(prefix, "atan2"), [](const T& y, const T& x) { return atan2(y, x); });
  mod.method(raw_method(prefix, "sinh"), [](const T& value) { return sinh(value); });
  mod.method(raw_method(prefix, "cosh"), [](const T& value) { return cosh(value); });
  mod.method(raw_method(prefix, "tanh"), [](const T& value) { return tanh(value); });
  mod.method(raw_method(prefix, "asinh"), [](const T& value) { return asinh(value); });
  mod.method(raw_method(prefix, "acosh"), [](const T& value) { return acosh(value); });
  mod.method(raw_method(prefix, "atanh"), [](const T& value) { return atanh(value); });
  mod.method(raw_method(prefix, "exp"), [](const T& value) { return exp(value); });
  mod.method(raw_method(prefix, "expm1"), [](const T& value) { return expm1(value); });
  mod.method(raw_method(prefix, "log"), [](const T& value) { return log(value); });
  mod.method(raw_method(prefix, "log10"), [](const T& value) { return log10(value); });
  mod.method(raw_method(prefix, "log1p"), [](const T& value) { return log1p(value); });
  mod.method(raw_method(prefix, "sqrt"), [](const T& value) { return sqrt(value); });
  mod.method(raw_method(prefix, "floor"), [](const T& value) { return floor(value); });
  mod.method(raw_method(prefix, "ceil"), [](const T& value) { return ceil(value); });
  mod.method(raw_method(prefix, "fabs"), [](const T& value) { return fabs(value); });
  mod.method(raw_method(prefix, "erf"), [](const T& value) { return erf(value); });
  mod.method(raw_method(prefix, "erfinv"), [](const T& value) { return erfinv(value); });
  mod.method(raw_method(prefix, "sign"), [](const T& value) { return sign(value); });
  mod.method(raw_method(prefix, "sq"), [](const T& value) { return sq(value); });
  mod.method(raw_method(prefix, "fmod"), [](const T& lhs, const T& rhs) { return fmod(lhs, rhs); });
  mod.method(raw_method(prefix, "remainder"), [](const T& lhs, const T& rhs) { return remainder(lhs, rhs); });
  mod.method(raw_method(prefix, "fmin"), [](const T& lhs, const T& rhs) { return fmin(lhs, rhs); });
  mod.method(raw_method(prefix, "fmax"), [](const T& lhs, const T& rhs) { return fmax(lhs, rhs); });
  mod.method(raw_method(prefix, "copysign"), [](const T& lhs, const T& rhs) { return copysign(lhs, rhs); });
  mod.method(raw_method(prefix, "constpow"), [](const T& lhs, const T& rhs) { return constpow(lhs, rhs); });
  mod.method(raw_method(prefix, "hypot"), [](const T& lhs, const T& rhs) { return hypot(lhs, rhs); });
  mod.method(raw_method(prefix, "if_else_zero"), [](const T& condition, const T& if_true) {
    return if_else_zero(condition, if_true);
  });
  mod.method(raw_method(prefix, "if_else"), [](const T& condition, const T& if_true, const T& if_false, const bool short_circuit) {
    return if_else(condition, if_true, if_false, short_circuit);
  });
  mod.method(raw_method(prefix, "conditional"), &matrix_conditional<T>);
  mod.method(raw_method(prefix, "norm_1"), [](const T& value) { return norm_1(value); });
  mod.method(raw_method(prefix, "norm_2"), [](const T& value) { return norm_2(value); });
  mod.method(raw_method(prefix, "norm_fro"), [](const T& value) { return norm_fro(value); });
  mod.method(raw_method(prefix, "norm_inf"), [](const T& value) { return norm_inf(value); });
  mod.method(raw_method(prefix, "det"), [](const T& value) { return det(value); });
  mod.method(raw_method(prefix, "trace"), [](const T& value) { return trace(value); });
  mod.method(raw_method(prefix, "diag"), [](const T& value) { return diag(value); });
  mod.method(raw_method(prefix, "inv"), [](const T& value) { return inv(value); });
  mod.method(raw_method(prefix, "inv_options"), [](const T& value, const std::string& lsolver, const GenericType& options) {
    return inv(value, lsolver, generic_as_dict(options, "inv options"));
  });
  mod.method(raw_method(prefix, "pinv"), [](const T& value) { return pinv(value); });
  mod.method(raw_method(prefix, "pinv_options"), [](const T& value, const std::string& lsolver, const GenericType& options) {
    return pinv(value, lsolver, generic_as_dict(options, "pinv options"));
  });
  mod.method(raw_method(prefix, "solve"), [](const T& matrix, const T& rhs) { return solve(matrix, rhs); });
  mod.method(raw_method(prefix, "solve_options"), [](const T& matrix, const T& rhs, const std::string& lsolver, const GenericType& options) {
    return solve(matrix, rhs, lsolver, generic_as_dict(options, "solve options"));
  });
  mod.method(raw_method(prefix, "mldivide"), [](const T& matrix, const T& rhs) { return mldivide(matrix, rhs); });
  mod.method(raw_method(prefix, "mrdivide"), [](const T& matrix, const T& rhs) { return mrdivide(matrix, rhs); });
  mod.method(raw_method(prefix, "sum1"), [](const T& value) { return sum1(value); });
  mod.method(raw_method(prefix, "sum2"), [](const T& value) { return sum2(value); });
  mod.method(raw_method(prefix, "sumsqr"), [](const T& value) { return sumsqr(value); });
  mod.method(raw_method(prefix, "mmin"), [](const T& value) { return mmin(value); });
  mod.method(raw_method(prefix, "mmax"), [](const T& value) { return mmax(value); });
  mod.method(raw_method(prefix, "cumsum"), [](const T& value, const std::int64_t axis) {
    return cumsum(value, static_cast<casadi_int>(axis));
  });
  mod.method(raw_method(prefix, "diff"), [](const T& value, const std::int64_t n, const std::int64_t axis) {
    return diff(value, checked_nonnegative(n, "n"), static_cast<casadi_int>(axis));
  });
  mod.method(raw_method(prefix, "nullspace"), [](const T& value) { return nullspace(value); });
  mod.method(raw_method(prefix, "polyval"), [](const T& polynomial, const T& value) {
    return polyval(polynomial, value);
  });
  mod.method(raw_method(prefix, "unite"), [](const T& lhs, const T& rhs) { return unite(lhs, rhs); });
  mod.method(raw_method(prefix, "tril2symm"), [](const T& value) { return tril2symm(value); });
  mod.method(raw_method(prefix, "triu2symm"), [](const T& value) { return triu2symm(value); });
  mod.method(raw_method(prefix, "logsumexp"), [](const T& value) { return logsumexp(value); });
  mod.method(raw_method(prefix, "soc"), [](const T& x, const T& y) { return soc(x, y); });
  mod.method(raw_method(prefix, "cross"), [](const T& lhs, const T& rhs, const std::int64_t dim) {
    return cross(lhs, rhs, static_cast<casadi_int>(dim));
  });
  mod.method(raw_method(prefix, "skew"), [](const T& value) { return skew(value); });
  mod.method(raw_method(prefix, "inv_skew"), [](const T& value) { return inv_skew(value); });
  mod.method(raw_method(prefix, "bilin"), [](const T& matrix, const T& x, const T& y) {
    return bilin(matrix, x, y);
  });
  mod.method(raw_method(prefix, "rank1"), [](const T& matrix, const T& alpha, const T& x, const T& y) {
    return rank1(matrix, alpha, x, y);
  });
  mod.method(raw_method(prefix, "dot"), [](const T& lhs, const T& rhs) { return dot(lhs, rhs); });

  if constexpr(!std::is_same_v<T, MX>)
  {
    mod.method(raw_method(prefix, "has_nz"), [](const T& value, const std::int64_t row, const std::int64_t col) {
      return value.has_nz(checked_index(row, "row"), checked_index(col, "col"));
    });
    mod.method(raw_method(prefix, "is_integer"), [](const T& value) { return value.is_integer(); });
    mod.method(raw_method(prefix, "has_zeros"), [](const T& value) { return value.has_zeros(); });
    mod.method(raw_method(prefix, "sparsify"), [](const T& value, const double tolerance) {
      return sparsify(value, tolerance);
    });
    mod.method(raw_method(prefix, "norm_inf_mul"), [](const T& lhs, const T& rhs) {
      return norm_inf_mul(lhs, rhs);
    });
    mod.method(raw_method(prefix, "all"), [](const T& value) { return all(value); });
    mod.method(raw_method(prefix, "any"), [](const T& value) { return any(value); });
  }
}

template<typename T>
void register_calculus(jlcxx::Module& mod, const std::string& prefix)
{
  mod.method(raw_method(prefix, "jacobian"), [](const T& expression, const T& variable) {
    return jacobian(expression, variable);
  });
  mod.method(raw_method(prefix, "gradient"), [](const T& expression, const T& variable) {
    return gradient(expression, variable);
  });
  mod.method(raw_method(prefix, "hessian"), &hessian_value<T>);
  mod.method(raw_method(prefix, "hessian_gradient"), &hessian_gradient<T>);
  mod.method(raw_method(prefix, "tangent"), &tangent_value<T>);
  mod.method(raw_method(prefix, "jtimes"), &jtimes_value<T>);
}

template<typename T>
void register_symbolic_utilities(jlcxx::Module& mod, const std::string& prefix)
{
  mod.method(raw_method(prefix, "symvar"), &symbolic_variables<T>);
  mod.method(raw_method(prefix, "depends_on"), [](const T& expression, const T& variable) {
    return depends_on(expression, variable);
  });
  mod.method(raw_method(prefix, "contains"), &symbolic_contains<T>);
  mod.method(raw_method(prefix, "contains_all"), &symbolic_contains_all<T>);
  mod.method(raw_method(prefix, "contains_any"), &symbolic_contains_any<T>);
  mod.method(raw_method(prefix, "substitute"), [](const T& expression, const T& variable, const T& replacement) {
    return substitute(expression, variable, replacement);
  });
  mod.method(raw_method(prefix, "substitute_vector"), &symbolic_substitute_vector<T>);
  mod.method(raw_method(prefix, "which_depends"), &symbolic_dependencies<T>);
  mod.method(raw_method(prefix, "simplify"), [](const T& expression) {
    return simplify(expression);
  });
  mod.method(raw_method(prefix, "n_nodes"), &symbolic_node_count<T>);
  mod.method(raw_method(prefix, "cse"), [](const T& expression) {
    return cse(expression);
  });
  mod.method(raw_method(prefix, "cse_vector"), &symbolic_cse_vector<T>);
  mod.method(raw_method(prefix, "jacobian_sparsity"), [](const T& expression, const T& variable) {
    return jacobian_sparsity(expression, variable);
  });
}

void register_matrix_bindings(jlcxx::Module& mod)
{
  register_matrix_common<SX>(mod, "sx");
  register_matrix_common<DM>(mod, "dm");
  register_matrix_common<MX>(mod, "mx");
  register_calculus<SX>(mod, "sx");
  register_calculus<MX>(mod, "mx");
  register_symbolic_utilities<SX>(mod, "sx");
  register_symbolic_utilities<MX>(mod, "mx");
  register_matrix_serialization<SX>(mod, "sx");
  register_matrix_serialization<DM>(mod, "dm");

  mod.method(raw_method("mx_graph_substitute"), &mx_graph_substitute);
  mod.method(raw_method("mx_graph_substitute_vector"), &mx_graph_substitute_vector);

  mod.method(raw_method("sx_sym"), [](const std::string& name) { return SX::sym(name); });
  mod.method(raw_method("sx_sym"), [](const std::string& name, const std::int64_t rows, const std::int64_t cols) {
    return SX::sym(name, checked_nonnegative(rows, "rows"), checked_nonnegative(cols, "cols"));
  });
  mod.method(raw_method("sx_sym_sparsity"), [](const std::string& name, const Sparsity& sp) {
    return SX::sym(name, sp);
  });

  mod.method(raw_method("mx_sym"), [](const std::string& name) { return MX::sym(name); });
  mod.method(raw_method("mx_sym"), [](const std::string& name, const std::int64_t rows, const std::int64_t cols) {
    return MX::sym(name, checked_nonnegative(rows, "rows"), checked_nonnegative(cols, "cols"));
  });
  mod.method(raw_method("mx_sym_sparsity"), [](const std::string& name, const Sparsity& sp) {
    return MX::sym(name, sp);
  });

  mod.method(raw_method("dm_dense"), &dense_from_vector<DM>);
  mod.method(raw_method("dm_full"), &dm_full);
  mod.method(raw_method("dm_scalar_value"), [](const DM& value) { return value.scalar(); });
  mod.method(raw_method("dm_to_file"), [](const DM& value, const std::string& filename, const std::string& format) {
    value.to_file(filename, format);
  });
  mod.method(raw_method("dm_from_file"), [](const std::string& filename, const std::string& format_hint) {
    return DM::from_file(filename, format_hint);
  });
}

} // namespace casadi_cxxwrap
