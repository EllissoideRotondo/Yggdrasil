#include "casadi_cxxwrap.hpp"

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
T hessian_value(const T& expression, const T& variable)
{
  T gradient;
  return hessian(expression, variable, gradient);
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
void register_matrix_common(jlcxx::Module& mod, const std::string& prefix)
{
  mod.method(raw_method(prefix, "scalar"), &scalar<T>);
  mod.method(raw_method(prefix, "zeros"), &zeros<T>);
  mod.method(raw_method(prefix, "ones"), &ones<T>);
  mod.method(raw_method(prefix, "eye"), &eye<T>);
  mod.method(raw_method(prefix, "string"), &to_string<T>);
  mod.method(raw_method(prefix, "rows"), &rows<T>);
  mod.method(raw_method(prefix, "cols"), &cols<T>);
  mod.method(raw_method(prefix, "nnz"), &nonzeros<T>);
  mod.method(raw_method(prefix, "numel"), &numel<T>);
  mod.method(raw_method(prefix, "is_empty"), [](const T& value) { return value.is_empty(); });
  mod.method(raw_method(prefix, "is_scalar"), [](const T& value) { return value.is_scalar(); });
  mod.method(raw_method(prefix, "get"), &element_at<T>);
  mod.method(raw_method(prefix, "sparsity"), [](const T& value) { return value.get_sparsity(); });

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
  mod.method(raw_method(prefix, "vertcat"), [](jlcxx::ArrayRef<T> values) { return vertcat(to_vector(values)); });
  mod.method(raw_method(prefix, "horzcat"), [](jlcxx::ArrayRef<T> values) { return horzcat(to_vector(values)); });

  mod.method(raw_method(prefix, "sin"), [](const T& value) { return sin(value); });
  mod.method(raw_method(prefix, "cos"), [](const T& value) { return cos(value); });
  mod.method(raw_method(prefix, "tan"), [](const T& value) { return tan(value); });
  mod.method(raw_method(prefix, "asin"), [](const T& value) { return asin(value); });
  mod.method(raw_method(prefix, "acos"), [](const T& value) { return acos(value); });
  mod.method(raw_method(prefix, "atan"), [](const T& value) { return atan(value); });
  mod.method(raw_method(prefix, "sinh"), [](const T& value) { return sinh(value); });
  mod.method(raw_method(prefix, "cosh"), [](const T& value) { return cosh(value); });
  mod.method(raw_method(prefix, "tanh"), [](const T& value) { return tanh(value); });
  mod.method(raw_method(prefix, "exp"), [](const T& value) { return exp(value); });
  mod.method(raw_method(prefix, "log"), [](const T& value) { return log(value); });
  mod.method(raw_method(prefix, "sqrt"), [](const T& value) { return sqrt(value); });
  mod.method(raw_method(prefix, "fabs"), [](const T& value) { return fabs(value); });
  mod.method(raw_method(prefix, "sq"), [](const T& value) { return sq(value); });
  mod.method(raw_method(prefix, "if_else"), [](const T& condition, const T& if_true, const T& if_false, const bool short_circuit) {
    return if_else(condition, if_true, if_false, short_circuit);
  });
  mod.method(raw_method(prefix, "norm_1"), [](const T& value) { return norm_1(value); });
  mod.method(raw_method(prefix, "norm_2"), [](const T& value) { return norm_2(value); });
  mod.method(raw_method(prefix, "norm_fro"), [](const T& value) { return norm_fro(value); });
  mod.method(raw_method(prefix, "norm_inf"), [](const T& value) { return norm_inf(value); });
  mod.method(raw_method(prefix, "det"), [](const T& value) { return det(value); });
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
  mod.method(raw_method(prefix, "sum1"), [](const T& value) { return sum1(value); });
  mod.method(raw_method(prefix, "sum2"), [](const T& value) { return sum2(value); });
  mod.method(raw_method(prefix, "dot"), [](const T& lhs, const T& rhs) { return dot(lhs, rhs); });
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
}

} // namespace casadi_cxxwrap
