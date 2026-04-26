#include "casadi_cxxwrap.hpp"

namespace casadi_cxxwrap
{
namespace
{

casadi::VariableType opti_variable_type(const std::int64_t value)
{
  switch(value)
  {
    case casadi::OPTI_VAR:
      return casadi::OPTI_VAR;
    case casadi::OPTI_PAR:
      return casadi::OPTI_PAR;
    case casadi::OPTI_DUAL_G:
      return casadi::OPTI_DUAL_G;
    default:
      throw std::invalid_argument("unknown Opti variable type");
  }
}

DMDict opti_dm_dict(jlcxx::ArrayRef<std::string> keys, jlcxx::ArrayRef<DM> values, const char* name)
{
  return DMDict(named_dict(keys, values, name));
}

} // namespace

Opti opti_new(const std::string& problem_type)
{
  return Opti(problem_type);
}

MX opti_variable(Opti& opti, const std::int64_t rows, const std::int64_t cols, const std::string& attribute)
{
  return opti.variable(checked_nonnegative(rows, "rows"), checked_nonnegative(cols, "cols"), attribute);
}

MX opti_variable_sparsity(Opti& opti, const Sparsity& sparsity, const std::string& attribute)
{
  return opti.variable(sparsity, attribute);
}

MX opti_variable_symbol(Opti& opti, const MX& symbol, const std::string& attribute)
{
  return opti.variable(symbol, attribute);
}

MX opti_parameter(Opti& opti, const std::int64_t rows, const std::int64_t cols, const std::string& attribute)
{
  return opti.parameter(checked_nonnegative(rows, "rows"), checked_nonnegative(cols, "cols"), attribute);
}

MX opti_parameter_sparsity(Opti& opti, const Sparsity& sparsity, const std::string& attribute)
{
  return opti.parameter(sparsity, attribute);
}

MX opti_parameter_symbol(Opti& opti, const MX& symbol, const std::string& attribute)
{
  return opti.parameter(symbol, attribute);
}

void opti_subject_to(Opti& opti, const MX& constraint, const GenericType& options)
{
  opti.subject_to(constraint, generic_as_dict(options, "Opti subject_to options"));
}

void opti_subject_to_vector(Opti& opti, jlcxx::ArrayRef<MX> constraints, const GenericType& options)
{
  opti.subject_to(to_vector(constraints), generic_as_dict(options, "Opti subject_to options"));
}

void opti_subject_to_scaled(Opti& opti, const MX& constraint, const DM& linear_scale, const GenericType& options)
{
  opti.subject_to(constraint, linear_scale, generic_as_dict(options, "Opti subject_to options"));
}

void opti_subject_to_vector_scaled(
  Opti& opti,
  jlcxx::ArrayRef<MX> constraints,
  const DM& linear_scale,
  const GenericType& options)
{
  opti.subject_to(to_vector(constraints), linear_scale, generic_as_dict(options, "Opti subject_to options"));
}

void opti_solver(
  Opti& opti,
  const std::string& solver,
  const GenericType& plugin_options,
  const GenericType& solver_options)
{
  opti.solver(
    solver,
    generic_as_dict(plugin_options, "Opti plugin options"),
    generic_as_dict(solver_options, "Opti solver options"));
}

void opti_set_initial_assignments(Opti& opti, jlcxx::ArrayRef<MX> assignments)
{
  opti.set_initial(to_vector(assignments));
}

void opti_set_value_assignments(Opti& opti, jlcxx::ArrayRef<MX> assignments)
{
  opti.set_value(to_vector(assignments));
}

DM opti_value_mx(const Opti& opti, const MX& expression, jlcxx::ArrayRef<MX> values)
{
  return opti.value(expression, to_vector(values));
}

DM opti_value_dm(const Opti& opti, const DM& expression, jlcxx::ArrayRef<MX> values)
{
  return opti.value(expression, to_vector(values));
}

DM opti_value_sx(const Opti& opti, const SX& expression, jlcxx::ArrayRef<MX> values)
{
  return opti.value(expression, to_vector(values));
}

Function opti_to_function(
  Opti& opti,
  const std::string& name,
  jlcxx::ArrayRef<MX> inputs,
  jlcxx::ArrayRef<MX> outputs,
  const GenericType& options)
{
  return opti.to_function(
    name,
    to_vector(inputs),
    to_vector(outputs),
    generic_as_dict(options, "Opti to_function options"));
}

Function opti_to_function_named(
  Opti& opti,
  const std::string& name,
  jlcxx::ArrayRef<MX> inputs,
  jlcxx::ArrayRef<MX> outputs,
  jlcxx::ArrayRef<std::string> input_names,
  jlcxx::ArrayRef<std::string> output_names,
  const GenericType& options)
{
  return opti.to_function(
    name,
    to_vector(inputs),
    to_vector(outputs),
    to_vector(input_names),
    to_vector(output_names),
    generic_as_dict(options, "Opti to_function options"));
}

Function opti_to_function_dict(
  Opti& opti,
  const std::string& name,
  jlcxx::ArrayRef<std::string> keys,
  jlcxx::ArrayRef<MX> values,
  jlcxx::ArrayRef<std::string> input_names,
  jlcxx::ArrayRef<std::string> output_names,
  const GenericType& options)
{
  return opti.to_function(
    name,
    MXDict(named_dict(keys, values, "Opti to_function dictionary")),
    to_vector(input_names),
    to_vector(output_names),
    generic_as_dict(options, "Opti to_function options"));
}

DM opti_sol_value_mx(const OptiSol& solution, const MX& expression, jlcxx::ArrayRef<MX> values)
{
  return solution.value(expression, to_vector(values));
}

DM opti_sol_value_dm(const OptiSol& solution, const DM& expression, jlcxx::ArrayRef<MX> values)
{
  return solution.value(expression, to_vector(values));
}

DM opti_sol_value_sx(const OptiSol& solution, const SX& expression, jlcxx::ArrayRef<MX> values)
{
  return solution.value(expression, to_vector(values));
}

void register_opti_bindings(jlcxx::Module& mod)
{
  mod.method(raw_method("opti_new"), &opti_new);
  mod.method(raw_method("opti_string"), [](const Opti& opti, const bool more) {
    return opti.get_str(more);
  });
  mod.method(raw_method("opti_variable"), &opti_variable);
  mod.method(raw_method("opti_variable_sparsity"), &opti_variable_sparsity);
  mod.method(raw_method("opti_variable_symbol"), &opti_variable_symbol);
  mod.method(raw_method("opti_parameter"), &opti_parameter);
  mod.method(raw_method("opti_parameter_sparsity"), &opti_parameter_sparsity);
  mod.method(raw_method("opti_parameter_symbol"), &opti_parameter_symbol);
  mod.method(raw_method("opti_minimize"), [](Opti& opti, const MX& objective, const double linear_scale) {
    opti.minimize(objective, linear_scale);
  });
  mod.method(raw_method("opti_subject_to"), &opti_subject_to);
  mod.method(raw_method("opti_subject_to_vector"), &opti_subject_to_vector);
  mod.method(raw_method("opti_subject_to_scaled"), &opti_subject_to_scaled);
  mod.method(raw_method("opti_subject_to_vector_scaled"), &opti_subject_to_vector_scaled);
  mod.method(raw_method("opti_subject_to_clear"), [](Opti& opti) {
    opti.subject_to();
  });
  mod.method(raw_method("opti_solver"), &opti_solver);
  mod.method(raw_method("opti_set_initial"), [](Opti& opti, const MX& expression, const DM& value) {
    opti.set_initial(expression, value);
  });
  mod.method(raw_method("opti_set_initial_assignments"), &opti_set_initial_assignments);
  mod.method(raw_method("opti_set_value"), [](Opti& opti, const MX& expression, const DM& value) {
    opti.set_value(expression, value);
  });
  mod.method(raw_method("opti_set_value_assignments"), &opti_set_value_assignments);
  mod.method(raw_method("opti_set_domain"), [](Opti& opti, const MX& expression, const std::string& domain) {
    opti.set_domain(expression, domain);
  });
  mod.method(raw_method("opti_set_linear_scale"), [](Opti& opti, const MX& expression, const DM& scale, const DM& offset) {
    opti.set_linear_scale(expression, scale, offset);
  });
  mod.method(raw_method("opti_solve"), [](Opti& opti) {
    return opti.solve();
  });
  mod.method(raw_method("opti_solve_limited"), [](Opti& opti) {
    return opti.solve_limited();
  });
  mod.method(raw_method("opti_value_mx"), &opti_value_mx);
  mod.method(raw_method("opti_value_dm"), &opti_value_dm);
  mod.method(raw_method("opti_value_sx"), &opti_value_sx);
  mod.method(raw_method("opti_stats"), [](const Opti& opti) {
    return GenericType(opti.stats());
  });
  mod.method(raw_method("opti_return_status"), [](const Opti& opti) {
    return opti.return_status();
  });
  mod.method(raw_method("opti_initial"), [](const Opti& opti) {
    return opti.initial();
  });
  mod.method(raw_method("opti_value_variables"), [](const Opti& opti) {
    return opti.value_variables();
  });
  mod.method(raw_method("opti_value_parameters"), [](const Opti& opti) {
    return opti.value_parameters();
  });
  mod.method(raw_method("opti_scale_helper"), [](const Opti& opti, const Function& helper) {
    return opti.scale_helper(helper);
  });
  mod.method(raw_method("opti_dual"), [](const Opti& opti, const MX& constraint) {
    return opti.dual(constraint);
  });
  mod.method(raw_method("opti_nx"), [](const Opti& opti) {
    return static_cast<std::int64_t>(opti.nx());
  });
  mod.method(raw_method("opti_np"), [](const Opti& opti) {
    return static_cast<std::int64_t>(opti.np());
  });
  mod.method(raw_method("opti_ng"), [](const Opti& opti) {
    return static_cast<std::int64_t>(opti.ng());
  });
  mod.method(raw_method("opti_x"), [](const Opti& opti) { return opti.x(); });
  mod.method(raw_method("opti_p"), [](const Opti& opti) { return opti.p(); });
  mod.method(raw_method("opti_g"), [](const Opti& opti) { return opti.g(); });
  mod.method(raw_method("opti_f"), [](const Opti& opti) { return opti.f(); });
  mod.method(raw_method("opti_lbg"), [](const Opti& opti) { return opti.lbg(); });
  mod.method(raw_method("opti_ubg"), [](const Opti& opti) { return opti.ubg(); });
  mod.method(raw_method("opti_x_linear_scale"), [](const Opti& opti) { return opti.x_linear_scale(); });
  mod.method(raw_method("opti_x_linear_scale_offset"), [](const Opti& opti) {
    return opti.x_linear_scale_offset();
  });
  mod.method(raw_method("opti_g_linear_scale"), [](const Opti& opti) { return opti.g_linear_scale(); });
  mod.method(raw_method("opti_f_linear_scale"), [](const Opti& opti) { return opti.f_linear_scale(); });
  mod.method(raw_method("opti_lam_g"), [](const Opti& opti) { return opti.lam_g(); });
  mod.method(raw_method("opti_to_function"), &opti_to_function);
  mod.method(raw_method("opti_to_function_named"), &opti_to_function_named);
  mod.method(raw_method("opti_to_function_dict"), &opti_to_function_dict);
  mod.method(raw_method("opti_bounded"), [](const MX& lb, const MX& expression, const MX& ub) {
    return Opti::bounded(lb, expression, ub);
  });
  mod.method(raw_method("opti_debug"), [](const Opti& opti) { return opti.debug(); });
  mod.method(raw_method("opti_advanced"), [](const Opti& opti) { return opti.advanced(); });
  mod.method(raw_method("opti_copy"), [](const Opti& opti) { return opti.copy(); });
  mod.method(raw_method("opti_update_user_dict"), [](Opti& opti, const MX& expression, const GenericType& meta) {
    opti.update_user_dict(expression, generic_as_dict(meta, "Opti user dictionary"));
  });
  mod.method(raw_method("opti_update_user_dict_vector"), [](Opti& opti, jlcxx::ArrayRef<MX> expressions, const GenericType& meta) {
    opti.update_user_dict(to_vector(expressions), generic_as_dict(meta, "Opti user dictionary"));
  });
  mod.method(raw_method("opti_user_dict"), [](const Opti& opti, const MX& expression) {
    return GenericType(opti.user_dict(expression));
  });

  mod.method(raw_method("opti_advanced_string"), [](const OptiAdvanced& opti, const bool more) {
    return opti.get_str(more);
  });
  mod.method(raw_method("opti_advanced_casadi_solver"), [](const OptiAdvanced& opti) {
    return opti.casadi_solver();
  });
  mod.method(raw_method("opti_advanced_is_parametric"), [](const OptiAdvanced& opti, const MX& expression) {
    return opti.is_parametric(expression);
  });
  mod.method(raw_method("opti_advanced_symvar"), [](const OptiAdvanced& opti) {
    return opti.symvar();
  });
  mod.method(raw_method("opti_advanced_symvar_expr"), [](const OptiAdvanced& opti, const MX& expression) {
    return opti.symvar(expression);
  });
  mod.method(raw_method("opti_advanced_symvar_expr_type"), [](const OptiAdvanced& opti, const MX& expression, const std::int64_t type) {
    return opti.symvar(expression, opti_variable_type(type));
  });
  mod.method(raw_method("opti_advanced_active_symvar"), [](const OptiAdvanced& opti, const std::int64_t type) {
    return opti.active_symvar(opti_variable_type(type));
  });
  mod.method(raw_method("opti_advanced_active_values"), [](const OptiAdvanced& opti, const std::int64_t type) {
    return opti.active_values(opti_variable_type(type));
  });
  mod.method(raw_method("opti_advanced_x_lookup"), [](const OptiAdvanced& opti, const std::int64_t index) {
    return opti.x_lookup(checked_index(index, "index"));
  });
  mod.method(raw_method("opti_advanced_g_lookup"), [](const OptiAdvanced& opti, const std::int64_t index) {
    return opti.g_lookup(checked_index(index, "index"));
  });
  mod.method(raw_method("opti_advanced_g_index_reduce_g"), [](const OptiAdvanced& opti, const std::int64_t index) {
    return static_cast<std::int64_t>(opti.g_index_reduce_g(checked_index(index, "index")));
  });
  mod.method(raw_method("opti_advanced_g_index_reduce_x"), [](const OptiAdvanced& opti, const std::int64_t index) {
    return static_cast<std::int64_t>(opti.g_index_reduce_x(checked_index(index, "index")));
  });
  mod.method(raw_method("opti_advanced_g_index_unreduce_g"), [](const OptiAdvanced& opti, const std::int64_t index) {
    return static_cast<std::int64_t>(opti.g_index_unreduce_g(checked_index(index, "index")));
  });
  mod.method(raw_method("opti_advanced_x_describe"), [](const OptiAdvanced& opti, const std::int64_t index, const GenericType& options) {
    return opti.x_describe(checked_index(index, "index"), generic_as_dict(options, "Opti x_describe options"));
  });
  mod.method(raw_method("opti_advanced_g_describe"), [](const OptiAdvanced& opti, const std::int64_t index, const GenericType& options) {
    return opti.g_describe(checked_index(index, "index"), generic_as_dict(options, "Opti g_describe options"));
  });
  mod.method(raw_method("opti_advanced_describe"), [](const OptiAdvanced& opti, const MX& expression, const std::int64_t indent, const GenericType& options) {
    return opti.describe(
      expression,
      checked_nonnegative(indent, "indent"),
      generic_as_dict(options, "Opti describe options"));
  });
  mod.method(raw_method("opti_advanced_show_infeasibilities"), [](const OptiAdvanced& opti, const double tolerance, const GenericType& options) {
    opti.show_infeasibilities(tolerance, generic_as_dict(options, "Opti show_infeasibilities options"));
  });
  mod.method(raw_method("opti_advanced_solve_prepare"), [](OptiAdvanced& opti) {
    opti.solve_prepare();
  });
  mod.method(raw_method("opti_advanced_solve_actual"), [](OptiAdvanced& opti, jlcxx::ArrayRef<std::string> keys, jlcxx::ArrayRef<DM> values) {
    return opti.solve_actual(opti_dm_dict(keys, values, "Opti solve_actual arguments"));
  });
  mod.method(raw_method("opti_advanced_arg"), [](const OptiAdvanced& opti) {
    return opti.arg();
  });
  mod.method(raw_method("opti_advanced_set_res"), [](OptiAdvanced& opti, jlcxx::ArrayRef<std::string> keys, jlcxx::ArrayRef<DM> values) {
    opti.res(opti_dm_dict(keys, values, "Opti result dictionary"));
  });
  mod.method(raw_method("opti_advanced_res"), [](const OptiAdvanced& opti) {
    return opti.res();
  });
  mod.method(raw_method("opti_advanced_constraints"), [](const OptiAdvanced& opti) {
    return opti.constraints();
  });
  mod.method(raw_method("opti_advanced_objective"), [](const OptiAdvanced& opti) {
    return opti.objective();
  });
  mod.method(raw_method("opti_advanced_baked_copy"), [](const OptiAdvanced& opti) {
    return opti.baked_copy();
  });
  mod.method(raw_method("opti_advanced_assert_empty"), [](const OptiAdvanced& opti) {
    opti.assert_empty();
  });
  mod.method(raw_method("opti_advanced_bake"), [](OptiAdvanced& opti) {
    opti.bake();
  });
  mod.method(raw_method("opti_advanced_mark_problem_dirty"), [](OptiAdvanced& opti, const bool flag) {
    opti.mark_problem_dirty(flag);
  });
  mod.method(raw_method("opti_advanced_problem_dirty"), [](const OptiAdvanced& opti) {
    return opti.problem_dirty();
  });
  mod.method(raw_method("opti_advanced_mark_solver_dirty"), [](OptiAdvanced& opti, const bool flag) {
    opti.mark_solver_dirty(flag);
  });
  mod.method(raw_method("opti_advanced_solver_dirty"), [](const OptiAdvanced& opti) {
    return opti.solver_dirty();
  });
  mod.method(raw_method("opti_advanced_mark_solved"), [](OptiAdvanced& opti, const bool flag) {
    opti.mark_solved(flag);
  });
  mod.method(raw_method("opti_advanced_solved"), [](const OptiAdvanced& opti) {
    return opti.solved();
  });
  mod.method(raw_method("opti_advanced_assert_solved"), [](const OptiAdvanced& opti) {
    opti.assert_solved();
  });
  mod.method(raw_method("opti_advanced_assert_baked"), [](const OptiAdvanced& opti) {
    opti.assert_baked();
  });
  mod.method(raw_method("opti_advanced_instance_number"), [](const OptiAdvanced& opti) {
    return static_cast<std::int64_t>(opti.instance_number());
  });

  mod.method(raw_method("opti_sol_string"), [](const OptiSol& solution, const bool more) {
    return solution.get_str(more);
  });
  mod.method(raw_method("opti_sol_value_mx"), &opti_sol_value_mx);
  mod.method(raw_method("opti_sol_value_dm"), &opti_sol_value_dm);
  mod.method(raw_method("opti_sol_value_sx"), &opti_sol_value_sx);
  mod.method(raw_method("opti_sol_value_variables"), [](const OptiSol& solution) {
    return solution.value_variables();
  });
  mod.method(raw_method("opti_sol_value_parameters"), [](const OptiSol& solution) {
    return solution.value_parameters();
  });
  mod.method(raw_method("opti_sol_stats"), [](const OptiSol& solution) {
    return GenericType(solution.stats());
  });
  mod.method(raw_method("opti_sol_opti"), [](const OptiSol& solution) {
    return solution.opti();
  });
}

} // namespace casadi_cxxwrap
