#include "casadi_cxxwrap.hpp"

namespace casadi_cxxwrap
{
DaeBuilder dae_builder_new(const std::string& name, const std::string& path, const GenericType& options)
{
  return DaeBuilder(name, path, generic_as_dict(options, "DaeBuilder options"));
}

MX dae_add(
  DaeBuilder& dae,
  const std::string& name,
  const std::string& causality,
  const std::string& variability,
  const GenericType& options)
{
  return dae.add(name, causality, variability, generic_as_dict(options, "DaeBuilder add options"));
}

MX dae_add_causality(
  DaeBuilder& dae,
  const std::string& name,
  const std::string& causality,
  const GenericType& options)
{
  return dae.add(name, causality, generic_as_dict(options, "DaeBuilder add options"));
}

MX dae_add_default(DaeBuilder& dae, const std::string& name, const GenericType& options)
{
  return dae.add(name, generic_as_dict(options, "DaeBuilder add options"));
}

void dae_add_existing(
  DaeBuilder& dae,
  const std::string& name,
  const std::string& causality,
  const std::string& variability,
  const MX& expression,
  const GenericType& options)
{
  dae.add(name, causality, variability, expression, generic_as_dict(options, "DaeBuilder add options"));
}

void dae_eq(DaeBuilder& dae, const MX& lhs, const MX& rhs, const GenericType& options)
{
  dae.eq(lhs, rhs, generic_as_dict(options, "DaeBuilder equation options"));
}

void dae_when(DaeBuilder& dae, const MX& condition, jlcxx::ArrayRef<std::string> equations, const GenericType& options)
{
  dae.when(condition, to_vector(equations), generic_as_dict(options, "DaeBuilder when options"));
}

Function dae_add_fun(
  DaeBuilder& dae,
  const std::string& name,
  jlcxx::ArrayRef<std::string> inputs,
  jlcxx::ArrayRef<std::string> outputs,
  const GenericType& options)
{
  return dae.add_fun(
    name,
    to_vector(inputs),
    to_vector(outputs),
    generic_as_dict(options, "DaeBuilder add_fun options"));
}

Function dae_add_fun_importer(
  DaeBuilder& dae,
  const std::string& name,
  const Importer& importer,
  const GenericType& options)
{
  return dae.add_fun(name, importer, generic_as_dict(options, "DaeBuilder add_fun options"));
}

Function dae_create_legacy(
  const DaeBuilder& dae,
  const std::string& name,
  jlcxx::ArrayRef<std::string> inputs,
  jlcxx::ArrayRef<std::string> outputs,
  const bool sx,
  const bool lifted_calls)
{
  return dae.create(name, to_vector(inputs), to_vector(outputs), sx, lifted_calls);
}

Function dae_create_named(
  const DaeBuilder& dae,
  const std::string& name,
  jlcxx::ArrayRef<std::string> inputs,
  jlcxx::ArrayRef<std::string> outputs,
  const GenericType& options)
{
  return dae.create(
    name,
    to_vector(inputs),
    to_vector(outputs),
    generic_as_dict(options, "DaeBuilder create options"));
}

Function dae_create_standard(const DaeBuilder& dae, const std::string& name, const GenericType& options)
{
  return dae.create(name, generic_as_dict(options, "DaeBuilder create options"));
}

Function dae_dependent_fun(
  const DaeBuilder& dae,
  const std::string& name,
  jlcxx::ArrayRef<std::string> inputs,
  jlcxx::ArrayRef<std::string> outputs)
{
  return dae.dependent_fun(name, to_vector(inputs), to_vector(outputs));
}

std::vector<std::int64_t> dae_dimension(const DaeBuilder& dae, const std::string& name)
{
  return from_casadi_int_vector(dae.dimension(name));
}

std::vector<std::string> dae_der_names(const DaeBuilder& dae, jlcxx::ArrayRef<std::string> names)
{
  return dae.der(to_vector(names));
}

std::vector<std::string> dae_pre_names(const DaeBuilder& dae, jlcxx::ArrayRef<std::string> names)
{
  return dae.pre(to_vector(names));
}

std::vector<double> dae_attribute_vector(
  const DaeBuilder& dae,
  const std::string& attribute,
  jlcxx::ArrayRef<std::string> names)
{
  return dae.attribute(attribute, to_vector(names));
}

void dae_set_attribute_vector(
  DaeBuilder& dae,
  const std::string& attribute,
  jlcxx::ArrayRef<std::string> names,
  jlcxx::ArrayRef<double> values)
{
  dae.set_attribute(attribute, to_vector(names), to_vector(values));
}

std::vector<double> dae_min_vector(const DaeBuilder& dae, jlcxx::ArrayRef<std::string> names)
{
  return dae.min(to_vector(names));
}

void dae_set_min_vector(DaeBuilder& dae, jlcxx::ArrayRef<std::string> names, jlcxx::ArrayRef<double> values)
{
  dae.set_min(to_vector(names), to_vector(values));
}

std::vector<double> dae_max_vector(const DaeBuilder& dae, jlcxx::ArrayRef<std::string> names)
{
  return dae.max(to_vector(names));
}

void dae_set_max_vector(DaeBuilder& dae, jlcxx::ArrayRef<std::string> names, jlcxx::ArrayRef<double> values)
{
  dae.set_max(to_vector(names), to_vector(values));
}

std::vector<double> dae_nominal_vector(const DaeBuilder& dae, jlcxx::ArrayRef<std::string> names)
{
  return dae.nominal(to_vector(names));
}

void dae_set_nominal_vector(DaeBuilder& dae, jlcxx::ArrayRef<std::string> names, jlcxx::ArrayRef<double> values)
{
  dae.set_nominal(to_vector(names), to_vector(values));
}

std::vector<double> dae_start_vector(const DaeBuilder& dae, jlcxx::ArrayRef<std::string> names)
{
  return dae.start(to_vector(names));
}

void dae_set_start_vector(DaeBuilder& dae, jlcxx::ArrayRef<std::string> names, jlcxx::ArrayRef<double> values)
{
  dae.set_start(to_vector(names), to_vector(values));
}

void dae_set_values(DaeBuilder& dae, jlcxx::ArrayRef<std::string> names, jlcxx::ArrayRef<double> values)
{
  dae.set(to_vector(names), to_vector(values));
}

void dae_set_string_values(DaeBuilder& dae, jlcxx::ArrayRef<std::string> names, jlcxx::ArrayRef<std::string> values)
{
  dae.set(to_vector(names), to_vector(values));
}

std::vector<GenericType> dae_get_values(const DaeBuilder& dae, jlcxx::ArrayRef<std::string> names)
{
  return dae.get(to_vector(names));
}

Sparsity dae_jac_sparsity(
  const DaeBuilder& dae,
  jlcxx::ArrayRef<std::string> outputs,
  jlcxx::ArrayRef<std::string> inputs)
{
  return dae.jac_sparsity(to_vector(outputs), to_vector(inputs));
}

void nlp_import_nl(NlpBuilder& nlp, const std::string& filename, const GenericType& options)
{
  nlp.import_nl(filename, generic_as_dict(options, "NlpBuilder import_nl options"));
}

void register_dae_builder_bindings(jlcxx::Module& mod)
{
  mod.method(raw_method("dae_builder_new"), &dae_builder_new);
  mod.method(raw_method("dae_string"), [](const DaeBuilder& dae, const bool more) {
    return dae.get_str(more);
  });
  mod.method(raw_method("dae_name"), [](const DaeBuilder& dae) { return dae.name(); });
  mod.method(raw_method("dae_time"), [](const DaeBuilder& dae) { return dae.time(); });
  mod.method(raw_method("dae_t"), [](const DaeBuilder& dae) { return dae.t_new(); });
  mod.method(raw_method("dae_x"), [](const DaeBuilder& dae) { return dae.x(); });
  mod.method(raw_method("dae_y"), [](const DaeBuilder& dae) { return dae.y(); });
  mod.method(raw_method("dae_z"), [](const DaeBuilder& dae) { return dae.z(); });
  mod.method(raw_method("dae_q"), [](const DaeBuilder& dae) { return dae.q(); });
  mod.method(raw_method("dae_u"), [](const DaeBuilder& dae) { return dae.u(); });
  mod.method(raw_method("dae_p"), [](const DaeBuilder& dae) { return dae.p(); });
  mod.method(raw_method("dae_c"), [](const DaeBuilder& dae) { return dae.c(); });
  mod.method(raw_method("dae_d"), [](const DaeBuilder& dae) { return dae.d(); });
  mod.method(raw_method("dae_w"), [](const DaeBuilder& dae) { return dae.w(); });
  mod.method(raw_method("dae_ode"), [](const DaeBuilder& dae) { return dae.ode(); });
  mod.method(raw_method("dae_alg"), [](const DaeBuilder& dae) { return dae.alg(); });
  mod.method(raw_method("dae_quad"), [](const DaeBuilder& dae) { return dae.quad(); });
  mod.method(raw_method("dae_zero"), [](const DaeBuilder& dae) { return dae.zero(); });
  mod.method(raw_method("dae_ydef"), [](const DaeBuilder& dae) { return dae.ydef(); });
  mod.method(raw_method("dae_set_y"), [](DaeBuilder& dae, jlcxx::ArrayRef<std::string> names) {
    dae.set_y(to_vector(names));
  });
  mod.method(raw_method("dae_rate"), [](const DaeBuilder& dae) { return dae.rate(); });
  mod.method(raw_method("dae_set_rate"), [](DaeBuilder& dae, jlcxx::ArrayRef<std::string> names) {
    dae.set_rate(to_vector(names));
  });
  mod.method(raw_method("dae_cdef"), [](const DaeBuilder& dae) { return dae.cdef(); });
  mod.method(raw_method("dae_ddef"), [](const DaeBuilder& dae) { return dae.ddef(); });
  mod.method(raw_method("dae_wdef"), [](const DaeBuilder& dae) { return dae.wdef(); });
  mod.method(raw_method("dae_init_lhs"), [](const DaeBuilder& dae) { return dae.init_lhs(); });
  mod.method(raw_method("dae_init_rhs"), [](const DaeBuilder& dae) { return dae.init_rhs(); });
  mod.method(raw_method("dae_outputs"), [](const DaeBuilder& dae) { return dae.outputs(); });
  mod.method(raw_method("dae_derivatives"), [](const DaeBuilder& dae) { return dae.derivatives(); });
  mod.method(raw_method("dae_initial_unknowns"), [](const DaeBuilder& dae) { return dae.initial_unknowns(); });
  mod.method(raw_method("dae_has_t"), [](const DaeBuilder& dae) { return dae.has_t(); });
  mod.method(raw_method("dae_has_rate"), [](const DaeBuilder& dae) { return dae.has_rate(); });
  mod.method(raw_method("dae_nx"), [](const DaeBuilder& dae) { return static_cast<std::int64_t>(dae.nx()); });
  mod.method(raw_method("dae_nz"), [](const DaeBuilder& dae) { return static_cast<std::int64_t>(dae.nz()); });
  mod.method(raw_method("dae_nq"), [](const DaeBuilder& dae) { return static_cast<std::int64_t>(dae.nq()); });
  mod.method(raw_method("dae_nzero"), [](const DaeBuilder& dae) { return static_cast<std::int64_t>(dae.nzero()); });
  mod.method(raw_method("dae_ny"), [](const DaeBuilder& dae) { return static_cast<std::int64_t>(dae.ny()); });
  mod.method(raw_method("dae_nu"), [](const DaeBuilder& dae) { return static_cast<std::int64_t>(dae.nu()); });
  mod.method(raw_method("dae_np"), [](const DaeBuilder& dae) { return static_cast<std::int64_t>(dae.np()); });
  mod.method(raw_method("dae_nc"), [](const DaeBuilder& dae) { return static_cast<std::int64_t>(dae.nc()); });
  mod.method(raw_method("dae_nd"), [](const DaeBuilder& dae) { return static_cast<std::int64_t>(dae.nd()); });
  mod.method(raw_method("dae_nw"), [](const DaeBuilder& dae) { return static_cast<std::int64_t>(dae.nw()); });
  mod.method(raw_method("dae_add"), &dae_add);
  mod.method(raw_method("dae_add_causality"), &dae_add_causality);
  mod.method(raw_method("dae_add_default"), &dae_add_default);
  mod.method(raw_method("dae_add_existing"), &dae_add_existing);
  mod.method(raw_method("dae_eq"), &dae_eq);
  mod.method(raw_method("dae_when"), &dae_when);
  mod.method(raw_method("dae_assign"), [](DaeBuilder& dae, const std::string& name, const MX& value) {
    return dae.assign(name, value);
  });
  mod.method(raw_method("dae_reinit"), [](DaeBuilder& dae, const std::string& name, const MX& value) {
    return dae.reinit(name, value);
  });
  mod.method(raw_method("dae_set_init"), [](DaeBuilder& dae, const std::string& name, const MX& value) {
    dae.set_init(name, value);
  });
  mod.method(raw_method("dae_sanity_check"), [](const DaeBuilder& dae) { dae.sanity_check(); });
  mod.method(raw_method("dae_reorder"), [](DaeBuilder& dae, const std::string& category, jlcxx::ArrayRef<std::string> names) {
    dae.reorder(category, to_vector(names));
  });
  mod.method(raw_method("dae_eliminate"), [](DaeBuilder& dae, const std::string& category) { dae.eliminate(category); });
  mod.method(raw_method("dae_sort"), [](DaeBuilder& dae, const std::string& category) { dae.sort(category); });
  mod.method(raw_method("dae_lift"), [](DaeBuilder& dae, const bool lift_shared, const bool lift_calls) {
    dae.lift(lift_shared, lift_calls);
  });
  mod.method(raw_method("dae_prune"), [](DaeBuilder& dae, const bool prune_p, const bool prune_u) {
    dae.prune(prune_p, prune_u);
  });
  mod.method(raw_method("dae_tear"), [](DaeBuilder& dae) { dae.tear(); });
  mod.method(raw_method("dae_add_fun"), &dae_add_fun);
  mod.method(raw_method("dae_add_fun_existing"), [](DaeBuilder& dae, const Function& f) { return dae.add_fun(f); });
  mod.method(raw_method("dae_add_fun_importer"), &dae_add_fun_importer);
  mod.method(raw_method("dae_has_fun"), [](const DaeBuilder& dae, const std::string& name) { return dae.has_fun(name); });
  mod.method(raw_method("dae_fun"), [](const DaeBuilder& dae, const std::string& name) { return dae.fun(name); });
  mod.method(raw_method("dae_fun_all"), [](const DaeBuilder& dae) { return dae.fun(); });
  mod.method(raw_method("dae_gather_fun"), [](DaeBuilder& dae, const std::int64_t max_depth) {
    dae.gather_fun(static_cast<casadi_int>(max_depth));
  });
  mod.method(raw_method("dae_provides_directional_derivatives"), [](const DaeBuilder& dae) {
    return dae.provides_directional_derivatives();
  });
  mod.method(raw_method("dae_load_fmi_description"), [](DaeBuilder& dae, const std::string& filename) {
    dae.load_fmi_description(filename);
  });
  mod.method(raw_method("dae_export_fmu"), [](DaeBuilder& dae, const GenericType& options) {
    return dae.export_fmu(generic_as_dict(options, "DaeBuilder export_fmu options"));
  });
  mod.method(raw_method("dae_add_lc"), [](DaeBuilder& dae, const std::string& name, jlcxx::ArrayRef<std::string> outputs) {
    dae.add_lc(name, to_vector(outputs));
  });
  mod.method(raw_method("dae_create_legacy"), &dae_create_legacy);
  mod.method(raw_method("dae_create_named"), &dae_create_named);
  mod.method(raw_method("dae_create_standard"), &dae_create_standard);
  mod.method(raw_method("dae_create_default"), [](const DaeBuilder& dae) { return dae.create(); });
  mod.method(raw_method("dae_dependent_fun"), &dae_dependent_fun);
  mod.method(raw_method("dae_transition"), [](const DaeBuilder& dae, const std::string& name) {
    return dae.transition(name);
  });
  mod.method(raw_method("dae_transition_index"), [](const DaeBuilder& dae, const std::string& name, const std::int64_t index) {
    return dae.transition(name, checked_index(index, "index"));
  });
  mod.method(raw_method("dae_transition_default"), [](const DaeBuilder& dae) { return dae.transition(); });
  mod.method(raw_method("dae_var"), [](const DaeBuilder& dae, const std::string& name) { return dae.var(name); });
  mod.method(raw_method("dae_der_names"), &dae_der_names);
  mod.method(raw_method("dae_der_name"), [](const DaeBuilder& dae, const std::string& name) { return dae.der(name); });
  mod.method(raw_method("dae_der_mx"), [](DaeBuilder& dae, const MX& expression) { return dae.der(expression); });
  mod.method(raw_method("dae_pre_names"), &dae_pre_names);
  mod.method(raw_method("dae_pre_name"), [](const DaeBuilder& dae, const std::string& name) { return dae.pre(name); });
  mod.method(raw_method("dae_pre_mx"), [](const DaeBuilder& dae, const MX& expression) { return dae.pre(expression); });
  mod.method(raw_method("dae_has_beq"), [](const DaeBuilder& dae, const std::string& name) { return dae.has_beq(name); });
  mod.method(raw_method("dae_beq"), [](const DaeBuilder& dae, const std::string& name) { return dae.beq(name); });
  mod.method(raw_method("dae_value_reference"), [](const DaeBuilder& dae, const std::string& name) {
    return static_cast<std::int64_t>(dae.value_reference(name));
  });
  mod.method(raw_method("dae_set_value_reference"), [](DaeBuilder& dae, const std::string& name, const std::int64_t value) {
    dae.set_value_reference(name, checked_index(value, "value_reference"));
  });
  mod.method(raw_method("dae_description"), [](const DaeBuilder& dae, const std::string& name) { return dae.description(name); });
  mod.method(raw_method("dae_set_description"), [](DaeBuilder& dae, const std::string& name, const std::string& value) {
    dae.set_description(name, value);
  });
  mod.method(raw_method("dae_type"), [](const DaeBuilder& dae, const std::string& name, const std::int64_t fmi_version) {
    return dae.type(name, checked_nonnegative(fmi_version, "fmi_version"));
  });
  mod.method(raw_method("dae_set_type"), [](DaeBuilder& dae, const std::string& name, const std::string& value) {
    dae.set_type(name, value);
  });
  mod.method(raw_method("dae_causality"), [](const DaeBuilder& dae, const std::string& name) { return dae.causality(name); });
  mod.method(raw_method("dae_set_causality"), [](DaeBuilder& dae, const std::string& name, const std::string& value) {
    dae.set_causality(name, value);
  });
  mod.method(raw_method("dae_variability"), [](const DaeBuilder& dae, const std::string& name) { return dae.variability(name); });
  mod.method(raw_method("dae_set_variability"), [](DaeBuilder& dae, const std::string& name, const std::string& value) {
    dae.set_variability(name, value);
  });
  mod.method(raw_method("dae_category"), [](const DaeBuilder& dae, const std::string& name) { return dae.category(name); });
  mod.method(raw_method("dae_set_category"), [](DaeBuilder& dae, const std::string& name, const std::string& value) {
    dae.set_category(name, value);
  });
  mod.method(raw_method("dae_initial"), [](const DaeBuilder& dae, const std::string& name) { return dae.initial(name); });
  mod.method(raw_method("dae_set_initial"), [](DaeBuilder& dae, const std::string& name, const std::string& value) {
    dae.set_initial(name, value);
  });
  mod.method(raw_method("dae_unit"), [](const DaeBuilder& dae, const std::string& name) { return dae.unit(name); });
  mod.method(raw_method("dae_set_unit"), [](DaeBuilder& dae, const std::string& name, const std::string& value) {
    dae.set_unit(name, value);
  });
  mod.method(raw_method("dae_display_unit"), [](const DaeBuilder& dae, const std::string& name) {
    return dae.display_unit(name);
  });
  mod.method(raw_method("dae_set_display_unit"), [](DaeBuilder& dae, const std::string& name, const std::string& value) {
    dae.set_display_unit(name, value);
  });
  mod.method(raw_method("dae_numel"), [](const DaeBuilder& dae, const std::string& name) {
    return static_cast<std::int64_t>(dae.numel(name));
  });
  mod.method(raw_method("dae_dimension"), &dae_dimension);
  mod.method(raw_method("dae_start_time"), [](const DaeBuilder& dae) { return dae.start_time(); });
  mod.method(raw_method("dae_set_start_time"), [](DaeBuilder& dae, const double value) { dae.set_start_time(value); });
  mod.method(raw_method("dae_stop_time"), [](const DaeBuilder& dae) { return dae.stop_time(); });
  mod.method(raw_method("dae_set_stop_time"), [](DaeBuilder& dae, const double value) { dae.set_stop_time(value); });
  mod.method(raw_method("dae_tolerance"), [](const DaeBuilder& dae) { return dae.tolerance(); });
  mod.method(raw_method("dae_set_tolerance"), [](DaeBuilder& dae, const double value) { dae.set_tolerance(value); });
  mod.method(raw_method("dae_step_size"), [](const DaeBuilder& dae) { return dae.step_size(); });
  mod.method(raw_method("dae_set_step_size"), [](DaeBuilder& dae, const double value) { dae.set_step_size(value); });
  mod.method(raw_method("dae_attribute"), [](const DaeBuilder& dae, const std::string& attribute, const std::string& name) {
    return dae.attribute(attribute, name);
  });
  mod.method(raw_method("dae_set_attribute"), [](DaeBuilder& dae, const std::string& attribute, const std::string& name, const double value) {
    dae.set_attribute(attribute, name, value);
  });
  mod.method(raw_method("dae_min"), [](const DaeBuilder& dae, const std::string& name) { return dae.min(name); });
  mod.method(raw_method("dae_set_min"), [](DaeBuilder& dae, const std::string& name, const double value) {
    dae.set_min(name, value);
  });
  mod.method(raw_method("dae_max"), [](const DaeBuilder& dae, const std::string& name) { return dae.max(name); });
  mod.method(raw_method("dae_set_max"), [](DaeBuilder& dae, const std::string& name, const double value) {
    dae.set_max(name, value);
  });
  mod.method(raw_method("dae_nominal"), [](const DaeBuilder& dae, const std::string& name) { return dae.nominal(name); });
  mod.method(raw_method("dae_set_nominal"), [](DaeBuilder& dae, const std::string& name, const double value) {
    dae.set_nominal(name, value);
  });
  mod.method(raw_method("dae_start"), [](const DaeBuilder& dae, const std::string& name) { return dae.start(name); });
  mod.method(raw_method("dae_set_start"), [](DaeBuilder& dae, const std::string& name, const double value) {
    dae.set_start(name, value);
  });
  mod.method(raw_method("dae_set_start_vector"), [](DaeBuilder& dae, const std::string& name, jlcxx::ArrayRef<double> value) {
    dae.set_start(name, to_vector(value));
  });
  mod.method(raw_method("dae_reset"), [](DaeBuilder& dae) { dae.reset(); });
  mod.method(raw_method("dae_set"), [](DaeBuilder& dae, const std::string& name, const double value) { dae.set(name, value); });
  mod.method(raw_method("dae_set_string"), [](DaeBuilder& dae, const std::string& name, const std::string& value) {
    dae.set(name, value);
  });
  mod.method(raw_method("dae_get"), [](const DaeBuilder& dae, const std::string& name) { return dae.get(name); });
  mod.method(raw_method("dae_attribute_vector"), &dae_attribute_vector);
  mod.method(raw_method("dae_set_attribute_vector"), &dae_set_attribute_vector);
  mod.method(raw_method("dae_min_vector"), &dae_min_vector);
  mod.method(raw_method("dae_set_min_vector"), &dae_set_min_vector);
  mod.method(raw_method("dae_max_vector"), &dae_max_vector);
  mod.method(raw_method("dae_set_max_vector"), &dae_set_max_vector);
  mod.method(raw_method("dae_nominal_vector"), &dae_nominal_vector);
  mod.method(raw_method("dae_set_nominal_vector"), &dae_set_nominal_vector);
  mod.method(raw_method("dae_start_vector"), &dae_start_vector);
  mod.method(raw_method("dae_set_start_names"), &dae_set_start_vector);
  mod.method(raw_method("dae_set_values"), &dae_set_values);
  mod.method(raw_method("dae_set_string_values"), &dae_set_string_values);
  mod.method(raw_method("dae_get_values"), &dae_get_values);
  mod.method(raw_method("dae_has"), [](const DaeBuilder& dae, const std::string& name) { return dae.has(name); });
  mod.method(raw_method("dae_all"), [](const DaeBuilder& dae) { return dae.all(); });
  mod.method(raw_method("dae_all_category"), [](const DaeBuilder& dae, const std::string& category) {
    return dae.all(category);
  });
  mod.method(raw_method("dae_oracle"), [](const DaeBuilder& dae, const bool sx, const bool eliminate_w, const bool lifted_calls) {
    return dae.oracle(sx, eliminate_w, lifted_calls);
  });
  mod.method(raw_method("dae_jac_sparsity"), &dae_jac_sparsity);
}

void register_nlp_builder_bindings(jlcxx::Module& mod)
{
  mod.method(raw_method("nlp_builder_string"), [](const NlpBuilder& nlp, const bool more) {
    return nlp.get_str(more);
  });
  mod.method(raw_method("nlp_builder_import_nl"), &nlp_import_nl);
  mod.method(raw_method("nlp_builder_get_x"), [](const NlpBuilder& nlp) { return nlp.x; });
  mod.method(raw_method("nlp_builder_set_x"), [](NlpBuilder& nlp, jlcxx::ArrayRef<MX> value) {
    nlp.x = to_vector(value);
  });
  mod.method(raw_method("nlp_builder_get_f"), [](const NlpBuilder& nlp) { return nlp.f; });
  mod.method(raw_method("nlp_builder_set_f"), [](NlpBuilder& nlp, const MX& value) { nlp.f = value; });
  mod.method(raw_method("nlp_builder_get_g"), [](const NlpBuilder& nlp) { return nlp.g; });
  mod.method(raw_method("nlp_builder_set_g"), [](NlpBuilder& nlp, jlcxx::ArrayRef<MX> value) {
    nlp.g = to_vector(value);
  });
  mod.method(raw_method("nlp_builder_get_x_lb"), [](const NlpBuilder& nlp) { return nlp.x_lb; });
  mod.method(raw_method("nlp_builder_set_x_lb"), [](NlpBuilder& nlp, jlcxx::ArrayRef<double> value) {
    nlp.x_lb = to_vector(value);
  });
  mod.method(raw_method("nlp_builder_get_x_ub"), [](const NlpBuilder& nlp) { return nlp.x_ub; });
  mod.method(raw_method("nlp_builder_set_x_ub"), [](NlpBuilder& nlp, jlcxx::ArrayRef<double> value) {
    nlp.x_ub = to_vector(value);
  });
  mod.method(raw_method("nlp_builder_get_g_lb"), [](const NlpBuilder& nlp) { return nlp.g_lb; });
  mod.method(raw_method("nlp_builder_set_g_lb"), [](NlpBuilder& nlp, jlcxx::ArrayRef<double> value) {
    nlp.g_lb = to_vector(value);
  });
  mod.method(raw_method("nlp_builder_get_g_ub"), [](const NlpBuilder& nlp) { return nlp.g_ub; });
  mod.method(raw_method("nlp_builder_set_g_ub"), [](NlpBuilder& nlp, jlcxx::ArrayRef<double> value) {
    nlp.g_ub = to_vector(value);
  });
  mod.method(raw_method("nlp_builder_get_x_init"), [](const NlpBuilder& nlp) { return nlp.x_init; });
  mod.method(raw_method("nlp_builder_set_x_init"), [](NlpBuilder& nlp, jlcxx::ArrayRef<double> value) {
    nlp.x_init = to_vector(value);
  });
  mod.method(raw_method("nlp_builder_get_lambda_init"), [](const NlpBuilder& nlp) { return nlp.lambda_init; });
  mod.method(raw_method("nlp_builder_set_lambda_init"), [](NlpBuilder& nlp, jlcxx::ArrayRef<double> value) {
    nlp.lambda_init = to_vector(value);
  });
  mod.method(raw_method("nlp_builder_get_discrete"), [](const NlpBuilder& nlp) { return nlp.discrete; });
  mod.method(raw_method("nlp_builder_set_discrete"), [](NlpBuilder& nlp, jlcxx::ArrayRef<bool> value) {
    nlp.discrete = to_vector(value);
  });
}

void register_builder_bindings(jlcxx::Module& mod)
{
  register_dae_builder_bindings(mod);
  register_nlp_builder_bindings(mod);
}

} // namespace casadi_cxxwrap
