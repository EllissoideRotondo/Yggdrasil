#include "casadi_cxxwrap.hpp"

namespace casadi_cxxwrap
{

Function nlpsol_sx(
  const std::string& name,
  const std::string& solver,
  jlcxx::ArrayRef<std::string> keys,
  jlcxx::ArrayRef<SX> values,
  const GenericType& options)
{
  return casadi::nlpsol(name, solver, SXDict(named_dict(keys, values, "NLP")), generic_as_dict(options, "nlpsol options"));
}

Function nlpsol_mx(
  const std::string& name,
  const std::string& solver,
  jlcxx::ArrayRef<std::string> keys,
  jlcxx::ArrayRef<MX> values,
  const GenericType& options)
{
  return casadi::nlpsol(name, solver, MXDict(named_dict(keys, values, "NLP")), generic_as_dict(options, "nlpsol options"));
}

Function nlpsol_filename(
  const std::string& name,
  const std::string& solver,
  const std::string& filename,
  const GenericType& options)
{
  return casadi::nlpsol(name, solver, filename, generic_as_dict(options, "nlpsol options"));
}

Function nlpsol_importer(
  const std::string& name,
  const std::string& solver,
  const Importer& importer,
  const GenericType& options)
{
  return casadi::nlpsol(name, solver, importer, generic_as_dict(options, "nlpsol options"));
}

Function nlpsol_builder(
  const std::string& name,
  const std::string& solver,
  const NlpBuilder& nlp,
  const GenericType& options)
{
  return casadi::nlpsol(name, solver, nlp, generic_as_dict(options, "nlpsol options"));
}

Function nlpsol_function(
  const std::string& name,
  const std::string& solver,
  const Function& nlp,
  const GenericType& options)
{
  return casadi::nlpsol(name, solver, nlp, generic_as_dict(options, "nlpsol options"));
}

Function qpsol_sx(
  const std::string& name,
  const std::string& solver,
  jlcxx::ArrayRef<std::string> keys,
  jlcxx::ArrayRef<SX> values,
  const GenericType& options)
{
  return casadi::qpsol(name, solver, SXDict(named_dict(keys, values, "QP")), generic_as_dict(options, "qpsol options"));
}

Function qpsol_mx(
  const std::string& name,
  const std::string& solver,
  jlcxx::ArrayRef<std::string> keys,
  jlcxx::ArrayRef<MX> values,
  const GenericType& options)
{
  return casadi::qpsol(name, solver, MXDict(named_dict(keys, values, "QP")), generic_as_dict(options, "qpsol options"));
}

Function conic_sp(
  const std::string& name,
  const std::string& solver,
  jlcxx::ArrayRef<std::string> keys,
  jlcxx::ArrayRef<Sparsity> values,
  const GenericType& options)
{
  return casadi::conic(name, solver, SpDict(named_dict(keys, values, "conic")), generic_as_dict(options, "conic options"));
}

std::string conic_debug_string(const Function& f)
{
  std::ostringstream out;
  casadi::conic_debug(f, out);
  return out.str();
}

Function rootfinder_sx(
  const std::string& name,
  const std::string& solver,
  jlcxx::ArrayRef<std::string> keys,
  jlcxx::ArrayRef<SX> values,
  const GenericType& options)
{
  return casadi::rootfinder(name, solver, SXDict(named_dict(keys, values, "rootfinder")), generic_as_dict(options, "rootfinder options"));
}

Function rootfinder_mx(
  const std::string& name,
  const std::string& solver,
  jlcxx::ArrayRef<std::string> keys,
  jlcxx::ArrayRef<MX> values,
  const GenericType& options)
{
  return casadi::rootfinder(name, solver, MXDict(named_dict(keys, values, "rootfinder")), generic_as_dict(options, "rootfinder options"));
}

Function rootfinder_function(
  const std::string& name,
  const std::string& solver,
  const Function& f,
  const GenericType& options)
{
  return casadi::rootfinder(name, solver, f, generic_as_dict(options, "rootfinder options"));
}

Function integrator_sx(
  const std::string& name,
  const std::string& solver,
  jlcxx::ArrayRef<std::string> keys,
  jlcxx::ArrayRef<SX> values,
  const GenericType& options)
{
  return casadi::integrator(name, solver, SXDict(named_dict(keys, values, "DAE")), generic_as_dict(options, "integrator options"));
}

Function integrator_mx(
  const std::string& name,
  const std::string& solver,
  jlcxx::ArrayRef<std::string> keys,
  jlcxx::ArrayRef<MX> values,
  const GenericType& options)
{
  return casadi::integrator(name, solver, MXDict(named_dict(keys, values, "DAE")), generic_as_dict(options, "integrator options"));
}

Function integrator_function(
  const std::string& name,
  const std::string& solver,
  const Function& dae,
  const GenericType& options)
{
  return casadi::integrator(name, solver, dae, generic_as_dict(options, "integrator options"));
}

Function integrator_sx_tf(
  const std::string& name,
  const std::string& solver,
  jlcxx::ArrayRef<std::string> keys,
  jlcxx::ArrayRef<SX> values,
  const double t0,
  const double tf,
  const GenericType& options)
{
  return casadi::integrator(name, solver, SXDict(named_dict(keys, values, "DAE")), t0, tf, generic_as_dict(options, "integrator options"));
}

Function integrator_mx_tf(
  const std::string& name,
  const std::string& solver,
  jlcxx::ArrayRef<std::string> keys,
  jlcxx::ArrayRef<MX> values,
  const double t0,
  const double tf,
  const GenericType& options)
{
  return casadi::integrator(name, solver, MXDict(named_dict(keys, values, "DAE")), t0, tf, generic_as_dict(options, "integrator options"));
}

Function integrator_function_tf(
  const std::string& name,
  const std::string& solver,
  const Function& dae,
  const double t0,
  const double tf,
  const GenericType& options)
{
  return casadi::integrator(name, solver, dae, t0, tf, generic_as_dict(options, "integrator options"));
}

Function integrator_sx_tout(
  const std::string& name,
  const std::string& solver,
  jlcxx::ArrayRef<std::string> keys,
  jlcxx::ArrayRef<SX> values,
  const double t0,
  jlcxx::ArrayRef<double> tout,
  const GenericType& options)
{
  return casadi::integrator(name, solver, SXDict(named_dict(keys, values, "DAE")), t0, to_vector(tout), generic_as_dict(options, "integrator options"));
}

Function integrator_mx_tout(
  const std::string& name,
  const std::string& solver,
  jlcxx::ArrayRef<std::string> keys,
  jlcxx::ArrayRef<MX> values,
  const double t0,
  jlcxx::ArrayRef<double> tout,
  const GenericType& options)
{
  return casadi::integrator(name, solver, MXDict(named_dict(keys, values, "DAE")), t0, to_vector(tout), generic_as_dict(options, "integrator options"));
}

Function integrator_function_tout(
  const std::string& name,
  const std::string& solver,
  const Function& dae,
  const double t0,
  jlcxx::ArrayRef<double> tout,
  const GenericType& options)
{
  return casadi::integrator(name, solver, dae, t0, to_vector(tout), generic_as_dict(options, "integrator options"));
}

void register_factory_bindings(jlcxx::Module& mod)
{
  mod.method(raw_method("nlpsol_sx"), &nlpsol_sx);
  mod.method(raw_method("nlpsol_mx"), &nlpsol_mx);
  mod.method(raw_method("nlpsol_filename"), &nlpsol_filename);
  mod.method(raw_method("nlpsol_importer"), &nlpsol_importer);
  mod.method(raw_method("nlpsol_builder"), &nlpsol_builder);
  mod.method(raw_method("nlpsol_function"), &nlpsol_function);
  mod.method(raw_method("nlpsol_in"), []() { return casadi::nlpsol_in(); });
  mod.method(raw_method("nlpsol_out"), []() { return casadi::nlpsol_out(); });
  mod.method(raw_method("nlpsol_in_index"), [](const std::int64_t index) { return casadi::nlpsol_in(checked_index(index, "index")); });
  mod.method(raw_method("nlpsol_out_index"), [](const std::int64_t index) { return casadi::nlpsol_out(checked_index(index, "index")); });
  mod.method(raw_method("nlpsol_n_in"), []() { return static_cast<std::int64_t>(casadi::nlpsol_n_in()); });
  mod.method(raw_method("nlpsol_n_out"), []() { return static_cast<std::int64_t>(casadi::nlpsol_n_out()); });
  mod.method(raw_method("nlpsol_default_in"), [](const std::int64_t index) { return casadi::nlpsol_default_in(checked_index(index, "index")); });
  mod.method(raw_method("nlpsol_default_in_all"), []() { return casadi::nlpsol_default_in(); });
  mod.method(raw_method("nlpsol_options"), [](const std::string& plugin) { return casadi::nlpsol_options(plugin); });
  mod.method(raw_method("nlpsol_option_type"), [](const std::string& plugin, const std::string& option) { return casadi::nlpsol_option_type(plugin, option); });
  mod.method(raw_method("nlpsol_option_info"), [](const std::string& plugin, const std::string& option) { return casadi::nlpsol_option_info(plugin, option); });
  mod.method(raw_method("has_nlpsol"), [](const std::string& plugin) { return casadi::has_nlpsol(plugin); });
  mod.method(raw_method("load_nlpsol"), [](const std::string& plugin) { casadi::load_nlpsol(plugin); });
  mod.method(raw_method("doc_nlpsol"), [](const std::string& plugin) { return casadi::doc_nlpsol(plugin); });

  mod.method(raw_method("qpsol_sx"), &qpsol_sx);
  mod.method(raw_method("qpsol_mx"), &qpsol_mx);

  mod.method(raw_method("conic_sparsity"), &conic_sp);
  mod.method(raw_method("conic_in"), []() { return casadi::conic_in(); });
  mod.method(raw_method("conic_out"), []() { return casadi::conic_out(); });
  mod.method(raw_method("conic_in_index"), [](const std::int64_t index) { return casadi::conic_in(checked_index(index, "index")); });
  mod.method(raw_method("conic_out_index"), [](const std::int64_t index) { return casadi::conic_out(checked_index(index, "index")); });
  mod.method(raw_method("conic_n_in"), []() { return static_cast<std::int64_t>(casadi::conic_n_in()); });
  mod.method(raw_method("conic_n_out"), []() { return static_cast<std::int64_t>(casadi::conic_n_out()); });
  mod.method(raw_method("conic_options"), [](const std::string& plugin) { return casadi::conic_options(plugin); });
  mod.method(raw_method("conic_option_type"), [](const std::string& plugin, const std::string& option) { return casadi::conic_option_type(plugin, option); });
  mod.method(raw_method("conic_option_info"), [](const std::string& plugin, const std::string& option) { return casadi::conic_option_info(plugin, option); });
  mod.method(raw_method("has_conic"), [](const std::string& plugin) { return casadi::has_conic(plugin); });
  mod.method(raw_method("load_conic"), [](const std::string& plugin) { casadi::load_conic(plugin); });
  mod.method(raw_method("doc_conic"), [](const std::string& plugin) { return casadi::doc_conic(plugin); });
  mod.method(raw_method("conic_debug_file"), [](const Function& f, const std::string& filename) {
    casadi::conic_debug(f, filename);
  });
  mod.method(raw_method("conic_debug_string"), &conic_debug_string);

  mod.method(raw_method("rootfinder_sx"), &rootfinder_sx);
  mod.method(raw_method("rootfinder_mx"), &rootfinder_mx);
  mod.method(raw_method("rootfinder_function"), &rootfinder_function);
  mod.method(raw_method("rootfinder_in"), []() { return casadi::rootfinder_in(); });
  mod.method(raw_method("rootfinder_out"), []() { return casadi::rootfinder_out(); });
  mod.method(raw_method("rootfinder_in_index"), [](const std::int64_t index) { return casadi::rootfinder_in(checked_index(index, "index")); });
  mod.method(raw_method("rootfinder_out_index"), [](const std::int64_t index) { return casadi::rootfinder_out(checked_index(index, "index")); });
  mod.method(raw_method("rootfinder_n_in"), []() { return static_cast<std::int64_t>(casadi::rootfinder_n_in()); });
  mod.method(raw_method("rootfinder_n_out"), []() { return static_cast<std::int64_t>(casadi::rootfinder_n_out()); });
  mod.method(raw_method("rootfinder_options"), [](const std::string& plugin) { return casadi::rootfinder_options(plugin); });
  mod.method(raw_method("rootfinder_option_type"), [](const std::string& plugin, const std::string& option) { return casadi::rootfinder_option_type(plugin, option); });
  mod.method(raw_method("rootfinder_option_info"), [](const std::string& plugin, const std::string& option) { return casadi::rootfinder_option_info(plugin, option); });
  mod.method(raw_method("has_rootfinder"), [](const std::string& plugin) { return casadi::has_rootfinder(plugin); });
  mod.method(raw_method("load_rootfinder"), [](const std::string& plugin) { casadi::load_rootfinder(plugin); });
  mod.method(raw_method("doc_rootfinder"), [](const std::string& plugin) { return casadi::doc_rootfinder(plugin); });

  mod.method(raw_method("integrator_sx"), &integrator_sx);
  mod.method(raw_method("integrator_mx"), &integrator_mx);
  mod.method(raw_method("integrator_function"), &integrator_function);
  mod.method(raw_method("integrator_sx_tf"), &integrator_sx_tf);
  mod.method(raw_method("integrator_mx_tf"), &integrator_mx_tf);
  mod.method(raw_method("integrator_function_tf"), &integrator_function_tf);
  mod.method(raw_method("integrator_sx_tout"), &integrator_sx_tout);
  mod.method(raw_method("integrator_mx_tout"), &integrator_mx_tout);
  mod.method(raw_method("integrator_function_tout"), &integrator_function_tout);
  mod.method(raw_method("integrator_in"), []() { return casadi::integrator_in(); });
  mod.method(raw_method("integrator_out"), []() { return casadi::integrator_out(); });
  mod.method(raw_method("integrator_in_index"), [](const std::int64_t index) { return casadi::integrator_in(checked_index(index, "index")); });
  mod.method(raw_method("integrator_out_index"), [](const std::int64_t index) { return casadi::integrator_out(checked_index(index, "index")); });
  mod.method(raw_method("integrator_n_in"), []() { return static_cast<std::int64_t>(casadi::integrator_n_in()); });
  mod.method(raw_method("integrator_n_out"), []() { return static_cast<std::int64_t>(casadi::integrator_n_out()); });
  mod.method(raw_method("dyn_in"), []() { return casadi::dyn_in(); });
  mod.method(raw_method("dyn_out"), []() { return casadi::dyn_out(); });
  mod.method(raw_method("dyn_in_index"), [](const std::int64_t index) { return casadi::dyn_in(checked_index(index, "index")); });
  mod.method(raw_method("dyn_out_index"), [](const std::int64_t index) { return casadi::dyn_out(checked_index(index, "index")); });
  mod.method(raw_method("dyn_n_in"), []() { return static_cast<std::int64_t>(casadi::dyn_n_in()); });
  mod.method(raw_method("dyn_n_out"), []() { return static_cast<std::int64_t>(casadi::dyn_n_out()); });
  mod.method(raw_method("event_in"), []() { return casadi::event_in(); });
  mod.method(raw_method("event_out"), []() { return casadi::event_out(); });
  mod.method(raw_method("has_integrator"), [](const std::string& plugin) { return casadi::has_integrator(plugin); });
  mod.method(raw_method("load_integrator"), [](const std::string& plugin) { casadi::load_integrator(plugin); });
  mod.method(raw_method("doc_integrator"), [](const std::string& plugin) { return casadi::doc_integrator(plugin); });

  mod.method(raw_method("expmsol"), [](const std::string& name, const std::string& solver, const Sparsity& A, const GenericType& options) {
    return casadi::expmsol(name, solver, A, generic_as_dict(options, "expmsol options"));
  });
  mod.method(raw_method("expmsol_n_in"), []() { return casadi::expm_n_in(); });
  mod.method(raw_method("expmsol_n_out"), []() { return casadi::expm_n_out(); });
  mod.method(raw_method("has_expm"), [](const std::string& plugin) { return casadi::has_expm(plugin); });
  mod.method(raw_method("load_expm"), [](const std::string& plugin) { casadi::load_expm(plugin); });
  mod.method(raw_method("doc_expm"), [](const std::string& plugin) { return casadi::doc_expm(plugin); });

  mod.method(raw_method("dplesol"), [](const std::string& name, const std::string& solver, jlcxx::ArrayRef<std::string> keys, jlcxx::ArrayRef<Sparsity> values, const GenericType& options) {
    return casadi::dplesol(name, solver, SpDict(named_dict(keys, values, "DPLE")), generic_as_dict(options, "dplesol options"));
  });
  mod.method(raw_method("dplesol_mx"), [](const MX& A, const MX& V, const std::string& solver, const GenericType& options) {
    return casadi::dplesol(A, V, solver, generic_as_dict(options, "dplesol options"));
  });
  mod.method(raw_method("dplesol_mx_vector"), [](jlcxx::ArrayRef<MX> A, jlcxx::ArrayRef<MX> V, const std::string& solver, const GenericType& options) {
    return casadi::dplesol(to_vector(A), to_vector(V), solver, generic_as_dict(options, "dplesol options"));
  });
  mod.method(raw_method("dplesol_dm_vector"), [](jlcxx::ArrayRef<DM> A, jlcxx::ArrayRef<DM> V, const std::string& solver, const GenericType& options) {
    return casadi::dplesol(to_vector(A), to_vector(V), solver, generic_as_dict(options, "dplesol options"));
  });
  mod.method(raw_method("dple_in"), []() { return casadi::dple_in(); });
  mod.method(raw_method("dple_out"), []() { return casadi::dple_out(); });
  mod.method(raw_method("dple_in_index"), [](const std::int64_t index) { return casadi::dple_in(checked_index(index, "index")); });
  mod.method(raw_method("dple_out_index"), [](const std::int64_t index) { return casadi::dple_out(checked_index(index, "index")); });
  mod.method(raw_method("dple_n_in"), []() { return static_cast<std::int64_t>(casadi::dple_n_in()); });
  mod.method(raw_method("dple_n_out"), []() { return static_cast<std::int64_t>(casadi::dple_n_out()); });
  mod.method(raw_method("has_dple"), [](const std::string& plugin) { return casadi::has_dple(plugin); });
  mod.method(raw_method("load_dple"), [](const std::string& plugin) { casadi::load_dple(plugin); });
  mod.method(raw_method("doc_dple"), [](const std::string& plugin) { return casadi::doc_dple(plugin); });
}

} // namespace casadi_cxxwrap
