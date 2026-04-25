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

void register_factory_bindings(jlcxx::Module& mod)
{
  mod.method(raw_method("nlpsol_sx"), &nlpsol_sx);
  mod.method(raw_method("nlpsol_mx"), &nlpsol_mx);
  mod.method(raw_method("nlpsol_in"), []() { return casadi::nlpsol_in(); });
  mod.method(raw_method("nlpsol_out"), []() { return casadi::nlpsol_out(); });
  mod.method(raw_method("has_nlpsol"), [](const std::string& plugin) { return casadi::has_nlpsol(plugin); });
  mod.method(raw_method("load_nlpsol"), [](const std::string& plugin) { casadi::load_nlpsol(plugin); });
  mod.method(raw_method("doc_nlpsol"), [](const std::string& plugin) { return casadi::doc_nlpsol(plugin); });

  mod.method(raw_method("qpsol_sx"), &qpsol_sx);
  mod.method(raw_method("qpsol_mx"), &qpsol_mx);
  mod.method(raw_method("conic_sparsity"), &conic_sp);
  mod.method(raw_method("conic_in"), []() { return casadi::conic_in(); });
  mod.method(raw_method("conic_out"), []() { return casadi::conic_out(); });
  mod.method(raw_method("has_conic"), [](const std::string& plugin) { return casadi::has_conic(plugin); });
  mod.method(raw_method("load_conic"), [](const std::string& plugin) { casadi::load_conic(plugin); });
  mod.method(raw_method("doc_conic"), [](const std::string& plugin) { return casadi::doc_conic(plugin); });

  mod.method(raw_method("rootfinder_sx"), &rootfinder_sx);
  mod.method(raw_method("rootfinder_mx"), &rootfinder_mx);
  mod.method(raw_method("rootfinder_in"), []() { return casadi::rootfinder_in(); });
  mod.method(raw_method("rootfinder_out"), []() { return casadi::rootfinder_out(); });
  mod.method(raw_method("has_rootfinder"), [](const std::string& plugin) { return casadi::has_rootfinder(plugin); });
  mod.method(raw_method("load_rootfinder"), [](const std::string& plugin) { casadi::load_rootfinder(plugin); });
  mod.method(raw_method("doc_rootfinder"), [](const std::string& plugin) { return casadi::doc_rootfinder(plugin); });

  mod.method(raw_method("integrator_sx"), &integrator_sx);
  mod.method(raw_method("integrator_mx"), &integrator_mx);
  mod.method(raw_method("integrator_sx_tf"), &integrator_sx_tf);
  mod.method(raw_method("integrator_mx_tf"), &integrator_mx_tf);
  mod.method(raw_method("integrator_sx_tout"), &integrator_sx_tout);
  mod.method(raw_method("integrator_mx_tout"), &integrator_mx_tout);
  mod.method(raw_method("integrator_in"), []() { return casadi::integrator_in(); });
  mod.method(raw_method("integrator_out"), []() { return casadi::integrator_out(); });
  mod.method(raw_method("has_integrator"), [](const std::string& plugin) { return casadi::has_integrator(plugin); });
  mod.method(raw_method("load_integrator"), [](const std::string& plugin) { casadi::load_integrator(plugin); });
  mod.method(raw_method("doc_integrator"), [](const std::string& plugin) { return casadi::doc_integrator(plugin); });
}

} // namespace casadi_cxxwrap
