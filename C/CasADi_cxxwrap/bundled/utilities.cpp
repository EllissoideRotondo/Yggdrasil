#include "casadi_cxxwrap.hpp"

namespace casadi_cxxwrap
{

template<typename T>
std::vector<std::string> typed_dict_keys(const std::map<std::string, T>& dict)
{
  std::vector<std::string> keys;
  keys.reserve(dict.size());
  for(const auto& entry : dict)
  {
    keys.push_back(entry.first);
  }
  return keys;
}

template<typename T>
std::vector<T> typed_dict_values(const std::map<std::string, T>& dict)
{
  std::vector<T> values;
  values.reserve(dict.size());
  for(const auto& entry : dict)
  {
    values.push_back(entry.second);
  }
  return values;
}

template<typename T>
bool typed_dict_has(const std::map<std::string, T>& dict, const std::string& key)
{
  return dict.find(key) != dict.end();
}

template<typename T>
T typed_dict_get(const std::map<std::string, T>& dict, const std::string& key, const char* name)
{
  const auto found = dict.find(key);
  if(found == dict.end())
  {
    throw std::out_of_range(std::string(name) + " key not found: " + key);
  }
  return found->second;
}

SXDict sx_dict_from_arrays(jlcxx::ArrayRef<std::string> keys, jlcxx::ArrayRef<SX> values)
{
  return SXDict(named_dict(keys, values, "SXDict"));
}

MXDict mx_dict_from_arrays(jlcxx::ArrayRef<std::string> keys, jlcxx::ArrayRef<MX> values)
{
  return MXDict(named_dict(keys, values, "MXDict"));
}

DMDict dm_dict_from_arrays(jlcxx::ArrayRef<std::string> keys, jlcxx::ArrayRef<DM> values)
{
  return DMDict(named_dict(keys, values, "DMDict"));
}

SXDictStatsResult sx_dae_reduce_index_dict(const SXDict& dae, const GenericType& options)
{
  Dict stats;
  SXDict value = casadi::dae_reduce_index(dae, stats, generic_as_dict(options, "dae_reduce_index options"));
  return {value, GenericType(stats)};
}

MXDictStatsResult mx_dae_reduce_index_dict(const MXDict& dae, const GenericType& options)
{
  Dict stats;
  MXDict value = casadi::dae_reduce_index(dae, stats, generic_as_dict(options, "dae_reduce_index options"));
  return {value, GenericType(stats)};
}

SXDictDaeMapResult sx_dae_map_semi_expl_dict(const SXDict& dae, const SXDict& dae_red)
{
  Function state_to_orig;
  Function phi;
  SXDict value = casadi::dae_map_semi_expl(dae, dae_red, state_to_orig, phi);
  return {value, state_to_orig, phi};
}

MXDictDaeMapResult mx_dae_map_semi_expl_dict(const MXDict& dae, const MXDict& dae_red)
{
  Function state_to_orig;
  Function phi;
  MXDict value = casadi::dae_map_semi_expl(dae, dae_red, state_to_orig, phi);
  return {value, state_to_orig, phi};
}

Function sx_dae_init_gen_dict(
  const SXDict& dae,
  const SXDict& dae_red,
  const std::string& init_solver,
  const DMDict& init_strength,
  const GenericType& init_solver_options)
{
  return casadi::dae_init_gen(
    dae,
    dae_red,
    init_solver,
    init_strength,
    generic_as_dict(init_solver_options, "dae_init_gen solver options"));
}

Function mx_dae_init_gen_dict(
  const MXDict& dae,
  const MXDict& dae_red,
  const std::string& init_solver,
  const DMDict& init_strength,
  const GenericType& init_solver_options)
{
  return casadi::dae_init_gen(
    dae,
    dae_red,
    init_solver,
    init_strength,
    generic_as_dict(init_solver_options, "dae_init_gen solver options"));
}

SXSimpleBoundsResult sx_detect_simple_bounds(
  const SX& x,
  const SX& p,
  const SX& g,
  const SX& lbg,
  const SX& ubg)
{
  std::vector<casadi_int> gi;
  SX lbx;
  SX ubx;
  Function lam_forward;
  Function lam_backward;
  casadi::detect_simple_bounds(x, p, g, lbg, ubg, gi, lbx, ubx, lam_forward, lam_backward);
  return {from_casadi_int_vector(gi), lbx, ubx, lam_forward, lam_backward};
}

MXSimpleBoundsResult mx_detect_simple_bounds(
  const MX& x,
  const MX& p,
  const MX& g,
  const MX& lbg,
  const MX& ubg)
{
  std::vector<casadi_int> gi;
  MX lbx;
  MX ubx;
  Function lam_forward;
  Function lam_backward;
  casadi::detect_simple_bounds(x, p, g, lbg, ubg, gi, lbx, ubx, lam_forward, lam_backward);
  return {from_casadi_int_vector(gi), lbx, ubx, lam_forward, lam_backward};
}

std::vector<DM> collocation_interpolators_raw(jlcxx::ArrayRef<double> tau)
{
  std::vector<std::vector<double>> C;
  std::vector<double> D;
  casadi::collocation_interpolators(to_vector(tau), C, D);
  return {DM(C), DM(D)};
}

std::vector<DM> collocation_coeff_raw(jlcxx::ArrayRef<double> tau)
{
  DM C;
  DM D;
  DM B;
  casadi::collocation_coeff(to_vector(tau), C, D, B);
  return {C, D, B};
}

Function simple_irk_raw(
  const Function& f,
  const std::int64_t N,
  const std::int64_t order,
  const std::string& scheme,
  const std::string& solver,
  const GenericType& solver_options)
{
  return casadi::simpleIRK(
    f,
    checked_positive(N, "N"),
    checked_positive(order, "order"),
    scheme,
    solver,
    generic_as_dict(solver_options, "simpleIRK solver options"));
}

Function simple_integrator_raw(const Function& f, const std::string& integrator, const GenericType& integrator_options)
{
  return casadi::simpleIntegrator(f, integrator, generic_as_dict(integrator_options, "simpleIntegrator options"));
}

void register_utility_bindings(jlcxx::Module& mod)
{
  mod.method(raw_method("sx_dict_new"), &sx_dict_from_arrays);
  mod.method(raw_method("sx_dict_size"), [](const SXDict& dict) { return static_cast<std::int64_t>(dict.size()); });
  mod.method(raw_method("sx_dict_keys"), &typed_dict_keys<SX>);
  mod.method(raw_method("sx_dict_values"), &typed_dict_values<SX>);
  mod.method(raw_method("sx_dict_has"), &typed_dict_has<SX>);
  mod.method(raw_method("sx_dict_get"), [](const SXDict& dict, const std::string& key) {
    return typed_dict_get(dict, key, "SXDict");
  });

  mod.method(raw_method("mx_dict_new"), &mx_dict_from_arrays);
  mod.method(raw_method("mx_dict_size"), [](const MXDict& dict) { return static_cast<std::int64_t>(dict.size()); });
  mod.method(raw_method("mx_dict_keys"), &typed_dict_keys<MX>);
  mod.method(raw_method("mx_dict_values"), &typed_dict_values<MX>);
  mod.method(raw_method("mx_dict_has"), &typed_dict_has<MX>);
  mod.method(raw_method("mx_dict_get"), [](const MXDict& dict, const std::string& key) {
    return typed_dict_get(dict, key, "MXDict");
  });

  mod.method(raw_method("dm_dict_new"), &dm_dict_from_arrays);
  mod.method(raw_method("dm_dict_size"), [](const DMDict& dict) { return static_cast<std::int64_t>(dict.size()); });
  mod.method(raw_method("dm_dict_keys"), &typed_dict_keys<DM>);
  mod.method(raw_method("dm_dict_values"), &typed_dict_values<DM>);
  mod.method(raw_method("dm_dict_has"), &typed_dict_has<DM>);
  mod.method(raw_method("dm_dict_get"), [](const DMDict& dict, const std::string& key) {
    return typed_dict_get(dict, key, "DMDict");
  });

  mod.method(raw_method("sx_dict_stats_value"), [](const SXDictStatsResult& result) { return result.value; });
  mod.method(raw_method("sx_dict_stats_stats"), [](const SXDictStatsResult& result) { return result.stats; });
  mod.method(raw_method("mx_dict_stats_value"), [](const MXDictStatsResult& result) { return result.value; });
  mod.method(raw_method("mx_dict_stats_stats"), [](const MXDictStatsResult& result) { return result.stats; });

  mod.method(raw_method("sx_dae_map_value"), [](const SXDictDaeMapResult& result) { return result.value; });
  mod.method(raw_method("sx_dae_map_state_to_orig"), [](const SXDictDaeMapResult& result) { return result.state_to_orig; });
  mod.method(raw_method("sx_dae_map_phi"), [](const SXDictDaeMapResult& result) { return result.phi; });
  mod.method(raw_method("mx_dae_map_value"), [](const MXDictDaeMapResult& result) { return result.value; });
  mod.method(raw_method("mx_dae_map_state_to_orig"), [](const MXDictDaeMapResult& result) { return result.state_to_orig; });
  mod.method(raw_method("mx_dae_map_phi"), [](const MXDictDaeMapResult& result) { return result.phi; });

  mod.method(raw_method("sx_simple_bounds_gi"), [](const SXSimpleBoundsResult& result) { return result.gi; });
  mod.method(raw_method("sx_simple_bounds_lbx"), [](const SXSimpleBoundsResult& result) { return result.lbx; });
  mod.method(raw_method("sx_simple_bounds_ubx"), [](const SXSimpleBoundsResult& result) { return result.ubx; });
  mod.method(raw_method("sx_simple_bounds_lam_forward"), [](const SXSimpleBoundsResult& result) { return result.lam_forward; });
  mod.method(raw_method("sx_simple_bounds_lam_backward"), [](const SXSimpleBoundsResult& result) { return result.lam_backward; });
  mod.method(raw_method("mx_simple_bounds_gi"), [](const MXSimpleBoundsResult& result) { return result.gi; });
  mod.method(raw_method("mx_simple_bounds_lbx"), [](const MXSimpleBoundsResult& result) { return result.lbx; });
  mod.method(raw_method("mx_simple_bounds_ubx"), [](const MXSimpleBoundsResult& result) { return result.ubx; });
  mod.method(raw_method("mx_simple_bounds_lam_forward"), [](const MXSimpleBoundsResult& result) { return result.lam_forward; });
  mod.method(raw_method("mx_simple_bounds_lam_backward"), [](const MXSimpleBoundsResult& result) { return result.lam_backward; });

  mod.method(raw_method("resource_path"), [](const Resource& resource) { return std::string(resource.path()); });
  mod.method(raw_method("resource_change_option"), [](Resource& resource, const std::string& option_name, const GenericType& option_value) {
    resource.change_option(option_name, option_value);
  });

  mod.method(raw_method("casadi_meta_version"), []() { return std::string(casadi::CasadiMeta::version()); });
  mod.method(raw_method("casadi_meta_git_revision"), []() { return std::string(casadi::CasadiMeta::git_revision()); });
  mod.method(raw_method("casadi_meta_git_describe"), []() { return std::string(casadi::CasadiMeta::git_describe()); });
  mod.method(raw_method("casadi_meta_feature_list"), []() { return std::string(casadi::CasadiMeta::feature_list()); });
  mod.method(raw_method("casadi_meta_build_type"), []() { return std::string(casadi::CasadiMeta::build_type()); });
  mod.method(raw_method("casadi_meta_compiler_id"), []() { return std::string(casadi::CasadiMeta::compiler_id()); });
  mod.method(raw_method("casadi_meta_compiler"), []() { return std::string(casadi::CasadiMeta::compiler()); });
  mod.method(raw_method("casadi_meta_compiler_flags"), []() { return std::string(casadi::CasadiMeta::compiler_flags()); });
  mod.method(raw_method("casadi_meta_modules"), []() { return std::string(casadi::CasadiMeta::modules()); });
  mod.method(raw_method("casadi_meta_plugins"), []() { return std::string(casadi::CasadiMeta::plugins()); });
  mod.method(raw_method("casadi_meta_install_prefix"), []() { return std::string(casadi::CasadiMeta::install_prefix()); });
  mod.method(raw_method("casadi_meta_shared_library_prefix"), []() { return std::string(casadi::CasadiMeta::shared_library_prefix()); });
  mod.method(raw_method("casadi_meta_shared_library_suffix"), []() { return std::string(casadi::CasadiMeta::shared_library_suffix()); });
  mod.method(raw_method("casadi_meta_object_file_suffix"), []() { return std::string(casadi::CasadiMeta::object_file_suffix()); });

  mod.method(raw_method("global_options_set_simplification_on_the_fly"), [](const bool value) {
    casadi::GlobalOptions::setSimplificationOnTheFly(value);
  });
  mod.method(raw_method("global_options_get_simplification_on_the_fly"), []() {
    return casadi::GlobalOptions::getSimplificationOnTheFly();
  });
  mod.method(raw_method("global_options_set_hierarchical_sparsity"), [](const bool value) {
    casadi::GlobalOptions::setHierarchicalSparsity(value);
  });
  mod.method(raw_method("global_options_get_hierarchical_sparsity"), []() {
    return casadi::GlobalOptions::getHierarchicalSparsity();
  });
  mod.method(raw_method("global_options_set_casadi_path"), [](const std::string& value) {
    casadi::GlobalOptions::setCasadiPath(value);
  });
  mod.method(raw_method("global_options_get_casadi_path"), []() {
    return casadi::GlobalOptions::getCasadiPath();
  });
  mod.method(raw_method("global_options_set_casadi_include_path"), [](const std::string& value) {
    casadi::GlobalOptions::setCasadiIncludePath(value);
  });
  mod.method(raw_method("global_options_get_casadi_include_path"), []() {
    return casadi::GlobalOptions::getCasadiIncludePath();
  });
  mod.method(raw_method("global_options_set_max_num_dir"), [](const std::int64_t value) {
    casadi::GlobalOptions::setMaxNumDir(checked_nonnegative(value, "max_num_dir"));
  });
  mod.method(raw_method("global_options_get_max_num_dir"), []() {
    return static_cast<std::int64_t>(casadi::GlobalOptions::getMaxNumDir());
  });
  mod.method(raw_method("global_options_set_copy_elision_min_size"), [](const std::int64_t value) {
    casadi::GlobalOptions::setCopyElisionMinSize(checked_casadi_int(value, "copy_elision_min_size"));
  });
  mod.method(raw_method("global_options_get_copy_elision_min_size"), []() {
    return static_cast<std::int64_t>(casadi::GlobalOptions::getCopyElisionMinSize());
  });

  mod.method(raw_method("collocation_points"), [](const std::int64_t order, const std::string& scheme) {
    return casadi::collocation_points(checked_positive(order, "order"), scheme);
  });
  mod.method(raw_method("collocation_interpolators"), &collocation_interpolators_raw);
  mod.method(raw_method("collocation_coeff"), &collocation_coeff_raw);
  mod.method(raw_method("simple_rk"), [](const Function& f, const std::int64_t N, const std::int64_t order) {
    return casadi::simpleRK(f, checked_positive(N, "N"), checked_positive(order, "order"));
  });
  mod.method(raw_method("simple_irk"), &simple_irk_raw);
  mod.method(raw_method("simple_integrator"), &simple_integrator_raw);

  mod.method(raw_method("sx_dae_reduce_index_dict"), &sx_dae_reduce_index_dict);
  mod.method(raw_method("mx_dae_reduce_index_dict"), &mx_dae_reduce_index_dict);
  mod.method(raw_method("sx_dae_reduce_index"), [](jlcxx::ArrayRef<std::string> keys, jlcxx::ArrayRef<SX> values, const GenericType& options) {
    return sx_dae_reduce_index_dict(sx_dict_from_arrays(keys, values), options);
  });
  mod.method(raw_method("mx_dae_reduce_index"), [](jlcxx::ArrayRef<std::string> keys, jlcxx::ArrayRef<MX> values, const GenericType& options) {
    return mx_dae_reduce_index_dict(mx_dict_from_arrays(keys, values), options);
  });
  mod.method(raw_method("sx_dae_map_semi_expl_dict"), &sx_dae_map_semi_expl_dict);
  mod.method(raw_method("mx_dae_map_semi_expl_dict"), &mx_dae_map_semi_expl_dict);
  mod.method(raw_method("sx_dae_map_semi_expl"), [](
    jlcxx::ArrayRef<std::string> dae_keys,
    jlcxx::ArrayRef<SX> dae_values,
    jlcxx::ArrayRef<std::string> dae_red_keys,
    jlcxx::ArrayRef<SX> dae_red_values) {
    return sx_dae_map_semi_expl_dict(sx_dict_from_arrays(dae_keys, dae_values), sx_dict_from_arrays(dae_red_keys, dae_red_values));
  });
  mod.method(raw_method("mx_dae_map_semi_expl"), [](
    jlcxx::ArrayRef<std::string> dae_keys,
    jlcxx::ArrayRef<MX> dae_values,
    jlcxx::ArrayRef<std::string> dae_red_keys,
    jlcxx::ArrayRef<MX> dae_red_values) {
    return mx_dae_map_semi_expl_dict(mx_dict_from_arrays(dae_keys, dae_values), mx_dict_from_arrays(dae_red_keys, dae_red_values));
  });
  mod.method(raw_method("sx_dae_init_gen_dict"), &sx_dae_init_gen_dict);
  mod.method(raw_method("mx_dae_init_gen_dict"), &mx_dae_init_gen_dict);
  mod.method(raw_method("sx_dae_init_gen"), [](
    jlcxx::ArrayRef<std::string> dae_keys,
    jlcxx::ArrayRef<SX> dae_values,
    jlcxx::ArrayRef<std::string> dae_red_keys,
    jlcxx::ArrayRef<SX> dae_red_values,
    const std::string& init_solver,
    jlcxx::ArrayRef<std::string> init_strength_keys,
    jlcxx::ArrayRef<DM> init_strength_values,
    const GenericType& init_solver_options) {
    return sx_dae_init_gen_dict(
      sx_dict_from_arrays(dae_keys, dae_values),
      sx_dict_from_arrays(dae_red_keys, dae_red_values),
      init_solver,
      dm_dict_from_arrays(init_strength_keys, init_strength_values),
      init_solver_options);
  });
  mod.method(raw_method("mx_dae_init_gen"), [](
    jlcxx::ArrayRef<std::string> dae_keys,
    jlcxx::ArrayRef<MX> dae_values,
    jlcxx::ArrayRef<std::string> dae_red_keys,
    jlcxx::ArrayRef<MX> dae_red_values,
    const std::string& init_solver,
    jlcxx::ArrayRef<std::string> init_strength_keys,
    jlcxx::ArrayRef<DM> init_strength_values,
    const GenericType& init_solver_options) {
    return mx_dae_init_gen_dict(
      mx_dict_from_arrays(dae_keys, dae_values),
      mx_dict_from_arrays(dae_red_keys, dae_red_values),
      init_solver,
      dm_dict_from_arrays(init_strength_keys, init_strength_values),
      init_solver_options);
  });
  mod.method(raw_method("sx_detect_simple_bounds"), &sx_detect_simple_bounds);
  mod.method(raw_method("mx_detect_simple_bounds"), &mx_detect_simple_bounds);
}

} // namespace casadi_cxxwrap
