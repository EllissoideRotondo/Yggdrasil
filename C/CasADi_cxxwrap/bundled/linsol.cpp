#include "casadi_cxxwrap.hpp"

namespace casadi_cxxwrap
{

void register_linsol_bindings(jlcxx::Module& mod)
{
  mod.method(raw_method("linsol_new"), [](const std::string& name, const std::string& solver, const Sparsity& sp, const GenericType& options) {
    return casadi::Linsol(name, solver, sp, generic_as_dict(options, "Linsol options"));
  });
  mod.method(raw_method("linsol_string"), &to_string<casadi::Linsol>);
  mod.method(raw_method("linsol_plugin_name"), [](const casadi::Linsol& linsol) { return linsol.plugin_name(); });
  mod.method(raw_method("linsol_solve_dm"), [](const casadi::Linsol& linsol, const DM& A, const DM& b, const bool tr) {
    return linsol.solve(A, b, tr);
  });
  mod.method(raw_method("linsol_solve_mx"), [](const casadi::Linsol& linsol, const MX& A, const MX& b, const bool tr) {
    return linsol.solve(A, b, tr);
  });
  mod.method(raw_method("linsol_sfact"), [](const casadi::Linsol& linsol, const DM& A) { linsol.sfact(A); });
  mod.method(raw_method("linsol_nfact"), [](const casadi::Linsol& linsol, const DM& A) { linsol.nfact(A); });
  mod.method(raw_method("linsol_neig"), [](const casadi::Linsol& linsol, const DM& A) {
    return static_cast<std::int64_t>(linsol.neig(A));
  });
  mod.method(raw_method("linsol_rank"), [](const casadi::Linsol& linsol, const DM& A) {
    return static_cast<std::int64_t>(linsol.rank(A));
  });

  mod.method(raw_method("has_linsol"), [](const std::string& plugin) { return casadi::has_linsol(plugin); });
  mod.method(raw_method("load_linsol"), [](const std::string& plugin) { casadi::load_linsol(plugin); });
  mod.method(raw_method("doc_linsol"), [](const std::string& plugin) { return casadi::doc_linsol(plugin); });
}

} // namespace casadi_cxxwrap
