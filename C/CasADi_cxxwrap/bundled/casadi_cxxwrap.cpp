#include "casadi_cxxwrap.hpp"

namespace
{

void register_types(jlcxx::Module& mod)
{
  using namespace casadi_cxxwrap;

  mod.add_type<SX>("SX");
  mod.add_type<DM>("DM");
  mod.add_type<MX>("MX");
  mod.add_type<Function>("CasadiFunction");
  mod.add_type<GenericType>("GenericType");
  mod.add_type<Importer>("Importer");
  mod.add_type<Opti>("Opti")
    .constructor<>()
    .constructor<const std::string&>();
  mod.add_type<OptiAdvanced>("OptiAdvanced");
  mod.add_type<OptiSol>("OptiSol");
  mod.add_type<DaeBuilder>("DaeBuilder")
    .constructor<>()
    .constructor<const std::string&>()
    .constructor<const std::string&, const std::string&>();
  mod.add_type<NlpBuilder>("NlpBuilder")
    .constructor<>();
  mod.add_type<Sparsity>("Sparsity");
  mod.add_type<CodeGenerator>("CodeGenerator")
    .constructor<const std::string&>();
}

} // namespace

JLCXX_MODULE define_julia_module(jlcxx::Module& mod)
{
  register_types(mod);

  casadi_cxxwrap::register_matrix_bindings(mod);
  casadi_cxxwrap::register_function_bindings(mod);
  casadi_cxxwrap::register_callback_bindings(mod);
  casadi_cxxwrap::register_generic_type_bindings(mod);
  casadi_cxxwrap::register_sparsity_bindings(mod);
  casadi_cxxwrap::register_factory_bindings(mod);
  casadi_cxxwrap::register_codegen_bindings(mod);
  casadi_cxxwrap::register_interpolant_bindings(mod);
  casadi_cxxwrap::register_opti_bindings(mod);
  casadi_cxxwrap::register_builder_bindings(mod);
}
