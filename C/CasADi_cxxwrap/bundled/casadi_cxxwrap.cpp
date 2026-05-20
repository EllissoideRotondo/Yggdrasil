#include "casadi_cxxwrap.hpp"

namespace
{

void register_types(jlcxx::Module& mod)
{
  using namespace casadi_cxxwrap;

  mod.add_type<SX>("SX");
  mod.add_type<DM>("DM");
  mod.add_type<DMDict>("DMDict");
  mod.add_type<SXDict>("SXDict");
  mod.add_type<MXDict>("MXDict");
  mod.add_type<MX>("MX");
  mod.add_type<Function>("CasadiFunction");
  mod.add_type<GenericType>("GenericType");
  mod.add_type<Importer>("Importer");
  mod.add_type<Resource>("Resource")
    .constructor<>()
    .constructor<const std::string&>();
  mod.add_type<XmlNode>("XmlNode")
    .constructor<>();
  mod.add_type<XmlFile>("XmlFile")
    .constructor<>()
    .constructor<const std::string&>();
  mod.add_type<StringSerializer>("StringSerializer")
    .constructor<>();
  mod.add_type<FileSerializer>("FileSerializer")
    .constructor<const std::string&>();
  mod.add_type<StringDeserializer>("StringDeserializer")
    .constructor<>()
    .constructor<const std::string&>();
  mod.add_type<FileDeserializer>("FileDeserializer")
    .constructor<const std::string&>();
  mod.add_type<SXDictStatsResult>("SXDictStatsResult");
  mod.add_type<MXDictStatsResult>("MXDictStatsResult");
  mod.add_type<SXDictDaeMapResult>("SXDictDaeMapResult");
  mod.add_type<MXDictDaeMapResult>("MXDictDaeMapResult");
  mod.add_type<SXSimpleBoundsResult>("SXSimpleBoundsResult");
  mod.add_type<MXSimpleBoundsResult>("MXSimpleBoundsResult");
  mod.add_type<SparsityMappingResult>("SparsityMappingResult");
  mod.add_type<IntVectorPairResult>("IntVectorPairResult");
  mod.add_type<SparsityLdlResult>("SparsityLdlResult");
  mod.add_type<SparsityQrResult>("SparsityQrResult");
  mod.add_type<SparsitySccResult>("SparsitySccResult");
  mod.add_type<SparsityBtfResult>("SparsityBtfResult");
  mod.add_type<SXSharedResult>("SXSharedResult");
  mod.add_type<MXSharedResult>("MXSharedResult");
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
  mod.add_type<casadi::Linsol>("Linsol");
  mod.add_type<Sparsity>("Sparsity");
  mod.add_type<CodeGenerator>("CodeGenerator")
    .constructor<const std::string&>();
}

} // namespace

JLCXX_MODULE define_julia_module(jlcxx::Module& mod)
{
  register_types(mod);

  casadi_cxxwrap::register_dm_dict_bindings(mod);
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
  casadi_cxxwrap::register_linsol_bindings(mod);
  casadi_cxxwrap::register_utility_bindings(mod);
  casadi_cxxwrap::register_serialization_bindings(mod);
}
