#include "casadi_cxxwrap.hpp"

namespace casadi_cxxwrap
{

void register_codegen_bindings(jlcxx::Module& mod)
{
  mod.method(raw_method("codegen_new"), [](const std::string& filename, const bool with_header, const bool main, const bool mex, const bool cpp) {
    return CodeGenerator(filename, make_codegen_options(with_header, main, mex, cpp));
  });
  mod.method(raw_method("codegen_new_options"), [](const std::string& filename, const GenericType& options) {
    return CodeGenerator(filename, generic_as_dict(options, "CodeGenerator options"));
  });
  mod.method(raw_method("codegen_add"), [](CodeGenerator& generator, const Function& f) {
    generator.add(f);
  });
  mod.method(raw_method("codegen_add"), [](CodeGenerator& generator, const Function& f, const bool with_jac_sparsity) {
    generator.add(f, with_jac_sparsity);
  });
  mod.method(raw_method("codegen_generate"), [](CodeGenerator& generator) {
    return generator.generate();
  });
  mod.method(raw_method("codegen_generate"), [](CodeGenerator& generator, const std::string& prefix) {
    return generator.generate(prefix);
  });
  mod.method(raw_method("codegen_dump"), [](CodeGenerator& generator) {
    return generator.dump();
  });
}

} // namespace casadi_cxxwrap
