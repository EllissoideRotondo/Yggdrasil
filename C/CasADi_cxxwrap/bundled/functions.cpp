#include "casadi_cxxwrap.hpp"

namespace casadi_cxxwrap
{

Function make_sx_function(const std::string& name, jlcxx::ArrayRef<SX> inputs, jlcxx::ArrayRef<SX> outputs)
{
  return Function(name, to_vector(inputs), to_vector(outputs));
}

Function make_sx_function_options(
  const std::string& name,
  jlcxx::ArrayRef<SX> inputs,
  jlcxx::ArrayRef<SX> outputs,
  const GenericType& options)
{
  return Function(name, to_vector(inputs), to_vector(outputs), generic_as_dict(options, "Function options"));
}

Function make_mx_function(const std::string& name, jlcxx::ArrayRef<MX> inputs, jlcxx::ArrayRef<MX> outputs)
{
  return Function(name, to_vector(inputs), to_vector(outputs));
}

Function make_mx_function_options(
  const std::string& name,
  jlcxx::ArrayRef<MX> inputs,
  jlcxx::ArrayRef<MX> outputs,
  const GenericType& options)
{
  return Function(name, to_vector(inputs), to_vector(outputs), generic_as_dict(options, "Function options"));
}

Function function_map(const Function& f, const std::int64_t n, const std::string& parallelization)
{
  return f.map(checked_nonnegative(n, "n"), parallelization);
}

Function function_map_threads(
  const Function& f,
  const std::int64_t n,
  const std::string& parallelization,
  const std::int64_t max_num_threads)
{
  return f.map(
    checked_nonnegative(n, "n"),
    parallelization,
    checked_nonnegative(max_num_threads, "max_num_threads"));
}

Function function_map_reduce_indices(
  const Function& f,
  const std::string& name,
  const std::string& parallelization,
  const std::int64_t n,
  jlcxx::ArrayRef<std::int64_t> reduce_in,
  jlcxx::ArrayRef<std::int64_t> reduce_out,
  const GenericType& options)
{
  return f.map(
    name,
    parallelization,
    checked_nonnegative(n, "n"),
    to_casadi_int_vector(reduce_in),
    to_casadi_int_vector(reduce_out),
    generic_as_dict(options, "map options"));
}

Function function_map_reduce_names(
  const Function& f,
  const std::string& name,
  const std::string& parallelization,
  const std::int64_t n,
  jlcxx::ArrayRef<std::string> reduce_in,
  jlcxx::ArrayRef<std::string> reduce_out,
  const GenericType& options)
{
  return f.map(
    name,
    parallelization,
    checked_nonnegative(n, "n"),
    to_vector(reduce_in),
    to_vector(reduce_out),
    generic_as_dict(options, "map options"));
}

Function function_map_reduce_mask(
  const Function& f,
  const std::int64_t n,
  jlcxx::ArrayRef<bool> reduce_in,
  jlcxx::ArrayRef<bool> reduce_out,
  const GenericType& options)
{
  return f.map(
    checked_nonnegative(n, "n"),
    to_vector(reduce_in),
    to_vector(reduce_out),
    generic_as_dict(options, "map options"));
}

Function function_mapaccum(
  const Function& f,
  const std::string& name,
  const std::int64_t n,
  const GenericType& options)
{
  return f.mapaccum(name, checked_nonnegative(n, "n"), generic_as_dict(options, "mapaccum options"));
}

Function function_mapaccum_naccum(
  const Function& f,
  const std::string& name,
  const std::int64_t n,
  const std::int64_t n_accum,
  const GenericType& options)
{
  return f.mapaccum(
    name,
    checked_nonnegative(n, "n"),
    checked_nonnegative(n_accum, "n_accum"),
    generic_as_dict(options, "mapaccum options"));
}

Function function_mapaccum_indices(
  const Function& f,
  const std::string& name,
  const std::int64_t n,
  jlcxx::ArrayRef<std::int64_t> accum_in,
  jlcxx::ArrayRef<std::int64_t> accum_out,
  const GenericType& options)
{
  return f.mapaccum(
    name,
    checked_nonnegative(n, "n"),
    to_casadi_int_vector(accum_in),
    to_casadi_int_vector(accum_out),
    generic_as_dict(options, "mapaccum options"));
}

Function function_mapaccum_names(
  const Function& f,
  const std::string& name,
  const std::int64_t n,
  jlcxx::ArrayRef<std::string> accum_in,
  jlcxx::ArrayRef<std::string> accum_out,
  const GenericType& options)
{
  return f.mapaccum(
    name,
    checked_nonnegative(n, "n"),
    to_vector(accum_in),
    to_vector(accum_out),
    generic_as_dict(options, "mapaccum options"));
}

Function function_mapaccum_default(const Function& f, const std::int64_t n, const GenericType& options)
{
  return f.mapaccum(checked_nonnegative(n, "n"), generic_as_dict(options, "mapaccum options"));
}

Function function_fold(const Function& f, const std::int64_t n, const GenericType& options)
{
  return f.fold(checked_nonnegative(n, "n"), generic_as_dict(options, "fold options"));
}

std::string function_serialize(const Function& f, const GenericType& options)
{
  return f.serialize(generic_as_dict(options, "serialize options"));
}

void function_save(const Function& f, const std::string& filename, const GenericType& options)
{
  f.save(filename, generic_as_dict(options, "save options"));
}

Function function_deserialize(const std::string& serialized)
{
  return Function::deserialize(serialized);
}

Function function_load(const std::string& filename)
{
  return Function::load(filename);
}

Importer importer_new(const std::string& name, const std::string& compiler, const GenericType& options)
{
  return Importer(name, compiler, generic_as_dict(options, "Importer options"));
}

Function external_name(const std::string& name, const GenericType& options)
{
  return casadi::external(name, generic_as_dict(options, "external options"));
}

Function external_binary(const std::string& name, const std::string& binary, const GenericType& options)
{
  return casadi::external(name, binary, generic_as_dict(options, "external options"));
}

Function external_importer(const std::string& name, const Importer& importer, const GenericType& options)
{
  return casadi::external(name, importer, generic_as_dict(options, "external options"));
}

void register_function_bindings(jlcxx::Module& mod)
{
  mod.method(raw_method("function_sx"), &make_sx_function);
  mod.method(raw_method("function_sx_options"), &make_sx_function_options);
  mod.method(raw_method("function_mx"), &make_mx_function);
  mod.method(raw_method("function_mx_options"), &make_mx_function_options);
  mod.method(raw_method("function_string"), &to_string<Function>);
  mod.method(raw_method("function_name"), [](const Function& f) { return f.name(); });
  mod.method(raw_method("function_n_in"), [](const Function& f) { return static_cast<std::int64_t>(f.n_in()); });
  mod.method(raw_method("function_n_out"), [](const Function& f) { return static_cast<std::int64_t>(f.n_out()); });
  mod.method(raw_method("function_name_in"), [](const Function& f, const std::int64_t index) {
    return f.name_in(checked_index(index, "index"));
  });
  mod.method(raw_method("function_name_out"), [](const Function& f, const std::int64_t index) {
    return f.name_out(checked_index(index, "index"));
  });
  mod.method(raw_method("function_size1_in"), [](const Function& f, const std::int64_t index) {
    return static_cast<std::int64_t>(f.size1_in(checked_index(index, "index")));
  });
  mod.method(raw_method("function_size2_in"), [](const Function& f, const std::int64_t index) {
    return static_cast<std::int64_t>(f.size2_in(checked_index(index, "index")));
  });
  mod.method(raw_method("function_size1_out"), [](const Function& f, const std::int64_t index) {
    return static_cast<std::int64_t>(f.size1_out(checked_index(index, "index")));
  });
  mod.method(raw_method("function_size2_out"), [](const Function& f, const std::int64_t index) {
    return static_cast<std::int64_t>(f.size2_out(checked_index(index, "index")));
  });
  mod.method(raw_method("function_nnz_in"), [](const Function& f, const std::int64_t index) {
    return static_cast<std::int64_t>(f.nnz_in(checked_index(index, "index")));
  });
  mod.method(raw_method("function_nnz_out"), [](const Function& f, const std::int64_t index) {
    return static_cast<std::int64_t>(f.nnz_out(checked_index(index, "index")));
  });
  mod.method(raw_method("function_numel_in"), [](const Function& f, const std::int64_t index) {
    return static_cast<std::int64_t>(f.numel_in(checked_index(index, "index")));
  });
  mod.method(raw_method("function_numel_out"), [](const Function& f, const std::int64_t index) {
    return static_cast<std::int64_t>(f.numel_out(checked_index(index, "index")));
  });
  mod.method(raw_method("function_sparsity_in"), [](const Function& f, const std::int64_t index) {
    return Sparsity(f.sparsity_in(checked_index(index, "index")));
  });
  mod.method(raw_method("function_sparsity_out"), [](const Function& f, const std::int64_t index) {
    return Sparsity(f.sparsity_out(checked_index(index, "index")));
  });
  mod.method(raw_method("function_call_dm"), [](const Function& f, jlcxx::ArrayRef<DM> args) {
    return f(to_vector(args));
  });
  mod.method(raw_method("function_call_sx"), [](const Function& f, jlcxx::ArrayRef<SX> args) {
    return f(to_vector(args));
  });
  mod.method(raw_method("function_call_mx"), [](const Function& f, jlcxx::ArrayRef<MX> args) {
    return f(to_vector(args));
  });
  mod.method(raw_method("function_jacobian"), [](const Function& f) { return f.jacobian(); });
  mod.method(raw_method("function_generate"), [](const Function& f, const std::string& filename) {
    return f.generate(filename);
  });
  mod.method(raw_method("function_generate"), [](const Function& f, const std::string& filename, const bool with_header, const bool main, const bool mex, const bool cpp) {
    return f.generate(filename, make_codegen_options(with_header, main, mex, cpp));
  });
  mod.method(raw_method("function_generate_options"), [](const Function& f, const std::string& filename, const GenericType& options) {
    return f.generate(filename, generic_as_dict(options, "generate options"));
  });
  mod.method(raw_method("function_map"), &function_map);
  mod.method(raw_method("function_map_threads"), &function_map_threads);
  mod.method(raw_method("function_map_reduce_indices"), &function_map_reduce_indices);
  mod.method(raw_method("function_map_reduce_names"), &function_map_reduce_names);
  mod.method(raw_method("function_map_reduce_mask"), &function_map_reduce_mask);
  mod.method(raw_method("function_mapaccum"), &function_mapaccum);
  mod.method(raw_method("function_mapaccum_naccum"), &function_mapaccum_naccum);
  mod.method(raw_method("function_mapaccum_indices"), &function_mapaccum_indices);
  mod.method(raw_method("function_mapaccum_names"), &function_mapaccum_names);
  mod.method(raw_method("function_mapaccum_default"), &function_mapaccum_default);
  mod.method(raw_method("function_fold"), &function_fold);
  mod.method(raw_method("function_serialize"), &function_serialize);
  mod.method(raw_method("function_save"), &function_save);
  mod.method(raw_method("function_deserialize"), &function_deserialize);
  mod.method(raw_method("function_load"), &function_load);

  mod.method(raw_method("importer_new"), &importer_new);
  mod.method(raw_method("importer_string"), &to_string<Importer>);
  mod.method(raw_method("importer_has_plugin"), [](const std::string& plugin) { return Importer::has_plugin(plugin); });
  mod.method(raw_method("importer_load_plugin"), [](const std::string& plugin) { Importer::load_plugin(plugin); });
  mod.method(raw_method("importer_doc"), [](const std::string& plugin) { return Importer::doc(plugin); });
  mod.method(raw_method("importer_plugin_name"), [](const Importer& importer) { return importer.plugin_name(); });
  mod.method(raw_method("importer_has_function"), [](const Importer& importer, const std::string& name) {
    return importer.has_function(name);
  });
  mod.method(raw_method("importer_has_meta"), [](const Importer& importer, const std::string& command, const std::int64_t index) {
    return importer.has_meta(command, static_cast<casadi_int>(index));
  });
  mod.method(raw_method("importer_get_meta"), [](const Importer& importer, const std::string& command, const std::int64_t index) {
    return importer.get_meta(command, static_cast<casadi_int>(index));
  });
  mod.method(raw_method("importer_inlined"), [](const Importer& importer, const std::string& name) {
    return importer.inlined(name);
  });
  mod.method(raw_method("importer_body"), [](const Importer& importer, const std::string& name) {
    return importer.body(name);
  });
  mod.method(raw_method("importer_library"), [](const Importer& importer) { return importer.library(); });
  mod.method(raw_method("external"), &external_name);
  mod.method(raw_method("external_binary"), &external_binary);
  mod.method(raw_method("external_importer"), &external_importer);
}

} // namespace casadi_cxxwrap
