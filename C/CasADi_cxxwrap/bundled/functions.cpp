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

Function make_sx_function_named(
  const std::string& name,
  jlcxx::ArrayRef<SX> inputs,
  jlcxx::ArrayRef<SX> outputs,
  jlcxx::ArrayRef<std::string> input_names,
  jlcxx::ArrayRef<std::string> output_names,
  const GenericType& options)
{
  return Function(
    name,
    to_vector(inputs),
    to_vector(outputs),
    to_vector(input_names),
    to_vector(output_names),
    generic_as_dict(options, "Function options"));
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

Function make_mx_function_named(
  const std::string& name,
  jlcxx::ArrayRef<MX> inputs,
  jlcxx::ArrayRef<MX> outputs,
  jlcxx::ArrayRef<std::string> input_names,
  jlcxx::ArrayRef<std::string> output_names,
  const GenericType& options)
{
  return Function(
    name,
    to_vector(inputs),
    to_vector(outputs),
    to_vector(input_names),
    to_vector(output_names),
    generic_as_dict(options, "Function options"));
}

Function function_from_file(const std::string& filename)
{
  return Function(filename);
}

std::vector<std::int64_t> function_size_pair(const std::pair<casadi_int, casadi_int>& value)
{
  return {
    static_cast<std::int64_t>(value.first),
    static_cast<std::int64_t>(value.second)};
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

Function function_slice(
  const Function& f,
  const std::string& name,
  jlcxx::ArrayRef<std::int64_t> order_in,
  jlcxx::ArrayRef<std::int64_t> order_out,
  const GenericType& options)
{
  return f.slice(
    name,
    to_casadi_int_vector(order_in),
    to_casadi_int_vector(order_out),
    generic_as_dict(options, "slice options"));
}

Function function_factory(
  const Function& f,
  const std::string& name,
  jlcxx::ArrayRef<std::string> inputs,
  jlcxx::ArrayRef<std::string> outputs,
  const GenericType& options)
{
  return f.factory(
    name,
    to_vector(inputs),
    to_vector(outputs),
    Function::AuxOut(),
    generic_as_dict(options, "factory options"));
}

Function function_conditional(
  const std::string& name,
  jlcxx::ArrayRef<Function> functions,
  const Function& default_function,
  const GenericType& options)
{
  return Function::conditional(
    name,
    to_vector(functions),
    default_function,
    generic_as_dict(options, "conditional options"));
}

Function function_conditional_single(const std::string& name, const Function& f, const GenericType& options)
{
  return Function::conditional(name, f, generic_as_dict(options, "conditional options"));
}

Function function_if_else(
  const std::string& name,
  const Function& true_function,
  const Function& false_function,
  const GenericType& options)
{
  return Function::if_else(name, true_function, false_function, generic_as_dict(options, "if_else options"));
}

Function function_jit(
  const std::string& name,
  const std::string& body,
  jlcxx::ArrayRef<std::string> input_names,
  jlcxx::ArrayRef<std::string> output_names,
  const GenericType& options)
{
  return Function::jit(
    name,
    body,
    to_vector(input_names),
    to_vector(output_names),
    generic_as_dict(options, "jit options"));
}

Function function_jit_sparsity(
  const std::string& name,
  const std::string& body,
  jlcxx::ArrayRef<std::string> input_names,
  jlcxx::ArrayRef<std::string> output_names,
  jlcxx::ArrayRef<Sparsity> input_sparsities,
  jlcxx::ArrayRef<Sparsity> output_sparsities,
  const GenericType& options)
{
  return Function::jit(
    name,
    body,
    to_vector(input_names),
    to_vector(output_names),
    to_vector(input_sparsities),
    to_vector(output_sparsities),
    generic_as_dict(options, "jit options"));
}

std::vector<bool> function_which_depends(
  const Function& f,
  const std::string& input_name,
  jlcxx::ArrayRef<std::string> output_names,
  const std::int64_t order,
  const bool transpose)
{
  return f.which_depends(
    input_name,
    to_vector(output_names),
    checked_nonnegative(order, "order"),
    transpose);
}

std::vector<bool> function_which_depends_one(
  const Function& f,
  const std::string& input_name,
  const std::string& output_name,
  const std::int64_t order,
  const bool transpose)
{
  return f.which_depends(
    input_name,
    std::vector<std::string>{output_name},
    checked_nonnegative(order, "order"),
    transpose);
}

std::string function_serialize(const Function& f, const GenericType& options)
{
  return f.serialize(generic_as_dict(options, "serialize options"));
}

void function_export_code_file(
  const Function& f,
  const std::string& language,
  const std::string& filename,
  const GenericType& options)
{
  f.export_code(language, filename, generic_as_dict(options, "export_code options"));
}

std::string function_export_code_string(const Function& f, const std::string& language, const GenericType& options)
{
  return f.export_code(language, generic_as_dict(options, "export_code options"));
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

std::vector<std::int64_t> function_instruction_input(const Function& f, const std::int64_t index)
{
  return from_casadi_int_vector(f.instruction_input(checked_index(index, "index")));
}

std::vector<std::int64_t> function_instruction_output(const Function& f, const std::int64_t index)
{
  return from_casadi_int_vector(f.instruction_output(checked_index(index, "index")));
}

std::vector<std::int64_t> function_work_sizes(const Function& f)
{
  size_t sz_arg = 0;
  size_t sz_res = 0;
  size_t sz_iw = 0;
  size_t sz_w = 0;
  f.sz_work(sz_arg, sz_res, sz_iw, sz_w);
  return {
    static_cast<std::int64_t>(sz_arg),
    static_cast<std::int64_t>(sz_res),
    static_cast<std::int64_t>(sz_iw),
    static_cast<std::int64_t>(sz_w)};
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
  mod.method(raw_method("function_sx_named"), &make_sx_function_named);
  mod.method(raw_method("function_mx"), &make_mx_function);
  mod.method(raw_method("function_mx_options"), &make_mx_function_options);
  mod.method(raw_method("function_mx_named"), &make_mx_function_named);
  mod.method(raw_method("function_from_file"), &function_from_file);
  mod.method(raw_method("function_string"), &to_string<Function>);
  mod.method(raw_method("function_name"), [](const Function& f) { return f.name(); });
  mod.method(raw_method("function_is_a"), [](const Function& f, const std::string& type, const bool recursive) {
    return f.is_a(type, recursive);
  });
  mod.method(raw_method("function_check_name"), [](const std::string& name) { return Function::check_name(name); });
  mod.method(raw_method("function_fix_name"), [](const std::string& name) { return Function::fix_name(name); });
  mod.method(raw_method("function_n_in"), [](const Function& f) { return static_cast<std::int64_t>(f.n_in()); });
  mod.method(raw_method("function_n_out"), [](const Function& f) { return static_cast<std::int64_t>(f.n_out()); });
  mod.method(raw_method("function_name_in_all"), [](const Function& f) { return f.name_in(); });
  mod.method(raw_method("function_name_out_all"), [](const Function& f) { return f.name_out(); });
  mod.method(raw_method("function_name_in"), [](const Function& f, const std::int64_t index) {
    return f.name_in(checked_index(index, "index"));
  });
  mod.method(raw_method("function_name_out"), [](const Function& f, const std::int64_t index) {
    return f.name_out(checked_index(index, "index"));
  });
  mod.method(raw_method("function_index_in"), [](const Function& f, const std::string& name) {
    return static_cast<std::int64_t>(f.index_in(name));
  });
  mod.method(raw_method("function_index_out"), [](const Function& f, const std::string& name) {
    return static_cast<std::int64_t>(f.index_out(name));
  });
  mod.method(raw_method("function_has_in"), [](const Function& f, const std::string& name) {
    return f.has_in(name);
  });
  mod.method(raw_method("function_has_out"), [](const Function& f, const std::string& name) {
    return f.has_out(name);
  });
  mod.method(raw_method("function_size_in"), [](const Function& f, const std::int64_t index) {
    return function_size_pair(f.size_in(checked_index(index, "index")));
  });
  mod.method(raw_method("function_size_out"), [](const Function& f, const std::int64_t index) {
    return function_size_pair(f.size_out(checked_index(index, "index")));
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
  mod.method(raw_method("function_nnz_in_total"), [](const Function& f) {
    return static_cast<std::int64_t>(f.nnz_in());
  });
  mod.method(raw_method("function_nnz_out"), [](const Function& f, const std::int64_t index) {
    return static_cast<std::int64_t>(f.nnz_out(checked_index(index, "index")));
  });
  mod.method(raw_method("function_nnz_out_total"), [](const Function& f) {
    return static_cast<std::int64_t>(f.nnz_out());
  });
  mod.method(raw_method("function_numel_in"), [](const Function& f, const std::int64_t index) {
    return static_cast<std::int64_t>(f.numel_in(checked_index(index, "index")));
  });
  mod.method(raw_method("function_numel_in_total"), [](const Function& f) {
    return static_cast<std::int64_t>(f.numel_in());
  });
  mod.method(raw_method("function_numel_out"), [](const Function& f, const std::int64_t index) {
    return static_cast<std::int64_t>(f.numel_out(checked_index(index, "index")));
  });
  mod.method(raw_method("function_numel_out_total"), [](const Function& f) {
    return static_cast<std::int64_t>(f.numel_out());
  });
  mod.method(raw_method("function_default_in"), [](const Function& f, const std::int64_t index) {
    return f.default_in(checked_index(index, "index"));
  });
  mod.method(raw_method("function_min_in"), [](const Function& f, const std::int64_t index) {
    return f.min_in(checked_index(index, "index"));
  });
  mod.method(raw_method("function_max_in"), [](const Function& f, const std::int64_t index) {
    return f.max_in(checked_index(index, "index"));
  });
  mod.method(raw_method("function_nominal_in"), [](const Function& f, const std::int64_t index) {
    return f.nominal_in(checked_index(index, "index"));
  });
  mod.method(raw_method("function_nominal_out"), [](const Function& f, const std::int64_t index) {
    return f.nominal_out(checked_index(index, "index"));
  });
  mod.method(raw_method("function_is_diff_in"), [](const Function& f, const std::int64_t index) {
    return f.is_diff_in(checked_index(index, "index"));
  });
  mod.method(raw_method("function_is_diff_out"), [](const Function& f, const std::int64_t index) {
    return f.is_diff_out(checked_index(index, "index"));
  });
  mod.method(raw_method("function_is_diff_in_all"), [](const Function& f) { return f.is_diff_in(); });
  mod.method(raw_method("function_is_diff_out_all"), [](const Function& f) { return f.is_diff_out(); });
  mod.method(raw_method("function_sparsity_in"), [](const Function& f, const std::int64_t index) {
    return Sparsity(f.sparsity_in(checked_index(index, "index")));
  });
  mod.method(raw_method("function_sparsity_out"), [](const Function& f, const std::int64_t index) {
    return Sparsity(f.sparsity_out(checked_index(index, "index")));
  });
  mod.method(raw_method("function_has_option"), [](const Function& f, const std::string& name) {
    return f.has_option(name);
  });
  mod.method(raw_method("function_change_option"), [](Function& f, const std::string& name, const GenericType& value) {
    f.change_option(name, value);
  });
  mod.method(raw_method("function_reset_dump_count"), [](Function& f) { f.reset_dump_count(); });
  mod.method(raw_method("function_uses_output"), [](const Function& f) { return f.uses_output(); });
  mod.method(raw_method("function_print_dimensions"), [](const Function& f) {
    std::ostringstream out;
    f.print_dimensions(out);
    return out.str();
  });
  mod.method(raw_method("function_print_options"), [](const Function& f) {
    std::ostringstream out;
    f.print_options(out);
    return out.str();
  });
  mod.method(raw_method("function_print_option"), [](const Function& f, const std::string& name) {
    std::ostringstream out;
    f.print_option(name, out);
    return out.str();
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
  mod.method(raw_method("function_forward"), [](const Function& f, const std::int64_t nfwd) {
    return f.forward(checked_nonnegative(nfwd, "nfwd"));
  });
  mod.method(raw_method("function_reverse"), [](const Function& f, const std::int64_t nadj) {
    return f.reverse(checked_nonnegative(nadj, "nadj"));
  });
  mod.method(raw_method("function_jac_sparsity"), [](const Function& f, const bool compact) {
    return f.jac_sparsity(compact);
  });
  mod.method(raw_method("function_jac_sparsity_block"), [](const Function& f, const std::int64_t output_index, const std::int64_t input_index, const bool compact) {
    return f.jac_sparsity(
      checked_index(output_index, "output index"),
      checked_index(input_index, "input index"),
      compact);
  });
  mod.method(raw_method("function_expand"), [](const Function& f) { return f.expand(); });
  mod.method(raw_method("function_expand_options"), [](const Function& f, const std::string& name, const GenericType& options) {
    return f.expand(name, generic_as_dict(options, "expand options"));
  });
  mod.method(raw_method("function_wrap"), [](const Function& f) { return f.wrap(); });
  mod.method(raw_method("function_wrap_as_needed"), [](const Function& f, const GenericType& options) {
    return f.wrap_as_needed(generic_as_dict(options, "wrap options"));
  });
  mod.method(raw_method("function_oracle"), [](const Function& f) { return f.oracle(); });
  mod.method(raw_method("function_factory"), &function_factory);
  mod.method(raw_method("function_generate"), [](const Function& f, const std::string& filename) {
    return f.generate(filename);
  });
  mod.method(raw_method("function_generate_string"), [](const Function& f, const GenericType& options) {
    return f.generate(generic_as_dict(options, "generate options"));
  });
  mod.method(raw_method("function_generate"), [](const Function& f, const std::string& filename, const bool with_header, const bool main, const bool mex, const bool cpp) {
    return f.generate(filename, make_codegen_options(with_header, main, mex, cpp));
  });
  mod.method(raw_method("function_generate_options"), [](const Function& f, const std::string& filename, const GenericType& options) {
    return f.generate(filename, generic_as_dict(options, "generate options"));
  });
  mod.method(raw_method("function_generate_dependencies"), [](const Function& f, const std::string& filename, const GenericType& options) {
    return f.generate_dependencies(filename, generic_as_dict(options, "generate_dependencies options"));
  });
  mod.method(raw_method("function_generate_in_write"), [](Function& f, const std::string& filename, jlcxx::ArrayRef<DM> args) {
    f.generate_in(filename, to_vector(args));
  });
  mod.method(raw_method("function_generate_in_read"), [](Function& f, const std::string& filename) {
    return f.generate_in(filename);
  });
  mod.method(raw_method("function_generate_out_write"), [](Function& f, const std::string& filename, jlcxx::ArrayRef<DM> args) {
    f.generate_out(filename, to_vector(args));
  });
  mod.method(raw_method("function_generate_out_read"), [](Function& f, const std::string& filename) {
    return f.generate_out(filename);
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
  mod.method(raw_method("function_mapsum_mx"), [](const Function& f, jlcxx::ArrayRef<MX> args, const std::string& parallelization) {
    return f.mapsum(to_vector(args), parallelization);
  });
  mod.method(raw_method("function_slice"), &function_slice);
  mod.method(raw_method("function_conditional"), &function_conditional);
  mod.method(raw_method("function_conditional_single"), &function_conditional_single);
  mod.method(raw_method("function_if_else"), &function_if_else);
  mod.method(raw_method("function_jit"), &function_jit);
  mod.method(raw_method("function_jit_sparsity"), &function_jit_sparsity);
  mod.method(raw_method("function_which_depends"), &function_which_depends);
  mod.method(raw_method("function_which_depends"), &function_which_depends_one);
  mod.method(raw_method("function_stats"), [](const Function& f, const std::int64_t mem) {
    return GenericType(f.stats(checked_nonnegative(mem, "mem")));
  });
  mod.method(raw_method("function_info"), [](const Function& f) {
    return GenericType(f.info());
  });
  mod.method(raw_method("function_serialize"), &function_serialize);
  mod.method(raw_method("function_export_code_file"), &function_export_code_file);
  mod.method(raw_method("function_export_code_string"), &function_export_code_string);
  mod.method(raw_method("function_save"), &function_save);
  mod.method(raw_method("function_deserialize"), &function_deserialize);
  mod.method(raw_method("function_load"), &function_load);
  mod.method(raw_method("function_sx_in"), [](const Function& f, const std::int64_t index) {
    return f.sx_in(checked_index(index, "index"));
  });
  mod.method(raw_method("function_sx_in_all"), [](const Function& f) { return f.sx_in(); });
  mod.method(raw_method("function_mx_in"), [](const Function& f, const std::int64_t index) {
    return f.mx_in(checked_index(index, "index"));
  });
  mod.method(raw_method("function_mx_in_all"), [](const Function& f) { return f.mx_in(); });
  mod.method(raw_method("function_sx_out"), [](const Function& f, const std::int64_t index) {
    return f.sx_out(checked_index(index, "index"));
  });
  mod.method(raw_method("function_sx_out_all"), [](const Function& f) { return f.sx_out(); });
  mod.method(raw_method("function_mx_out"), [](const Function& f, const std::int64_t index) {
    return f.mx_out(checked_index(index, "index"));
  });
  mod.method(raw_method("function_mx_out_all"), [](const Function& f) { return f.mx_out(); });
  mod.method(raw_method("function_nz_from_in"), [](const Function& f, jlcxx::ArrayRef<DM> args) {
    return f.nz_from_in(to_vector(args));
  });
  mod.method(raw_method("function_nz_from_out"), [](const Function& f, jlcxx::ArrayRef<DM> args) {
    return f.nz_from_out(to_vector(args));
  });
  mod.method(raw_method("function_nz_to_in"), [](const Function& f, jlcxx::ArrayRef<double> args) {
    return f.nz_to_in(to_vector(args));
  });
  mod.method(raw_method("function_nz_to_out"), [](const Function& f, jlcxx::ArrayRef<double> args) {
    return f.nz_to_out(to_vector(args));
  });
  mod.method(raw_method("function_has_free"), [](const Function& f) { return f.has_free(); });
  mod.method(raw_method("function_get_free"), [](const Function& f) { return f.get_free(); });
  mod.method(raw_method("function_free_sx"), [](const Function& f) { return f.free_sx(); });
  mod.method(raw_method("function_free_mx"), [](const Function& f) { return f.free_mx(); });
  mod.method(raw_method("function_n_nodes"), [](const Function& f) {
    return static_cast<std::int64_t>(f.n_nodes());
  });
  mod.method(raw_method("function_n_instructions"), [](const Function& f) {
    return static_cast<std::int64_t>(f.n_instructions());
  });
  mod.method(raw_method("function_instruction_id"), [](const Function& f, const std::int64_t index) {
    return static_cast<std::int64_t>(f.instruction_id(checked_index(index, "index")));
  });
  mod.method(raw_method("function_instruction_input"), &function_instruction_input);
  mod.method(raw_method("function_instruction_output"), &function_instruction_output);
  mod.method(raw_method("function_instruction_constant"), [](const Function& f, const std::int64_t index) {
    return f.instruction_constant(checked_index(index, "index"));
  });
  mod.method(raw_method("function_instruction_mx"), [](const Function& f, const std::int64_t index) {
    return f.instruction_MX(checked_index(index, "index"));
  });
  mod.method(raw_method("function_instructions_sx"), [](const Function& f) { return f.instructions_sx(); });
  mod.method(raw_method("function_has_spfwd"), [](const Function& f) { return f.has_spfwd(); });
  mod.method(raw_method("function_has_sprev"), [](const Function& f) { return f.has_sprev(); });
  mod.method(raw_method("function_sz_arg"), [](const Function& f) { return static_cast<std::int64_t>(f.sz_arg()); });
  mod.method(raw_method("function_sz_res"), [](const Function& f) { return static_cast<std::int64_t>(f.sz_res()); });
  mod.method(raw_method("function_sz_iw"), [](const Function& f) { return static_cast<std::int64_t>(f.sz_iw()); });
  mod.method(raw_method("function_sz_w"), [](const Function& f) { return static_cast<std::int64_t>(f.sz_w()); });
  mod.method(raw_method("function_sz_work"), &function_work_sizes);
  mod.method(raw_method("function_cache"), [](const Function& f) {
    return GenericType(f.cache());
  });
  mod.method(raw_method("function_get_function_names"), [](const Function& f) { return f.get_function(); });
  mod.method(raw_method("function_get_function"), [](const Function& f, const std::string& name) {
    return f.get_function(name);
  });
  mod.method(raw_method("function_has_function"), [](const Function& f, const std::string& name) {
    return f.has_function(name);
  });
  mod.method(raw_method("function_find_functions"), [](const Function& f, const std::int64_t max_depth) {
    return f.find_functions(static_cast<casadi_int>(max_depth));
  });
  mod.method(raw_method("function_find_function"), [](const Function& f, const std::string& name, const std::int64_t max_depth) {
    return f.find_function(name, static_cast<casadi_int>(max_depth));
  });
  mod.method(raw_method("function_assert_size_in"), [](const Function& f, const std::int64_t index, const std::int64_t rows, const std::int64_t cols) {
    f.assert_size_in(checked_index(index, "index"), checked_nonnegative(rows, "rows"), checked_nonnegative(cols, "cols"));
  });
  mod.method(raw_method("function_assert_size_out"), [](const Function& f, const std::int64_t index, const std::int64_t rows, const std::int64_t cols) {
    f.assert_size_out(checked_index(index, "index"), checked_nonnegative(rows, "rows"), checked_nonnegative(cols, "cols"));
  });
  mod.method(raw_method("function_assert_sparsity_out"), [](const Function& f, const std::int64_t index, const Sparsity& sp, const std::int64_t n, const bool allow_all_zero_sparse) {
    f.assert_sparsity_out(
      checked_index(index, "index"),
      sp,
      checked_nonnegative(n, "n"),
      allow_all_zero_sparse);
  });

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
