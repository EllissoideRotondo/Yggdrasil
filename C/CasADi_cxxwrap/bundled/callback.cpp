#include "casadi_cxxwrap.hpp"

namespace casadi_cxxwrap
{
namespace
{

std::vector<std::string> callback_names(
  jlcxx::ArrayRef<std::string> names,
  const std::size_t count,
  const char* prefix)
{
  if(names.size() != 0 && names.size() != count)
  {
    throw std::invalid_argument(std::string(prefix) + " names must be empty or match the number of entries");
  }

  std::vector<std::string> out;
  out.reserve(count);
  for(std::size_t i = 0; i != count; ++i)
  {
    out.push_back(names.size() == 0 ? std::string(prefix) + std::to_string(i) : names[i]);
  }
  return out;
}

std::vector<DM> callback_result_to_vector(jl_value_t* value, const std::size_t expected)
{
  if(value == nullptr)
  {
    throw std::runtime_error("Julia callback returned a null value");
  }

  std::vector<DM> out;
  if(jl_is_array(value))
  {
    jlcxx::ArrayRef<DM> values(reinterpret_cast<jl_array_t*>(value));
    if(values.size() != expected)
    {
      throw std::runtime_error("Julia callback returned the wrong number of outputs");
    }

    out.reserve(values.size());
    for(std::size_t i = 0; i != values.size(); ++i)
    {
      out.push_back(values[i]);
    }
    return out;
  }

  if(expected == 1)
  {
    out.push_back(jlcxx::unbox<DM>(value));
    return out;
  }

  throw std::runtime_error(
    "Julia callback must return a Vector{DM} for multiple outputs, got " +
    jlcxx::julia_type_name(reinterpret_cast<jl_value_t*>(jl_typeof(value))));
}

jl_value_t* call_julia_function(jl_value_t* function, jl_value_t* argument)
{
  using JuliaFunctionPointer = decltype(jl_get_function(jl_base_module, "identity"));
  return jl_call1(reinterpret_cast<JuliaFunctionPointer>(function), argument);
}

class JuliaCallback final : public Callback
{
public:
  JuliaCallback(
    const std::string& name,
    jl_value_t* evaluator,
    std::vector<Sparsity> input_sparsities,
    std::vector<Sparsity> output_sparsities,
    std::vector<std::string> input_names,
    std::vector<std::string> output_names,
    const Dict& options,
    Function jacobian = Function())
    : evaluator_(evaluator),
      input_sparsities_(std::move(input_sparsities)),
      output_sparsities_(std::move(output_sparsities)),
      input_names_(std::move(input_names)),
      output_names_(std::move(output_names)),
      jacobian_(std::move(jacobian)),
      has_jacobian_(!jacobian_.is_null())
  {
    if(evaluator_ == nullptr)
    {
      throw std::invalid_argument("callback evaluator cannot be null");
    }
    if(!jl_subtype(reinterpret_cast<jl_value_t*>(jl_typeof(evaluator_)), reinterpret_cast<jl_value_t*>(jl_function_type)))
    {
      throw std::invalid_argument("callback evaluator must be a Julia Function");
    }
    jlcxx::protect_from_gc(evaluator_);
    construct(name, options);
  }

  ~JuliaCallback() override
  {
    jlcxx::unprotect_from_gc(evaluator_);
  }

  std::vector<DM> eval(const std::vector<DM>& arg) const override
  {
    jlcxx::Array<DM> args;
    for(const DM& value : arg)
    {
      args.push_back(value);
    }

    jl_value_t* args_value = reinterpret_cast<jl_value_t*>(args.wrapped());
    jl_value_t* result = nullptr;
    JL_GC_PUSH2(&args_value, &result);
    result = call_julia_function(evaluator_, args_value);
    if(jl_exception_occurred())
    {
      jl_call2(jl_get_function(jl_base_module, "showerror"), jl_stderr_obj(), jl_exception_occurred());
      jl_printf(jl_stderr_stream(), "\n");
      JL_GC_POP();
      throw std::runtime_error("Julia callback evaluation failed");
    }

    std::vector<DM> out = callback_result_to_vector(result, output_sparsities_.size());
    JL_GC_POP();
    return out;
  }

  casadi_int get_n_in() override
  {
    return static_cast<casadi_int>(input_sparsities_.size());
  }

  casadi_int get_n_out() override
  {
    return static_cast<casadi_int>(output_sparsities_.size());
  }

  Sparsity get_sparsity_in(const casadi_int i) override
  {
    return input_sparsities_.at(static_cast<std::size_t>(i));
  }

  Sparsity get_sparsity_out(const casadi_int i) override
  {
    return output_sparsities_.at(static_cast<std::size_t>(i));
  }

  std::string get_name_in(const casadi_int i) override
  {
    return input_names_.at(static_cast<std::size_t>(i));
  }

  std::string get_name_out(const casadi_int i) override
  {
    return output_names_.at(static_cast<std::size_t>(i));
  }

  bool has_jacobian() const override
  {
    return has_jacobian_;
  }

  Function get_jacobian(
    const std::string&,
    const std::vector<std::string>&,
    const std::vector<std::string>&,
    const Dict&) const override
  {
    return jacobian_;
  }

private:
  jl_value_t* evaluator_;
  std::vector<Sparsity> input_sparsities_;
  std::vector<Sparsity> output_sparsities_;
  std::vector<std::string> input_names_;
  std::vector<std::string> output_names_;
  Function jacobian_;
  bool has_jacobian_;
};

std::vector<std::shared_ptr<JuliaCallback>>& callback_registry()
{
  static std::vector<std::shared_ptr<JuliaCallback>> registry;
  return registry;
}

Function store_callback(const std::shared_ptr<JuliaCallback>& callback)
{
  callback_registry().push_back(callback);
  const Function& function = *callback;
  return function;
}

} // namespace

Function make_callback(
  const std::string& name,
  jl_value_t* evaluator,
  jlcxx::ArrayRef<Sparsity> input_sparsities,
  jlcxx::ArrayRef<Sparsity> output_sparsities,
  jlcxx::ArrayRef<std::string> input_names,
  jlcxx::ArrayRef<std::string> output_names,
  const GenericType& options)
{
  return store_callback(std::make_shared<JuliaCallback>(
    name,
    evaluator,
    to_vector(input_sparsities),
    to_vector(output_sparsities),
    callback_names(input_names, input_sparsities.size(), "i"),
    callback_names(output_names, output_sparsities.size(), "o"),
    generic_as_dict(options, "Callback options")));
}

Function make_callback_with_jacobian(
  const std::string& name,
  jl_value_t* evaluator,
  jlcxx::ArrayRef<Sparsity> input_sparsities,
  jlcxx::ArrayRef<Sparsity> output_sparsities,
  jlcxx::ArrayRef<std::string> input_names,
  jlcxx::ArrayRef<std::string> output_names,
  const GenericType& options,
  const Function& jacobian)
{
  return store_callback(std::make_shared<JuliaCallback>(
    name,
    evaluator,
    to_vector(input_sparsities),
    to_vector(output_sparsities),
    callback_names(input_names, input_sparsities.size(), "i"),
    callback_names(output_names, output_sparsities.size(), "o"),
    generic_as_dict(options, "Callback options"),
    jacobian));
}

void register_callback_bindings(jlcxx::Module& mod)
{
  mod.method(raw_method("callback"), &make_callback);
  mod.method(raw_method("callback_jacobian"), &make_callback_with_jacobian);
  mod.method(raw_method("callback_registry_size"), []() {
    return static_cast<std::int64_t>(callback_registry().size());
  });
}

} // namespace casadi_cxxwrap
