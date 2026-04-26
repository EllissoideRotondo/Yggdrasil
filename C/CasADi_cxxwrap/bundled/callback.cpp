#include "casadi_cxxwrap.hpp"

#include <mutex>

namespace casadi_cxxwrap
{
namespace
{

using DerivativeFunctionMap = std::map<casadi_int, Function>;
using JacSparsityMap = std::map<std::pair<casadi_int, casadi_int>, Sparsity>;

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

DerivativeFunctionMap derivative_function_map(
  jlcxx::ArrayRef<std::int64_t> orders,
  jlcxx::ArrayRef<Function> functions,
  const char* name)
{
  if(orders.size() != functions.size())
  {
    throw std::invalid_argument(std::string(name) + " orders and functions must have the same length");
  }

  DerivativeFunctionMap out;
  for(std::size_t i = 0; i != orders.size(); ++i)
  {
    const auto order = static_cast<casadi_int>(checked_nonnegative(orders[i], name));
    const auto inserted = out.emplace(order, functions[i]);
    if(!inserted.second)
    {
      throw std::invalid_argument(std::string(name) + " orders must be unique");
    }
  }
  return out;
}

JacSparsityMap jac_sparsity_map(
  jlcxx::ArrayRef<std::int64_t> output_indices,
  jlcxx::ArrayRef<std::int64_t> input_indices,
  jlcxx::ArrayRef<Sparsity> sparsities)
{
  if(output_indices.size() != input_indices.size() || output_indices.size() != sparsities.size())
  {
    throw std::invalid_argument("Jacobian sparsity output indices, input indices, and sparsities must have the same length");
  }

  JacSparsityMap out;
  for(std::size_t i = 0; i != output_indices.size(); ++i)
  {
    const auto output_index = static_cast<casadi_int>(checked_index(output_indices[i], "output index"));
    const auto input_index = static_cast<casadi_int>(checked_index(input_indices[i], "input index"));
    const auto inserted = out.emplace(std::make_pair(output_index, input_index), sparsities[i]);
    if(!inserted.second)
    {
      throw std::invalid_argument("Jacobian sparsity blocks must be unique");
    }
  }
  return out;
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
    Function jacobian = Function(),
    DerivativeFunctionMap forward_functions = DerivativeFunctionMap(),
    DerivativeFunctionMap reverse_functions = DerivativeFunctionMap(),
    JacSparsityMap jac_sparsities = JacSparsityMap(),
    bool uses_output = false)
    : evaluator_(evaluator),
      input_sparsities_(std::move(input_sparsities)),
      output_sparsities_(std::move(output_sparsities)),
      input_names_(std::move(input_names)),
      output_names_(std::move(output_names)),
      jacobian_(std::move(jacobian)),
      forward_functions_(std::move(forward_functions)),
      reverse_functions_(std::move(reverse_functions)),
      jac_sparsities_(std::move(jac_sparsities)),
      has_jacobian_(!jacobian_.is_null()),
      uses_output_(uses_output)
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

  bool uses_output() const override
  {
    return uses_output_;
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

  bool has_forward(const casadi_int nfwd) const override
  {
    return forward_functions_.find(nfwd) != forward_functions_.end();
  }

  Function get_forward(
    const casadi_int nfwd,
    const std::string&,
    const std::vector<std::string>&,
    const std::vector<std::string>&,
    const Dict&) const override
  {
    const auto it = forward_functions_.find(nfwd);
    if(it == forward_functions_.end())
    {
      throw std::out_of_range("Julia callback has no forward derivative for the requested order");
    }
    return it->second;
  }

  bool has_reverse(const casadi_int nadj) const override
  {
    return reverse_functions_.find(nadj) != reverse_functions_.end();
  }

  Function get_reverse(
    const casadi_int nadj,
    const std::string&,
    const std::vector<std::string>&,
    const std::vector<std::string>&,
    const Dict&) const override
  {
    const auto it = reverse_functions_.find(nadj);
    if(it == reverse_functions_.end())
    {
      throw std::out_of_range("Julia callback has no reverse derivative for the requested order");
    }
    return it->second;
  }

  bool has_jac_sparsity(const casadi_int oind, const casadi_int iind) const override
  {
    return jac_sparsities_.find(std::make_pair(oind, iind)) != jac_sparsities_.end();
  }

  Sparsity get_jac_sparsity(const casadi_int oind, const casadi_int iind, const bool) const override
  {
    const auto it = jac_sparsities_.find(std::make_pair(oind, iind));
    if(it == jac_sparsities_.end())
    {
      throw std::out_of_range("Julia callback has no Jacobian sparsity for the requested block");
    }
    return it->second;
  }

private:
  jl_value_t* evaluator_;
  std::vector<Sparsity> input_sparsities_;
  std::vector<Sparsity> output_sparsities_;
  std::vector<std::string> input_names_;
  std::vector<std::string> output_names_;
  Function jacobian_;
  DerivativeFunctionMap forward_functions_;
  DerivativeFunctionMap reverse_functions_;
  JacSparsityMap jac_sparsities_;
  bool has_jacobian_;
  bool uses_output_;
};

std::vector<std::shared_ptr<JuliaCallback>>& callback_registry()
{
  static std::vector<std::shared_ptr<JuliaCallback>> registry;
  return registry;
}

std::mutex& callback_registry_mutex()
{
  static std::mutex mutex;
  return mutex;
}

Function store_callback(const std::shared_ptr<JuliaCallback>& callback)
{
  std::lock_guard<std::mutex> lock(callback_registry_mutex());
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

Function make_callback_derivatives(
  const std::string& name,
  jl_value_t* evaluator,
  jlcxx::ArrayRef<Sparsity> input_sparsities,
  jlcxx::ArrayRef<Sparsity> output_sparsities,
  jlcxx::ArrayRef<std::string> input_names,
  jlcxx::ArrayRef<std::string> output_names,
  const GenericType& options,
  const Function& jacobian,
  jlcxx::ArrayRef<std::int64_t> forward_orders,
  jlcxx::ArrayRef<Function> forward_functions,
  jlcxx::ArrayRef<std::int64_t> reverse_orders,
  jlcxx::ArrayRef<Function> reverse_functions,
  jlcxx::ArrayRef<std::int64_t> jac_sparsity_output_indices,
  jlcxx::ArrayRef<std::int64_t> jac_sparsity_input_indices,
  jlcxx::ArrayRef<Sparsity> jac_sparsities,
  const bool uses_output)
{
  return store_callback(std::make_shared<JuliaCallback>(
    name,
    evaluator,
    to_vector(input_sparsities),
    to_vector(output_sparsities),
    callback_names(input_names, input_sparsities.size(), "i"),
    callback_names(output_names, output_sparsities.size(), "o"),
    generic_as_dict(options, "Callback options"),
    jacobian,
    derivative_function_map(forward_orders, forward_functions, "forward"),
    derivative_function_map(reverse_orders, reverse_functions, "reverse"),
    jac_sparsity_map(jac_sparsity_output_indices, jac_sparsity_input_indices, jac_sparsities),
    uses_output));
}

void register_callback_bindings(jlcxx::Module& mod)
{
  mod.method(raw_method("callback"), &make_callback);
  mod.method(raw_method("callback_jacobian"), &make_callback_with_jacobian);
  mod.method(raw_method("callback_derivatives"), &make_callback_derivatives);
  mod.method(raw_method("callback_registry_size"), []() {
    std::lock_guard<std::mutex> lock(callback_registry_mutex());
    return static_cast<std::int64_t>(callback_registry().size());
  });
}

} // namespace casadi_cxxwrap
