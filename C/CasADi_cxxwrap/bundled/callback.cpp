#include "casadi_cxxwrap.hpp"

#include <algorithm>
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
    jl_array_t* const array = reinterpret_cast<jl_array_t*>(value);
    const std::size_t size = jl_array_len(array);
    if(size != expected)
    {
      throw std::runtime_error("Julia callback returned the wrong number of outputs");
    }

    jl_value_t* const element_type = reinterpret_cast<jl_value_t*>(jl_array_eltype(value));
    jl_value_t* const dm_type = reinterpret_cast<jl_value_t*>(jlcxx::julia_base_type<DM>());
    if(element_type != dm_type)
    {
      throw std::runtime_error(
        "Julia callback must return Vector{DM}, got " +
        jlcxx::julia_type_name(reinterpret_cast<jl_value_t*>(jl_typeof(value))));
    }

    out.reserve(size);
    for(std::size_t i = 0; i != size; ++i)
    {
      jl_value_t* element = jl_array_ptr_ref(array, i);
      if(element == nullptr)
      {
        throw std::runtime_error("Julia callback returned a Vector with a null element");
      }
      out.push_back(jlcxx::unbox<DM>(element));
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
    const auto order = checked_nonnegative(orders[i], name);
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
    const auto output_index = checked_nonnegative(output_indices[i], "output index");
    const auto input_index = checked_nonnegative(input_indices[i], "input index");
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
    require_julia_callable(evaluator_);
    jlcxx::protect_from_gc(evaluator_);
    try
    {
      construct(name, options);
    }
    catch(...)
    {
      jlcxx::unprotect_from_gc(evaluator_);
      throw;
    }
  }

  ~JuliaCallback() override
  {
    jlcxx::unprotect_from_gc(evaluator_);
  }

  std::vector<DM> eval(const std::vector<DM>& arg) const override
  {
    require_julia_thread(
      "JuliaCallback cannot be evaluated on a non-Julia thread. "
      "Function::map with 'thread' parallelism is not supported when a Julia callback is involved.");

    jlcxx::Array<DM> args;
    for(const DM& value : arg)
    {
      args.push_back(value);
    }

    jl_value_t* args_value = reinterpret_cast<jl_value_t*>(args.wrapped());
    jl_value_t* result = nullptr;
    JL_GC_PUSH2(&args_value, &result);
    try
    {
      result = call_julia_function(evaluator_, args_value);
    }
    catch(...)
    {
      JL_GC_POP();
      throw;
    }
    if(jl_value_t* exception = jl_exception_occurred())
    {
      jl_call2(jl_get_function(jl_base_module, "showerror"), jl_stderr_obj(), exception);
      jl_printf(jl_stderr_stream(), "\n");
      jl_exception_clear();
      JL_GC_POP();
      throw std::runtime_error("Julia callback evaluation failed");
    }

    std::vector<DM> out;
    try
    {
      out = callback_result_to_vector(result, output_sparsities_.size());
    }
    catch(...)
    {
      JL_GC_POP();
      throw;
    }
    JL_GC_POP();
    return out;
  }

  casadi_int get_n_in() override
  {
    return checked_casadi_int_size(input_sparsities_.size(), "callback input count");
  }

  casadi_int get_n_out() override
  {
    return checked_casadi_int_size(output_sparsities_.size(), "callback output count");
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

std::int64_t release_callback_function_if_unused(const Function& function, const casadi_int max_count)
{
  std::lock_guard<std::mutex> lock(callback_registry_mutex());
  auto& registry = callback_registry();
  const auto old_size = registry.size();
  registry.erase(
    std::remove_if(
      registry.begin(),
      registry.end(),
      [&function, max_count](const std::shared_ptr<JuliaCallback>& callback) {
        if(!callback || callback->getCount() > max_count)
        {
          return false;
        }
        const Function& stored = *callback;
        return stored == function;
      }),
    registry.end());
  return static_cast<std::int64_t>(old_size - registry.size());
}

std::int64_t sweep_unused_callbacks()
{
  std::lock_guard<std::mutex> lock(callback_registry_mutex());
  auto& registry = callback_registry();
  const auto old_size = registry.size();
  registry.erase(
    std::remove_if(
      registry.begin(),
      registry.end(),
      [](const std::shared_ptr<JuliaCallback>& callback) {
        return !callback || callback->getCount() <= 1;
      }),
    registry.end());
  return static_cast<std::int64_t>(old_size - registry.size());
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
  mod.method(raw_method("callback_release_function_if_unused"), [](const Function& function) {
    // max_count=2: the registry holds one CasADi reference, the caller holds one.
    // A getCount() of at most 2 means no other user holds the function.
    return release_callback_function_if_unused(function, 2);
  });
  mod.method(raw_method("callback_sweep_unused"), []() {
    return sweep_unused_callbacks();
  });
  mod.method(raw_method("callback_clear_registry"), []() {
    std::lock_guard<std::mutex> lock(callback_registry_mutex());
    auto& registry = callback_registry();
    const auto old_size = registry.size();
    registry.clear();
    return static_cast<std::int64_t>(old_size);
  });
  mod.method(raw_method("callback_release_name"), [](const std::string& name) {
    std::lock_guard<std::mutex> lock(callback_registry_mutex());
    auto& registry = callback_registry();
    const auto old_size = registry.size();
    registry.erase(
      std::remove_if(
        registry.begin(),
        registry.end(),
        [&name](const std::shared_ptr<JuliaCallback>& callback) {
          return callback && callback->name() == name && callback->getCount() <= 1;
        }),
      registry.end());
    return static_cast<std::int64_t>(old_size - registry.size());
  });
}

} // namespace casadi_cxxwrap
