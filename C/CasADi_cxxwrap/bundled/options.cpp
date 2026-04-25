#include "casadi_cxxwrap.hpp"

namespace casadi_cxxwrap
{

GenericType generic_null()
{
  return GenericType();
}

GenericType generic_bool(const bool value)
{
  return GenericType(value);
}

GenericType generic_int(const std::int64_t value)
{
  return GenericType(static_cast<casadi_int>(value));
}

GenericType generic_double(const double value)
{
  return GenericType(value);
}

GenericType generic_from_string(const std::string& value)
{
  return GenericType(value);
}

GenericType generic_bool_vector(jlcxx::ArrayRef<bool> values)
{
  return GenericType(to_vector(values));
}

GenericType generic_int_vector(jlcxx::ArrayRef<std::int64_t> values)
{
  return GenericType(to_casadi_int_vector(values));
}

GenericType generic_double_vector(jlcxx::ArrayRef<double> values)
{
  return GenericType(to_vector(values));
}

GenericType generic_string_vector(jlcxx::ArrayRef<std::string> values)
{
  return GenericType(to_vector(values));
}

GenericType generic_function(const Function& value)
{
  return GenericType(value);
}

GenericType generic_function_vector(jlcxx::ArrayRef<Function> values)
{
  return GenericType(to_vector(values));
}

GenericType generic_vector(jlcxx::ArrayRef<GenericType> values)
{
  return GenericType(to_vector(values));
}

GenericType generic_dict(jlcxx::ArrayRef<std::string> keys, jlcxx::ArrayRef<GenericType> values)
{
  if(keys.size() != values.size())
  {
    throw std::invalid_argument("GenericType dictionary keys and values must have the same length");
  }

  Dict out;
  for(std::size_t i = 0; i != keys.size(); ++i)
  {
    out[keys[i]] = values[i];
  }
  return GenericType(out);
}

std::int64_t generic_type_id(const GenericType& value)
{
  return static_cast<std::int64_t>(value.getType());
}

std::string generic_description(const GenericType& value)
{
  return value.get_description();
}

std::string generic_repr(const GenericType& value)
{
  return to_string(value);
}

bool generic_to_bool(const GenericType& value)
{
  return static_cast<bool>(value);
}

std::int64_t generic_to_int(const GenericType& value)
{
  return static_cast<std::int64_t>(static_cast<casadi_int>(value));
}

double generic_to_double(const GenericType& value)
{
  return static_cast<double>(value);
}

std::string generic_to_string(const GenericType& value)
{
  return static_cast<std::string>(value);
}

std::vector<std::int64_t> generic_to_int_vector(const GenericType& value)
{
  const auto raw = static_cast<std::vector<casadi_int>>(value);
  std::vector<std::int64_t> out;
  out.reserve(raw.size());
  for(const auto item : raw)
  {
    out.push_back(static_cast<std::int64_t>(item));
  }
  return out;
}

std::vector<double> generic_to_double_vector(const GenericType& value)
{
  return static_cast<std::vector<double>>(value);
}

std::vector<std::string> generic_to_string_vector(const GenericType& value)
{
  return static_cast<std::vector<std::string>>(value);
}

std::int64_t generic_dict_size(const GenericType& value)
{
  return static_cast<std::int64_t>(generic_as_dict(value, "GenericType dictionary").size());
}

bool generic_dict_has(const GenericType& value, const std::string& key)
{
  const auto& dict = generic_as_dict(value, "GenericType dictionary");
  return dict.find(key) != dict.end();
}

GenericType generic_dict_get(const GenericType& value, const std::string& key)
{
  const auto& dict = generic_as_dict(value, "GenericType dictionary");
  const auto it = dict.find(key);
  if(it == dict.end())
  {
    throw std::out_of_range("GenericType dictionary has no key '" + key + "'");
  }
  return it->second;
}

const Dict& generic_as_dict(const GenericType& value, const char* name)
{
  if(value.getType() != casadi::OT_DICT)
  {
    throw std::invalid_argument(std::string(name) + " must be a CasADi Dict GenericType");
  }
  return static_cast<const Dict&>(value);
}

void register_generic_type_bindings(jlcxx::Module& mod)
{
  mod.method(raw_method("generic_null"), &generic_null);
  mod.method(raw_method("generic_bool"), &generic_bool);
  mod.method(raw_method("generic_int"), &generic_int);
  mod.method(raw_method("generic_double"), &generic_double);
  mod.method(raw_method("generic_from_string"), &generic_from_string);
  mod.method(raw_method("generic_bool_vector"), &generic_bool_vector);
  mod.method(raw_method("generic_int_vector"), &generic_int_vector);
  mod.method(raw_method("generic_double_vector"), &generic_double_vector);
  mod.method(raw_method("generic_string_vector"), &generic_string_vector);
  mod.method(raw_method("generic_function"), &generic_function);
  mod.method(raw_method("generic_function_vector"), &generic_function_vector);
  mod.method(raw_method("generic_vector"), &generic_vector);
  mod.method(raw_method("generic_dict"), &generic_dict);
  mod.method(raw_method("generic_type_id"), &generic_type_id);
  mod.method(raw_method("generic_description"), &generic_description);
  mod.method(raw_method("generic_repr"), &generic_repr);
  mod.method(raw_method("generic_to_bool"), &generic_to_bool);
  mod.method(raw_method("generic_to_int"), &generic_to_int);
  mod.method(raw_method("generic_to_double"), &generic_to_double);
  mod.method(raw_method("generic_to_string"), &generic_to_string);
  mod.method(raw_method("generic_to_int_vector"), &generic_to_int_vector);
  mod.method(raw_method("generic_to_double_vector"), &generic_to_double_vector);
  mod.method(raw_method("generic_to_string_vector"), &generic_to_string_vector);
  mod.method(raw_method("generic_dict_size"), &generic_dict_size);
  mod.method(raw_method("generic_dict_has"), &generic_dict_has);
  mod.method(raw_method("generic_dict_get"), &generic_dict_get);
}

} // namespace casadi_cxxwrap
