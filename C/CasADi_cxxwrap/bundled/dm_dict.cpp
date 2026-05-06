#include "casadi_cxxwrap.hpp"

namespace casadi_cxxwrap
{

std::vector<std::string> dm_dict_keys(const DMDict& dict)
{
  std::vector<std::string> keys;
  keys.reserve(dict.size());
  for(const auto& entry : dict)
  {
    keys.push_back(entry.first);
  }
  return keys;
}

std::vector<DM> dm_dict_values(const DMDict& dict)
{
  std::vector<DM> values;
  values.reserve(dict.size());
  for(const auto& entry : dict)
  {
    values.push_back(entry.second);
  }
  return values;
}

bool dm_dict_has(const DMDict& dict, const std::string& key)
{
  return dict.find(key) != dict.end();
}

DM dm_dict_get(const DMDict& dict, const std::string& key)
{
  const auto found = dict.find(key);
  if(found == dict.end())
  {
    throw std::out_of_range("DMDict key not found: " + key);
  }
  return found->second;
}

void register_dm_dict_bindings(jlcxx::Module& mod)
{
  mod.method(raw_method("dm_dict_size"), [](const DMDict& dict) {
    return static_cast<std::int64_t>(dict.size());
  });
  mod.method(raw_method("dm_dict_keys"), &dm_dict_keys);
  mod.method(raw_method("dm_dict_values"), &dm_dict_values);
  mod.method(raw_method("dm_dict_has"), &dm_dict_has);
  mod.method(raw_method("dm_dict_get"), &dm_dict_get);
}

} // namespace casadi_cxxwrap
