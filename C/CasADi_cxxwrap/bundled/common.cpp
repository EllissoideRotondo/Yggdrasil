#include "casadi_cxxwrap.hpp"

namespace casadi_cxxwrap
{

int checked_nonnegative(const std::int64_t value, const char* name)
{
  if(value < 0)
  {
    throw std::out_of_range(std::string(name) + " must be non-negative");
  }
  return static_cast<int>(value);
}

int checked_index(const std::int64_t value, const char* name)
{
  if(value < 0)
  {
    throw std::out_of_range(std::string(name) + " must be non-negative");
  }
  return static_cast<int>(value);
}

std::string raw_method(const std::string& name)
{
  return name + "_raw";
}

std::string raw_method(const std::string& prefix, const std::string& name)
{
  return prefix + "_" + name + "_raw";
}

std::vector<casadi_int> to_casadi_int_vector(jlcxx::ArrayRef<std::int64_t> values)
{
  std::vector<casadi_int> out;
  out.reserve(values.size());
  for(std::size_t i = 0; i != values.size(); ++i)
  {
    out.push_back(static_cast<casadi_int>(values[i]));
  }
  return out;
}

std::vector<std::int64_t> from_casadi_int_vector(const std::vector<casadi_int>& values)
{
  std::vector<std::int64_t> out;
  out.reserve(values.size());
  for(const auto value : values)
  {
    out.push_back(static_cast<std::int64_t>(value));
  }
  return out;
}

Dict make_codegen_options(const bool with_header, const bool main, const bool mex, const bool cpp)
{
  Dict opts;
  opts["with_header"] = with_header;
  opts["main"] = main;
  opts["mex"] = mex;
  opts["cpp"] = cpp;
  return opts;
}

} // namespace casadi_cxxwrap
