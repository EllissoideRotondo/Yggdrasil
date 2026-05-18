#include "casadi_cxxwrap.hpp"

#include <limits>

namespace casadi_cxxwrap
{

casadi_int checked_casadi_int(const std::int64_t value, const char* name)
{
  if constexpr(std::numeric_limits<casadi_int>::digits < std::numeric_limits<std::int64_t>::digits)
  {
    if(
      value < static_cast<std::int64_t>(std::numeric_limits<casadi_int>::min()) ||
      value > static_cast<std::int64_t>(std::numeric_limits<casadi_int>::max()))
    {
      throw std::out_of_range(std::string(name) + " is outside the casadi_int range");
    }
  }
  return static_cast<casadi_int>(value);
}

casadi_int checked_casadi_int_size(const std::size_t value, const char* name)
{
  if constexpr(std::numeric_limits<casadi_int>::digits < std::numeric_limits<std::size_t>::digits)
  {
    if(value > static_cast<std::size_t>(std::numeric_limits<casadi_int>::max()))
    {
      throw std::out_of_range(std::string(name) + " is outside the casadi_int range");
    }
  }
  return static_cast<casadi_int>(value);
}

std::int64_t checked_int64_size(const std::size_t value, const char* name)
{
  if(value > static_cast<std::size_t>(std::numeric_limits<std::int64_t>::max()))
  {
    throw std::out_of_range(std::string(name) + " is outside the Int64 range");
  }
  return static_cast<std::int64_t>(value);
}

casadi_int checked_nonnegative(const std::int64_t value, const char* name)
{
  const auto converted = checked_casadi_int(value, name);
  if(converted < 0)
  {
    throw std::out_of_range(std::string(name) + " must be non-negative");
  }
  return converted;
}

casadi_int checked_positive(const std::int64_t value, const char* name)
{
  const auto converted = checked_casadi_int(value, name);
  if(converted <= 0)
  {
    throw std::out_of_range(std::string(name) + " must be positive");
  }
  return converted;
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
    out.push_back(checked_casadi_int(values[i], "value"));
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
