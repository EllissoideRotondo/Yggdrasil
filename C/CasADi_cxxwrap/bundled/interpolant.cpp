#include "casadi_cxxwrap.hpp"

namespace casadi_cxxwrap
{
namespace
{

std::vector<std::vector<double>> interpolant_grid(
  jlcxx::ArrayRef<std::int64_t> grid_lengths,
  jlcxx::ArrayRef<double> grid_values)
{
  std::vector<std::vector<double>> grid;
  grid.reserve(grid_lengths.size());

  std::size_t offset = 0;
  for(std::size_t dim = 0; dim != grid_lengths.size(); ++dim)
  {
    const auto length = checked_nonnegative(grid_lengths[dim], "grid length");
    if(offset + static_cast<std::size_t>(length) > grid_values.size())
    {
      throw std::invalid_argument("grid lengths exceed the number of supplied grid values");
    }

    std::vector<double> current;
    current.reserve(static_cast<std::size_t>(length));
    for(int i = 0; i != length; ++i)
    {
      current.push_back(grid_values[offset++]);
    }
    grid.push_back(std::move(current));
  }

  if(offset != grid_values.size())
  {
    throw std::invalid_argument("grid lengths do not consume all supplied grid values");
  }
  return grid;
}

} // namespace

Function interpolant_values(
  const std::string& name,
  const std::string& solver,
  jlcxx::ArrayRef<std::int64_t> grid_lengths,
  jlcxx::ArrayRef<double> grid_values,
  jlcxx::ArrayRef<double> values,
  const GenericType& options)
{
  return casadi::interpolant(
    name,
    solver,
    interpolant_grid(grid_lengths, grid_values),
    to_vector(values),
    generic_as_dict(options, "interpolant options"));
}

Function interpolant_parametric_coefficients(
  const std::string& name,
  const std::string& solver,
  jlcxx::ArrayRef<std::int64_t> grid_lengths,
  jlcxx::ArrayRef<double> grid_values,
  const std::int64_t output_dimension,
  const GenericType& options)
{
  return casadi::interpolant(
    name,
    solver,
    interpolant_grid(grid_lengths, grid_values),
    checked_nonnegative(output_dimension, "output_dimension"),
    generic_as_dict(options, "interpolant options"));
}

Function interpolant_parametric_grid(
  const std::string& name,
  const std::string& solver,
  jlcxx::ArrayRef<std::int64_t> grid_dimensions,
  const std::int64_t output_dimension,
  const GenericType& options)
{
  return casadi::interpolant(
    name,
    solver,
    to_casadi_int_vector(grid_dimensions),
    checked_nonnegative(output_dimension, "output_dimension"),
    generic_as_dict(options, "interpolant options"));
}

Function interpolant_parametric_grid_values(
  const std::string& name,
  const std::string& solver,
  jlcxx::ArrayRef<std::int64_t> grid_dimensions,
  jlcxx::ArrayRef<double> values,
  const GenericType& options)
{
  return casadi::interpolant(
    name,
    solver,
    to_casadi_int_vector(grid_dimensions),
    to_vector(values),
    generic_as_dict(options, "interpolant options"));
}

void register_interpolant_bindings(jlcxx::Module& mod)
{
  mod.method(raw_method("interpolant_values"), &interpolant_values);
  mod.method(raw_method("interpolant_parametric_coefficients"), &interpolant_parametric_coefficients);
  mod.method(raw_method("interpolant_parametric_grid"), &interpolant_parametric_grid);
  mod.method(raw_method("interpolant_parametric_grid_values"), &interpolant_parametric_grid_values);
  mod.method(raw_method("has_interpolant"), [](const std::string& plugin) {
    return casadi::has_interpolant(plugin);
  });
  mod.method(raw_method("load_interpolant"), [](const std::string& plugin) {
    casadi::load_interpolant(plugin);
  });
  mod.method(raw_method("doc_interpolant"), [](const std::string& plugin) {
    return casadi::doc_interpolant(plugin);
  });
}

} // namespace casadi_cxxwrap
