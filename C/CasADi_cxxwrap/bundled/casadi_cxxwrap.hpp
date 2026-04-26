#pragma once

#include <casadi/casadi.hpp>

#include <jlcxx/array.hpp>
#include <jlcxx/functions.hpp>
#include <jlcxx/jlcxx.hpp>
#include <jlcxx/stl.hpp>

#include <cstdint>
#include <map>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace casadi_cxxwrap
{
using casadi::CodeGenerator;
using casadi::Callback;
using casadi::DM;
using casadi::DaeBuilder;
using casadi::DMDict;
using casadi::Dict;
using casadi::Function;
using casadi::GenericType;
using casadi::Importer;
using casadi::MX;
using casadi::MXDict;
using casadi::NlpBuilder;
using casadi::Opti;
using casadi::OptiAdvanced;
using casadi::OptiSol;
using casadi::Sparsity;
using casadi::SpDict;
using casadi::SX;
using casadi::SXDict;

int checked_nonnegative(std::int64_t value, const char* name);
int checked_index(std::int64_t value, const char* name);

std::string raw_method(const std::string& name);
std::string raw_method(const std::string& prefix, const std::string& name);

std::vector<casadi_int> to_casadi_int_vector(jlcxx::ArrayRef<std::int64_t> values);
std::vector<std::int64_t> from_casadi_int_vector(const std::vector<casadi_int>& values);

template<typename T>
std::vector<T> to_vector(jlcxx::ArrayRef<T> values)
{
  std::vector<T> out;
  out.reserve(values.size());
  for(std::size_t i = 0; i != values.size(); ++i)
  {
    out.push_back(values[i]);
  }
  return out;
}

template<typename T>
std::string to_string(const T& value)
{
  std::ostringstream out;
  out << value;
  return out.str();
}

Dict make_codegen_options(bool with_header, bool main, bool mex, bool cpp);
const Dict& generic_as_dict(const GenericType& value, const char* name);

template<typename T>
std::map<std::string, T> named_dict(jlcxx::ArrayRef<std::string> keys, jlcxx::ArrayRef<T> values, const char* name)
{
  if(keys.size() != values.size())
  {
    throw std::invalid_argument(std::string(name) + " keys and values must have the same length");
  }

  std::map<std::string, T> out;
  for(std::size_t i = 0; i != keys.size(); ++i)
  {
    out[keys[i]] = values[i];
  }
  return out;
}

void register_matrix_bindings(jlcxx::Module& mod);
void register_generic_type_bindings(jlcxx::Module& mod);
void register_sparsity_bindings(jlcxx::Module& mod);
void register_function_bindings(jlcxx::Module& mod);
void register_callback_bindings(jlcxx::Module& mod);
void register_factory_bindings(jlcxx::Module& mod);
void register_codegen_bindings(jlcxx::Module& mod);
void register_interpolant_bindings(jlcxx::Module& mod);
void register_opti_bindings(jlcxx::Module& mod);
void register_builder_bindings(jlcxx::Module& mod);

} // namespace casadi_cxxwrap
