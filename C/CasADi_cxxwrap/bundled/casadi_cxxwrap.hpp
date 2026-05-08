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
using casadi::OptiCallback;
using casadi::OptiSol;
using casadi::Sparsity;
using casadi::SpDict;
using casadi::Linsol;
using casadi::FileDeserializer;
using casadi::FileSerializer;
using casadi::StringDeserializer;
using casadi::StringSerializer;
using casadi::SX;
using casadi::SXDict;
using casadi::XmlFile;
using casadi::XmlNode;

struct SXDictStatsResult
{
  SXDict value;
  GenericType stats;
};

struct MXDictStatsResult
{
  MXDict value;
  GenericType stats;
};

struct SXDictDaeMapResult
{
  SXDict value;
  Function state_to_orig;
  Function phi;
};

struct MXDictDaeMapResult
{
  MXDict value;
  Function state_to_orig;
  Function phi;
};

struct SXSimpleBoundsResult
{
  std::vector<std::int64_t> gi;
  SX lbx;
  SX ubx;
  Function lam_forward;
  Function lam_backward;
};

struct MXSimpleBoundsResult
{
  std::vector<std::int64_t> gi;
  MX lbx;
  MX ubx;
  Function lam_forward;
  Function lam_backward;
};

struct SparsityMappingResult
{
  Sparsity value;
  std::vector<std::int64_t> mapping;
};

struct IntVectorPairResult
{
  std::vector<std::int64_t> first;
  std::vector<std::int64_t> second;
};

struct SparsityLdlResult
{
  Sparsity lt;
  std::vector<std::int64_t> permutation;
};

struct SparsityQrResult
{
  Sparsity v;
  Sparsity r;
  std::vector<std::int64_t> prinv;
  std::vector<std::int64_t> pc;
};

struct SparsitySccResult
{
  std::int64_t components;
  std::vector<std::int64_t> index;
  std::vector<std::int64_t> offset;
};

struct SparsityBtfResult
{
  std::int64_t blocks;
  std::vector<std::int64_t> rowperm;
  std::vector<std::int64_t> colperm;
  std::vector<std::int64_t> rowblock;
  std::vector<std::int64_t> colblock;
  std::vector<std::int64_t> coarse_rowblock;
  std::vector<std::int64_t> coarse_colblock;
};

struct SXSharedResult
{
  std::vector<SX> expressions;
  std::vector<SX> variables;
  std::vector<SX> definitions;
};

struct MXSharedResult
{
  std::vector<MX> expressions;
  std::vector<MX> variables;
  std::vector<MX> definitions;
};

casadi_int checked_casadi_int(std::int64_t value, const char* name);
casadi_int checked_nonnegative(std::int64_t value, const char* name);
casadi_int checked_positive(std::int64_t value, const char* name);
casadi_int checked_index(std::int64_t value, const char* name);

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
void register_dm_dict_bindings(jlcxx::Module& mod);
void register_linsol_bindings(jlcxx::Module& mod);
void register_utility_bindings(jlcxx::Module& mod);
void register_serialization_bindings(jlcxx::Module& mod);

} // namespace casadi_cxxwrap
