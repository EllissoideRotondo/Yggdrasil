#include "casadi_cxxwrap.hpp"

#include <type_traits>

namespace casadi_cxxwrap
{

template<typename Serializer>
void serializer_pack_sparsity(Serializer& serializer, const Sparsity& value)
{
  serializer.pack(value);
}

template<typename Serializer>
void serializer_pack_sx(Serializer& serializer, const SX& value)
{
  serializer.pack(value);
}

template<typename Serializer>
void serializer_pack_mx(Serializer& serializer, const MX& value)
{
  serializer.pack(value);
}

template<typename Serializer>
void serializer_pack_dm(Serializer& serializer, const DM& value)
{
  serializer.pack(value);
}

template<typename Serializer>
void serializer_pack_linsol(Serializer& serializer, const Linsol& value)
{
  serializer.pack(value);
}

template<typename Serializer>
void serializer_pack_function(Serializer& serializer, const Function& value)
{
  serializer.pack(value);
}

template<typename Serializer>
void serializer_pack_generic(Serializer& serializer, const GenericType& value)
{
  serializer.pack(value);
}

template<typename Serializer>
void serializer_pack_int(Serializer& serializer, const std::int64_t value)
{
  const casadi_int converted = checked_casadi_int(value, "value");
  serializer.pack(converted);
}

template<typename Serializer>
void serializer_pack_double(Serializer& serializer, const double value)
{
  serializer.pack(value);
}

template<typename Serializer>
void serializer_pack_string(Serializer& serializer, const std::string& value)
{
  serializer.pack(value);
}

template<typename Serializer>
void serializer_pack_sparsity_vector(Serializer& serializer, jlcxx::ArrayRef<Sparsity> values)
{
  serializer.pack(to_vector(values));
}

template<typename Serializer>
void serializer_pack_sx_vector(Serializer& serializer, jlcxx::ArrayRef<SX> values)
{
  serializer.pack(to_vector(values));
}

template<typename Serializer>
void serializer_pack_mx_vector(Serializer& serializer, jlcxx::ArrayRef<MX> values)
{
  serializer.pack(to_vector(values));
}

template<typename Serializer>
void serializer_pack_dm_vector(Serializer& serializer, jlcxx::ArrayRef<DM> values)
{
  serializer.pack(to_vector(values));
}

template<typename Serializer>
void serializer_pack_linsol_vector(Serializer& serializer, jlcxx::ArrayRef<Linsol> values)
{
  serializer.pack(to_vector(values));
}

template<typename Serializer>
void serializer_pack_function_vector(Serializer& serializer, jlcxx::ArrayRef<Function> values)
{
  serializer.pack(to_vector(values));
}

template<typename Serializer>
void serializer_pack_generic_vector(Serializer& serializer, jlcxx::ArrayRef<GenericType> values)
{
  serializer.pack(to_vector(values));
}

template<typename Serializer>
void serializer_pack_int_vector(Serializer& serializer, jlcxx::ArrayRef<std::int64_t> values)
{
  serializer.pack(to_casadi_int_vector(values));
}

template<typename Serializer>
void serializer_pack_double_vector(Serializer& serializer, jlcxx::ArrayRef<double> values)
{
  serializer.pack(to_vector(values));
}

template<typename Serializer>
void serializer_pack_string_vector(Serializer& serializer, jlcxx::ArrayRef<std::string> values)
{
  serializer.pack(to_vector(values));
}

template<typename Deserializer>
std::int64_t deserializer_pop_type(Deserializer& deserializer)
{
  return static_cast<std::int64_t>(deserializer.pop_type());
}

template<typename Deserializer>
Sparsity deserializer_unpack_sparsity(Deserializer& deserializer)
{
  return deserializer.unpack_sparsity();
}

template<typename Deserializer>
SX deserializer_unpack_sx(Deserializer& deserializer)
{
  return deserializer.unpack_sx();
}

template<typename Deserializer>
MX deserializer_unpack_mx(Deserializer& deserializer)
{
  return deserializer.unpack_mx();
}

template<typename Deserializer>
DM deserializer_unpack_dm(Deserializer& deserializer)
{
  return deserializer.unpack_dm();
}

template<typename Deserializer>
Linsol deserializer_unpack_linsol(Deserializer& deserializer)
{
  return deserializer.unpack_linsol();
}

template<typename Deserializer>
Function deserializer_unpack_function(Deserializer& deserializer)
{
  return deserializer.unpack_function();
}

template<typename Deserializer>
GenericType deserializer_unpack_generic(Deserializer& deserializer)
{
  return deserializer.unpack_generictype();
}

template<typename Deserializer>
std::int64_t deserializer_unpack_int(Deserializer& deserializer)
{
  return static_cast<std::int64_t>(deserializer.unpack_int());
}

template<typename Deserializer>
double deserializer_unpack_double(Deserializer& deserializer)
{
  return deserializer.unpack_double();
}

template<typename Deserializer>
std::string deserializer_unpack_string(Deserializer& deserializer)
{
  return deserializer.unpack_string();
}

template<typename Deserializer>
std::vector<Sparsity> deserializer_unpack_sparsity_vector(Deserializer& deserializer)
{
  return deserializer.unpack_sparsity_vector();
}

template<typename Deserializer>
std::vector<SX> deserializer_unpack_sx_vector(Deserializer& deserializer)
{
  return deserializer.unpack_sx_vector();
}

template<typename Deserializer>
std::vector<MX> deserializer_unpack_mx_vector(Deserializer& deserializer)
{
  return deserializer.unpack_mx_vector();
}

template<typename Deserializer>
std::vector<DM> deserializer_unpack_dm_vector(Deserializer& deserializer)
{
  return deserializer.unpack_dm_vector();
}

template<typename Deserializer>
std::vector<Linsol> deserializer_unpack_linsol_vector(Deserializer& deserializer)
{
  return deserializer.unpack_linsol_vector();
}

template<typename Deserializer>
std::vector<Function> deserializer_unpack_function_vector(Deserializer& deserializer)
{
  return deserializer.unpack_function_vector();
}

template<typename Deserializer>
std::vector<GenericType> deserializer_unpack_generic_vector(Deserializer& deserializer)
{
  return deserializer.unpack_generictype_vector();
}

template<typename Deserializer>
std::vector<std::int64_t> deserializer_unpack_int_vector(Deserializer& deserializer)
{
  return from_casadi_int_vector(deserializer.unpack_int_vector());
}

template<typename Deserializer>
std::vector<double> deserializer_unpack_double_vector(Deserializer& deserializer)
{
  return deserializer.unpack_double_vector();
}

template<typename Deserializer>
std::vector<std::string> deserializer_unpack_string_vector(Deserializer& deserializer)
{
  return deserializer.unpack_string_vector();
}

std::string xml_node_string(const XmlNode& node)
{
  return to_string(node);
}

std::string xml_node_dump_string(const XmlNode& node, const std::int64_t indent)
{
  std::ostringstream out;
  node.dump(out, checked_nonnegative(indent, "indent"));
  return out.str();
}

XmlNode xml_node_new(const std::string& name, const std::string& text, const std::string& comment, const std::int64_t line)
{
  XmlNode node;
  node.name = name;
  node.text = text;
  node.comment = comment;
  node.line = checked_casadi_int(line, "line");
  return node;
}

XmlNode xml_node_child_at(const XmlNode& node, const std::int64_t index)
{
  return node[static_cast<std::size_t>(checked_nonnegative(index, "index"))];
}

void xml_node_add_child(XmlNode& node, const XmlNode& child)
{
  node.children.push_back(child);
}

void xml_node_set_attribute_int(XmlNode& node, const std::string& name, const std::int64_t value)
{
  node.set_attribute(name, checked_casadi_int(value, "attribute"));
}

std::int64_t xml_node_attribute_int(const XmlNode& node, const std::string& name)
{
  return static_cast<std::int64_t>(node.attribute<casadi_int>(name));
}

std::int64_t xml_node_attribute_int_default(const XmlNode& node, const std::string& name, const std::int64_t default_value)
{
  const casadi_int converted = checked_casadi_int(default_value, "default_value");
  return static_cast<std::int64_t>(node.attribute<casadi_int>(name, converted));
}

std::vector<std::int64_t> xml_node_attribute_int_vector(const XmlNode& node, const std::string& name)
{
  return from_casadi_int_vector(node.attribute<std::vector<casadi_int>>(name));
}

std::vector<std::int64_t> xml_node_text_int_vector(const XmlNode& node)
{
  std::vector<casadi_int> value;
  node.get(&value);
  return from_casadi_int_vector(value);
}

template<typename Serializer>
void register_serializer_methods(jlcxx::Module& mod, const std::string& prefix)
{
  if constexpr(std::is_same_v<Serializer, StringSerializer>)
  {
    mod.method(raw_method(prefix + "_serializer_encode"), [](Serializer& serializer) { return serializer.encode(); });
  }
  mod.method(raw_method(prefix + "_serializer_pack_sparsity"), &serializer_pack_sparsity<Serializer>);
  mod.method(raw_method(prefix + "_serializer_pack_sx"), &serializer_pack_sx<Serializer>);
  mod.method(raw_method(prefix + "_serializer_pack_mx"), &serializer_pack_mx<Serializer>);
  mod.method(raw_method(prefix + "_serializer_pack_dm"), &serializer_pack_dm<Serializer>);
  mod.method(raw_method(prefix + "_serializer_pack_linsol"), &serializer_pack_linsol<Serializer>);
  mod.method(raw_method(prefix + "_serializer_pack_function"), &serializer_pack_function<Serializer>);
  mod.method(raw_method(prefix + "_serializer_pack_generic"), &serializer_pack_generic<Serializer>);
  mod.method(raw_method(prefix + "_serializer_pack_int"), &serializer_pack_int<Serializer>);
  mod.method(raw_method(prefix + "_serializer_pack_double"), &serializer_pack_double<Serializer>);
  mod.method(raw_method(prefix + "_serializer_pack_string"), &serializer_pack_string<Serializer>);
  mod.method(raw_method(prefix + "_serializer_pack_sparsity_vector"), &serializer_pack_sparsity_vector<Serializer>);
  mod.method(raw_method(prefix + "_serializer_pack_sx_vector"), &serializer_pack_sx_vector<Serializer>);
  mod.method(raw_method(prefix + "_serializer_pack_mx_vector"), &serializer_pack_mx_vector<Serializer>);
  mod.method(raw_method(prefix + "_serializer_pack_dm_vector"), &serializer_pack_dm_vector<Serializer>);
  mod.method(raw_method(prefix + "_serializer_pack_linsol_vector"), &serializer_pack_linsol_vector<Serializer>);
  mod.method(raw_method(prefix + "_serializer_pack_function_vector"), &serializer_pack_function_vector<Serializer>);
  mod.method(raw_method(prefix + "_serializer_pack_generic_vector"), &serializer_pack_generic_vector<Serializer>);
  mod.method(raw_method(prefix + "_serializer_pack_int_vector"), &serializer_pack_int_vector<Serializer>);
  mod.method(raw_method(prefix + "_serializer_pack_double_vector"), &serializer_pack_double_vector<Serializer>);
  mod.method(raw_method(prefix + "_serializer_pack_string_vector"), &serializer_pack_string_vector<Serializer>);
}

template<typename Deserializer>
void register_deserializer_methods(jlcxx::Module& mod, const std::string& raw_prefix)
{
  mod.method(raw_method(raw_prefix + "_pop_type"), &deserializer_pop_type<Deserializer>);
  mod.method(raw_method(raw_prefix + "_unpack_sparsity"), &deserializer_unpack_sparsity<Deserializer>);
  mod.method(raw_method(raw_prefix + "_unpack_sx"), &deserializer_unpack_sx<Deserializer>);
  mod.method(raw_method(raw_prefix + "_unpack_mx"), &deserializer_unpack_mx<Deserializer>);
  mod.method(raw_method(raw_prefix + "_unpack_dm"), &deserializer_unpack_dm<Deserializer>);
  mod.method(raw_method(raw_prefix + "_unpack_linsol"), &deserializer_unpack_linsol<Deserializer>);
  mod.method(raw_method(raw_prefix + "_unpack_function"), &deserializer_unpack_function<Deserializer>);
  mod.method(raw_method(raw_prefix + "_unpack_generic"), &deserializer_unpack_generic<Deserializer>);
  mod.method(raw_method(raw_prefix + "_unpack_int"), &deserializer_unpack_int<Deserializer>);
  mod.method(raw_method(raw_prefix + "_unpack_double"), &deserializer_unpack_double<Deserializer>);
  mod.method(raw_method(raw_prefix + "_unpack_string"), &deserializer_unpack_string<Deserializer>);
  mod.method(raw_method(raw_prefix + "_unpack_sparsity_vector"), &deserializer_unpack_sparsity_vector<Deserializer>);
  mod.method(raw_method(raw_prefix + "_unpack_sx_vector"), &deserializer_unpack_sx_vector<Deserializer>);
  mod.method(raw_method(raw_prefix + "_unpack_mx_vector"), &deserializer_unpack_mx_vector<Deserializer>);
  mod.method(raw_method(raw_prefix + "_unpack_dm_vector"), &deserializer_unpack_dm_vector<Deserializer>);
  mod.method(raw_method(raw_prefix + "_unpack_linsol_vector"), &deserializer_unpack_linsol_vector<Deserializer>);
  mod.method(raw_method(raw_prefix + "_unpack_function_vector"), &deserializer_unpack_function_vector<Deserializer>);
  mod.method(raw_method(raw_prefix + "_unpack_generic_vector"), &deserializer_unpack_generic_vector<Deserializer>);
  mod.method(raw_method(raw_prefix + "_unpack_int_vector"), &deserializer_unpack_int_vector<Deserializer>);
  mod.method(raw_method(raw_prefix + "_unpack_double_vector"), &deserializer_unpack_double_vector<Deserializer>);
  mod.method(raw_method(raw_prefix + "_unpack_string_vector"), &deserializer_unpack_string_vector<Deserializer>);
  mod.method(raw_method(raw_prefix + "_reset"), [](Deserializer& deserializer) { deserializer.reset(); });
}

void register_utility_xml_methods(jlcxx::Module& mod)
{
  mod.method(raw_method("xml_node_new"), &xml_node_new);
  mod.method(raw_method("xml_node_string"), &xml_node_string);
  mod.method(raw_method("xml_node_dump_string"), &xml_node_dump_string);
  mod.method(raw_method("xml_node_name"), [](const XmlNode& node) { return node.name; });
  mod.method(raw_method("xml_node_set_name"), [](XmlNode& node, const std::string& name) { node.name = name; });
  mod.method(raw_method("xml_node_text"), [](const XmlNode& node) { return node.text; });
  mod.method(raw_method("xml_node_set_text"), [](XmlNode& node, const std::string& text) { node.text = text; });
  mod.method(raw_method("xml_node_comment"), [](const XmlNode& node) { return node.comment; });
  mod.method(raw_method("xml_node_set_comment"), [](XmlNode& node, const std::string& comment) { node.comment = comment; });
  mod.method(raw_method("xml_node_line"), [](const XmlNode& node) { return static_cast<std::int64_t>(node.line); });
  mod.method(raw_method("xml_node_set_line"), [](XmlNode& node, const std::int64_t line) {
    node.line = checked_casadi_int(line, "line");
  });
  mod.method(raw_method("xml_node_size"), [](const XmlNode& node) { return static_cast<std::int64_t>(node.size()); });
  mod.method(raw_method("xml_node_children"), [](const XmlNode& node) { return node.children; });
  mod.method(raw_method("xml_node_child_at"), &xml_node_child_at);
  mod.method(raw_method("xml_node_child_named"), [](const XmlNode& node, const std::string& name) { return node[name]; });
  mod.method(raw_method("xml_node_has_child"), [](const XmlNode& node, const std::string& name) { return node.has_child(name); });
  mod.method(raw_method("xml_node_child_names"), [](const XmlNode& node) { return node.child_names(); });
  mod.method(raw_method("xml_node_add_child"), &xml_node_add_child);
  mod.method(raw_method("xml_node_attribute_names"), [](const XmlNode& node) { return node.attribute_names(); });
  mod.method(raw_method("xml_node_has_attribute"), [](const XmlNode& node, const std::string& name) {
    return node.has_attribute(name);
  });
  mod.method(raw_method("xml_node_set_attribute_string"), [](XmlNode& node, const std::string& name, const std::string& value) {
    node.set_attribute(name, value);
  });
  mod.method(raw_method("xml_node_set_attribute_bool"), [](XmlNode& node, const std::string& name, const bool value) {
    node.set_attribute(name, value ? "true" : "false");
  });
  mod.method(raw_method("xml_node_set_attribute_int"), &xml_node_set_attribute_int);
  mod.method(raw_method("xml_node_set_attribute_double"), [](XmlNode& node, const std::string& name, const double value) {
    node.set_attribute(name, value);
  });
  mod.method(raw_method("xml_node_set_attribute_int_vector"), [](XmlNode& node, const std::string& name, jlcxx::ArrayRef<std::int64_t> values) {
    node.set_attribute(name, to_casadi_int_vector(values));
  });
  mod.method(raw_method("xml_node_attribute_string"), [](const XmlNode& node, const std::string& name) {
    return node.attribute<std::string>(name);
  });
  mod.method(raw_method("xml_node_attribute_string_default"), [](const XmlNode& node, const std::string& name, const std::string& default_value) {
    return node.attribute<std::string>(name, default_value);
  });
  mod.method(raw_method("xml_node_attribute_bool"), [](const XmlNode& node, const std::string& name) {
    return node.attribute<bool>(name);
  });
  mod.method(raw_method("xml_node_attribute_bool_default"), [](const XmlNode& node, const std::string& name, const bool default_value) {
    return node.attribute<bool>(name, default_value);
  });
  mod.method(raw_method("xml_node_attribute_int"), &xml_node_attribute_int);
  mod.method(raw_method("xml_node_attribute_int_default"), &xml_node_attribute_int_default);
  mod.method(raw_method("xml_node_attribute_double"), [](const XmlNode& node, const std::string& name) {
    return node.attribute<double>(name);
  });
  mod.method(raw_method("xml_node_attribute_double_default"), [](const XmlNode& node, const std::string& name, const double default_value) {
    return node.attribute<double>(name, default_value);
  });
  mod.method(raw_method("xml_node_attribute_int_vector"), &xml_node_attribute_int_vector);
  mod.method(raw_method("xml_node_attribute_string_vector"), [](const XmlNode& node, const std::string& name) {
    return node.attribute<std::vector<std::string>>(name);
  });
  mod.method(raw_method("xml_node_text_string"), [](const XmlNode& node) {
    std::string value;
    node.get(&value);
    return value;
  });
  mod.method(raw_method("xml_node_text_bool"), [](const XmlNode& node) {
    bool value = false;
    node.get(&value);
    return value;
  });
  mod.method(raw_method("xml_node_text_int"), [](const XmlNode& node) {
    casadi_int value = 0;
    node.get(&value);
    return static_cast<std::int64_t>(value);
  });
  mod.method(raw_method("xml_node_text_double"), [](const XmlNode& node) {
    double value = 0.0;
    node.get(&value);
    return value;
  });
  mod.method(raw_method("xml_node_text_int_vector"), &xml_node_text_int_vector);
  mod.method(raw_method("xml_node_text_string_vector"), [](const XmlNode& node) {
    std::vector<std::string> value;
    node.get(&value);
    return value;
  });
  mod.method(raw_method("xml_file_new"), [](const std::string& plugin) { return XmlFile(plugin); });
  mod.method(raw_method("xml_file_load_plugin"), [](const std::string& plugin) { XmlFile::load_plugin(plugin); });
  mod.method(raw_method("xml_file_doc"), [](const std::string& plugin) { return XmlFile::doc(plugin); });
  mod.method(raw_method("xml_file_string"), &to_string<XmlFile>);
  mod.method(raw_method("xml_file_parse"), [](XmlFile& file, const std::string& filename) { return file.parse(filename); });
  mod.method(raw_method("xml_file_dump"), [](XmlFile& file, const std::string& filename, const XmlNode& node) {
    file.dump(filename, node);
  });
}

void register_serialization_bindings(jlcxx::Module& mod)
{
  mod.method(raw_method("serializer_type_to_string"), [](const std::int64_t value) {
    return StringSerializer::type_to_string(
      static_cast<StringSerializer::SerializationType>(checked_casadi_int(value, "serialization_type")));
  });

  register_serializer_methods<StringSerializer>(mod, "string");
  register_serializer_methods<FileSerializer>(mod, "file");
  register_deserializer_methods<StringDeserializer>(mod, "string_deserializer");
  register_deserializer_methods<FileDeserializer>(mod, "file_deserializer");
  mod.method(raw_method("string_deserializer_decode"), [](StringDeserializer& deserializer, const std::string& value) {
    deserializer.decode(value);
  });
  register_utility_xml_methods(mod);
}

} // namespace casadi_cxxwrap
