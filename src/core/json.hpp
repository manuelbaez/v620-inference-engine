// Small recursive-descent JSON parser. It is enough for safetensors headers,
// config.json and tokenizer.json. Numbers are kept as double plus int64.
#pragma once

#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

namespace qw {

class Json {
public:
    enum Type { Null, Bool, Number, String, Array, Object };

    Type type = Null;
    bool b = false;
    double num = 0;
    int64_t i64 = 0;
    std::string str;
    std::vector<Json> arr;
    std::vector<std::pair<std::string, Json>> obj;  // insertion order kept

    static Json parse(std::string_view text);

    bool is_null() const { return type == Null; }
    bool has(std::string_view key) const { return find(key) != nullptr; }
    const Json *find(std::string_view key) const;
    const Json &operator[](std::string_view key) const;  // throws if missing
    const Json &operator[](size_t i) const;
    size_t size() const { return type == Array ? arr.size() : obj.size(); }

    int64_t as_int() const;
    double as_double() const;
    const std::string &as_str() const;
    bool as_bool() const;
};

}  // namespace qw
