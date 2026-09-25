#include "core/json.hpp"

#include <cmath>
#include <cstdlib>

#include "core/common.hpp"

namespace qw {

namespace {

struct Parser {
    std::string_view s;
    size_t p = 0;

    [[noreturn]] void error(const char *what) const {
        fail(std::string("json: ") + what + " at offset " + std::to_string(p));
    }

    void ws() {
        while (p < s.size() && (s[p] == ' ' || s[p] == '\n' || s[p] == '\r' || s[p] == '\t')) ++p;
    }

    char peek() {
        ws();
        if (p >= s.size()) error("unexpected end");
        return s[p];
    }

    void expect(char c) {
        if (peek() != c) error("unexpected character");
        ++p;
    }

    static void put_utf8(std::string &out, uint32_t cp) {
        if (cp < 0x80) {
            out += char(cp);
        } else if (cp < 0x800) {
            out += char(0xc0 | (cp >> 6));
            out += char(0x80 | (cp & 0x3f));
        } else if (cp < 0x10000) {
            out += char(0xe0 | (cp >> 12));
            out += char(0x80 | ((cp >> 6) & 0x3f));
            out += char(0x80 | (cp & 0x3f));
        } else {
            out += char(0xf0 | (cp >> 18));
            out += char(0x80 | ((cp >> 12) & 0x3f));
            out += char(0x80 | ((cp >> 6) & 0x3f));
            out += char(0x80 | (cp & 0x3f));
        }
    }

    uint32_t hex4() {
        if (p + 4 > s.size()) error("short \\u escape");
        uint32_t v = 0;
        for (int i = 0; i < 4; ++i) {
            char c = s[p++];
            v <<= 4;
            if (c >= '0' && c <= '9') v |= uint32_t(c - '0');
            else if (c >= 'a' && c <= 'f') v |= uint32_t(c - 'a' + 10);
            else if (c >= 'A' && c <= 'F') v |= uint32_t(c - 'A' + 10);
            else error("bad hex digit");
        }
        return v;
    }

    std::string string() {
        expect('"');
        std::string out;
        for (;;) {
            if (p >= s.size()) error("unterminated string");
            char c = s[p++];
            if (c == '"') break;
            if (c != '\\') {
                out += c;
                continue;
            }
            if (p >= s.size()) error("bad escape");
            char e = s[p++];
            switch (e) {
                case '"': out += '"'; break;
                case '\\': out += '\\'; break;
                case '/': out += '/'; break;
                case 'b': out += '\b'; break;
                case 'f': out += '\f'; break;
                case 'n': out += '\n'; break;
                case 'r': out += '\r'; break;
                case 't': out += '\t'; break;
                case 'u': {
                    uint32_t cp = hex4();
                    if (cp >= 0xd800 && cp < 0xdc00 && p + 6 <= s.size() && s[p] == '\\' &&
                        s[p + 1] == 'u') {
                        p += 2;
                        uint32_t lo = hex4();
                        cp = 0x10000 + ((cp - 0xd800) << 10) + (lo - 0xdc00);
                    }
                    put_utf8(out, cp);
                    break;
                }
                default: error("bad escape");
            }
        }
        return out;
    }

    Json value() {
        Json j;
        char c = peek();
        if (c == '{') {
            ++p;
            j.type = Json::Object;
            if (peek() == '}') {
                ++p;
                return j;
            }
            for (;;) {
                std::string k = string();
                expect(':');
                j.obj.emplace_back(std::move(k), value());
                char d = peek();
                ++p;
                if (d == '}') break;
                if (d != ',') error("expected , or }");
            }
        } else if (c == '[') {
            ++p;
            j.type = Json::Array;
            if (peek() == ']') {
                ++p;
                return j;
            }
            for (;;) {
                j.arr.push_back(value());
                char d = peek();
                ++p;
                if (d == ']') break;
                if (d != ',') error("expected , or ]");
            }
        } else if (c == '"') {
            j.type = Json::String;
            j.str = string();
        } else if (s.compare(p, 4, "true") == 0) {
            p += 4;
            j.type = Json::Bool;
            j.b = true;
        } else if (s.compare(p, 5, "false") == 0) {
            p += 5;
            j.type = Json::Bool;
        } else if (s.compare(p, 4, "null") == 0) {
            p += 4;
        } else {
            size_t start = p;
            bool is_int = true;
            if (s[p] == '-') ++p;
            while (p < s.size()) {
                char d = s[p];
                if (d >= '0' && d <= '9') {
                    ++p;
                } else if (d == '.' || d == 'e' || d == 'E' || d == '+' || d == '-') {
                    is_int = false;
                    ++p;
                } else {
                    break;
                }
            }
            if (p == start) error("bad value");
            std::string tok(s.substr(start, p - start));
            j.type = Json::Number;
            j.num = std::strtod(tok.c_str(), nullptr);
            j.i64 = is_int ? std::strtoll(tok.c_str(), nullptr, 10) : int64_t(j.num);
        }
        return j;
    }
};

}  // namespace

Json Json::parse(std::string_view text) {
    Parser ps{text};
    Json j = ps.value();
    ps.ws();
    if (ps.p != text.size()) ps.error("trailing data");
    return j;
}

const Json *Json::find(std::string_view key) const {
    if (type != Object) return nullptr;
    for (auto &kv : obj)
        if (kv.first == key) return &kv.second;
    return nullptr;
}

const Json &Json::operator[](std::string_view key) const {
    const Json *j = find(key);
    if (!j) fail("json: missing key " + std::string(key));
    return *j;
}

const Json &Json::operator[](size_t i) const {
    if (type != Array || i >= arr.size()) fail("json: bad array index");
    return arr[i];
}

int64_t Json::as_int() const {
    if (type != Number) fail("json: not a number");
    return i64;
}

double Json::as_double() const {
    if (type != Number) fail("json: not a number");
    return num;
}

const std::string &Json::as_str() const {
    if (type != String) fail("json: not a string");
    return str;
}

bool Json::as_bool() const {
    if (type != Bool) fail("json: not a bool");
    return b;
}

}  // namespace qw
