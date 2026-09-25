// Prints n-gram ids for a token list so tests/ngram_ref.py can diff them
// against a direct port of the reference implementation.
//   test_ngram [ids]   (default: a fixed sequence with EOS tokens in it)
#include <cstdio>
#include <sstream>
#include <string>

#include "core/ple.hpp"

int main(int argc, char **argv) {
    std::vector<int32_t> toks = {760, 6511, 248044, 314, 9338, 248044, 248044, 369, 1, 2, 3, 248319};
    if (argc > 1) {
        toks.clear();
        std::stringstream ss(argv[1]);
        std::string t;
        while (std::getline(ss, t, ',')) toks.push_back(std::stoi(t));
    }
    qw::NgramHasher h;
    // Split into two chunks to exercise the history path.
    size_t cut = toks.size() / 2;
    std::vector<int32_t> a(toks.begin(), toks.begin() + long(cut)), b(toks.begin() + long(cut), toks.end());
    auto ia = h.ids_for({}, a);
    auto ib = h.ids_for(a, b);
    ia.insert(ia.end(), ib.begin(), ib.end());
    for (auto &ids : ia) {
        for (auto v : ids) std::printf("%lld ", (long long)v);
        std::printf("\n");
    }
}
