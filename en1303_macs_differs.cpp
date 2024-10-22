#include <getopt.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cassert>
#include <iostream>
#include <locale>
#include <random>

#ifndef DEBUG
#define DEBUG 0
#endif

using ull = unsigned long long;

// A key is a sequence of n cut heights.
template <unsigned long n>
using key = std::array<unsigned char, n>;

template <unsigned long h>
using freqs = std::array<unsigned char, h>;

template <unsigned long n>
using samples_t = std::vector<key<n>>;

// A reservoir for reservoir sampling. `samples` is the samples we've accepted
// into the reservoir, while `seen` is the number of samples we've been offered.
template <unsigned long n>
struct Reservoir {
  std::minstd_rand0 gen{0xFEDE123};
  samples_t<n> samples = {};
  int seen = 0;
};

// If nonzero, display a sample of size `sample_size` of valid keys.
unsigned int sample_size = 0;

// Checks that we have no bitting depth accounts for more than 50%
// of the key's cuts.
template <unsigned long n, unsigned long h>
bool slow_en_1303_fiddy(key<n>& arr) {
  std::array<int, h> buffer;
  for (int& x : buffer) x = 0;
  for (int x : arr) ++buffer[x];
  for (int v : buffer)
    if (v > n / 2) return false;
  return true;
}

// Checks that we have no three consecutive identical depths.
template <unsigned long n, unsigned long h>
bool slow_en_1303_no_consecutive_3(key<n>& arr) {
  for (int i = 0; i < n - 2; ++i) {
    if (arr[i] == arr[i + 1] && arr[i + 1] == arr[i + 2]) return false;
  }
  return true;
}

// Checks that a key meets EN-1303 requirements.
template <unsigned long n, unsigned long h>
bool slow_en_1303(key<n>& arr) {
  return slow_en_1303_fiddy<n, h>(arr) &&
         slow_en_1303_no_consecutive_3<n, h>(arr);
}

// Given h reservoirs of n samples each, combine them into a single reservoir
// of size n.
template <unsigned long n, unsigned long h>
samples_t<n> combine_reservoirs(std::array<Reservoir<n>, h>& reservoirs) {
  samples_t<n> all_samples;
  int all_seen = 0;
  for (auto& r : reservoirs) {
    all_seen += r.samples.size();
    all_samples.insert(all_samples.end(), r.samples.begin(), r.samples.end());
  }

  Reservoir<n> r;
  int i = 0;
  while (i < sample_size && i < all_samples.size()) {
    r.samples.push_back(all_samples[i]);
    ++i;
  }

  while (i < all_samples.size()) {
    std::uniform_int_distribution<int> dist(0, all_seen + 1);
    int idx = dist(r.gen);
    if (idx < sample_size) {
      r.samples[idx] = all_samples[i];
    }
    ++i;
  }

  return r.samples;
}

// Possibly add a given key to the reservoir.
template <unsigned long n>
void maybe_sample(const key<n>& arr, Reservoir<n>& r) {
  if (r.samples.size() < sample_size) {
    r.samples.push_back(arr);
    ++r.seen;
    return;
  }

  std::uniform_int_distribution<int> dist(0, r.seen + 1);
  int idx = dist(r.gen);
  if (idx < sample_size) {
    r.samples[idx] = arr;
  }
  ++r.seen;
}

// Returns whether there are any three consecutive cuts at the same depth.
template <unsigned long n, unsigned long h>
bool en_1303_no_consecutive_3(key<n>& a, int size) {
  if (size < 3) return true;
  return !(a[size - 1] == a[size - 2] && a[size - 2] == a[size - 3]);
}

// Returns whether a key meets the MACS restriction (adjacent cuts differ by at
// most `macs` heights), using cuts between `begin` and `end`.
template <unsigned long n>
bool check_macs(const key<n>& arr, int macs, int begin = 0, int end = n) {
  for (int i = begin; i < end - 1; ++i) {
    if (std::abs(arr[i] - arr[i + 1]) > macs) return false;
  }
  return true;
}

// Computes the number of physical and legal keys that have `arr` as a prefix of
// length `i`, that have total length n. These are added to `legal`. `r` is a reservoir
// where to offer samples when we find full keys, and `f` describes the state of `arr`.
template <unsigned long n, unsigned long h>
void rec(key<n>& arr, int i, int macs, ull& legal, Reservoir<n>& r,
         freqs<h>& f) {
  if (!en_1303_no_consecutive_3<n, h>(arr, i)) return;
  if (i == n) {
    ++legal;
#if DEBUG
    if (!(check_macs(arr, macs))) assert(false);
    if (!(slow_en_1303<n, h>(arr))) assert(false);
#endif
    if (sample_size) maybe_sample(arr, r);
    return;
  }
  for (int j = 0; j < h; ++j) {
    if (f[j] + 1 > n / 2) continue;
    if (i && std::abs(arr[i - 1] - j) > macs) continue;
    arr[i] = j;
    ++f[j];
    rec<n, h>(arr, i + 1, macs, legal, r, f);
    --f[j];
  }
  return;
}

template<unsigned long n>
struct CalcResult {
  ull legal_keys;
  samples_t<n> samples;
};

// Computes the number of physical and legal keys, given a MACS restriction. The
// counts are then added to `legal_global`.
template <unsigned long n, unsigned long h>
CalcResult<n> calc(int macs) {
  // Reservoirs for reservoir sampling.
  std::array<Reservoir<n>, h> reservoirs = {};
  std::atomic<ull> legal_keys = 0;
 #pragma omp parallel for num_threads(h)
  for (int i = 0; i < h; ++i) {
    key<n> combination = {static_cast<unsigned char>(i)};
    freqs<h> f = {};
    f[i] = 1;
    ull legal = 0;
    rec<n, h>(combination, 1, macs, legal, reservoirs[i], f);
    legal_keys += legal;
  }

  return CalcResult<n>{
    .legal_keys = legal_keys,
    .samples = combine_reservoirs(reservoirs)
  };
}

struct space_out : std::numpunct<char> {
  char do_thousands_sep() const { return ','; }
  std::string do_grouping() const { return "\3"; }
};

int main(int argc, char** argv) {
  const struct option longopts[] = {{"sample-size", 1, 0, 's'}, {0, 0, 0, 0}};
  int opt;
  while ((opt = getopt_long(argc, argv, "s:", longopts, nullptr)) != -1) {
    switch (opt) {
      case 's':
        sample_size = atoi(optarg);
        break;
    }
  }
  constexpr int n = 10;
  constexpr int k = 6;
  constexpr int macs = 4;
  std::cout << "n = " << n << ", k = " << k << ", macs = " << macs << std::endl;
  const auto result = calc<n, k>(macs);
  std::cout.imbue(std::locale(std::cout.getloc(), new space_out));
  std::cout << "Legal keys: " << result.legal_keys << std::endl;
  if (sample_size > 0) {
    std::cout << "Samples: " << std::endl;
    for (int i = 0; i < result.samples.size(); ++i) {
      const auto& v = result.samples[i];
      for (int j = 0; j < v.size(); ++j) {
        if (j > 0) std::cout << " ";
        std::cout << int(v[j]);
      }
      std::cout << std::endl;
    }
  }
}
