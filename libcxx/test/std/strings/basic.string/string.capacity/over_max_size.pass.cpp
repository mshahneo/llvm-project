//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: no-exceptions

// After changing the alignment of the allocated pointer from 16 to 8, the exception
// thrown is no longer `bad_alloc` but instead length_error on systems using new
// headers but a dylib that doesn't contain 04ce0ba.
//
// XFAIL: using-built-library-before-llvm-19

// <string>

// size_type max_size() const; // constexpr since C++20

#include <string>
#include <cassert>
#include <stdexcept>

#include "test_macros.h"
#include "min_allocator.h"

template <class S>
void test(const S& s) {
  assert(s.max_size() >= s.size());
  S s2(s);
  const std::size_t sz = s2.max_size() + 1;
  try {
    s2.resize(sz, 'x');
  } catch (const std::length_error&) {
    return;
  }
  assert(false);
}

template <class S>
void test_string() {
  test(S());
  test(S("123"));
  test(S("12345678901234567890123456789012345678901234567890"));
}

bool test() {
  test_string<std::string>();
#if TEST_STD_VER >= 11
  test_string<std::basic_string<char, std::char_traits<char>, min_allocator<char>>>();
#endif

  return true;
}

int main(int, char**) {
  test();

  return 0;
}
