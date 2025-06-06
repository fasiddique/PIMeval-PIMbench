// Test: Test reduction sum
// Copyright (c) 2024 University of Virginia
// This file is licensed under the MIT License.
// See the LICENSE file in the root of this repository for more details.

#include "libpimeval.h"
#include <iostream>
#include <vector>
#include <algorithm>
#include <bitset>
#include <cassert>
#include <cstdlib>
#include <cstdio>
#include <limits>


// test UINT32 reduction sum
bool testRedSum(PimDeviceEnum deviceType)
{
  unsigned numRanks = 2;
  unsigned numBankPerRank = 2;
  unsigned numSubarrayPerBank = 8;
  unsigned numRows = 1024;
  unsigned numCols = 8192;

  uint64_t numElements = 65536;
  std::vector<unsigned> src(numElements);
  std::vector<unsigned> dest(numElements);
  unsigned sum32 = 0;
  uint64_t sum64 = 0;
  unsigned sumRanged32 = 0;
  uint64_t sumRanged64 = 0;
  unsigned idxBegin = 12345;
  unsigned idxEnd = 22222;
  for (uint64_t i = 0; i < numElements; ++i) {
    src[i] = std::numeric_limits<unsigned>::max() - i;  // test when sum is greater than unsigned max
    sum32 += src[i];
    sum64 += src[i];
    if (i >= idxBegin && i < idxEnd) {
      sumRanged32 += src[i];
      sumRanged64 += src[i];
    }
  }

  PimStatus status = pimCreateDevice(deviceType, numRanks, numBankPerRank, numSubarrayPerBank, numRows, numCols);
  assert(status == PIM_OK);

  // test a few iterations
  bool ok = true;
  for (int iter = 0; iter < 2; ++iter) {
    PimObjId obj = pimAlloc(PIM_ALLOC_AUTO, numElements, PIM_UINT32);
    assert(obj != -1);

    status = pimCopyHostToDevice((void*)src.data(), obj);
    assert(status == PIM_OK);
    uint64_t sum = 0;
    status = pimRedSum(obj, static_cast<void*>(&sum));
    assert(status == PIM_OK);

    uint64_t sumRanged = 0;
    status = pimRedSum(obj, static_cast<void*>(&sumRanged), idxBegin, idxEnd);
    assert(status == PIM_OK);

    std::cout << "Result: RedSum: PIM " << sum << " expected 32-bit " << sum32 << " 64-bit " << sum64 << std::endl;
    std::cout << "Result: RedSumRanged: PIM " << sumRanged << " expected 32-bit " << sumRanged32 << " 64-bit " << sumRanged64 << std::endl;

    // results are 64 bit but not 32 bit
    if (sum == sum64 && sumRanged == sumRanged64) {
      std::cout << "Passed!" << std::endl;
    } else {
      std::cout << "Failed!" << std::endl;
      ok = false;
    }

    pimFree(obj);
  }

  pimShowStats();
  pimResetStats();
  pimDeleteDevice();
  return ok;
}

// test BOOL reduction sum
bool testRedSumBool(PimDeviceEnum deviceType)
{
  unsigned numRanks = 2;
  unsigned numBankPerRank = 2;
  unsigned numSubarrayPerBank = 8;
  unsigned numRows = 1024;
  unsigned numCols = 8192;

  uint64_t numElements = 65536 * 32; // multiply by 32 to match #bytes of UINT32 redsum for testing
  std::vector<uint8_t> src(numElements); // use uint8_t on host for bool
  std::vector<uint8_t> dest(numElements);
  uint64_t sum64 = 0;
  uint64_t sumRanged64 = 0;
  unsigned idxBegin = 12345;
  unsigned idxEnd = 22222;
  for (uint64_t i = 0; i < numElements; ++i) {
    src[i] = i % 2; // 0 or 1 only
    sum64 += src[i];
    if (i >= idxBegin && i < idxEnd) {
      sumRanged64 += src[i];
    }
  }

  PimStatus status = pimCreateDevice(deviceType, numRanks, numBankPerRank, numSubarrayPerBank, numRows, numCols);
  assert(status == PIM_OK);

  // test a few iterations
  bool ok = true;
  for (int iter = 0; iter < 2; ++iter) {
    PimObjId obj = pimAlloc(PIM_ALLOC_AUTO, numElements, PIM_BOOL);  // non-associated
    assert(obj != -1);

    status = pimCopyHostToDevice((void*)src.data(), obj);
    assert(status == PIM_OK);
    uint64_t sum = 0;
    status = pimRedSum(obj, static_cast<void*>(&sum));
    assert(status == PIM_OK);

    uint64_t sumRanged = 0;
    status = pimRedSum(obj, static_cast<void*>(&sumRanged), idxBegin, idxEnd);
    assert(status == PIM_OK);

    std::cout << "Result: RedSum: PIM " << sum << " expected 64-bit " << sum64 << std::endl;
    std::cout << "Result: RedSumRanged: PIM " << sumRanged << " expected 64-bit " << sumRanged64 << std::endl;

    // results are 64 bit but not 32 bit
    if (sum == sum64 && sumRanged == sumRanged64) {
      std::cout << "Passed!" << std::endl;
    } else {
      std::cout << "Failed!" << std::endl;
      ok = false;
    }

    pimFree(obj);
  }

  pimShowStats();
  pimResetStats();
  pimDeleteDevice();
  return ok;
}

// test BOOL reduction sum with mixed data type association and padding
bool testRedSumBoolPadded(PimDeviceEnum deviceType)
{
  unsigned numRanks = 2;
  unsigned numBankPerRank = 2;
  unsigned numSubarrayPerBank = 8;
  unsigned numRows = 1024;
  unsigned numCols = 8192;

  uint64_t numElements = 65536;
  std::vector<uint8_t> src(numElements); // use uint8_t on host for bool
  std::vector<uint8_t> dest(numElements);
  uint64_t sum64 = 0;
  uint64_t sumRanged64 = 0;
  unsigned idxBegin = 12345;
  unsigned idxEnd = 22222;
  for (uint64_t i = 0; i < numElements; ++i) {
    src[i] = i % 2; // 0 or 1 only
    sum64 += src[i];
    if (i >= idxBegin && i < idxEnd) {
      sumRanged64 += src[i];
    }
  }

  PimStatus status = pimCreateDevice(deviceType, numRanks, numBankPerRank, numSubarrayPerBank, numRows, numCols);
  assert(status == PIM_OK);

  // test a few iterations
  bool ok = true;
  for (int iter = 0; iter < 2; ++iter) {
    PimObjId objInt = pimAlloc(PIM_ALLOC_AUTO, numElements, PIM_INT32);
    assert(objInt != -1);
    PimObjId obj = pimAllocAssociated(objInt, PIM_BOOL);  // associated
    assert(obj != -1);

    status = pimCopyHostToDevice((void*)src.data(), obj);
    assert(status == PIM_OK);
    uint64_t sum = 0;
    status = pimRedSum(obj, static_cast<void*>(&sum));
    assert(status == PIM_OK);

    uint64_t sumRanged = 0;
    status = pimRedSum(obj, static_cast<void*>(&sumRanged), idxBegin, idxEnd);
    assert(status == PIM_OK);

    std::cout << "Result: RedSum: PIM " << sum << " expected 64-bit " << sum64 << std::endl;
    std::cout << "Result: RedSumRanged: PIM " << sumRanged << " expected 64-bit " << sumRanged64 << std::endl;

    // results are 64 bit but not 32 bit
    if (sum == sum64 && sumRanged == sumRanged64) {
      std::cout << "Passed!" << std::endl;
    } else {
      std::cout << "Failed!" << std::endl;
      ok = false;
    }

    pimFree(obj);
  }

  pimShowStats();
  pimResetStats();
  pimDeleteDevice();
  return ok;
}

int main()
{
  std::cout << "PIM Regression Test: Reduction Sum" << std::endl;

  bool ok = true;

  ok &= testRedSum(PIM_DEVICE_BITSIMD_V);
  ok &= testRedSumBool(PIM_DEVICE_BITSIMD_V);
  ok &= testRedSumBoolPadded(PIM_DEVICE_BITSIMD_V);

  ok &= testRedSum(PIM_DEVICE_FULCRUM);
  ok &= testRedSumBool(PIM_DEVICE_FULCRUM);
  ok &= testRedSumBoolPadded(PIM_DEVICE_FULCRUM);

  ok &= testRedSum(PIM_DEVICE_BANK_LEVEL);
  ok &= testRedSumBool(PIM_DEVICE_BANK_LEVEL);
  ok &= testRedSumBoolPadded(PIM_DEVICE_BANK_LEVEL);

  std::cout << (ok ? "ALL PASSED!" : "FAILED!") << std::endl;

  return 0;
}

