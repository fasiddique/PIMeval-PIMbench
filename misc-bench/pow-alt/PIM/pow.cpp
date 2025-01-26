// Test: C++ version of pow (x, n)
// Copyright (c) 2024 University of Virginia
// This file is licensed under the MIT License.
// See the LICENSE file in the root of this repository for more details.

#include <iostream>
#include <vector>
#include <getopt.h>
#include <stdint.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <sys/mman.h>
#include <unistd.h>
#include <cstring>
#include <cassert>
#include <cinttypes>
#if defined(_OPENMP)
#include <omp.h>
#endif

#include "../../util.h"
#include "libpimeval.h"

// Params ---------------------------------------------------------------------
typedef struct Params
{
  uint64_t dataSize;
  uint64_t n;
  char *configFile;
  char *inputFile;
  bool shouldVerify;
} Params;

void usage()
{
  fprintf(stderr,
          "\nUsage:  ./pow.out [options]"
          "\n"
          "\n    -l    vector length (default=2048 elements)"
          "\n    -n    exponent (default=2)"
          "\n    -c    dramsim config file"
          "\n    -i    input file containing two vectors (default=generates vectors with random numbers)"
          "\n    -v    t = verifies PIM output with host output. (default=false)"
          "\n");
}

struct Params getInputParams(int argc, char **argv)
{
  struct Params p;
  p.dataSize = 2048;
  p.n = 2;
  p.configFile = nullptr;
  p.inputFile = nullptr;
  p.shouldVerify = false;

  int opt;
  while ((opt = getopt(argc, argv, "h:l:n:c:i:v:")) >= 0)
  {
    switch (opt)
    {
    case 'h':
      usage();
      exit(0);
      break;
    case 'l':
      p.dataSize = strtoull(optarg, NULL, 0);
      break;
    case 'n':
      p.n = strtoull(optarg, NULL, 0);
      break;
    case 'c':
      p.configFile = optarg;
      break;
    case 'i':
      p.inputFile = optarg;
      break;
    case 'v':
      p.shouldVerify = (*optarg == 't') ? true : false;
      break;
    default:
      fprintf(stderr, "\nUnrecognized option!\n");
      usage();
      exit(0);
    }
  }
  return p;
}

void pow(uint64_t vectorLength, uint64_t exponent, const std::vector<int> src, std::vector<int> &dst)
{
  PimObjId srcObj = pimAlloc(PIM_ALLOC_AUTO, vectorLength, PIM_INT32);
  assert(srcObj != -1);
  PimObjId dstObj = pimAllocAssociated(srcObj, PIM_INT32);
  assert(dstObj != -1);

  PimStatus status = pimCopyHostToDevice((void *) src.data(), srcObj);
  assert(status == PIM_OK);

  status = pimCopyHostToDevice((void *) src.data(), dstObj);
  assert(status == PIM_OK);
  
  dst.resize(vectorLength);

  status = pimPow(srcObj, dstObj, exponent);
  assert(status == PIM_OK);

  dst.resize(vectorLength);
  status = pimCopyDeviceToHost(dstObj, (void *) dst.data());
  assert(status == PIM_OK);
  
  pimFree(dstObj);
}

int main(int argc, char *argv[])
{
  struct Params params = getInputParams(argc, argv);
  std::vector<int> src, dst;
  uint64_t vectorLength = params.dataSize;
  
  if (params.inputFile == nullptr)
  {
    src.resize(vectorLength);
    
    getVector(vectorLength, src);
  }
  else
  {
    std::cerr << "Reading from the input file is not implemented yet for the input matrix" << std::endl;
    return 1;
  }

  std::printf("Performing power of %" PRIu64 " on %" PRIu64 " data points\n", params.n, vectorLength);

  if (!createDevice(params.configFile))
  {
    return 1;
  }

  pow(vectorLength, params.n, src, dst);

  if (params.shouldVerify)
  {  
    int errorFlag = 0;

    #pragma omp parallel for 
    for (uint64_t i = 0; i < vectorLength; ++i) 
    {
      int res = std::pow(src[i], params.n);    
      if (res != dst[i])
      {
        std::cout << "Wrong answer at index " << i << " | Wrong PIM answer = " << dst[i] << " (Baseline expected = " << res << ")" << std::endl;
        errorFlag = 1;
      }
    }
    if (!errorFlag)
    {
      std::cout << "Correct!" << std::endl;
    }
  }

  pimShowStats();

  return 0;
}
