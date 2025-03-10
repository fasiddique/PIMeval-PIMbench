// Test: C++ version of matrix matrix multiplication
// Copyright (c) 2024 University of Virginia
// This file is licensed under the MIT License.
// See the LICENSE file in the root of this repository for more details.

#include <iostream>
#include <vector>
#include <getopt.h>
#include <stdint.h>
#include <iomanip>
#if defined(_OPENMP)
#include <omp.h>
#endif

#include "util.h"
#include "libpimeval.h"

// Params ---------------------------------------------------------------------
typedef struct Params
{
  uint64_t row, columnA, columnB;
  char *configFile;
  char *inputFile;
  bool shouldVerify;
} Params;

void usage()
{
  fprintf(stderr,
          "\nUsage:  ./gemm.out [options]"
          "\n"
          "\n    -r    matrix1 row (default=65536 elements)"
          "\n    -d    matrix1 column (default=65536 elements)"
          "\n    -z    matrix2 column (default=65536 elements)"
          "\n    -c    dramsim config file"
          "\n    -i    input file containing two vectors (default=generates vector with random numbers)"
          "\n    -v    t = verifies PIM output with host output. (default=false)"
          "\n");
}

struct Params getInputParams(int argc, char **argv)
{
  struct Params p;
  p.row = 65536;
  p.columnA = 65536;
  p.columnB = 65536;
  p.configFile = nullptr;
  p.inputFile = nullptr;
  p.shouldVerify = false;

  int opt;
  while ((opt = getopt(argc, argv, "h:r:d:z:c:i:v:")) >= 0)
  {
    switch (opt)
    {
    case 'h':
      usage();
      exit(0);
      break;
    case 'r':
      p.row = strtoull(optarg, NULL, 0);
      break;
    case 'd':
      p.columnA = strtoull(optarg, NULL, 0);
      break;
    case 'z':
      p.columnB = strtoull(optarg, NULL, 0);
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

void gemv(uint64_t row, uint64_t col, std::vector<int> &srcVector, std::vector<std::vector<int>> &srcMatrix, std::vector<int64_t> &dst)
{
  PimObjId srcObj1 = pimAlloc(PIM_ALLOC_AUTO, col, PIM_INT32);
  if (srcObj1 == -1)
  {
    std::cout << "Abort" << std::endl;
    return;
  }
  PimObjId srcObj2 = pimAllocAssociated(srcObj1, PIM_INT32);
  if (srcObj2 == -1)
  {
    std::cout << "Abort" << std::endl;
    return;
  }

  PimObjId dstObj = pimAllocAssociated(srcObj1, PIM_INT32);
  if (dstObj == -1)
  {
    std::cout << "Abort" << std::endl;
    return;
  }

  PimStatus status = pimCopyHostToDevice((void *)srcVector.data(), srcObj2);
  for (uint64_t i = 0; i < row; ++i)
  {
    status = pimCopyHostToDevice((void *)srcMatrix[i].data(), srcObj1);
    if (status != PIM_OK)
    {
      std::cout << "Abort" << std::endl;
      return;
    }

    status = pimMul(srcObj1, srcObj2, dstObj);
    if (status != PIM_OK)
    {
      std::cout << "Abort" << std::endl;
      return;
    }

    status = pimRedSum(dstObj, static_cast<void*>(&dst[i]));
    if (status != PIM_OK)
    {
      std::cout << "Abort" << std::endl;
      return;
    }
  }

  pimFree(srcObj1);
  pimFree(srcObj2);
  pimFree(dstObj);
}

void transposeMatrix(uint64_t row, uint64_t col, std::vector<std::vector<int>> &srcMatrix, std::vector<std::vector<int>> &dstMatrix)
{
#pragma omp parallel for
  for (uint64_t i = 0; i < col; ++i)
  {
    for (uint64_t j = 0; j < row; ++j)
    {
      dstMatrix[i][j] = srcMatrix[j][i];
    }
  }
}

void gemm(uint64_t row, uint64_t colA, uint64_t colB, std::vector<std::vector<int>> &srcMatrixA, std::vector<std::vector<int>> &srcMatrixB, std::vector<std::vector<int64_t>> &dstMatrix, bool shouldVerify)
{
  dstMatrix.resize(row, std::vector<int64_t>(colB, 0));
  vector<std::vector<int>> srcMatrixAT(colA, std::vector<int>(row, 0)), srcMatrixBT(colB, std::vector<int>(colA, 0));
  transposeMatrix(colA, colB, srcMatrixB, srcMatrixBT);
  for (uint64_t i = 0; i < row; ++i)
  {
    gemv(colB, colA, srcMatrixA[i], srcMatrixBT, dstMatrix[i]);
  }
  if (shouldVerify)
  {
    cout << "Starting verification......\n";
    std::vector<std::vector<int64_t>> C(row, std::vector<int64_t>(colB, 0));
    for (uint64_t i = 0; i < row; ++i)
    {
      for (uint64_t j = 0; j < colB; ++j)
      {
        for (uint64_t k = 0; k < colA; ++k)
        {
            C[i][j] += srcMatrixA[i][k] * srcMatrixB[k][j];
        }
      }
    }
    bool shouldContinue = true;
    for (uint64_t i = 0; i < row && shouldContinue; ++i)
    {
      for (uint64_t j = 0; j < colB; ++j)
      {
        if (C[i][j] != dstMatrix[i][j])
        {
          std::cout << "Error: Incorrect Result.\nHost: " << C[i][j] << "\t PIM: " << dstMatrix[i][j] << "\n";
          shouldContinue = false;
          break;
        }
      }
    }
  }
}

int main(int argc, char *argv[])
{
  struct Params params = getInputParams(argc, argv);
  std::cout << "Row: " << params.row << " Column: " << params.columnA << "\n";

  std::vector<int> srcVector, resultVector;
  std::vector<std::vector<int>> srcMatrixA, srcMatrixB;
  std::vector<std::vector<int64_t>> dstMatrix;
  if (params.inputFile == nullptr)
  {
    getMatrix(params.row, params.columnA, 0, srcMatrixA);
    getMatrix(params.columnA, params.columnB, 0, srcMatrixB);
  }
  else
  {
    std::cout << "Reading from input file is not implemented yet." << std::endl;
    return 0;
  }

  if (!createDevice(params.configFile))
    return 1;

  // TODO: Check if vector can fit in one iteration. Otherwise need to run in multiple iteration.
  gemm(params.row, params.columnA, params.columnB, srcMatrixA, srcMatrixB, dstMatrix, params.shouldVerify);

  pimShowStats();

  return 0;
}
