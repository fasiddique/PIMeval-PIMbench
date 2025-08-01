// File: pimPerfEnergyBankLevel.cc
// PIMeval Simulator - Performance Energy Models
// Copyright (c) 2024 University of Virginia
// This file is licensed under the MIT License.
// See the LICENSE file in the root of this repository for more details.

#include "pimPerfEnergyBankLevel.h"
#include "pimCmd.h"
#include <cstdio>
#include <cmath>


//! @brief  Perf energy model of bank-level PIM for func1
pimeval::perfEnergy
pimPerfEnergyBankLevel::getPerfEnergyForFunc1(PimCmdEnum cmdType, const pimObjInfo& obj, const pimObjInfo& objDest) const
{
  double msRuntime = 0.0;
  double mjEnergy = 0.0;
  double msRead = 0.0;
  double msWrite = 0.0;
  double msCompute = 0.0;
  unsigned numPass = obj.getMaxNumRegionsPerCore();
  unsigned bitsPerElement = obj.getBitsPerElement(PimBitWidth::ACTUAL);
  uint64_t totalOp = 0;
  if (cmdType == PimCmdEnum::CONVERT_TYPE) {
    // for type conversion, ALU parallelism is determined by the wider data type
    bitsPerElement = std::max(bitsPerElement, objDest.getBitsPerElement(PimBitWidth::ACTUAL));
  }
  unsigned numCores = obj.isLoadBalanced() ? obj.getNumCoreAvailable() : obj.getNumCoresUsed();

  unsigned maxElementsPerRegion = obj.getMaxElementsPerRegion();
  double numberOfOperationPerElement = ((double)bitsPerElement / m_blimpCoreBitWidth);
  unsigned minElementPerRegion = obj.isLoadBalanced() ? (std::ceil(obj.getNumElements() * 1.0 / numCores) - (maxElementsPerRegion * (numPass - 1))) : maxElementsPerRegion;
  // How many iteration require to read / write max elements per region
  unsigned maxGDLItr = std::ceil(maxElementsPerRegion * bitsPerElement * 1.0 / m_GDLWidth);
  unsigned minGDLItr = std::ceil(minElementPerRegion * bitsPerElement * 1.0 / m_GDLWidth);
  unsigned numBankPerChip = numCores / m_numChipsPerRank;
  double activateMS = minGDLItr * m_tGDL < m_tRAS * m_tCK ? m_tRAS * m_tCK : m_tACT; // Use tRAS if GDL is less than tRAS
  // for scalar operations an extra read is required to read the scalar value
  switch (cmdType)
  {
    case PimCmdEnum::COPY_O2O:
    {
      msRead = ((m_tACT + m_tPRE + maxGDLItr * m_tGDL) * (numPass - 1)) + (activateMS + m_tPRE + (minGDLItr * m_tGDL));
      msWrite = ((m_tACT + m_tPRE + maxGDLItr * m_tGDL) * (numPass - 1)) + (activateMS + m_tPRE + (minGDLItr * m_tGDL));
      msCompute = 0;
      msRuntime = msRead + msWrite + msCompute;
      mjEnergy = numPass * numCores * (m_eACT + m_ePRE) * 2;
      mjEnergy += ((m_eR * maxGDLItr * (numPass-1) * numBankPerChip * m_numRanks) + (m_eR * minGDLItr * numBankPerChip * m_numRanks));
      mjEnergy += ((m_eW * maxGDLItr * (numPass-1) * numBankPerChip * m_numRanks) + (m_eW * minGDLItr * numBankPerChip * m_numRanks));
      mjEnergy += m_pBChip * m_numChipsPerRank * m_numRanks * msRuntime;
      break;
    }
    case PimCmdEnum::POPCOUNT:
    case PimCmdEnum::ABS:
    case PimCmdEnum::BIT_SLICE_EXTRACT:
    case PimCmdEnum::BIT_SLICE_INSERT:
    case PimCmdEnum::CONVERT_TYPE:
    {
      if (cmdType == PimCmdEnum::BIT_SLICE_EXTRACT) {
        // Assume on ALU cycle to do this for now
        // numberOfOperationPerElement *= 2; // 1 shift, 1 and
      } else if (cmdType == PimCmdEnum::BIT_SLICE_INSERT) {
        // Assume on ALU cycle to do this for now
        // numberOfOperationPerElement *= 5; // 2 shifts, 1 not, 1 and, 1 or
      }
      // Refer to fulcrum documentation
      msRead = (m_tACT + m_tPRE) * (numPass - 1) + (activateMS + m_tPRE);
      msWrite = ((m_tACT + m_tPRE + maxGDLItr * m_tGDL) * (numPass - 1)) + (activateMS + m_tPRE + (minGDLItr * m_tGDL));
      msCompute = (maxElementsPerRegion * m_blimpLatency * numberOfOperationPerElement * (numPass - 1)) + (minElementPerRegion * m_blimpLatency * numberOfOperationPerElement);
      msRuntime = msRead + msWrite + msCompute;
      mjEnergy = ((m_eACT + m_ePRE) * 2 + (maxElementsPerRegion * m_blimpArithmeticEnergy * numberOfOperationPerElement)) * numCores * (numPass - 1);
      mjEnergy += ((m_eACT + m_ePRE) * 2 + (minElementPerRegion * m_blimpArithmeticEnergy * numberOfOperationPerElement)) * numCores;
      mjEnergy += (m_eR * maxGDLItr * (numPass-1) * numBankPerChip * m_numRanks + (m_eR * minGDLItr * numBankPerChip * m_numRanks));
      mjEnergy += (m_eW * maxGDLItr * (numPass-1) * numBankPerChip * m_numRanks + (m_eW * minGDLItr * numBankPerChip * m_numRanks));
      mjEnergy += m_pBChip * m_numChipsPerRank * m_numRanks * msRuntime;
      totalOp = obj.getNumElements();
      break;
    }
    case PimCmdEnum::ADD_SCALAR:
    case PimCmdEnum::SUB_SCALAR:
    case PimCmdEnum::MUL_SCALAR:
    case PimCmdEnum::DIV_SCALAR:
    {
      msRead = (m_tACT + m_tPRE) * (numPass - 1) + (activateMS + m_tPRE) + m_tR + m_tGDL;
      msWrite = ((m_tACT + m_tPRE + maxGDLItr * m_tGDL) * (numPass - 1)) + (activateMS + m_tPRE + (minGDLItr * m_tGDL));
      msCompute = (maxElementsPerRegion * m_blimpLatency * numberOfOperationPerElement * (numPass - 1)) + (minElementPerRegion * m_blimpLatency * numberOfOperationPerElement);
      msRuntime = msRead + msWrite + msCompute;
      mjEnergy = ((m_eACT + m_ePRE) * 2 + (maxElementsPerRegion * m_blimpArithmeticEnergy * numberOfOperationPerElement)) * numCores * (numPass - 1);
      mjEnergy += ((m_eACT + m_ePRE) * 2 + (minElementPerRegion * m_blimpArithmeticEnergy * numberOfOperationPerElement)) * numCores;
      mjEnergy += (m_eR * maxGDLItr * (numPass-1) * numBankPerChip * m_numRanks + (m_eR * minGDLItr * numBankPerChip * m_numRanks)) + (m_eAP * numCores + m_eR * numBankPerChip * m_numRanks);
      mjEnergy += (m_eW * maxGDLItr * (numPass-1) * numBankPerChip * m_numRanks + (m_eW * minGDLItr * numBankPerChip * m_numRanks));
      mjEnergy += m_pBChip * m_numChipsPerRank * m_numRanks * msRuntime;
      totalOp = obj.getNumElements();
      break;
    }
    case PimCmdEnum::AND_SCALAR:
    case PimCmdEnum::OR_SCALAR:
    case PimCmdEnum::XOR_SCALAR:
    case PimCmdEnum::XNOR_SCALAR:
    case PimCmdEnum::GT_SCALAR:
    case PimCmdEnum::LT_SCALAR:
    case PimCmdEnum::EQ_SCALAR:
    case PimCmdEnum::NE_SCALAR:
    case PimCmdEnum::MIN_SCALAR:
    case PimCmdEnum::MAX_SCALAR:
    {
      msRead = (m_tACT + m_tPRE) * (numPass - 1) + m_tR + m_tGDL + activateMS + m_tPRE;
      msWrite = ((m_tACT + m_tPRE + maxGDLItr * m_tGDL) * (numPass - 1)) + (activateMS + m_tPRE + (minGDLItr * m_tGDL));
      msCompute = (maxElementsPerRegion * m_blimpLatency * numberOfOperationPerElement * (numPass - 1)) + (minElementPerRegion * m_blimpLatency * numberOfOperationPerElement);
      msRuntime = msRead + msWrite + msCompute;
      mjEnergy = (((m_eACT + m_ePRE) * 2) +  (maxElementsPerRegion * m_blimpLogicalEnergy * numberOfOperationPerElement)) * numCores * (numPass - 1);
      mjEnergy += (((m_eACT + m_ePRE) * 2) + (minElementPerRegion * m_blimpLogicalEnergy * numberOfOperationPerElement)) * numCores;
      mjEnergy += (m_eR * maxGDLItr * (numPass-1) * numBankPerChip * m_numRanks + (m_eR * minGDLItr * numBankPerChip * m_numRanks)) + (m_eAP * numCores + m_eR * numBankPerChip * m_numRanks);
      mjEnergy += (m_eW * maxGDLItr * (numPass-1) * numBankPerChip * m_numRanks + (m_eW * minGDLItr * numBankPerChip * m_numRanks));
      mjEnergy += m_pBChip * m_numChipsPerRank * m_numRanks * msRuntime;
      totalOp = obj.getNumElements();
      break;
    }
    case PimCmdEnum::SHIFT_BITS_L:
    case PimCmdEnum::SHIFT_BITS_R:
    {
      msRead = (m_tACT + m_tPRE) * (numPass - 1) + (activateMS + m_tPRE);
      msWrite = ((m_tACT + m_tPRE + maxGDLItr * m_tGDL) * (numPass - 1)) + (activateMS + m_tPRE + (minGDLItr * m_tGDL));
      msCompute = (maxElementsPerRegion * m_blimpLatency * numberOfOperationPerElement * (numPass - 1)) + (minElementPerRegion * m_blimpLatency * numberOfOperationPerElement);
      msRuntime = msRead + msWrite + msCompute;
      mjEnergy = (((m_eACT + m_ePRE) * 2) +  (maxElementsPerRegion * m_blimpLogicalEnergy * numberOfOperationPerElement)) * numCores * (numPass - 1);
      mjEnergy += (((m_eACT + m_ePRE) * 2) + (minElementPerRegion * m_blimpLogicalEnergy * numberOfOperationPerElement)) * numCores;
      mjEnergy += (m_eR * maxGDLItr * (numPass-1) * numBankPerChip * m_numRanks + (m_eR * minGDLItr * numBankPerChip * m_numRanks));
      mjEnergy += (m_eW * maxGDLItr * (numPass-1) * numBankPerChip * m_numRanks + (m_eW * minGDLItr * numBankPerChip * m_numRanks));
      mjEnergy += m_pBChip * m_numChipsPerRank * m_numRanks * msRuntime;
      totalOp = obj.getNumElements();
      break;
    }
    case PimCmdEnum::AES_SBOX:
    case PimCmdEnum::AES_INVERSE_SBOX:
    {
      // NOTE:
      // Although the Processing Element (PE) is 32 bits wide and can theoretically perform four 8-bit operations in parallel,
      // in the case of these LUT-based commands (e.g., AES S-box or inverse S-box), each operation is treated as a single,
      // independent access driven by an 8-bit input.
      //
      // If the operation instead made full use of the 32-bit PE width to process four 8-bit inputs in parallel
      // then numberOfOperationPerElement would be 0.25. However, such parallelism is not modeled here due to the limitation of the LUT.
      // Therefore, for the uint8 data type, we set numberOfOperationPerElement = 1 because each 8-bit input
      // corresponds to one logical LUT access, and we assume that this access is not vectorized across multiple inputs
      // within a single PE execution. In other words, we model the cost at the granularity of one element per operation.
      numberOfOperationPerElement = 1;
      msRead = (m_tACT + m_tPRE) * (numPass - 1) + (activateMS + m_tPRE);
      msWrite = ((m_tACT + m_tPRE + maxGDLItr * m_tGDL) * (numPass - 1)) + (activateMS + m_tPRE + (minGDLItr * m_tGDL));
      msCompute = (maxElementsPerRegion * m_blimpLatency * numberOfOperationPerElement * (numPass - 1)) + (minElementPerRegion * m_blimpLatency * numberOfOperationPerElement);
      msRuntime = msRead + msWrite + msCompute;
      mjEnergy = ((m_eAP * 2) +  (maxElementsPerRegion * m_blimpLogicalEnergy * numberOfOperationPerElement)) * numCores * (numPass - 1);
      mjEnergy += ((m_eAP * 2) + (minElementPerRegion * m_blimpLogicalEnergy * numberOfOperationPerElement)) * numCores;
      mjEnergy += (m_eR * maxGDLItr * (numPass-1) * numBankPerChip * m_numRanks + (m_eR * minGDLItr * numBankPerChip * m_numRanks));
      mjEnergy += (m_eW * maxGDLItr * (numPass-1) * numBankPerChip * m_numRanks + (m_eW * minGDLItr * numBankPerChip * m_numRanks));
      mjEnergy += m_pBChip * m_numChipsPerRank * m_numRanks * msRuntime;
      totalOp = obj.getNumElements();
      break;
    }
    default:
      printf("PIM-Warning: Perf energy model not available for PIM command %s\n", pimCmd::getName(cmdType, "").c_str());
      break;
  }

  return pimeval::perfEnergy(msRuntime, mjEnergy, msRead, msWrite, msCompute, totalOp);
}

//! @brief  Perf energy model of bank-level PIM for func2
pimeval::perfEnergy
pimPerfEnergyBankLevel::getPerfEnergyForFunc2(PimCmdEnum cmdType, const pimObjInfo& obj, const pimObjInfo& objSrc2, const pimObjInfo& objDest) const
{
  double msRuntime = 0.0;
  double mjEnergy = 0.0;
  double msRead = 0.0;
  double msWrite = 0.0;
  double msCompute = 0.0;
  unsigned numPass = obj.getMaxNumRegionsPerCore();
  unsigned bitsPerElement = obj.getBitsPerElement(PimBitWidth::ACTUAL);
  unsigned numCoresUsed = obj.isLoadBalanced() ? obj.getNumCoreAvailable() : obj.getNumCoresUsed();

  unsigned maxElementsPerRegion = obj.getMaxElementsPerRegion();
  double numberOfOperationPerElement = ((double)bitsPerElement / m_blimpCoreBitWidth);
  unsigned minElementPerRegion = obj.isLoadBalanced() ? (std::ceil(obj.getNumElements() * 1.0 / numCoresUsed) - (maxElementsPerRegion * (numPass - 1))) : maxElementsPerRegion;
  // How many iteration require to read / write max elements per region
  unsigned maxGDLItr = std::ceil(maxElementsPerRegion * bitsPerElement * 1.0 / m_GDLWidth);
  unsigned minGDLItr = std::ceil(minElementPerRegion * bitsPerElement * 1.0 / m_GDLWidth);
  uint64_t totalOp = 0;
  unsigned numBankPerChip = numCoresUsed / m_numChipsPerRank;
  double activateMS = minGDLItr * m_tGDL < m_tRAS * m_tCK ? m_tRAS * m_tCK : m_tACT; // Use tRAS if GDL is less than tRAS

  switch (cmdType)
  {
    case PimCmdEnum::ADD:
    case PimCmdEnum::SUB:
    case PimCmdEnum::MUL:
    case PimCmdEnum::DIV:
    {
      msRead = ((2 * (m_tACT + m_tPRE)) + (maxGDLItr * m_tGDL)) * (numPass - 1) + ((2 * (activateMS + m_tPRE)) + (minGDLItr * m_tGDL));
      msWrite = ((m_tACT + m_tPRE) + (maxGDLItr * m_tGDL)) * (numPass - 1) + ((activateMS + m_tPRE) + (minGDLItr * m_tGDL));
      msCompute = (maxElementsPerRegion * m_blimpLatency * numberOfOperationPerElement * (numPass - 1)) + (minElementPerRegion * m_blimpLatency * numberOfOperationPerElement);
      msRuntime = msRead + msWrite + msCompute;
      mjEnergy = (((m_eACT + m_ePRE) * 3) + (maxElementsPerRegion * m_blimpArithmeticEnergy * numberOfOperationPerElement)) * numCoresUsed * (numPass - 1);
      mjEnergy += (((m_eACT + m_ePRE) * 3) + (minElementPerRegion * m_blimpArithmeticEnergy * numberOfOperationPerElement)) * numCoresUsed;
      mjEnergy += ((m_eR * 2 * maxGDLItr * (numPass-1)) + (m_eR * 2 * minGDLItr)) * numBankPerChip * m_numRanks;
      mjEnergy += ((m_eW * maxGDLItr * (numPass-1)) + (m_eW * minGDLItr)) * numBankPerChip * m_numRanks;
      mjEnergy += m_pBChip * m_numChipsPerRank * m_numRanks * msRuntime;
      totalOp = obj.getNumElements();
      break;
    }
    case PimCmdEnum::SCALED_ADD:
    {
      /**
       * Performs a multiply-add operation on rows in DRAM.
       *
       * This command executes the following steps:
       * 1. Multiply the elements of a source row by a scalar value.
       * 2. Add the result of the multiplication to the elements of another row.
       * 3. Write the final result back to a row in DRAM.
       *
       * Performance Optimizations:
       * - While performing the multiplication, the next row to be added can be fetched without any additional overhead.
       * - During the addition, the next row to be multiplied can be fetched concurrently.
       *
       * As a result, only one read operation is necessary for the entire pass.
      */
      msRead = ((m_tACT + m_tPRE) * 2) * (numPass - 1) + (m_tR + m_tGDL) + (activateMS + m_tPRE);
      msWrite = ((m_tACT + m_tPRE) + (maxGDLItr * m_tGDL)) * (numPass - 1) + ((activateMS + m_tPRE) + (minGDLItr * m_tGDL));
      msCompute = (maxElementsPerRegion * m_blimpLatency * numberOfOperationPerElement * 2 * (numPass - 1)) + (minElementPerRegion * m_blimpLatency * numberOfOperationPerElement * 2);
      msRuntime = msRead + msWrite + msCompute;
      mjEnergy = (((m_eACT + m_ePRE) * 3) + (maxElementsPerRegion * m_blimpArithmeticEnergy * numberOfOperationPerElement * 2)) * numCoresUsed * (numPass - 1);
      mjEnergy += (((m_eACT + m_ePRE) * 3) + (minElementPerRegion * m_blimpArithmeticEnergy * numberOfOperationPerElement * 2)) * numCoresUsed;
      mjEnergy += ((m_eR * 2 * maxGDLItr * (numPass-1)) + (m_eR * 2 * minGDLItr)) * numBankPerChip * m_numRanks + (m_eAP * numCoresUsed + m_eR * numBankPerChip * m_numRanks);
      mjEnergy += ((m_eW * maxGDLItr * (numPass-1)) + (m_eW * minGDLItr)) * numBankPerChip * m_numRanks;
      mjEnergy += m_pBChip * m_numChipsPerRank * m_numRanks * msRuntime;
      totalOp = obj.getNumElements() * 2;
      break;
    }
    case PimCmdEnum::AND:
    case PimCmdEnum::OR:
    case PimCmdEnum::XOR:
    case PimCmdEnum::XNOR:
    case PimCmdEnum::GT:
    case PimCmdEnum::LT:
    case PimCmdEnum::EQ:
    case PimCmdEnum::NE:
    case PimCmdEnum::MIN:
    case PimCmdEnum::MAX:
    case PimCmdEnum::COND_BROADCAST:
    case PimCmdEnum::COND_SELECT:
    case PimCmdEnum::COND_SELECT_SCALAR:
    {
      msRead = ((2 * (m_tACT + m_tPRE)) + (maxGDLItr * m_tGDL)) * (numPass - 1) + ((2 * (activateMS + m_tPRE)) + (minGDLItr * m_tGDL));
      msWrite = ((m_tACT + m_tPRE) + (maxGDLItr * m_tGDL)) * (numPass - 1) + ((activateMS + m_tPRE) + (minGDLItr * m_tGDL));
      msCompute = (maxElementsPerRegion * m_blimpLatency * numberOfOperationPerElement * (numPass - 1)) + (minElementPerRegion * m_blimpLatency * numberOfOperationPerElement);
      msRuntime = msRead + msWrite + msCompute;
      mjEnergy = (((m_eACT + m_ePRE) * 3) + (maxElementsPerRegion * m_blimpLogicalEnergy * numberOfOperationPerElement)) * numCoresUsed * (numPass - 1);
      mjEnergy += (((m_eACT + m_ePRE) * 3) + (minElementPerRegion * m_blimpLogicalEnergy * numberOfOperationPerElement)) * numCoresUsed;
      mjEnergy += ((m_eR * 2 * maxGDLItr * (numPass-1)) + (m_eR * 2 * minGDLItr)) * numBankPerChip * m_numRanks;
      mjEnergy += ((m_eW * maxGDLItr * (numPass-1)) + (m_eW * minGDLItr)) * numBankPerChip * m_numRanks;
      mjEnergy += m_pBChip * m_numChipsPerRank * m_numRanks * msRuntime;
      totalOp = obj.getNumElements();
      break;
    }
    default:
      printf("PIM-Warning: Perf energy model not available for PIM command %s\n", pimCmd::getName(cmdType, "").c_str());
      break;
  }
  return pimeval::perfEnergy(msRuntime, mjEnergy, msRead, msWrite, msCompute, totalOp);
}

//! @brief  Perf energy model of bank-level PIM for reduction sum
pimeval::perfEnergy
pimPerfEnergyBankLevel::getPerfEnergyForReduction(PimCmdEnum cmdType, const pimObjInfo& obj, unsigned numPass) const
{
  double msRuntime = 0.0;
  double mjEnergy = 0.0;
  double msRead = 0.0;
  double msWrite = 0.0;
  double msCompute = 0.0;
  unsigned bitsPerElement = obj.getBitsPerElement(PimBitWidth::ACTUAL);
  unsigned maxElementsPerRegion = obj.getMaxElementsPerRegion();
  unsigned numCore = obj.isLoadBalanced() ? obj.getNumCoreAvailable() : obj.getNumCoresUsed();
  double cpuTDP = 225; // W; AMD EPYC 9124 16 core
  unsigned minElementPerRegion = obj.isLoadBalanced() ? (std::ceil(obj.getNumElements() * 1.0 / numCore) - (maxElementsPerRegion * (numPass - 1))) : maxElementsPerRegion;
  // How many iteration require to read / write max elements per region
  unsigned maxGDLItr = std::ceil(maxElementsPerRegion * bitsPerElement * 1.0 / m_GDLWidth);
  unsigned minGDLItr = std::ceil(minElementPerRegion * bitsPerElement * 1.0 / m_GDLWidth);
  uint64_t totalOp = 0;
  unsigned numBankPerChip = numCore / m_numChipsPerRank;
  double activateMS = minGDLItr * m_tGDL < m_tRAS * m_tCK ? m_tRAS * m_tCK : m_tACT; // Use tRAS if GDL is less than tRAS

  switch (cmdType) {
    case PimCmdEnum::REDSUM:
    case PimCmdEnum::REDSUM_RANGE:
    case PimCmdEnum::REDMIN:
    case PimCmdEnum::REDMIN_RANGE:
    case PimCmdEnum::REDMAX:
    case PimCmdEnum::REDMAX_RANGE:
    {
      // How many iteration require to read / write max elements per region
      double numberOfOperationPerElement = ((double)bitsPerElement / m_blimpCoreBitWidth);
      msRead = (m_tACT + m_tPRE) * (numPass - 1) + (activateMS + m_tPRE);
      // reduction for all regions assuming 16 core AMD EPYC 9124
      double aggregateMs = static_cast<double>(obj.getNumCoresUsed()) / 2300000;
      msCompute = (maxElementsPerRegion * m_blimpLatency * numberOfOperationPerElement * (numPass - 1)) + (minElementPerRegion * m_blimpLatency * numberOfOperationPerElement) + aggregateMs;
      msRuntime = msRead + msWrite + msCompute;

      // Refer to fulcrum documentation
      mjEnergy = ((m_eACT + m_ePRE) + (maxElementsPerRegion * m_blimpArithmeticEnergy * numberOfOperationPerElement)) * (numPass - 1) * numCore;
      mjEnergy += ((m_eACT + m_ePRE) + (minElementPerRegion * m_blimpArithmeticEnergy * numberOfOperationPerElement)) * numCore;
      mjEnergy += aggregateMs * cpuTDP;
      mjEnergy += ((m_eR * maxGDLItr * (numPass-1)) + (m_eR * minGDLItr)) * numBankPerChip;
      mjEnergy += m_pBChip * m_numChipsPerRank * m_numRanks * msRuntime;
      totalOp = obj.getNumElements();
      break;
    }
    default:
      printf("PIM-Warning: Unsupported reduction command for bank-level PIM: %s\n", pimCmd::getName(cmdType, "").c_str());
      break;
    }

  return pimeval::perfEnergy(msRuntime, mjEnergy, msRead, msWrite, msCompute, totalOp);
}

//! @brief  Perf energy model of bank-level PIM for broadcast
pimeval::perfEnergy
pimPerfEnergyBankLevel::getPerfEnergyForBroadcast(PimCmdEnum cmdType, const pimObjInfo& obj) const
{
  double msRuntime = 0.0;
  double mjEnergy = 0.0;
  double msRead = 0.0;
  double msWrite = 0.0;
  double msCompute = 0.0;
  unsigned numPass = obj.getMaxNumRegionsPerCore();
  unsigned numCore = obj.isLoadBalanced() ? obj.getNumCoreAvailable() : obj.getNumCoresUsed();
  unsigned bitsPerElement = obj.getBitsPerElement(PimBitWidth::ACTUAL);
  unsigned maxElementsPerRegion = obj.getMaxElementsPerRegion();
  unsigned minElementPerRegion = obj.isLoadBalanced() ? (std::ceil(obj.getNumElements() * 1.0 / obj.getNumCoreAvailable()) - (maxElementsPerRegion * (numPass - 1))) : maxElementsPerRegion;
  // How many iteration require to read / write max elements per region
  unsigned maxGDLItr = std::ceil(maxElementsPerRegion * bitsPerElement * 1.0 / m_GDLWidth);
  unsigned minGDLItr = std::ceil(minElementPerRegion * bitsPerElement * 1.0 / m_GDLWidth);
  unsigned numBankPerChip = numCore / m_numChipsPerRank;
  double activateMS = minGDLItr * m_tGDL < m_tRAS * m_tCK ? m_tRAS * m_tCK : m_tACT; // Use tRAS if GDL is less than tRAS
  uint64_t totalOp = 0;
  msWrite = ((m_tACT + m_tPRE) + (maxGDLItr * m_tGDL)) * (numPass - 1) + ((activateMS + m_tPRE) + (minGDLItr * m_tGDL));

  msRuntime = msRead + msWrite + msCompute;
  mjEnergy = (m_eACT + m_ePRE) * numPass * numCore;
  mjEnergy += (m_eW * maxGDLItr * (numPass-1) + m_eW * minGDLItr) * numBankPerChip;
  mjEnergy += m_pBChip * m_numChipsPerRank * m_numRanks * msRuntime;
  return pimeval::perfEnergy(msRuntime, mjEnergy, msRead, msWrite, msCompute, totalOp);
}

// TODO: This needs to be revisited
//! @brief  Perf energy model of bank-level PIM for rotate
pimeval::perfEnergy
pimPerfEnergyBankLevel::getPerfEnergyForRotate(PimCmdEnum cmdType, const pimObjInfo& obj) const
{
  double msRuntime = 0.0;
  double mjEnergy = 0.0;
  double msRead = 0.0;
  double msWrite = 0.0;
  double msCompute = 0.0;
  unsigned numPass = obj.getMaxNumRegionsPerCore();
  unsigned bitsPerElement = obj.getBitsPerElement(PimBitWidth::ACTUAL);
  unsigned numRegions = obj.getRegions().size();
  uint64_t totalOp = 0;
  // boundary handling - assume two times copying between device and host for boundary elements
  pimeval::perfEnergy perfEnergyBT = getPerfEnergyForBytesTransfer(PimCmdEnum::COPY_D2H, numRegions * bitsPerElement / 8);

  // rotate within subarray:
  // For every bit: Read row to SA; move SA to R1; Shift R1 by N steps; Move R1 to SA; Write SA to row
  // TODO: separate bank level and GDL
  // TODO: energy unimplemented
  // TODO: perf per watt
  msRuntime = (m_tR + (bitsPerElement + 2) * m_tL + m_tW); // for one pass
  msRuntime *= numPass;
  mjEnergy = (m_eAP + (bitsPerElement + 2) * m_eL) * numPass;
  msRuntime += 2 * perfEnergyBT.m_msRuntime;
  mjEnergy += 2 * perfEnergyBT.m_mjEnergy;
  printf("PIM-Warning: Perf energy model is not precise for PIM command %s\n", pimCmd::getName(cmdType, "").c_str());

  return pimeval::perfEnergy(msRuntime, mjEnergy, msRead, msWrite, msCompute, totalOp);
}

//! @brief  Perf energy model of bank-level PIM for prefix-sum
pimeval::perfEnergy
pimPerfEnergyBankLevel::getPerfEnergyForPrefixSum(PimCmdEnum cmdType, const pimObjInfo& obj) const
{
  double msRuntime = 0.0;
  double mjEnergy = 0.0;
  double msRead = 0.0;
  double msWrite = 0.0;
  double msCompute = 0.0;
  unsigned numPass = obj.getMaxNumRegionsPerCore();
  unsigned bitsPerElement = obj.getBitsPerElement(PimBitWidth::ACTUAL);
  unsigned maxElementsPerRegion = obj.getMaxElementsPerRegion();
  unsigned numCore = obj.isLoadBalanced() ? obj.getNumCoreAvailable() : obj.getNumCoresUsed();
  double cpuTDP = 225; // W; AMD EPYC 9124 16 core
  unsigned minElementPerRegion = obj.isLoadBalanced() ? (std::ceil(obj.getNumElements() * 1.0 / numCore) - (maxElementsPerRegion * (numPass - 1))) : maxElementsPerRegion;
  // How many iteration require to read / write max elements per region
  unsigned maxGDLItr = std::ceil(maxElementsPerRegion * bitsPerElement * 1.0 / m_GDLWidth);
  unsigned minGDLItr = std::ceil(minElementPerRegion * bitsPerElement * 1.0 / m_GDLWidth);
  uint64_t totalOp = 0;
  unsigned numBankPerChip = numCore / m_numChipsPerRank;
  double activateMS = minGDLItr * m_tGDL < m_tRAS * m_tCK ? m_tRAS * m_tCK : m_tACT; // Use tRAS if GDL is less than tRAS
  switch (cmdType) {
    case PimCmdEnum::PREFIX_SUM:
    {
      /**
       * Performs prefix sum: dstVec[i] = dstVec[i-1] + srcVec[i]
       *
       * Execution Steps:
       * 1. Each bank performs a local prefix sum on its portion of the data.
       * 2. The host CPU fetches the final value from each subarray using `n`
       * DRAM READ. Here, `n = number of banks`.
       * 3. The host CPU aggregates these values (i.e., computes the prefix sum
       * across banks).
       * 4. The host CPU writes the aggregated values back to DRAM using `n`
       * DRAM WRITE.
       * 5. Each bank updates its elements using the received value to complete
       * the final prefix sum.
       *
       * Performance Model:
       * - While performing addition, the next row can be
       * fetched concurrently. As a result, `msRead = 2 * m_tR` (multiplied by
       * two because, two prefix sum iterations are required).
       * - `aggregateMs` models the time for host-side aggregation.
       * - `hostRW` accounts for host read/write overhead, including DRAM tR,
       * tW, and GDL delays.
       *
       */

      // How many iteration require to read / write max elements per region
      double numberOfOperationPerElement = ((double)bitsPerElement / m_blimpCoreBitWidth);
      msRead = (2 * numPass - 1) * (m_tACT + m_tPRE) + 2 * (activateMS + m_tPRE);
      msWrite = (2 * numPass - 1) * (m_tACT + m_tPRE) + 2 *(activateMS + m_tPRE);

      // reduction for all regions assuming 16 core AMD EPYC 9124
      double aggregateMs = static_cast<double>(obj.getNumCoresUsed()) / 2300000;
      double hostRW = (obj.getNumCoresUsed() * 1.0 / m_numChipsPerRank) * (m_tR + m_tW + (m_tGDL * 2));
      
      msCompute = (maxElementsPerRegion * m_blimpLatency * numberOfOperationPerElement * (numPass - 1)) + (minElementPerRegion * m_blimpLatency * numberOfOperationPerElement) + aggregateMs + hostRW;
      msRuntime = msRead + msWrite + msCompute;

      // Refer to fulcrum documentation
      mjEnergy = ((m_eACT + m_ePRE) + (maxElementsPerRegion * m_blimpArithmeticEnergy * numberOfOperationPerElement)) * (numPass - 1) * numCore * 2;
      mjEnergy += ((m_eACT + m_ePRE) + (minElementPerRegion * m_blimpArithmeticEnergy * numberOfOperationPerElement)) * numCore * 2;
      mjEnergy += aggregateMs * cpuTDP + ((obj.getNumCoresUsed() * 1.0 / m_numChipsPerRank) * ((2 * m_eAP)  + m_eR + m_eW));
      mjEnergy += ((m_eR * maxGDLItr * (numPass-1)) + (m_eR * minGDLItr)) * numBankPerChip * m_numRanks * 2;
      mjEnergy += ((m_eW * maxGDLItr * (numPass-1)) + (m_eW * minGDLItr)) * numBankPerChip * m_numRanks * 2;
      mjEnergy += m_pBChip * m_numChipsPerRank * m_numRanks * msRuntime;
      totalOp = obj.getNumElements() * 2;
      break;
    }
    default:
      printf("PIM-Warning: Unsupported reduction command for bank-level PIM: %s\n", pimCmd::getName(cmdType, "").c_str());
      break;
    }
  return pimeval::perfEnergy(msRuntime, mjEnergy, msRead, msWrite, msCompute, totalOp);
}