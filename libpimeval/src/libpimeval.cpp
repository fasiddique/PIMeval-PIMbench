// File: libpimeval.cpp
// PIMeval Simulator - Library Interface
// Copyright (c) 2024 University of Virginia
// This file is licensed under the MIT License.
// See the LICENSE file in the root of this repository for more details.

#include "libpimeval.h"
#include "pimSim.h"
#include "pimUtils.h"

//! @brief  Create a PIM device
PimStatus
pimCreateDevice(PimDeviceEnum deviceType, unsigned numRanks, unsigned numBankPerRank, unsigned numSubarrayPerBank, unsigned numRows, unsigned numCols, unsigned bufferSize)
{
  bool ok = pimSim::get()->createDevice(deviceType, numRanks, numBankPerRank, numSubarrayPerBank, numRows, numCols, bufferSize);
  return ok ? PIM_OK : PIM_ERROR;
}

//! @brief  Create a PIM device from config file
PimStatus
pimCreateDeviceFromConfig(PimDeviceEnum deviceType, const char* configFileName)
{
  bool ok = pimSim::get()->createDeviceFromConfig(deviceType, configFileName);
  return ok ? PIM_OK : PIM_ERROR;
}

//! @brief  Get PIM device properties
PimStatus
pimGetDeviceProperties(PimDeviceProperties* deviceProperties)
{
  bool ok = pimSim::get()->getDeviceProperties(deviceProperties);
  return ok ? PIM_OK : PIM_ERROR;
}

//! @brief  Delete a PIM device
PimStatus
pimDeleteDevice()
{
  bool ok = pimSim::get()->deleteDevice();
  return ok ? PIM_OK : PIM_ERROR;
}

PimStatus
pimPrefixSum(PimObjId src, PimObjId dest)
{
  bool ok = pimSim::get()->pimPrefixSum(src, dest);
  return ok ? PIM_OK : PIM_ERROR;
}

//! @brief  Start timer for a PIM kernel to measure CPU runtime and DRAM refresh
void
pimStartTimer()
{
  pimSim::get()->startKernelTimer();
}

//! @brief  End timer for a PIM kernel to measure CPU runtime and DRAM refresh
void
pimEndTimer()
{
  pimSim::get()->endKernelTimer();
}

//! @brief  Show PIM command stats
void
pimShowStats()
{
  pimSim::get()->showStats();
}

//! @brief  Reset PIM command stats
void
pimResetStats()
{
  pimSim::get()->resetStats();
}

//! @brief  Is analysis mode. Call this after device creation
bool
pimIsAnalysisMode()
{
  return pimSim::get()->isAnalysisMode();
}

//! @brief  Allocate a PIM resource
PimObjId
pimAlloc(PimAllocEnum allocType, uint64_t numElements, PimDataType dataType)
{
  return pimSim::get()->pimAlloc(allocType, numElements, dataType);
}

//! @brief  Allocate a PIM resource, with an associated object as reference
PimObjId
pimAllocAssociated(PimObjId assocId, PimDataType dataType)
{
  return pimSim::get()->pimAllocAssociated(assocId, dataType);
}

//! @brief  Allocate a global buffer for broadcasting data to all PIM cores
PimObjId
pimAllocBuffer(uint32_t numElements, PimDataType dataType)
{
  return pimSim::get()->pimAllocBuffer(numElements, dataType);
}

//! @brief  Free a PIM resource
PimStatus
pimFree(PimObjId obj)
{
  bool ok = pimSim::get()->pimFree(obj);
  return ok ? PIM_OK : PIM_ERROR;
}

//! @brief  Create an obj referencing to a range of an existing obj
PimObjId
pimCreateRangedRef(PimObjId refId, uint64_t idxBegin, uint64_t idxEnd)
{
  return pimSim::get()->pimCreateRangedRef(refId, idxBegin, idxEnd);
}

//! @brief  Create an obj referencing to negation of an existing obj based on dual-contact memory cells
PimObjId
pimCreateDualContactRef(PimObjId refId)
{
  return pimSim::get()->pimCreateDualContactRef(refId);
}

//! @brief  Copy data from main memory to PIM device for a range of elements within the PIM object
PimStatus
pimCopyHostToDevice(void* src, PimObjId dest, uint64_t idxBegin, uint64_t idxEnd)
{
  bool ok = pimSim::get()->pimCopyMainToDevice(src, dest, idxBegin, idxEnd);
  return ok ? PIM_OK : PIM_ERROR;
}

//! @brief  Copy data from PIM device to main memory for a range of elements within the PIM object
PimStatus
pimCopyDeviceToHost(PimObjId src, void* dest, uint64_t idxBegin, uint64_t idxEnd)
{
  bool ok = pimSim::get()->pimCopyDeviceToMain(src, dest, idxBegin, idxEnd);
  return ok ? PIM_OK : PIM_ERROR;
}

//! @brief  Copy data from main memory to PIM device with type for a range of elements within the PIM object
PimStatus
pimCopyHostToDeviceWithType(PimCopyEnum copyType, void* src, PimObjId dest, uint64_t idxBegin, uint64_t idxEnd)
{
  bool ok = pimSim::get()->pimCopyMainToDeviceWithType(copyType, src, dest, idxBegin, idxEnd);
  return ok ? PIM_OK : PIM_ERROR;
}

//! @brief  Copy data from PIM device to main memory with type for a range of elements within the PIM object
PimStatus
pimCopyDeviceToHostWithType(PimCopyEnum copyType, PimObjId src, void* dest, uint64_t idxBegin, uint64_t idxEnd)
{
  bool ok = pimSim::get()->pimCopyDeviceToMainWithType(copyType, src, dest, idxBegin, idxEnd);
  return ok ? PIM_OK : PIM_ERROR;
}

//! @brief  Copy data from PIM device to device for a range of elements within the PIM object
PimStatus
pimCopyDeviceToDevice(PimObjId src, PimObjId dest, uint64_t idxBegin, uint64_t idxEnd)
{
  bool ok = pimSim::get()->pimCopyDeviceToDevice(src, dest, idxBegin, idxEnd);
  return ok ? PIM_OK : PIM_ERROR;
}

PimStatus pimCopyObjectToObject(PimObjId src, PimObjId dest)
{
  bool ok = pimSim::get()->pimCopyObjectToObject(src, dest);
  return ok ? PIM_OK : PIM_ERROR;
}

//! @brief  Convert data type between two associated PIM objects of different data types
PimStatus pimConvertType(PimObjId src, PimObjId dest)
{
  bool ok = pimSim::get()->pimConvertType(src, dest);
  return ok ? PIM_OK : PIM_ERROR;
}

//! @brief  Load vector with a signed int value
PimStatus
pimBroadcastInt(PimObjId dest, int64_t value)
{
  bool ok = pimSim::get()->pimBroadcast(dest, value);
  return ok ? PIM_OK : PIM_ERROR;
}

//! @brief  Load vector with an unsigned int value
PimStatus
pimBroadcastUInt(PimObjId dest, uint64_t value)
{
  bool ok = pimSim::get()->pimBroadcast(dest, value);
  return ok ? PIM_OK : PIM_ERROR;
}

//! @brief  Load vector with a float32 value
PimStatus
pimBroadcastFP(PimObjId dest, float value)
{
  bool ok = pimSim::get()->pimBroadcast(dest, value);
  return ok ? PIM_OK : PIM_ERROR;
}

//! @brief  PIM add
PimStatus
pimAdd(PimObjId src1, PimObjId src2, PimObjId dest)
{
  bool ok = pimSim::get()->pimAdd(src1, src2, dest);
  return ok ? PIM_OK : PIM_ERROR;
}

//! @brief  PIM sub
PimStatus
pimSub(PimObjId src1, PimObjId src2, PimObjId dest)
{
  bool ok = pimSim::get()->pimSub(src1, src2, dest);
  return ok ? PIM_OK : PIM_ERROR;
}

//! @brief  PIM div
PimStatus
pimDiv(PimObjId src1, PimObjId src2, PimObjId dest)
{
  bool ok = pimSim::get()->pimDiv(src1, src2, dest);
  return ok ? PIM_OK : PIM_ERROR;
}

//! @brief  PIM not
PimStatus
pimNot(PimObjId src, PimObjId dest)
{
  bool ok = pimSim::get()->pimNot(src, dest);;
  return ok ? PIM_OK : PIM_ERROR;
}

//! @brief  PIM or
PimStatus
pimOr(PimObjId src1, PimObjId src2, PimObjId dest)
{
  bool ok = pimSim::get()->pimOr(src1, src2, dest);
  return ok ? PIM_OK : PIM_ERROR;
}

//! @brief  PIM and
PimStatus
pimAnd(PimObjId src1, PimObjId src2, PimObjId dest)
{
  bool ok = pimSim::get()->pimAnd(src1, src2, dest);
  return ok ? PIM_OK : PIM_ERROR;
}

//! @brief  PIM xor
PimStatus
pimXor(PimObjId src1, PimObjId src2, PimObjId dest)
{
  bool ok = pimSim::get()->pimXor(src1, src2, dest);
  return ok ? PIM_OK : PIM_ERROR;
}

//! @brief  PIM xnor
PimStatus
pimXnor(PimObjId src1, PimObjId src2, PimObjId dest)
{
  bool ok = pimSim::get()->pimXnor(src1, src2, dest);
  return ok ? PIM_OK : PIM_ERROR;
}

//! @brief  PIM abs
PimStatus
pimAbs(PimObjId src, PimObjId dest)
{
  bool ok = pimSim::get()->pimAbs(src, dest);;
  return ok ? PIM_OK : PIM_ERROR;
}

//! @brief  PIM multiplication
PimStatus
pimMul(PimObjId src1, PimObjId src2, PimObjId dest)
{
  bool ok = pimSim::get()->pimMul(src1, src2, dest);
  return ok ? PIM_OK : PIM_ERROR;
}

//! @brief  PIM GT
PimStatus
pimGT(PimObjId src1, PimObjId src2, PimObjId dest)
{
  bool ok = pimSim::get()->pimGT(src1, src2, dest);
  return ok ? PIM_OK : PIM_ERROR;
}

//! @brief  PIM LT
PimStatus
pimLT(PimObjId src1, PimObjId src2, PimObjId dest)
{
  bool ok = pimSim::get()->pimLT(src1, src2, dest);
  return ok ? PIM_OK : PIM_ERROR;
}

//! @brief  PIM EQ
PimStatus
pimEQ(PimObjId src1, PimObjId src2, PimObjId dest)
{
  bool ok = pimSim::get()->pimEQ(src1, src2, dest);
  return ok ? PIM_OK : PIM_ERROR;
}

//! @brief  PIM NE
PimStatus
pimNE(PimObjId src1, PimObjId src2, PimObjId dest)
{
  bool ok = pimSim::get()->pimNE(src1, src2, dest);
  return ok ? PIM_OK : PIM_ERROR;
}

//! @brief  PIM Min
PimStatus
pimMin(PimObjId src1, PimObjId src2, PimObjId dest)
{
  bool ok = pimSim::get()->pimMin(src1, src2, dest);
  return ok ? PIM_OK : PIM_ERROR;
}

//! @brief  PIM Max
PimStatus
pimMax(PimObjId src1, PimObjId src2, PimObjId dest)
{
  bool ok = pimSim::get()->pimMax(src1, src2, dest);
  return ok ? PIM_OK : PIM_ERROR;
}

PimStatus pimAddScalar(PimObjId src, PimObjId dest, uint64_t scalarValue)
{
  bool ok = pimSim::get()->pimAdd(src, dest, scalarValue);
  return ok ? PIM_OK : PIM_ERROR;
}

PimStatus pimSubScalar(PimObjId src, PimObjId dest, uint64_t scalarValue)
{
  bool ok = pimSim::get()->pimSub(src, dest, scalarValue);
  return ok ? PIM_OK : PIM_ERROR;
}

PimStatus pimMulScalar(PimObjId src, PimObjId dest, uint64_t scalarValue)
{
  bool ok = pimSim::get()->pimMul(src, dest, scalarValue);
  return ok ? PIM_OK : PIM_ERROR;
}

PimStatus pimDivScalar(PimObjId src, PimObjId dest, uint64_t scalarValue)
{
  bool ok = pimSim::get()->pimDiv(src, dest, scalarValue);
  return ok ? PIM_OK : PIM_ERROR;
}

PimStatus pimAndScalar(PimObjId src, PimObjId dest, uint64_t scalarValue)
{
  bool ok = pimSim::get()->pimAnd(src, dest, scalarValue);
  return ok ? PIM_OK : PIM_ERROR;
}

PimStatus pimOrScalar(PimObjId src, PimObjId dest, uint64_t scalarValue)
{
  bool ok = pimSim::get()->pimOr(src, dest, scalarValue);
  return ok ? PIM_OK : PIM_ERROR;
}

PimStatus pimXorScalar(PimObjId src, PimObjId dest, uint64_t scalarValue)
{
  bool ok = pimSim::get()->pimXor(src, dest, scalarValue);
  return ok ? PIM_OK : PIM_ERROR;
}

PimStatus pimXnorScalar(PimObjId src, PimObjId dest, uint64_t scalarValue)
{
  bool ok = pimSim::get()->pimXnor(src, dest, scalarValue);
  return ok ? PIM_OK : PIM_ERROR;
}

PimStatus pimGTScalar(PimObjId src, PimObjId dest, uint64_t scalarValue)
{
  bool ok = pimSim::get()->pimGT(src, dest, scalarValue);
  return ok ? PIM_OK : PIM_ERROR;
}

PimStatus pimLTScalar(PimObjId src, PimObjId dest, uint64_t scalarValue)
{
  bool ok = pimSim::get()->pimLT(src, dest, scalarValue);
  return ok ? PIM_OK : PIM_ERROR;
}

PimStatus pimEQScalar(PimObjId src, PimObjId dest, uint64_t scalarValue)
{
  bool ok = pimSim::get()->pimEQ(src, dest, scalarValue);
  return ok ? PIM_OK : PIM_ERROR;
}

PimStatus pimNEScalar(PimObjId src, PimObjId dest, uint64_t scalarValue)
{
  bool ok = pimSim::get()->pimNE(src, dest, scalarValue);
  return ok ? PIM_OK : PIM_ERROR;
}

PimStatus pimMinScalar(PimObjId src, PimObjId dest, uint64_t scalarValue)
{
  bool ok = pimSim::get()->pimMin(src, dest, scalarValue);
  return ok ? PIM_OK : PIM_ERROR;
}

PimStatus pimMaxScalar(PimObjId src, PimObjId dest, uint64_t scalarValue)
{
  bool ok = pimSim::get()->pimMax(src, dest, scalarValue);
  return ok ? PIM_OK : PIM_ERROR;
}

PimStatus pimScaledAdd(PimObjId src1, PimObjId src2, PimObjId dest, uint64_t scalarValue) 
{
  bool ok = pimSim::get()->pimScaledAdd(src1, src2, dest, scalarValue);
  return ok ? PIM_OK : PIM_ERROR;
}

//! @brief  PIM Pop Count
PimStatus
pimPopCount(PimObjId src, PimObjId dest)
{
  bool ok = pimSim::get()->pimPopCount(src, dest);
  return ok ? PIM_OK : PIM_ERROR;
}

//! @brief  Extract a bit slice from a data vector. Dest must be BOOL type
PimStatus
pimBitSliceExtract(PimObjId src, PimObjId destBool, unsigned bitIdx)
{
  bool ok = pimSim::get()->pimBitSliceExtract(src, destBool, bitIdx);
  return ok ? PIM_OK : PIM_ERROR;
}

//! @brief  Insert a bit slice to a data vector. Src must be BOOL type
PimStatus
pimBitSliceInsert(PimObjId srcBool, PimObjId dest, unsigned bitIdx)
{
  bool ok = pimSim::get()->pimBitSliceInsert(srcBool, dest, bitIdx);
  return ok ? PIM_OK : PIM_ERROR;
}

//! @brief  Conditional copy: dest[i] = cond ? src[i] : dest[i]
PimStatus
pimCondCopy(PimObjId condBool, PimObjId src, PimObjId dest)
{
  bool ok = pimSim::get()->pimCondCopy(condBool, src, dest);
  return ok ? PIM_OK : PIM_ERROR;
}

//! @brief  Conditional broadcast: dest[i] = cond ? scalar : dest[i]
PimStatus
pimCondBroadcast(PimObjId condBool, uint64_t scalarBits, PimObjId dest)
{
  bool ok = pimSim::get()->pimCondBroadcast(condBool, scalarBits, dest);
  return ok ? PIM_OK : PIM_ERROR;
}

//! @brief  Conditional select: dest[i] = cond ? src1[i] : src2[i]
PimStatus
pimCondSelect(PimObjId condBool, PimObjId src1, PimObjId src2, PimObjId dest)
{
  bool ok = pimSim::get()->pimCondSelect(condBool, src1, src2, dest);
  return ok ? PIM_OK : PIM_ERROR;
}

//! @brief  Conditional select scalar: dest[i] = cond ? src1[i] : scalar
PimStatus
pimCondSelectScalar(PimObjId condBool, PimObjId src1, uint64_t scalarBits, PimObjId dest)
 {
  bool ok = pimSim::get()->pimCondSelectScalar(condBool, src1, scalarBits, dest);
  return ok ? PIM_OK : PIM_ERROR;
 }

//! @brief  AES Sbox: dest[i] = lut[src[i]]
PimStatus 
pimAesSbox(PimObjId src, PimObjId dest, const std::vector<uint8_t>& lut)
{
  bool ok = pimSim::get()->pimAesSbox(src, dest, lut);
  return ok ? PIM_OK : PIM_ERROR;
}

//! @brief  AES Sbox: dest[i] = lut[src[i]] (similar to AES sbox, different in perforamance and energy model for the bit-serial architecture)
PimStatus 
pimAesInverseSbox(PimObjId src, PimObjId dest, const std::vector<uint8_t>& lut)
{
  bool ok = pimSim::get()->pimAesInverseSbox(src, dest, lut);
  return ok ? PIM_OK : PIM_ERROR;
}

// Implementation of min reduction
PimStatus pimRedMin(PimObjId src, void* min, uint64_t idxBegin, uint64_t idxEnd) {
    bool ok = pimSim::get()->pimRedMin(src, min, idxBegin, idxEnd);
    return ok ? PIM_OK : PIM_ERROR;
}

// Implementation of max reduction
PimStatus pimRedMax(PimObjId src, void* max, uint64_t idxBegin, uint64_t idxEnd) {
    bool ok = pimSim::get()->pimRedMax(src, max, idxBegin, idxEnd);
    return ok ? PIM_OK : PIM_ERROR;
}

//! @brief  PIM MAC operation: dest += src1 * src2
PimStatus pimMAC(PimObjId src1, PimObjId src2, void *dest)
{
  bool ok = pimSim::get()->pimMAC(src1, src2, dest);
  return ok ? PIM_OK : PIM_ERROR;
}

//! @brief  PIM reduction sum for signed int. Result returned to a host variable
PimStatus
pimRedSum(PimObjId src, void* sum, uint64_t idxBegin, uint64_t idxEnd)
{
  bool ok = pimSim::get()->pimRedSum(src, sum, idxBegin, idxEnd);
  return ok ? PIM_OK : PIM_ERROR;
}

//! @brief  Rotate all elements of an obj by one step to the right
PimStatus
pimRotateElementsRight(PimObjId src)
{
  bool ok = pimSim::get()->pimRotateElementsRight(src);
  return ok ? PIM_OK : PIM_ERROR;
}

//! @brief  Rotate all elements of an obj by one step to the left
PimStatus
pimRotateElementsLeft(PimObjId src)
{
  bool ok = pimSim::get()->pimRotateElementsLeft(src);
  return ok ? PIM_OK : PIM_ERROR;
}

//! @brief  Shift elements of an obj by one step to the right and fill zero
PimStatus
pimShiftElementsRight(PimObjId src)
{
  bool ok = pimSim::get()->pimShiftElementsRight(src);
  return ok ? PIM_OK : PIM_ERROR;
}

//! @brief  Shift elements of an obj by one step to the left and fill zero
PimStatus
pimShiftElementsLeft(PimObjId src)
{
  bool ok = pimSim::get()->pimShiftElementsLeft(src);
  return ok ? PIM_OK : PIM_ERROR;
}

//! @brief  Shift bits of each elements of an obj by shiftAmount to the right. This currently implements arithmetic shift.
PimStatus
pimShiftBitsRight(PimObjId src, PimObjId dest, unsigned shiftAmount)
{
  bool ok = pimSim::get()->pimShiftBitsRight(src, dest, shiftAmount);
  return ok ? PIM_OK : PIM_ERROR;
}

//! @brief  Shift bits of each elements of an obj by shiftAmount to the left.
PimStatus
pimShiftBitsLeft(PimObjId src, PimObjId dest, unsigned shiftAmount)
{
  bool ok = pimSim::get()->pimShiftBitsLeft(src, dest, shiftAmount);
  return ok ? PIM_OK : PIM_ERROR;
}

//! @brief  Execute fused PIM APIs
PimStatus
pimFuse(PimProg prog)
{
  bool ok = pimSim::get()->pimFuse(prog);
  return ok ? PIM_OK : PIM_ERROR;
}

//! @brief  BitSIMD-V: Read a row to SA
PimStatus
pimOpReadRowToSa(PimObjId src, unsigned ofst)
{
  bool ok = pimSim::get()->pimOpReadRowToSa(src, ofst);
  return ok ? PIM_OK : PIM_ERROR;
}

//! @brief  BitSIMD-V: Write SA to a row
PimStatus
pimOpWriteSaToRow(PimObjId src, unsigned ofst)
{
  bool ok = pimSim::get()->pimOpWriteSaToRow(src, ofst);
  return ok ? PIM_OK : PIM_ERROR;
}

//! @brief  BitSIMD-V: Triple row activation to SA
PimStatus
pimOpTRA(PimObjId src1, unsigned ofst1, PimObjId src2, unsigned ofst2, PimObjId src3, unsigned ofst3)
{
  bool ok = pimSim::get()->pimOpTRA(src1, ofst1, src2, ofst2, src3, ofst3);
  return ok ? PIM_OK : PIM_ERROR;
}

//! @brief  BitSIMD-V: Move value between two regs
PimStatus
pimOpMove(PimObjId objId, PimRowReg src, PimRowReg dest)
{
  bool ok = pimSim::get()->pimOpMove(objId, src, dest);
  return ok ? PIM_OK : PIM_ERROR;
}

//! @brief  BitSIMD-V: Set value of a reg
PimStatus
pimOpSet(PimObjId objId, PimRowReg src, bool val)
{
  bool ok = pimSim::get()->pimOpSet(objId, src, val);
  return ok ? PIM_OK : PIM_ERROR;
}

//! @brief  BitSIMD-V: Not of a reg
PimStatus
pimOpNot(PimObjId objId, PimRowReg src, PimRowReg dest)
{
  bool ok = pimSim::get()->pimOpNot(objId, src, dest);
  return ok ? PIM_OK : PIM_ERROR;
}

//! @brief  BitSIMD-V: And of two regs
PimStatus
pimOpAnd(PimObjId objId, PimRowReg src1, PimRowReg src2, PimRowReg dest)
{
  bool ok = pimSim::get()->pimOpAnd(objId, src1, src2, dest);
  return ok ? PIM_OK : PIM_ERROR;
}

//! @brief  BitSIMD-V: Or of two regs
PimStatus
pimOpOr(PimObjId objId, PimRowReg src1, PimRowReg src2, PimRowReg dest)
{
  bool ok = pimSim::get()->pimOpOr(objId, src1, src2, dest);
  return ok ? PIM_OK : PIM_ERROR;
}

//! @brief  BitSIMD-V: Nand of two regs
PimStatus
pimOpNand(PimObjId objId, PimRowReg src1, PimRowReg src2, PimRowReg dest)
{
  bool ok = pimSim::get()->pimOpNand(objId, src1, src2, dest);
  return ok ? PIM_OK : PIM_ERROR;
}

//! @brief  BitSIMD-V: Nor of two regs
PimStatus
pimOpNor(PimObjId objId, PimRowReg src1, PimRowReg src2, PimRowReg dest)
{
  bool ok = pimSim::get()->pimOpNor(objId, src1, src2, dest);
  return ok ? PIM_OK : PIM_ERROR;
}

//! @brief  BitSIMD-V: Xor of two regs
PimStatus
pimOpXor(PimObjId objId, PimRowReg src1, PimRowReg src2, PimRowReg dest)
{
  bool ok = pimSim::get()->pimOpXor(objId, src1, src2, dest);
  return ok ? PIM_OK : PIM_ERROR;
}

//! @brief  BitSIMD-V: Xnor of two regs
PimStatus
pimOpXnor(PimObjId objId, PimRowReg src1, PimRowReg src2, PimRowReg dest)
{
  bool ok = pimSim::get()->pimOpXnor(objId, src1, src2, dest);
  return ok ? PIM_OK : PIM_ERROR;
}

//! @brief  BitSIMD-V: Maj of three regs
PimStatus
pimOpMaj(PimObjId objId, PimRowReg src1, PimRowReg src2, PimRowReg src3, PimRowReg dest)
{
  bool ok = pimSim::get()->pimOpMaj(objId, src1, src2, src3, dest);
  return ok ? PIM_OK : PIM_ERROR;
}

//! @brief  BitSIMD-V: Conditional selecion: dest = cond ? src1 : src2
PimStatus
pimOpSel(PimObjId objId, PimRowReg cond, PimRowReg src1, PimRowReg src2, PimRowReg dest)
{
  bool ok = pimSim::get()->pimOpSel(objId, cond, src1, src2, dest);
  return ok ? PIM_OK : PIM_ERROR;
}

//! @brief  BitSIMD-V: Rotate a reg to the right, using srcId for range
PimStatus
pimOpRotateRH(PimObjId objId, PimRowReg src)
{
  bool ok = pimSim::get()->pimOpRotateRH(objId, src);
  return ok ? PIM_OK : PIM_ERROR;
}

//! @brief  BitSIMD-V: Rotate a reg to the left, using srcId for range
PimStatus
pimOpRotateLH(PimObjId objId, PimRowReg src)
{
  bool ok = pimSim::get()->pimOpRotateLH(objId, src);
  return ok ? PIM_OK : PIM_ERROR;
}

// @brief  SIMDRAM: AP operation
PimStatus
pimOpAP(int numSrc, ...)
{
  va_list args;
  va_start(args, numSrc);
  bool ok = pimSim::get()->pimOpAP(numSrc, args);
  va_end(args);
  return ok ? PIM_OK : PIM_ERROR;
}

// @brief  SIMDRAM: AAP operation
PimStatus
pimOpAAP(int numSrc, int numDest, ...)
{
  va_list args;
  va_start(args, numDest);
  bool ok = pimSim::get()->pimOpAAP(numSrc, numDest, args);
  va_end(args);
  return ok ? PIM_OK : PIM_ERROR;
}

