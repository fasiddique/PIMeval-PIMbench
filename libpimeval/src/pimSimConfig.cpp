// File: pimSimConfig.cpp
// PIMeval Simulator - PIM Simulator Configurations
// Copyright (c) 2024 University of Virginia
// This file is licensed under the MIT License.
// See the LICENSE file in the root of this repository for more details.

#include "pimSimConfig.h"
#include "pimUtils.h"
#include <algorithm>
#include <filesystem>
#include <iomanip>
#include <string>
#include <thread>
#include <unordered_map>
#include <filesystem>


//! @brief  Init PIMeval simulation configuration parameters at device creation
bool
pimSimConfig::init(PimDeviceEnum deviceType,
    unsigned numRanks, unsigned numBankPerRank, unsigned numSubarrayPerBank,
    unsigned numRowPerSubarray, unsigned numColPerSubarray, unsigned bufferSize)
{
  reset();  // always reset before init
  return deriveConfig(deviceType, "",
                      numRanks, numBankPerRank, numSubarrayPerBank,
                      numRowPerSubarray, numColPerSubarray, bufferSize);
}

//! @brief  Init PIMeval simulation configuration parameters at device creation
bool
pimSimConfig::init(PimDeviceEnum deviceType, const std::string& configFilePath)
{
  reset();  // always reset before init
  return deriveConfig(deviceType, configFilePath);
}

//! @brief  Show all configuration parameters
void
pimSimConfig::show() const
{
  std::printf("----------------------------------------\n");
  std::printf("PIM-Config: Debug Flags = 0x%x\n", m_debug);
  std::printf("PIM-Config: Simulator Config File: %s\n",
            (m_simConfigFile.empty() ? "<NONE>" : m_simConfigFile.c_str()));
  std::printf("PIM-Config: Memory Config File: %s\n",
            (m_memConfigFile.empty() ? "<DEFAULT>" : m_memConfigFile.c_str()));
  std::printf("PIM-Config: Memory Protocol: %s\n", pimUtils::pimProtocolEnumToStr(m_memoryProtocol).c_str());

  std::printf("PIM-Config: Current Device = %s, Simulation Target = %s\n", 
            pimUtils::pimDeviceEnumToStr(m_deviceType).c_str(),
            pimUtils::pimDeviceEnumToStr(m_simTarget).c_str());

  std::printf("PIM-Config: #ranks = %u, #banksPerRank = %u, #subarraysPerBank = %u, #rowsPerSubarray = %u, #colsPerSubarray = %u",
            m_numRanks, m_numBankPerRank, m_numSubarrayPerBank, m_numRowPerSubarray, m_numColPerSubarray);
  if (m_bufferSize > 0) std::printf(", bufferSize = %uB", m_bufferSize);
  std::printf("\n");

  std::printf("PIM-Config: Number of Threads = %u\n", m_numThreads);
  std::printf("PIM-Config: Load Balanced = %s\n", m_loadBalanced ? "1" : "0");
  std::printf("----------------------------------------\n");
}

//! @brief  Derive PIMeval simulation configuration parameters with priority rules
bool
pimSimConfig::deriveConfig(PimDeviceEnum deviceType,
    const std::string& configFilePath,
    unsigned numRanks, unsigned numBankPerRank, unsigned numSubarrayPerBank,
    unsigned numRowPerSubarray, unsigned numColPerSubarray,
    unsigned bufferSize)
{
  bool ok = true;

  // Derive debug flags first
  ok = ok & deriveDebug();

  // Read environment variables
  m_envParams = readEnvVars();

  // Derive simulator config file
  ok = ok & deriveSimConfigFile(configFilePath);

  // Read config file parameters
  m_cfgParams = readSimConfigFileParams();

  // Derive other configuration parameters in order
  ok = ok & deriveDeviceType(deviceType);
  ok = ok & deriveSimTarget();
  ok = ok & deriveMemConfigFile();
  ok = ok & deriveDimensions(numRanks, numBankPerRank, numSubarrayPerBank, numRowPerSubarray, numColPerSubarray, bufferSize);
  ok = ok & deriveNumThreads();
  ok = ok & deriveMiscEnvVars();
  ok = ok & deriveLoadBalance();

  // Show summary
  show();
  if (!ok) {
    std::printf("PIM-Error: Please resolve incorrect PIMeval configuration.\n");
  }
  m_isInit = true;
  return ok;
}

//! @brief  Derive Params: Debug Flags
bool
pimSimConfig::deriveDebug()
{
  m_debug = 0;
  std::string envVal;
  bool hasEnv = pimUtils::getEnvVar(m_envVarDebug, envVal);
  if (hasEnv && !envVal.empty()) {
    bool ok = pimUtils::convertStringToUnsigned(envVal, m_debug);
    if (!ok) {
      std::printf("PIM-Error: Incorrect environment variable: %s = %s\n", m_envVarDebug.c_str(), envVal.c_str());
      return false;
    }
  }
  return true;
}

//! @brief  Read config env vars
std::unordered_map<std::string, std::string>
pimSimConfig::readEnvVars() const
{
  std::unordered_map<std::string, std::string> params;
  params = pimUtils::readParamsFromEnvVars(m_envVarList);

  if (m_debug & pimSimConfig::DEBUG_PARAMS) {
    for (const auto& [key, val] : params) {
      std::printf("PIM-Debug: Environment variable: %s = %s\n", key.c_str(), val.c_str());
    }
  }

  return params;
}

//! @brief  Derive Params: Simulator Configuration File
bool
pimSimConfig::deriveSimConfigFile(const std::string& configFilePath)
{
  m_simConfigFile.clear();

  // If a config file is specified through APIs, use it. Otherwise check env var
  if (!configFilePath.empty()) {
    m_simConfigFile = configFilePath;
  } else if (m_envParams.find(m_envVarSimConfig) != m_envParams.end()) {
    m_simConfigFile = m_envParams.at(m_envVarSimConfig);
  }
  if (!m_simConfigFile.empty()) {
    if (!std::filesystem::exists(m_simConfigFile)) {
      std::printf("PIM-Error: Cannot find simulator config file: %s\n", m_simConfigFile.c_str());
      return false;
    }
  }
  return true;
}

//! @brief  Read config file params
std::unordered_map<std::string, std::string>
pimSimConfig::readSimConfigFileParams() const
{
  std::unordered_map<std::string, std::string> params;
  if (!m_simConfigFile.empty()) {
    params = pimUtils::readParamsFromConfigFile(m_simConfigFile);

    if (m_debug & pimSimConfig::DEBUG_PARAMS) {
      for (const auto& [key, val] : params) {
        std::printf("PIM-Debug: Simulator config file parameter: %s = %s\n", key.c_str(), val.c_str());
      }
    }
  }
  return params;
}

//! @brief  Derive Params: Device Type
bool
pimSimConfig::deriveDeviceType(PimDeviceEnum deviceType)
{
  m_deviceType = deviceType;
  return true;
}

//! @brief  Derive Params: Simulation Target
bool
pimSimConfig::deriveSimTarget()
{
  // If device type is not functional, always use it as simulation target
  m_simTarget = m_deviceType;

  if (m_deviceType == PIM_FUNCTIONAL) {
    bool hasVal = false;
    std::string val;
    // Check simulator config file
    if (m_simTarget == PIM_DEVICE_NONE || m_simTarget == PIM_FUNCTIONAL) {
      val = pimUtils::getOptionalParam(m_cfgParams, m_cfgVarSimTarget, hasVal);
      if (hasVal) {
        m_simTarget = pimUtils::strToPimDeviceEnum(val);
        if (m_simTarget == PIM_DEVICE_NONE) {
          std::printf("PIM-Error: Incorrect config file parameter: %s=%s\n", m_cfgVarSimTarget.c_str(), val.c_str());
          return false;
        }
      }
    }
    // Check env var
    if (m_simTarget == PIM_DEVICE_NONE || m_simTarget == PIM_FUNCTIONAL) {
      val = pimUtils::getOptionalParam(m_envParams, m_envVarSimTarget, hasVal);
      if (hasVal) {
        m_simTarget = pimUtils::strToPimDeviceEnum(val);
        if (m_simTarget == PIM_DEVICE_NONE) {
          std::printf("PIM-Error: Incorrect environment variable: %s=%s\n", m_envVarSimTarget.c_str(), val.c_str());
          return false;
        }
      }
    }
    // Check macro
    if (m_simTarget == PIM_DEVICE_NONE || m_simTarget == PIM_FUNCTIONAL) {
      // from 'make PIM_SIM_TARGET=...'
      #if defined(PIM_SIM_TARGET)
      m_simTarget = PIM_SIM_TARGET;
      #endif
    }
    // Use default
    if (m_simTarget == PIM_DEVICE_NONE || m_simTarget == PIM_FUNCTIONAL) {
      m_simTarget = DEFAULT_SIM_TARGET;
    }
  }

  return true;
}

//! @brief  Derive Params: Memory Config File
bool
pimSimConfig::deriveMemConfigFile()
{
  m_memConfigFile.clear();

  // Read config file and env
  if (m_cfgParams.find(m_cfgVarMemConfig) != m_cfgParams.end()) {
    m_memConfigFile = m_cfgParams.at(m_cfgVarMemConfig);
  } else if (m_envParams.find(m_envVarMemConfig) != m_envParams.end()) {
    m_memConfigFile = m_envParams.at(m_envVarMemConfig);
  }
  if (!m_memConfigFile.empty()) {
    if (!std::filesystem::exists(m_memConfigFile)) {
      // Try to find it in the same directory of sim config file
      std::string configFilePath = pimUtils::getDirectoryPath(m_simConfigFile);
      if (std::filesystem::exists(configFilePath + "/" + m_memConfigFile)) {
        m_memConfigFile = configFilePath + "/" + m_memConfigFile;
      } else {
        std::printf("PIM-Error: Cannot find memory config file: %s\n", m_memConfigFile.c_str());
        return false;
      }
    }

    // Determine memory protocol from memory config file. This is not sim config file.
    std::unordered_map<std::string, std::string> memParams = pimUtils::readParamsFromConfigFile(m_memConfigFile);
    if (memParams.find("protocol") != memParams.end()) {
      std::string protocol = memParams.at("protocol");
      if (protocol == "DDR3" || protocol == "DDR4" || protocol == "DDR5") {
        m_memoryProtocol = PIM_DEVICE_PROTOCOL_DDR;
      } else if (protocol == "LPDDR3" || protocol == "LPDDR4") {
        m_memoryProtocol = PIM_DEVICE_PROTOCOL_LPDDR;
      } else if (protocol == "HBM" || protocol == "HBM2") {
        m_memoryProtocol = PIM_DEVICE_PROTOCOL_HBM;
      } else if (protocol == "GDDR5" || protocol == "GDDR5X" || protocol == "GDDR6") {
        m_memoryProtocol = PIM_DEVICE_PROTOCOL_GDDR;
      } else {
        std::printf("PIM-Error: Unknown protocol %s in memory config file: %s\n", protocol.c_str(), m_memConfigFile.c_str());
        return false;
      }
    } else {
      std::printf("PIM-Error: Missing protocol parameter in memory config file: %s\n", m_memConfigFile.c_str());
      return false;
    }
  }
  return true;
}

//! @brief  Derive Params: A Specific PIM Memory Dimension
bool
pimSimConfig::deriveDimension(const std::string& cfgVar, const std::string& envVar, const unsigned apiVal, const unsigned defVal, unsigned& retVal)
{
  retVal = 0; // auto derived

  bool hasVal = false;
  std::string valStr;

  // Check config file. Zero will be ignored
  valStr = pimUtils::getOptionalParam(m_cfgParams, cfgVar, hasVal);
  if (hasVal) {
    unsigned val = 0;
    bool ok = pimUtils::convertStringToUnsigned(valStr, val);
    if (!ok || val == 0) {
      std::printf("PIM-Error: Incorrect config file parameter: %s=%s\n", cfgVar.c_str(), valStr.c_str());
      return false;
    }
    if (val > 0) {
      retVal = val;
      return true;
    }
  }

  // Check env var. Zero will be ignored
  valStr = pimUtils::getOptionalParam(m_envParams, envVar, hasVal);
  if (hasVal) {
    unsigned val = 0;
    bool ok = pimUtils::convertStringToUnsigned(valStr, val);
    if (!ok) {
      std::printf("PIM-Error: Incorrect environment variable: %s=%s\n", envVar.c_str(), valStr.c_str());
      return false;
    }
    if (val > 0) {
      retVal = val;
      return true;
    }
  }

  // Check value from APIs
  retVal = (apiVal > 0) ? apiVal : defVal;
  return true;
}

//! @brief  Derive Params: PIM Memory Dimensions
bool
pimSimConfig::deriveDimensions(unsigned numRanks, unsigned numBankPerRank, unsigned numSubarrayPerBank, unsigned numRowPerSubarray, unsigned numColPerSubarray, unsigned bufferSize)
{
  bool ok = true;
  ok = ok & deriveDimension(m_cfgVarNumRanks, m_envVarNumRanks, numRanks, DEFAULT_NUM_RANKS, m_numRanks);
  ok = ok & deriveDimension(m_cfgVarNumBankPerRank, m_envVarNumBankPerRank, numBankPerRank, DEFAULT_NUM_BANK_PER_RANK, m_numBankPerRank);
  ok = ok & deriveDimension(m_cfgVarNumSubarrayPerBank, m_envVarNumSubarrayPerBank, numSubarrayPerBank, DEFAULT_NUM_SUBARRAY_PER_BANK, m_numSubarrayPerBank);
  ok = ok & deriveDimension(m_cfgVarNumRowPerSubarray, m_envVarNumRowPerSubarray, numRowPerSubarray, DEFAULT_NUM_ROW_PER_SUBARRAY, m_numRowPerSubarray);
  ok = ok & deriveDimension(m_cfgVarNumColPerSubarray, m_envVarNumColPerSubarray, numColPerSubarray, DEFAULT_NUM_COL_PER_SUBARRAY, m_numColPerSubarray);
  ok = ok & deriveDimension(m_cfgVarBufferSize, m_envVarBufferSize, bufferSize, DEFAULT_BUFFER_SIZE, m_bufferSize);
  if (m_numRanks == 0 || m_numBankPerRank == 0 || m_numSubarrayPerBank == 0 || m_numRowPerSubarray == 0 || m_numColPerSubarray == 0) {
    std::printf("PIM-Error: Memory dimension parameter cannot be 0\n");
    ok = false;
  }
  if (m_simTarget != PIM_DEVICE_AIM && m_bufferSize > 0) {
    std::printf("PIM-Error: PIM Device %s does not support any on-chip buffer.\n", pimUtils::pimDeviceEnumToStr(m_simTarget).c_str());
    ok = false;
  }
  return ok;
}

//! @brief  Derive Params: Max number of threads
bool
pimSimConfig::deriveNumThreads()
{
  m_numThreads = 0; // auto derived

  bool hasVal = false;
  std::string valStr;

  // Check config file
  if (m_numThreads == 0) {
    valStr = pimUtils::getOptionalParam(m_cfgParams, m_cfgVarMaxNumThreads, hasVal);
    if (hasVal) {
      unsigned val = 0;
      bool ok = pimUtils::convertStringToUnsigned(valStr, val);
      if (!ok) {
        std::printf("PIM-Error: Incorrect config file parameter: %s=%s\n", m_cfgVarMaxNumThreads.c_str(), valStr.c_str());
        return false;
      }
      if (val > 0) {
        m_numThreads = val;
      }
    }
  }

  // Check env var. Zero will be ignored
  if (m_numThreads == 0) {
    valStr = pimUtils::getOptionalParam(m_envParams, m_envVarMaxNumThreads, hasVal);
    if (hasVal) {
      unsigned val = 0;
      bool ok = pimUtils::convertStringToUnsigned(valStr, val);
      if (!ok) {
        std::printf("PIM-Error: Incorrect environment variable: %s=%s\n", m_envVarMaxNumThreads.c_str(), valStr.c_str());
        return false;
      }
      if (val > 0) {
        m_numThreads = val;
      }
    }
  }

  // Check hardware concurrency
  unsigned hwThreads = std::thread::hardware_concurrency();
  if (m_debug & pimSimConfig::DEBUG_PARAMS) {
    std::printf("PIM-Debug: Maximum number of threads = %u, hardware concurrency = %u\n", m_numThreads, hwThreads);
  }
  if (m_numThreads == 0) {
    m_numThreads = hwThreads;
  } else {
    m_numThreads = std::min(m_numThreads, hwThreads);
  }
  // Safety check
  if (m_numThreads < 1) {
    m_numThreads = 1;
  }
  return true;
}

//! @brief  Derive Params: Misc Env Vars
bool
pimSimConfig::deriveMiscEnvVars()
{
  bool hasVal = false;
  std::string valStr;

  // Analysis Mode
  m_analysisMode = false;  // off by default
  valStr = pimUtils::getOptionalParam(m_envParams, m_envVarAnalysisMode, hasVal);
  if (hasVal) {
    if (valStr != "0" && valStr != "1") {
      std::printf("PIM-Error: Incorrect environment variable: %s=%s\n", m_envVarAnalysisMode.c_str(), valStr.c_str());
      return false;
    }
    m_analysisMode = (valStr == "1");
  }
  if (m_analysisMode) {
    std::printf("PIM-Warning: Running analysis only mode. Ignoring computation for fast performance and energy analysis.\n");
  }

  return true;
}

//! @brief  Derive Params: Load balance - Distribute data evenly among parallel cores during allocation
bool
pimSimConfig::deriveLoadBalance()
{
  m_loadBalanced = true;  // on by default

  // Check config file then env variable
  bool hasVal = false;
  std::string valStr = pimUtils::getOptionalParam(m_cfgParams, m_cfgVarLoadBalance, hasVal);
  if (hasVal) {
    if (valStr != "0" && valStr != "1") {
      std::printf("PIM-Error: Incorrect config file parameter: %s=%s\n", m_cfgVarLoadBalance.c_str(), valStr.c_str());
      return false;
    }
    m_loadBalanced = (valStr == "1");
  } else {
    valStr = pimUtils::getOptionalParam(m_envParams, m_envVarLoadBalance, hasVal);
    if (hasVal) {
      if (valStr != "0" && valStr != "1") {
        std::printf("PIM-Error: Incorrect environment variable: %s=%s\n", m_envVarLoadBalance.c_str(), valStr.c_str());
        return false;
      }
      m_loadBalanced = (valStr == "1");
    }
  }
  return true;
}

