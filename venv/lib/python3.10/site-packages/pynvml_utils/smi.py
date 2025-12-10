# ============================================================================ #
# Copyright (c) 2011-2024, NVIDIA Corporation.  All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
#    * Redistributions of source code must retain the above copyright notice,
#      this list of conditions and the following disclaimer.
#    * Redistributions in binary form must reproduce the above copyright
#      notice, this list of conditions and the following disclaimer in the
#      documentation and/or other materials provided with the distribution.
#    * Neither the name of the NVIDIA Corporation nor the names of its
#      contributors may be used to endorse or promote products derived from
#      this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF
# THE POSSIBILITY OF SUCH DAMAGE.
# ============================================================================ #

"""
nvidia_smi

Sample code that attempts to reproduce the output of nvidia-smi -q -x
For many cases the output should match.
Each query parameter is documented within nvdia-smi --help-query-gpu

From Code:
DeviceQuery()
DeviceQuery("--help")
DeviceQuery("--help_query_gpu")
DeviceQuery("pci.bus_id,memory.total,memory.free")
DeviceQuery([NVSMI_PCI_BUS_ID, NVSMI_MEMORY_TOTAL, NVSMI_MEMORY_FREE])

XmlDeviceQuery()
XmlDeviceQuery("--help")
XmlDeviceQuery("--help_query_gpu")
"""

from pynvml import *
import datetime
import collections
import time
from threading import Thread

## ========================================================================== ##
##                                                                            ##
##                              Enumerations                                  ##
##                                                                            ##
## ========================================================================== ##

#Details and descriptions for enumerations in help_query_gpu.txt
NVSMI_ALL = -1

NVSMI_TIMESTAMP = 1
NVSMI_DRIVER_VERSION = 2
NVSMI_COUNT = 3
NVSMI_NAME = 4
NVSMI_SERIALNUMBER = 5
NVSMI_UUID = 6
NVSMI_PCI_BUS_ID = 7
NVSMI_PCI_DOMAIN = 8
NVSMI_PCI_BUS = 9
NVSMI_PCI_DEVICE = 10
NVSMI_PCI_DEVICE_ID = 11
NVSMI_PCI_SUBDEVICE_ID = 12
NVSMI_PCI_LINK_GEN_CUR = 13
NVSMI_PCI_LINK_GEN_MAX = 14
NVSMI_PCI_LINK_WIDTH_CUR = 15
NVSMI_PCI_LINK_WIDTH_MAX = 16
NVSMI_INDEX = 17
NVSMI_DISPLAY_MODE = 18
NVSMI_DISPLAY_ACTIVE = 19
NVSMI_PERSISTENCE_MODE = 20
NVSMI_ACCT_MODE = 21
NVSMI_ACCT_BUFFER_SIZE = 22
NVSMI_DRIVER_MODEL_CUR = 23
NVSMI_DRIVER_MODEL_PENDING = 24
NVSMI_VBIOS_VER = 25
NVSMI_BOARD_ID = 26

NVSMI_INFOROM_IMG = 190
NVSMI_INFOROM_OEM = 191
NVSMI_INFOROM_ECC = 192
NVSMI_INFOROM_PWR = 193

NVSMI_GOM_CUR = 30
NVSMI_GOM_PENDING = 31

NVSMI_FAN_SPEED = 32
NVSMI_PSTATE = 33

NVSMI_MEMORY_TOTAL = 50
NVSMI_MEMORY_FREE = 51
NVSMI_MEMORY_USED = 52
NVSMI_COMPUTE_MODE = 53
NVSMI_MEMORY_BAR1 = 54

NVSMI_UTILIZATION_GPU = 60
NVSMI_UTILIZATION_MEM = 61
NVSMI_UTILIZATION_ENCODER = 62
NVSMI_UTILIZATION_DECODER = 63

NVSMI_ENCODER_STATS_SESSIONCOUNT = 260
NVSMI_ENCODER_STATS_AVG_EPS = 261
NVSMI_ENCODER_STATS_AVG_LATENCY = 262

NVSMI_ECC_MODE_CUR = 70
NVSMI_ECC_MODE_PENDING = 71

NVSMI_ECC_ERROR_CORRECTED_VOLATILE_DEV_MEM = 80
NVSMI_ECC_ERROR_CORRECTED_VOLATILE_REGFILE = 81
NVSMI_ECC_ERROR_CORRECTED_VOLATILE_L1CACHE = 82
NVSMI_ECC_ERROR_CORRECTED_VOLATILE_L2CACHE = 83
NVSMI_ECC_ERROR_CORRECTED_VOLATILE_TEXTURE = 84
NVSMI_ECC_ERROR_CORRECTED_VOLATILE_TOTAL = 85

NVSMI_ECC_ERROR_CORRECTED_AGGREGATE_DEV_MEM = 90
NVSMI_ECC_ERROR_CORRECTED_AGGREGATE_REGFILE = 91
NVSMI_ECC_ERROR_CORRECTED_AGGREGATE_L1CACHE = 92
NVSMI_ECC_ERROR_CORRECTED_AGGREGATE_L2CACHE = 93
NVSMI_ECC_ERROR_CORRECTED_AGGREGATE_TEXTURE = 94
NVSMI_ECC_ERROR_CORRECTED_AGGREGATE_TOTAL = 95

NVSMI_ECC_ERROR_UNCORRECTED_VOLATILE_DEV_MEM = 100
NVSMI_ECC_ERROR_UNCORRECTED_VOLATILE_REGFILE = 101
NVSMI_ECC_ERROR_UNCORRECTED_VOLATILE_L1CACHE = 102
NVSMI_ECC_ERROR_UNCORRECTED_VOLATILE_L2CACHE = 103
NVSMI_ECC_ERROR_UNCORRECTED_VOLATILE_TEXTURE = 104
NVSMI_ECC_ERROR_UNCORRECTED_VOLATILE_TOTAL = 105

NVSMI_ECC_ERROR_UNCORRECTED_AGGREGATE_DEV_MEM = 110
NVSMI_ECC_ERROR_UNCORRECTED_AGGREGATE_REGFILE = 111
NVSMI_ECC_ERROR_UNCORRECTED_AGGREGATE_L1CACHE = 112
NVSMI_ECC_ERROR_UNCORRECTED_AGGREGATE_L2CACHE = 113
NVSMI_ECC_ERROR_UNCORRECTED_AGGREGATE_TEXTURE = 114
NVSMI_ECC_ERROR_UNCORRECTED_AGGREGATE_TOTAL = 115

NVSMI_RETIREDPAGES_SINGLE_BIT_ECC_COUNT = 120
NVSMI_RETIREDPAGES_DOUBLE_BIT_ECC_COUNT = 121
NVSMI_RETIREDPAGES_PENDING = 122

NVSMI_CLOCK_THROTTLE_REASONS_SUPPORTED = 170
NVSMI_CLOCK_THROTTLE_REASONS_ACTIVE = 171
NVSMI_CLOCK_THROTTLE_REASONS_IDLE = 172
NVSMI_CLOCK_THROTTLE_REASONS_APP_SETTING = 173
NVSMI_CLOCK_THROTTLE_REASONS_SW_PWR_CAP = 174
NVSMI_CLOCK_THROTTLE_REASONS_HW_SLOWDOWN = 175

NVSMI_CLOCK_THROTTLE_REASONS_HW_THERMAL_SLOWDOWN = 176
NVSMI_CLOCK_THROTTLE_REASONS_HW_PWR_BRAKE_SLOWDOWN = 177
NVSMI_CLOCK_THROTTLE_REASONS_SW_THERMAL_SLOWDOWN = 178
NVSMI_CLOCK_THROTTLE_REASONS_SYNC_BOOST = 179

NVSMI_TEMPERATURE_GPU = 130
NVSMI_TEMPERATURE_MEMORY = 131

NVSMI_POWER_MGMT = 140
NVSMI_POWER_DRAW = 141
NVSMI_POWER_LIMIT = 142
NVSMI_POWER_LIMIT_ENFORCED = 143
NVSMI_POWER_LIMIT_DEFAULT = 144
NVSMI_POWER_LIMIT_MIN = 145
NVSMI_POWER_LIMIT_MAX = 146

NVSMI_CLOCKS_GRAPHICS_CUR = 150
NVSMI_CLOCKS_SM_CUR = 151
NVSMI_CLOCKS_MEMORY_CUR = 152
NVSMI_CLOCKS_VIDEO_CUR = 153
NVSMI_CLOCKS_APPL_GRAPHICS = 154
NVSMI_CLOCKS_APPL_MEMORY = 155
NVSMI_CLOCKS_APPL_GRAPHICS_DEFAULT = 156
NVSMI_CLOCKS_APPL_MEMORY_DEFAULT = 157
NVSMI_CLOCKS_GRAPHICS_MAX = 158
NVSMI_CLOCKS_SM_MAX = 159
NVSMI_CLOCKS_MEMORY_MAX = 160

NVSMI_COMPUTE_APPS = 200
NVSMI_ACCOUNTED_APPS = 201
NVSMI_CLOCKS_POLICY = 202
NVSMI_CLOCKS_SUPPORTED = 203

#Details and descriptions for enumerations in help_query_gpu.txt
NVSMI_QUERY_GPU = {
    "timestamp":NVSMI_TIMESTAMP,
    "driver_version":NVSMI_DRIVER_VERSION,
    "count":NVSMI_COUNT,
    "name":NVSMI_NAME,
    "gpu_name":NVSMI_NAME,
    "serial":NVSMI_SERIALNUMBER,
    "gpu_serial":NVSMI_SERIALNUMBER,
    "uuid":NVSMI_UUID,
    "gpu_uuid":NVSMI_UUID,
    "pci.bus_id":NVSMI_PCI_BUS_ID,
    "gpu_bus_id":NVSMI_PCI_BUS_ID,
    "pci.domain":NVSMI_PCI_DOMAIN,
    "pci.bus":NVSMI_PCI_BUS,
    "pci.device":NVSMI_PCI_DEVICE,
    "pci.device_id":NVSMI_PCI_DEVICE_ID,
    "pci.sub_device_id":NVSMI_PCI_SUBDEVICE_ID,
    "pcie.link.gen.current":NVSMI_PCI_LINK_GEN_CUR,
    "pcie.link.gen.max":NVSMI_PCI_LINK_GEN_MAX,
    "pcie.link.width.current":NVSMI_PCI_LINK_WIDTH_CUR,
    "pcie.link.width.max":NVSMI_PCI_LINK_WIDTH_MAX,
    "index":NVSMI_INDEX,
    "display_mode":NVSMI_DISPLAY_MODE,
    "display_active":NVSMI_DISPLAY_ACTIVE,
    "persistence_mode":NVSMI_PERSISTENCE_MODE,
    "accounting.mode":NVSMI_ACCT_MODE,
    "accounting.buffer_size":NVSMI_ACCT_BUFFER_SIZE,
    "driver_model.current":NVSMI_DRIVER_MODEL_CUR,
    "driver_model.pending":NVSMI_DRIVER_MODEL_PENDING,
    "vbios_version":NVSMI_VBIOS_VER,
    "inforom.img":NVSMI_INFOROM_IMG,
    "inforom.image":NVSMI_INFOROM_IMG,
    "inforom.oem":NVSMI_INFOROM_OEM,
    "inforom.ecc":NVSMI_INFOROM_ECC,
    "inforom.pwr":NVSMI_INFOROM_PWR,
    "inforom.power":NVSMI_INFOROM_PWR,
    "gom.current":NVSMI_GOM_CUR,
    "gpu_operation_mode.current":NVSMI_GOM_CUR,
    "gom.pending":NVSMI_GOM_PENDING,
    "gpu_operation_mode.pending":NVSMI_GOM_PENDING,
    "fan.speed":NVSMI_FAN_SPEED,
    "pstate":NVSMI_PSTATE,
    "clocks_throttle_reasons.supported":NVSMI_CLOCK_THROTTLE_REASONS_SUPPORTED,
    "clocks_throttle_reasons.active":NVSMI_CLOCK_THROTTLE_REASONS_ACTIVE,
    "clocks_throttle_reasons.gpu_idle":NVSMI_CLOCK_THROTTLE_REASONS_IDLE,
    "clocks_throttle_reasons.applications_clocks_setting":NVSMI_CLOCK_THROTTLE_REASONS_APP_SETTING,
    "clocks_throttle_reasons.sw_power_cap":NVSMI_CLOCK_THROTTLE_REASONS_SW_PWR_CAP,
    "clocks_throttle_reasons.hw_slowdown":NVSMI_CLOCK_THROTTLE_REASONS_HW_SLOWDOWN,
    "clocks_throttle_reasons.hw_thermal_slowdown":NVSMI_CLOCK_THROTTLE_REASONS_HW_THERMAL_SLOWDOWN,
    "clocks_throttle_reasons.hw_power_brake_slowdown":NVSMI_CLOCK_THROTTLE_REASONS_HW_PWR_BRAKE_SLOWDOWN,
    "clocks_throttle_reasons.sw_thermal_slowdown":NVSMI_CLOCK_THROTTLE_REASONS_SW_THERMAL_SLOWDOWN,
    "clocks_throttle_reasons.sync_boost":NVSMI_CLOCK_THROTTLE_REASONS_SYNC_BOOST,
    "memory.total":NVSMI_MEMORY_TOTAL,
    "memory.used":NVSMI_MEMORY_USED,
    "memory.free":NVSMI_MEMORY_FREE,
    "compute_mode":NVSMI_COMPUTE_MODE,
    "utilization.gpu":NVSMI_UTILIZATION_GPU,
    "utilization.memory":NVSMI_UTILIZATION_MEM,
    "encoder.stats.sessionCount":NVSMI_ENCODER_STATS_SESSIONCOUNT,
    "encoder.stats.averageFps":NVSMI_ENCODER_STATS_AVG_EPS,
    "encoder.stats.averageLatency":NVSMI_ENCODER_STATS_AVG_LATENCY,
    "ecc.mode.current":NVSMI_ECC_MODE_CUR,
    "ecc.mode.pending":NVSMI_ECC_MODE_PENDING,
    "ecc.errors.corrected.volatile.device_memory":NVSMI_ECC_ERROR_CORRECTED_VOLATILE_DEV_MEM,
    "ecc.errors.corrected.volatile.register_file":NVSMI_ECC_ERROR_CORRECTED_VOLATILE_REGFILE,
    "ecc.errors.corrected.volatile.l1_cache":NVSMI_ECC_ERROR_CORRECTED_VOLATILE_L1CACHE,
    "ecc.errors.corrected.volatile.l2_cache":NVSMI_ECC_ERROR_CORRECTED_VOLATILE_L2CACHE,
    "ecc.errors.corrected.volatile.texture_memory":NVSMI_ECC_ERROR_CORRECTED_VOLATILE_TEXTURE,
    "ecc.errors.corrected.volatile.total":NVSMI_ECC_ERROR_CORRECTED_VOLATILE_TOTAL,
    "ecc.errors.corrected.aggregate.device_memory":NVSMI_ECC_ERROR_CORRECTED_AGGREGATE_DEV_MEM,
    "ecc.errors.corrected.aggregate.register_file":NVSMI_ECC_ERROR_CORRECTED_AGGREGATE_REGFILE,
    "ecc.errors.corrected.aggregate.l1_cache":NVSMI_ECC_ERROR_CORRECTED_AGGREGATE_L1CACHE,
    "ecc.errors.corrected.aggregate.l2_cache":NVSMI_ECC_ERROR_CORRECTED_AGGREGATE_L2CACHE,
    "ecc.errors.corrected.aggregate.texture_memory":NVSMI_ECC_ERROR_CORRECTED_AGGREGATE_TEXTURE,
    "ecc.errors.corrected.aggregate.total":NVSMI_ECC_ERROR_CORRECTED_AGGREGATE_TOTAL,
    "ecc.errors.uncorrected.volatile.device_memory":NVSMI_ECC_ERROR_UNCORRECTED_VOLATILE_DEV_MEM,
    "ecc.errors.uncorrected.volatile.register_file":NVSMI_ECC_ERROR_UNCORRECTED_VOLATILE_REGFILE,
    "ecc.errors.uncorrected.volatile.l1_cache":NVSMI_ECC_ERROR_UNCORRECTED_VOLATILE_L1CACHE,
    "ecc.errors.uncorrected.volatile.l2_cache":NVSMI_ECC_ERROR_UNCORRECTED_VOLATILE_L2CACHE,
    "ecc.errors.uncorrected.volatile.texture_memory":NVSMI_ECC_ERROR_UNCORRECTED_VOLATILE_TEXTURE,
    "ecc.errors.uncorrected.volatile.total":NVSMI_ECC_ERROR_UNCORRECTED_VOLATILE_TOTAL,
    "ecc.errors.uncorrected.aggregate.device_memory":NVSMI_ECC_ERROR_UNCORRECTED_AGGREGATE_DEV_MEM,
    "ecc.errors.uncorrected.aggregate.register_file":NVSMI_ECC_ERROR_UNCORRECTED_AGGREGATE_REGFILE,
    "ecc.errors.uncorrected.aggregate.l1_cache":NVSMI_ECC_ERROR_UNCORRECTED_AGGREGATE_L1CACHE,
    "ecc.errors.uncorrected.aggregate.l2_cache":NVSMI_ECC_ERROR_UNCORRECTED_AGGREGATE_L2CACHE,
    "ecc.errors.uncorrected.aggregate.texture_memory":NVSMI_ECC_ERROR_UNCORRECTED_AGGREGATE_TEXTURE,
    "ecc.errors.uncorrected.aggregate.total":NVSMI_ECC_ERROR_UNCORRECTED_AGGREGATE_TOTAL,
    "retired_pages.single_bit_ecc.count":NVSMI_RETIREDPAGES_SINGLE_BIT_ECC_COUNT,
    "retired_pages.sbe":NVSMI_RETIREDPAGES_SINGLE_BIT_ECC_COUNT,
    "retired_pages.double_bit.count":NVSMI_RETIREDPAGES_DOUBLE_BIT_ECC_COUNT,
    "retired_pages.dbe":NVSMI_RETIREDPAGES_DOUBLE_BIT_ECC_COUNT,
    "retired_pages.pending":NVSMI_RETIREDPAGES_PENDING,
    "temperature.gpu":NVSMI_TEMPERATURE_GPU,
    "temperature.memory":NVSMI_TEMPERATURE_MEMORY,
    "power.management":NVSMI_POWER_MGMT,
    "power.draw":NVSMI_POWER_DRAW,
    "power.limit":NVSMI_POWER_LIMIT,
    "enforced.power.limit":NVSMI_POWER_LIMIT_ENFORCED,
    "power.default_limit":NVSMI_POWER_LIMIT_DEFAULT,
    "power.min_limit":NVSMI_POWER_LIMIT_MIN,
    "power.max_limit":NVSMI_POWER_LIMIT_MAX,
    "clocks.current.graphics":NVSMI_CLOCKS_GRAPHICS_CUR,
    "clocks.gr":NVSMI_CLOCKS_GRAPHICS_CUR,
    "clocks.current.sm":NVSMI_CLOCKS_SM_CUR,
    "clocks.sm":NVSMI_CLOCKS_SM_CUR,
    "clocks.current.memory":NVSMI_CLOCKS_MEMORY_CUR,
    "clocks.mem":NVSMI_CLOCKS_MEMORY_CUR,
    "clocks.current.video":NVSMI_CLOCKS_VIDEO_CUR,
    "clocks.video":NVSMI_CLOCKS_VIDEO_CUR,
    "clocks.applications.graphics":NVSMI_CLOCKS_APPL_GRAPHICS,
    "clocks.applications.gr":NVSMI_CLOCKS_APPL_GRAPHICS,
    "clocks.applications.memory":NVSMI_CLOCKS_APPL_MEMORY,
    "clocks.applications.mem":NVSMI_CLOCKS_APPL_MEMORY,
    "clocks.default_applications.graphics":NVSMI_CLOCKS_APPL_GRAPHICS_DEFAULT,
    "clocks.default_applications.gr":NVSMI_CLOCKS_APPL_GRAPHICS_DEFAULT,
    "clocks.default_applications.memory":NVSMI_CLOCKS_APPL_MEMORY_DEFAULT,
    "clocks.default_applications.mem":NVSMI_CLOCKS_APPL_MEMORY_DEFAULT,
    "clocks.max.graphics":NVSMI_CLOCKS_GRAPHICS_MAX,
    "clocks.max.gr":NVSMI_CLOCKS_GRAPHICS_MAX,
    "clocks.max.sm":NVSMI_CLOCKS_SM_MAX,
    "clocks.max.sm":NVSMI_CLOCKS_SM_MAX,
    "clocks.max.memory":NVSMI_CLOCKS_MEMORY_MAX,
    "clocks.max.mem":NVSMI_CLOCKS_MEMORY_MAX,
    "supported-clocks":NVSMI_CLOCKS_SUPPORTED,
    "compute-apps":NVSMI_COMPUTE_APPS,
    "accounted-apps":NVSMI_ACCOUNTED_APPS,
    "clocks":NVSMI_CLOCKS_POLICY,
}

NVSMI_BRAND_NAMES = {NVML_BRAND_UNKNOWN :  "Unknown",
                     NVML_BRAND_QUADRO  :  "Quadro",
                     NVML_BRAND_TESLA   :  "Tesla",
                     NVML_BRAND_NVS     :  "NVS",
                     NVML_BRAND_GRID    :  "Grid",
                     NVML_BRAND_GEFORCE :  "GeForce",
}

## ========================================================================== ##
##                                                                            ##
##                             nvidia_smi Class                               ##
##                                                                            ##
## ========================================================================== ##

class nvidia_smi:
  __instance = None
  __handles = None

  class loop_async:
      __last_result = None
      __task = None
      __abort = False
      __callback_chain = None

      def __init__(self, time_in_milliseconds=1, filter=None, callback=None):
          self.__abort = False
          self.__callback_chain = callback

          self.__task = Thread(target = nvidia_smi.loop_async.__loop_task, args = (self, time_in_milliseconds, filter, nvidia_smi.loop_async.__callback))
          self.__task.start()

      def __del__(self):
          self.__abort = True
          self.__callback_chain = None

      @staticmethod
      def __loop_task(async_results, time_in_milliseconds=1, filter=None, callback=None):
          delay_seconds = time_in_milliseconds / 1000
          nvsmi = nvidia_smi.getInstance()

          while async_results.is_aborted() == False:
              results = nvsmi.DeviceQuery(filter)
              async_results.__last_results = results
              if (callback is not None):
                  callback(async_results, results)

              time.sleep(delay_seconds)

      def __callback(self, result):
          self.__last_result = result
          if (self.__callback_chain is not None):
              self.__callback_chain(self, result)

      def cancel(self):
          self.__abort = True
          if (self.__task is not None):
              self.__task.join()

      def is_aborted(self):
          return self.__abort

      def result(self):
          return self.__last_result

  @staticmethod
  def getInstance():
      ''' Static access method. '''
      if nvidia_smi.__instance == None:
          nvidia_smi()
      return nvidia_smi.__instance

  @staticmethod
  def loop(time_in_milliseconds=1, filter=None, callback=None):
      return nvidia_smi.loop_async(time_in_milliseconds, filter, callback)

  def __init__(self):
      ''' Virtually private constructor. '''
      if nvidia_smi.__instance != None:
         raise Exception("This class is a singleton, use getInstance()")
      else:
         nvidia_smi.__instance = self

      if nvidia_smi.__handles == None:
         nvidia_smi.__handles = nvidia_smi.__initialize_nvml()

  @staticmethod
  def __initialize_nvml():
    ''' Initialize NVML bindings. '''
    nvmlInit()
    deviceCount = nvmlDeviceGetCount()
    handles = {}
    for i in range(0, deviceCount):
        handles[i] = nvmlDeviceGetHandleByIndex(i)
    return handles

  def __del__(self):
    if (nvidia_smi.__instance != None) and (nvidia_smi.__instance == self):
        del(nvidia_smi.__instance)
        nvidia_smi.__instance = None
        nvidia_smi.__handles = None
        nvmlShutdown()

  #
  # Helper functions
  #

  @staticmethod
  def __fromDeviceQueryString(queryString):
      parameters = queryString.split(",")
      values = []
      for p in parameters:
          ps = p.strip()
          if (ps in NVSMI_QUERY_GPU):
            values.append(NVSMI_QUERY_GPU[ps])

      return values


  @staticmethod
  def __xmlGetEccByType(handle, counterType, errorType):
      strResult = ''

      try:
          deviceMemory = nvmlDeviceGetMemoryErrorCounter(handle, errorType, counterType,
                                                         NVML_MEMORY_LOCATION_DEVICE_MEMORY)
      except NVMLError as err:
          deviceMemory = nvidia_smi.__handleError(err)
      strResult += '          <device_memory>' + nvidia_smi.__toString(deviceMemory) + '</device_memory>\n'

      try:
          registerFile = nvmlDeviceGetMemoryErrorCounter(handle, errorType, counterType,
                                                         NVML_MEMORY_LOCATION_REGISTER_FILE)
      except NVMLError as err:
          registerFile = nvidia_smi.__handleError(err)

      strResult += '          <register_file>' + nvidia_smi.__toString(registerFile) + '</register_file>\n'

      try:
          l1Cache = nvmlDeviceGetMemoryErrorCounter(handle, errorType, counterType,
                                                    NVML_MEMORY_LOCATION_L1_CACHE)
      except NVMLError as err:
          l1Cache = nvidia_smi.__handleError(err)
      strResult += '          <l1_cache>' + nvidia_smi.__toString(l1Cache) + '</l1_cache>\n'

      try:
          l2Cache = nvmlDeviceGetMemoryErrorCounter(handle, errorType, counterType,
                                                    NVML_MEMORY_LOCATION_L2_CACHE)
      except NVMLError as err:
          l2Cache = nvidia_smi.__handleError(err)
      strResult += '          <l2_cache>' + nvidia_smi.__toString(l2Cache) + '</l2_cache>\n'

      try:
          textureMemory = nvmlDeviceGetMemoryErrorCounter(handle, errorType, counterType,
                                                          NVML_MEMORY_LOCATION_TEXTURE_MEMORY)
      except NVMLError as err:
          textureMemory = nvidia_smi.__handleError(err)
      strResult += '          <texture_memory>' + nvidia_smi.__toString(textureMemory) + '</texture_memory>\n'

      try:
          count = nvidia_smi.__toString(nvmlDeviceGetTotalEccErrors(handle, errorType, counterType))
      except NVMLError as err:
          count = nvidia_smi.__handleError(err)
      strResult += '          <total>' + count + '</total>\n'

      return strResult

  @staticmethod
  def __GetEccByType(handle, counterType, errorType):
      strResult = ''

      eccByType = {}
      try:
          deviceMemory = nvmlDeviceGetMemoryErrorCounter(handle, errorType, counterType,
                                                         NVML_MEMORY_LOCATION_DEVICE_MEMORY)
      except NVMLError as err:
          deviceMemory = nvidia_smi.__handleError(err)

      eccByType['device_memory'] = deviceMemory

      try:
          registerFile = nvmlDeviceGetMemoryErrorCounter(handle, errorType, counterType,
                                                         NVML_MEMORY_LOCATION_REGISTER_FILE)
      except NVMLError as err:
          registerFile = nvidia_smi.__handleError(err)

      eccByType['register_file'] = registerFile

      try:
          l1Cache = nvmlDeviceGetMemoryErrorCounter(handle, errorType, counterType,
                                                    NVML_MEMORY_LOCATION_L1_CACHE)
      except NVMLError as err:
          l1Cache = nvidia_smi.__handleError(err)
      eccByType['l1_cache'] = l1Cache

      try:
          l2Cache = nvmlDeviceGetMemoryErrorCounter(handle, errorType, counterType,
                                                    NVML_MEMORY_LOCATION_L2_CACHE)
      except NVMLError as err:
          l2Cache = nvidia_smi.__handleError(err)
      eccByType['l2_cache'] = l2Cache

      try:
          textureMemory = nvmlDeviceGetMemoryErrorCounter(handle, errorType, counterType,
                                                          NVML_MEMORY_LOCATION_TEXTURE_MEMORY)
      except NVMLError as err:
          textureMemory = nvidia_smi.__handleError(err)
      eccByType['texture_memory'] = textureMemory

      try:
          count = nvidia_smi.__toString(nvmlDeviceGetTotalEccErrors(handle, errorType, counterType))
      except NVMLError as err:
          count = nvidia_smi.__handleError(err)
      eccByType['total'] = count

      return eccByType

  @staticmethod
  def __xmlGetEccByCounter(handle, counterType, filter):
      eccByCounter = ''
      if (NVSMI_ALL in filter or
          NVSMI_ECC_ERROR_CORRECTED_VOLATILE_DEV_MEM in filter or
          NVSMI_ECC_ERROR_CORRECTED_VOLATILE_REGFILE in filter or
          NVSMI_ECC_ERROR_CORRECTED_VOLATILE_L1CACHE in filter or
          NVSMI_ECC_ERROR_CORRECTED_VOLATILE_L2CACHE in filter or
          NVSMI_ECC_ERROR_CORRECTED_VOLATILE_TEXTURE in filter or
          NVSMI_ECC_ERROR_CORRECTED_AGGREGATE_DEV_MEM in filter or
          NVSMI_ECC_ERROR_CORRECTED_AGGREGATE_REGFILE in filter or
          NVSMI_ECC_ERROR_CORRECTED_AGGREGATE_L1CACHE in filter or
          NVSMI_ECC_ERROR_CORRECTED_AGGREGATE_L2CACHE in filter or
          NVSMI_ECC_ERROR_CORRECTED_AGGREGATE_TEXTURE in filter or
          NVSMI_ECC_ERROR_CORRECTED_AGGREGATE_TOTAL in filter):
          eccByCounter += '        <single_bit>\n'
          eccByCounter += nvidia_smi.__toString(nvidia_smi.__xmlGetEccByType(handle, counterType, NVML_MEMORY_ERROR_TYPE_CORRECTED))
          eccByCounter += '        </single_bit>\n'

      if (NVSMI_ALL in filter or
          NVSMI_ECC_ERROR_CORRECTED_VOLATILE_DEV_MEM in filter or
          NVSMI_ECC_ERROR_CORRECTED_VOLATILE_REGFILE in filter or
          NVSMI_ECC_ERROR_CORRECTED_VOLATILE_L1CACHE in filter or
          NVSMI_ECC_ERROR_CORRECTED_VOLATILE_L2CACHE in filter or
          NVSMI_ECC_ERROR_CORRECTED_VOLATILE_TEXTURE in filter or
          NVSMI_ECC_ERROR_CORRECTED_AGGREGATE_DEV_MEM in filter or
          NVSMI_ECC_ERROR_CORRECTED_AGGREGATE_REGFILE in filter or
          NVSMI_ECC_ERROR_CORRECTED_AGGREGATE_L1CACHE in filter or
          NVSMI_ECC_ERROR_CORRECTED_AGGREGATE_L2CACHE in filter or
          NVSMI_ECC_ERROR_CORRECTED_AGGREGATE_TEXTURE in filter or
          NVSMI_ECC_ERROR_CORRECTED_AGGREGATE_TOTAL in filter):
          eccByCounter += '        <double_bit>\n'
          eccByCounter += nvidia_smi.__toString(nvidia_smi.__xmlGetEccByType(handle, counterType, NVML_MEMORY_ERROR_TYPE_UNCORRECTED))
          eccByCounter += '        </double_bit>\n'

      return eccByCounter

  @staticmethod
  def __GetEccByCounter(handle, counterType, filter):
      eccByCounter = {}

      if (NVSMI_ALL in filter or
          NVSMI_ECC_ERROR_CORRECTED_VOLATILE_DEV_MEM in filter or
          NVSMI_ECC_ERROR_CORRECTED_VOLATILE_REGFILE in filter or
          NVSMI_ECC_ERROR_CORRECTED_VOLATILE_L1CACHE in filter or
          NVSMI_ECC_ERROR_CORRECTED_VOLATILE_L2CACHE in filter or
          NVSMI_ECC_ERROR_CORRECTED_VOLATILE_TEXTURE in filter or
          NVSMI_ECC_ERROR_CORRECTED_AGGREGATE_DEV_MEM in filter or
          NVSMI_ECC_ERROR_CORRECTED_AGGREGATE_REGFILE in filter or
          NVSMI_ECC_ERROR_CORRECTED_AGGREGATE_L1CACHE in filter or
          NVSMI_ECC_ERROR_CORRECTED_AGGREGATE_L2CACHE in filter or
          NVSMI_ECC_ERROR_CORRECTED_AGGREGATE_TEXTURE in filter or
          NVSMI_ECC_ERROR_CORRECTED_AGGREGATE_TOTAL in filter):
          eccByCounter['single_bit'] = nvidia_smi.__GetEccByType(handle, counterType, NVML_MEMORY_ERROR_TYPE_CORRECTED)

      if (NVSMI_ALL in filter or
          NVSMI_ECC_ERROR_CORRECTED_VOLATILE_DEV_MEM in filter or
          NVSMI_ECC_ERROR_CORRECTED_VOLATILE_REGFILE in filter or
          NVSMI_ECC_ERROR_CORRECTED_VOLATILE_L1CACHE in filter or
          NVSMI_ECC_ERROR_CORRECTED_VOLATILE_L2CACHE in filter or
          NVSMI_ECC_ERROR_CORRECTED_VOLATILE_TEXTURE in filter or
          NVSMI_ECC_ERROR_CORRECTED_AGGREGATE_DEV_MEM in filter or
          NVSMI_ECC_ERROR_CORRECTED_AGGREGATE_REGFILE in filter or
          NVSMI_ECC_ERROR_CORRECTED_AGGREGATE_L1CACHE in filter or
          NVSMI_ECC_ERROR_CORRECTED_AGGREGATE_L2CACHE in filter or
          NVSMI_ECC_ERROR_CORRECTED_AGGREGATE_TEXTURE in filter or
          NVSMI_ECC_ERROR_CORRECTED_AGGREGATE_TOTAL in filter):
          eccByCounter['double_bit'] = nvidia_smi.__GetEccByType(handle, counterType, NVML_MEMORY_ERROR_TYPE_UNCORRECTED)

      return eccByCounter

  @staticmethod
  def __xmlGetEcc(handle, filter):
      ecc = ''
      includeEcc = False
      if (NVSMI_ALL in filter or
          NVSMI_ECC_ERROR_CORRECTED_VOLATILE_DEV_MEM in filter or
          NVSMI_ECC_ERROR_CORRECTED_VOLATILE_REGFILE in filter or
          NVSMI_ECC_ERROR_CORRECTED_VOLATILE_L1CACHE in filter or
          NVSMI_ECC_ERROR_CORRECTED_VOLATILE_L2CACHE in filter or
          NVSMI_ECC_ERROR_CORRECTED_VOLATILE_TEXTURE in filter or
          NVSMI_ECC_ERROR_UNCORRECTED_VOLATILE_DEV_MEM in filter or
          NVSMI_ECC_ERROR_UNCORRECTED_VOLATILE_REGFILE in filter or
          NVSMI_ECC_ERROR_UNCORRECTED_VOLATILE_L1CACHE in filter or
          NVSMI_ECC_ERROR_UNCORRECTED_VOLATILE_L2CACHE in filter or
          NVSMI_ECC_ERROR_UNCORRECTED_VOLATILE_TEXTURE in filter or
          NVSMI_ECC_ERROR_UNCORRECTED_VOLATILE_TOTAL in filter  ):
          ecc += '      <volatile>\n'
          ecc += nvidia_smi.__toString(nvidia_smi.__xmlGetEccByCounter(handle, NVML_VOLATILE_ECC, filter))
          ecc += '      </volatile>\n'
          includeEcc = True

      if (NVSMI_ALL in filter or
          NVSMI_ECC_ERROR_CORRECTED_AGGREGATE_DEV_MEM in filter or
          NVSMI_ECC_ERROR_CORRECTED_AGGREGATE_REGFILE in filter or
          NVSMI_ECC_ERROR_CORRECTED_AGGREGATE_L1CACHE in filter or
          NVSMI_ECC_ERROR_CORRECTED_AGGREGATE_L2CACHE in filter or
          NVSMI_ECC_ERROR_CORRECTED_AGGREGATE_TEXTURE in filter or
          NVSMI_ECC_ERROR_CORRECTED_AGGREGATE_TOTAL in filter or
          NVSMI_ECC_ERROR_UNCORRECTED_AGGREGATE_DEV_MEM in filter or
          NVSMI_ECC_ERROR_UNCORRECTED_AGGREGATE_REGFILE in filter or
          NVSMI_ECC_ERROR_UNCORRECTED_AGGREGATE_L1CACHE in filter or
          NVSMI_ECC_ERROR_UNCORRECTED_AGGREGATE_L2CACHE in filter or
          NVSMI_ECC_ERROR_UNCORRECTED_AGGREGATE_TEXTURE in filter or
          NVSMI_ECC_ERROR_UNCORRECTED_AGGREGATE_TOTAL in filter ):
          ecc += '      <aggregate>\n'
          ecc += nvidia_smi.__toString(nvidia_smi.__xmlGetEccByCounter(handle, NVML_AGGREGATE_ECC, filter))
          ecc += '      </aggregate>\n'
          includeEcc = True

      return ecc if len(ecc) > 0 else None, includeEcc

  @staticmethod
  def __GetEcc( handle, filter):
      ecc = {}
      includeEcc = False
      if (NVSMI_ALL in filter or
          NVSMI_ECC_ERROR_CORRECTED_VOLATILE_DEV_MEM in filter or
          NVSMI_ECC_ERROR_CORRECTED_VOLATILE_REGFILE in filter or
          NVSMI_ECC_ERROR_CORRECTED_VOLATILE_L1CACHE in filter or
          NVSMI_ECC_ERROR_CORRECTED_VOLATILE_L2CACHE in filter or
          NVSMI_ECC_ERROR_CORRECTED_VOLATILE_TEXTURE in filter or
          NVSMI_ECC_ERROR_UNCORRECTED_VOLATILE_DEV_MEM in filter or
          NVSMI_ECC_ERROR_UNCORRECTED_VOLATILE_REGFILE in filter or
          NVSMI_ECC_ERROR_UNCORRECTED_VOLATILE_L1CACHE in filter or
          NVSMI_ECC_ERROR_UNCORRECTED_VOLATILE_L2CACHE in filter or
          NVSMI_ECC_ERROR_UNCORRECTED_VOLATILE_TEXTURE in filter or
          NVSMI_ECC_ERROR_UNCORRECTED_VOLATILE_TOTAL in filter  ):
          ecc['volatile'] = nvidia_smi.__GetEccByCounter(handle, NVML_VOLATILE_ECC, filter)
          includeEcc = True

      if (NVSMI_ALL in filter or
          NVSMI_ECC_ERROR_CORRECTED_AGGREGATE_DEV_MEM in filter or
          NVSMI_ECC_ERROR_CORRECTED_AGGREGATE_REGFILE in filter or
          NVSMI_ECC_ERROR_CORRECTED_AGGREGATE_L1CACHE in filter or
          NVSMI_ECC_ERROR_CORRECTED_AGGREGATE_L2CACHE in filter or
          NVSMI_ECC_ERROR_CORRECTED_AGGREGATE_TEXTURE in filter or
          NVSMI_ECC_ERROR_CORRECTED_AGGREGATE_TOTAL in filter or
          NVSMI_ECC_ERROR_UNCORRECTED_AGGREGATE_DEV_MEM in filter or
          NVSMI_ECC_ERROR_UNCORRECTED_AGGREGATE_REGFILE in filter or
          NVSMI_ECC_ERROR_UNCORRECTED_AGGREGATE_L1CACHE in filter or
          NVSMI_ECC_ERROR_UNCORRECTED_AGGREGATE_L2CACHE in filter or
          NVSMI_ECC_ERROR_UNCORRECTED_AGGREGATE_TEXTURE in filter or
          NVSMI_ECC_ERROR_UNCORRECTED_AGGREGATE_TOTAL in filter ):
          ecc['aggregate'] = nvidia_smi.__GetEccByCounter(handle, NVML_AGGREGATE_ECC, filter)
          includeEcc = True

      return ecc if len(ecc.values()) > 0 else None, includeEcc

  @staticmethod
  def __xmlGetRetiredPagesByCause(handle, cause):
      retiredPagedByCause = ''

      error = None
      count = 0
      try:
          pages = nvmlDeviceGetRetiredPages(handle, cause)
          count = sum(map(len, pages)) #[py2] count = nvidia_smi.__toString(len(pages))
      except NVMLError as err:
          error = nvidia_smi.__handleError(err)
          pages = None

      retiredPagedByCause += '        <retired_count>' + nvidia_smi.__toString(count) + '</retired_count>\n'
      if pages is not None:
          retiredPagedByCause += '        <retired_page_addresses>\n'
          for page in pages:
              retiredPagedByCause += '          <retired_page_address>' + "0x%016x" % page + '</retired_page_address>\n'
          retiredPagedByCause += '        </retired_page_addresses>\n'
      else:
          retiredPagedByCause += '        <retired_page_addresses>' + error + '</retired_page_addresses>\n'

      return retiredPagedByCause if count > 0 else ''

  @staticmethod
  def __GetRetiredPagesByCause(handle, cause):
      retiredPagedByCause = {}

      error = None
      count = 0
      try:
          pages = nvmlDeviceGetRetiredPages(handle, cause)
      except NVMLError as err:
          error = nvidia_smi.__handleError(err)
          pages = None

      retiredPageAddresses = {}
      if pages is not None:
          ii = 1
          for page in pages:
              retiredPageAddresses['retired_page_address_'+str(ii)] = "0x%016x" % page
              ii+=1
              count+=1
      if error is not None:
          retiredPageAddresses['Error'] = error

      retiredPagedByCause['retired_count'] = count
      retiredPagedByCause['retired_page_addresses'] = retiredPageAddresses if len(retiredPageAddresses.values()) > 0 else None

      return retiredPagedByCause if count > 0 else None

  @staticmethod
  def __xmlGetRetiredPages(handle, filter):
      retiredPages = ''
      includeRetiredPages = False

      causes = [ "multiple_single_bit_retirement", "double_bit_retirement" ]
      for idx in range(NVML_PAGE_RETIREMENT_CAUSE_COUNT):
          if (NVSMI_ALL in filter or
              (NVSMI_RETIREDPAGES_SINGLE_BIT_ECC_COUNT in filter and idx == 0) or
              (NVSMI_RETIREDPAGES_DOUBLE_BIT_ECC_COUNT in filter and idx == 1)):
              retiredPages += '      <' + causes[idx] + '>\n'
              retiredPages += nvidia_smi.__xmlGetRetiredPagesByCause(handle, idx)
              retiredPages += '      </' + causes[idx] + '>\n'
              includeRetiredPages = True

      if (NVSMI_ALL in filter or NVSMI_RETIREDPAGES_PENDING in filter):
          retiredPages += '      <pending_retirement>'
          try:
              if NVML_FEATURE_DISABLED == nvmlDeviceGetRetiredPagesPendingStatus(handle):
                  retiredPages += "No"
              else:
                  retiredPages += "Yes"
          except NVMLError as err:
              retiredPages += nvidia_smi.__handleError(err)
          retiredPages += '</pending_retirement>\n'
          includeRetiredPages = True

      return retiredPages if len(retiredPages) > 0 else None, includeRetiredPages

  @staticmethod
  def __GetRetiredPages(handle, filter):
      retiredPages = {}
      includeRetiredPages = False

      causes = [ "multiple_single_bit_retirement", "double_bit_retirement" ]
      for idx in range(NVML_PAGE_RETIREMENT_CAUSE_COUNT):
          if (NVSMI_ALL in filter or
              (NVSMI_RETIREDPAGES_SINGLE_BIT_ECC_COUNT in filter and idx == 0) or
              (NVSMI_RETIREDPAGES_DOUBLE_BIT_ECC_COUNT in filter and idx == 1)):
              retiredPages[causes[idx]] = nvidia_smi.__GetRetiredPagesByCause(handle, idx)
              includeRetiredPages = True

      if (NVSMI_ALL in filter or NVSMI_RETIREDPAGES_PENDING in filter):
          pending_retirement = ''
          try:
              if NVML_FEATURE_DISABLED == nvmlDeviceGetRetiredPagesPendingStatus(handle):
                  pending_retirement = "No"
              else:
                  pending_retirement = "Yes"
          except NVMLError as err:
              pending_retirement = nvidia_smi.__handleError(err)
          retiredPages['pending_retirement'] = pending_retirement
          includeRetiredPages = True

      return retiredPages if len(retiredPages.values()) > 0 else None, includeRetiredPages

  @staticmethod
  def __toStrGOM(mode):
      if mode == NVML_GOM_ALL_ON:
          return "All On";
      elif mode == NVML_GOM_COMPUTE:
          return "Compute";
      elif mode == NVML_GOM_LOW_DP:
          return "Low Double Precision";
      else:
          return "Unknown";

  @staticmethod
  def __xmlGetClocksThrottleReasons(handle):
      throttleReasons = [
              [nvmlClocksThrottleReasonGpuIdle,           "clocks_throttle_reason_gpu_idle"],
              [nvmlClocksThrottleReasonUserDefinedClocks, "clocks_throttle_reason_user_defined_clocks"],
              [nvmlClocksThrottleReasonApplicationsClocksSetting, "clocks_throttle_reason_applications_clocks_setting"],
              [nvmlClocksThrottleReasonSwPowerCap,        "clocks_throttle_reason_sw_power_cap"],
              [nvmlClocksThrottleReasonHwSlowdown,        "clocks_throttle_reason_hw_slowdown"],
              [nvmlClocksThrottleReasonUnknown,           "clocks_throttle_reason_unknown"]
              ];

      strResult = ''

      try:
          supportedClocksThrottleReasons = nvmlDeviceGetSupportedClocksThrottleReasons(handle);
          clocksThrottleReasons = nvmlDeviceGetCurrentClocksThrottleReasons(handle);
          strResult += '    <clocks_throttle_reasons>\n'
          for (mask, name) in throttleReasons:
              if (name != "clocks_throttle_reason_user_defined_clocks"):
                  if (mask & supportedClocksThrottleReasons):
                      val = "Active" if mask & clocksThrottleReasons else "Not Active";
                  else:
                      val = "N/A" #nvidia_smi.__handleError(NVML_ERROR_NOT_SUPPORTED);
                  strResult += "      <%s>%s</%s>\n" % (name, val, name);
          strResult += '    </clocks_throttle_reasons>\n'
      except NVMLError as err:
          strResult += '    <clocks_throttle_reasons>%s</clocks_throttle_reasons>\n' % (nvidia_smi.__handleError(err));

      return strResult;

  @staticmethod
  def __GetClocksThrottleReasons(handle):
      throttleReasons = [
              [nvmlClocksThrottleReasonGpuIdle,           "clocks_throttle_reason_gpu_idle"],
              [nvmlClocksThrottleReasonUserDefinedClocks, "clocks_throttle_reason_user_defined_clocks"],
              [nvmlClocksThrottleReasonApplicationsClocksSetting, "clocks_throttle_reason_applications_clocks_setting"],
              [nvmlClocksThrottleReasonSwPowerCap,        "clocks_throttle_reason_sw_power_cap"],
              [nvmlClocksThrottleReasonHwSlowdown,        "clocks_throttle_reason_hw_slowdown"],
              [nvmlClocksThrottleReasonUnknown,           "clocks_throttle_reason_unknown"]
              ];

      clockThrottleReasons = {}

      try:
          supportedClocksThrottleReasons = nvmlDeviceGetSupportedClocksThrottleReasons(handle);
          clocksThrottleReasons = nvmlDeviceGetCurrentClocksThrottleReasons(handle);
          for (mask, name) in throttleReasons:
              if (name != "clocks_throttle_reason_user_defined_clocks"):
                  if (mask & supportedClocksThrottleReasons):
                      val = "Active" if mask & clocksThrottleReasons else "Not Active";
                  else:
                      val = "N/A" #nvidia_smi.__handleError(NVML_ERROR_NOT_SUPPORTED);
                  clockThrottleReasons[name]= val;
      except NVMLError as err:
          clockThrottleReasons['Error'] = nvidia_smi.__handleError(err);

      return clockThrottleReasons if len(clockThrottleReasons.values()) > 0 else None

  #
  # Converts errors into string messages
  #
  @staticmethod
  def __handleError(err):
      if (err.value == NVML_ERROR_NOT_SUPPORTED):
          return "N/A"
      else:
          return err.__str__()

  @staticmethod
  def __toString(val):
      if (isinstance(val, bytes)):
          return val.decode("utf-8")
      return str(val)


  @classmethod
  def XmlDeviceQuery(self, filter=None):
      '''
      Provides a Python interface to GPU management and monitoring functions.

      This is a wrapper around the NVML library.
      For information about the NVML library, see the NVML developer page
      http://developer.nvidia.com/nvidia-management-library-nvml

      Examples:
      ---------------------------------------------------------------------------
      For all elements as in XML format.  Similiar to nvisia-smi -q -x

      $ XmlDeviceQuery()

      ---------------------------------------------------------------------------
      For XML of filtered elements by string name.
      Similiar ot nvidia-smi --query-gpu=pci.bus_id,memory.total,memory.free
      See help_query_gpu.txt or XmlDeviceQuery("--help_query_gpu") for available filter elements

      $ XmlDeviceQuery("pci.bus_id,memory.total,memory.free")

      ---------------------------------------------------------------------------
      For XML of filtered elements by enumeration value.
      See help_query_gpu.txt or XmlDeviceQuery("--help_query_gpu") for available filter elements

      $ XmlDeviceQuery([NVSMI_PCI_BUS_ID, NVSMI_MEMORY_TOTAL, NVSMI_MEMORY_FREE])

      '''

      if (filter is None):
          filter = [NVSMI_ALL]
      elif (isinstance(filter, str)):
          if (filter == "--help") or (filter == "-h"):
              return nvidia_smi.XmlDeviceQuery.__doc__
          elif (filter == "--help-query-gpu"):
              with open("help_query_gpu.txt", 'r') as fin:
                  return fin.read()
          else:
              filter = nvidia_smi.__fromDeviceQueryString(filter)
      else:
          filter = list(filter)

      strResult = ''
      try:
          strResult += '<?xml version="1.0" ?>\n'
          strResult += '<!DOCTYPE nvidia_smi_log SYSTEM "nvsmi_device_v4.dtd">\n'
          strResult += '<nvidia_smi>\n'

          if (NVSMI_ALL in filter or NVSMI_TIMESTAMP in filter):
              strResult += '  <timestamp>' + nvidia_smi.__toString(datetime.date.today()) + '</timestamp>\n'
          if (NVSMI_ALL in filter or NVSMI_DRIVER_VERSION in filter):
              strResult += '  <driver_version>' + nvidia_smi.__toString(nvmlSystemGetDriverVersion()) + '</driver_version>\n'

          deviceCount = nvmlDeviceGetCount()
          if (NVSMI_ALL in filter or NVSMI_COUNT in filter):
              strResult += '  <count>' + nvidia_smi.__toString(deviceCount) + '</count>\n'

          for i in range(0, deviceCount):
              handle = self.__handles[i]

              pciInfo = nvmlDeviceGetPciInfo(handle)

              gpuInfo = ''

              if (NVSMI_ALL in filter or NVSMI_PCI_BUS_ID in filter):
                  gpuInfo += '  <id>%s</id>\n' % pciInfo.busId

              if (NVSMI_ALL in filter or NVSMI_NAME in filter):
                  gpuInfo += '    <product_name>' + nvidia_smi.__toString(nvmlDeviceGetName(handle)) + '</product_name>\n'

                  try:
                      # if nvmlDeviceGetBrand() succeeds it is guaranteed to be in the dictionary
                      brandName = NVSMI_BRAND_NAMES[nvmlDeviceGetBrand(handle)]
                  except NVMLError as err:
                      brandName = nvidia_smi.__handleError(err)


                  gpuInfo += '    <product_brand>' + brandName + '</product_brand>\n'

              if (NVSMI_ALL in filter or NVSMI_DISPLAY_MODE in filter):
                  try:
                      state = ('Enabled' if (nvmlDeviceGetDisplayMode(handle) != 0) else 'Disabled')
                  except NVMLError as err:
                      state = nvidia_smi.__handleError(err)

                  gpuInfo += '    <display_mode>' + state + '</display_mode>\n'

              if (NVSMI_ALL in filter or NVSMI_DISPLAY_ACTIVE in filter):
                  try:
                      state = ('Enabled' if (nvmlDeviceGetDisplayActive(handle) != 0) else 'Disabled')
                  except NVMLError as err:
                      state = nvidia_smi.__handleError(err)

                  gpuInfo += '    <display_active>' + state + '</display_active>\n'

              if (NVSMI_ALL in filter or NVSMI_PERSISTENCE_MODE in filter):
                  try:
                      mode = 'Enabled' if (nvmlDeviceGetPersistenceMode(handle) != 0) else 'Disabled'
                  except NVMLError as err:
                      mode = nvidia_smi.__handleError(err)

                  gpuInfo += '    <persistence_mode>' + mode + '</persistence_mode>\n'

              if (NVSMI_ALL in filter or NVSMI_ACCT_MODE in filter):
                  try:
                      mode = 'Enabled' if (nvmlDeviceGetAccountingMode(handle) != 0) else 'Disabled'
                  except NVMLError as err:
                      mode = nvidia_smi.__handleError(err)

                  gpuInfo += '    <accounting_mode>' + mode + '</accounting_mode>\n'

              if (NVSMI_ALL in filter or NVSMI_ACCT_BUFFER_SIZE in filter):
                  try:
                      bufferSize = nvidia_smi.__toString(nvmlDeviceGetAccountingBufferSize(handle))
                  except NVMLError as err:
                      bufferSize = nvidia_smi.__handleError(err)

                  gpuInfo += '    <accounting_mode_buffer_size>' + bufferSize + '</accounting_mode_buffer_size>\n'

              driverModel = ''
              includeDriverModel = False

              if (NVSMI_ALL in filter or NVSMI_DRIVER_MODEL_CUR in filter):
                  try:
                      current = 'WDDM' if (nvmlDeviceGetCurrentDriverModel(handle) == NVML_DRIVER_WDDM) else 'TCC'
                  except NVMLError as err:
                      current = nvidia_smi.__handleError(err)
                  driverModel += '      <current_dm>' + current + '</current_dm>\n'
                  includeDriverModel = True

              if (NVSMI_ALL in filter or NVSMI_DRIVER_MODEL_PENDING in filter):
                try:
                    pending = 'WDDM' if (nvmlDeviceGetPendingDriverModel(handle) == NVML_DRIVER_WDDM) else 'TCC'
                except NVMLError as err:
                    pending = nvidia_smi.__handleError(err)
                    driverModel += '      <pending_dm>' + pending + '</pending_dm>\n'
                    includeDriverModel = True

              if includeDriverModel:
                  gpuInfo += '    <driver_model>\n'
                  gpuInfo += driverModel
                  gpuInfo += '    </driver_model>\n'

              if (NVSMI_ALL in filter or NVSMI_SERIALNUMBER in filter):
                  try:
                      serial = nvmlDeviceGetSerial(handle)
                  except NVMLError as err:
                      serial = nvidia_smi.__handleError(err)

                  gpuInfo += '    <serial>' + nvidia_smi.__toString(serial) + '</serial>\n'

              if (NVSMI_ALL in filter or NVSMI_UUID in filter):
                  try:
                      uuid = nvmlDeviceGetUUID(handle)
                  except NVMLError as err:
                      uuid = nvidia_smi.__handleError(err)

                  gpuInfo += '    <uuid>' + nvidia_smi.__toString(uuid) + '</uuid>\n'

              if (NVSMI_ALL in filter or NVSMI_INDEX in filter):
                  try:
                      minor_number = nvmlDeviceGetMinorNumber(handle)
                  except NVMLError as err:
                      minor_number = nvidia_smi.__handleError(err)

                  gpuInfo += '    <minor_number>' + nvidia_smi.__toString(minor_number) + '</minor_number>\n'

              if (NVSMI_ALL in filter or NVSMI_VBIOS_VER in filter):
                  try:
                      vbios = nvmlDeviceGetVbiosVersion(handle)
                  except NVMLError as err:
                      vbios = nvidia_smi.__handleError(err)

                  gpuInfo += '    <vbios_version>' + nvidia_smi.__toString(vbios) + '</vbios_version>\n'

              if (NVSMI_ALL in filter or NVSMI_VBIOS_VER in filter):
                  try:
                      multiGpuBool = nvmlDeviceGetMultiGpuBoard(handle)
                  except NVMLError as err:
                      multiGpuBool = nvidia_smi.__handleError(err);

                  if multiGpuBool == "N/A":
                      gpuInfo += '    <multigpu_board>' + 'N/A' + '</multigpu_board>\n'
                  elif multiGpuBool:
                      gpuInfo += '    <multigpu_board>' + 'Yes' + '</multigpu_board>\n'
                  else:
                      gpuInfo += '    <multigpu_board>' + 'No' + '</multigpu_board>\n'

              if (NVSMI_ALL in filter or NVSMI_BOARD_ID in filter):
                  try:
                      boardId = nvmlDeviceGetBoardId(handle)
                  except NVMLError as err:
                      boardId = nvidia_smi.__handleError(err)

                  try:
                      hexBID = "0x%x" % boardId
                  except:
                      hexBID = boardId

                  gpuInfo += '    <board_id>' + hexBID + '</board_id>\n'

              inforomVersion = ''
              includeInforom = False
              if (NVSMI_ALL in filter or NVSMI_INFOROM_IMG in filter):
                  try:
                      img = nvmlDeviceGetInforomImageVersion(handle)
                  except NVMLError as err:
                      img = nvidia_smi.__handleError(err)

                  inforomVersion += '      <img_version>' + nvidia_smi.__toString(img) + '</img_version>\n'
                  includeInforom = True

              if (NVSMI_ALL in filter or NVSMI_INFOROM_OEM in filter):
                  try:
                      oem = nvmlDeviceGetInforomVersion(handle, NVML_INFOROM_OEM)
                  except NVMLError as err:
                      oem = nvidia_smi.__handleError(err)

                  inforomVersion += '      <oem_object>' + nvidia_smi.__toString(oem) + '</oem_object>\n'
                  includeInforom = True

              if (NVSMI_ALL in filter or NVSMI_INFOROM_ECC in filter):
                  try:
                      ecc = nvmlDeviceGetInforomVersion(handle, NVML_INFOROM_ECC)
                  except NVMLError as err:
                      ecc = nvidia_smi.__handleError(err)

                  inforomVersion += '      <ecc_object>' + nvidia_smi.__toString(ecc) + '</ecc_object>\n'
                  includeInforom = True

              if (NVSMI_ALL in filter or NVSMI_INFOROM_PWR in filter):
                  try:
                      pwr = nvmlDeviceGetInforomVersion(handle, NVML_INFOROM_POWER)
                  except NVMLError as err:
                      pwr = nvidia_smi.__handleError(err)

                  inforomVersion += '      <pwr_object>' + nvidia_smi.__toString(pwr) + '</pwr_object>\n'
                  includeInforom = True

              if includeInforom:
                  gpuInfo += '    <inforom_version>\n'
                  gpuInfo += inforomVersion
                  gpuInfo += '    </inforom_version>\n'

              gpuOperationMode = ''
              includeGOM = False
              if (NVSMI_ALL in filter or NVSMI_INFOROM_PWR in filter):
                  try:
                      current = nvidia_smi.__toStrGOM(nvmlDeviceGetCurrentGpuOperationMode(handle))
                  except NVMLError as err:
                      current = nvidia_smi.__handleError(err)
                  gpuOperationMode += '      <current_gom>' + nvidia_smi.__toString(current) + '</current_gom>\n'
                  includeGOM = True

              if (NVSMI_ALL in filter or NVSMI_INFOROM_PWR in filter):
                  try:
                      pending = nvidia_smi.__toStrGOM(nvmlDeviceGetPendingGpuOperationMode(handle))
                  except NVMLError as err:
                      pending = nvidia_smi.__handleError(err)

                  gpuOperationMode += '      <pending_gom>' + nvidia_smi.__toString(pending) + '</pending_gom>\n'
                  includeGOM = True

              if includeGOM:
                  gpuInfo += '    <gpu_operation_mode>\n'
                  gpuInfo += gpuOperationMode
                  gpuInfo += '    </gpu_operation_mode>\n'

              pci = ''
              includePci = False

              if (NVSMI_ALL in filter or NVSMI_PCI_BUS in filter):
                  pci += '      <pci_bus>%02X</pci_bus>\n' % pciInfo.bus
                  includePci = True

              if (NVSMI_ALL in filter or NVSMI_PCI_DEVICE in filter):
                  pci += '      <pci_device>%02X</pci_device>\n' % pciInfo.device
                  includePci = True

              if (NVSMI_ALL in filter or NVSMI_PCI_DOMAIN in filter):
                  pci += '      <pci_domain>%04X</pci_domain>\n' % pciInfo.domain
                  includePci = True

              if (NVSMI_ALL in filter or NVSMI_PCI_DEVICE_ID in filter):
                  pci += '      <pci_device_id>%08X</pci_device_id>\n' % (pciInfo.pciDeviceId)
                  includePci = True

              if (NVSMI_ALL in filter or NVSMI_PCI_BUS_ID in filter):
                  pci += '      <pci_bus_id>' + nvidia_smi.__toString(pciInfo.busId) + '</pci_bus_id>\n'
                  includePci = True

              if (NVSMI_ALL in filter or NVSMI_PCI_SUBDEVICE_ID in filter):
                  pci += '      <pci_sub_system_id>%08X</pci_sub_system_id>\n' % (pciInfo.pciSubSystemId)
                  includePci = True

              pciGpuLinkInfo = ''
              includeLinkInfo = False
              pciGen = ''
              includeGen = False

              if (NVSMI_ALL in filter or NVSMI_PCI_LINK_GEN_MAX in filter):
                  try:
                      gen = nvidia_smi.__toString(nvmlDeviceGetMaxPcieLinkGeneration(handle))
                  except NVMLError as err:
                      gen = nvidia_smi.__handleError(err)

                  pciGen += '          <max_link_gen>' + gen + '</max_link_gen>\n'
                  includeGen = True

              if (NVSMI_ALL in filter or NVSMI_PCI_LINK_GEN_CUR in filter):
                  try:
                      gen = nvidia_smi.__toString(nvmlDeviceGetCurrPcieLinkGeneration(handle))
                  except NVMLError as err:
                      gen = nvidia_smi.__handleError(err)

                  pciGen += '          <current_link_gen>' + gen + '</current_link_gen>\n'
                  includeGen = True

              if includeGen:
                  pciGpuLinkInfo += '        <pcie_gen>\n'
                  pciGpuLinkInfo += pciGen
                  pciGpuLinkInfo += '        </pcie_gen>\n'
                  includeLinkInfo = True

              pciLinkWidths = ''
              includeLinkWidths = False

              if (NVSMI_ALL in filter or NVSMI_PCI_LINK_WIDTH_MAX in filter):
                  try:
                      width = nvidia_smi.__toString(nvmlDeviceGetMaxPcieLinkWidth(handle)) + 'x'
                  except NVMLError as err:
                      width = nvidia_smi.__handleError(err)

                  pciLinkWidths += '          <max_link_width>' + width + '</max_link_width>\n'
                  includeLinkWidths = True

              if (NVSMI_ALL in filter or NVSMI_PCI_LINK_WIDTH_CUR in filter):
                  try:
                      width = nvidia_smi.__toString(nvmlDeviceGetCurrPcieLinkWidth(handle)) + 'x'
                  except NVMLError as err:
                      width = nvidia_smi.__handleError(err)

                  pciLinkWidths += '          <current_link_width>' + width + '</current_link_width>\n'
                  includeLinkWidths = True

              if includeLinkWidths:
                  pciGpuLinkInfo += '        <link_widths>\n'
                  pciGpuLinkInfo += pciLinkWidths
                  pciGpuLinkInfo += '        </link_widths>\n'
                  includeLinkInfo = True

              if includeLinkInfo:
                  pci += '      <pci_gpu_link_info>\n'
                  pci += pciGpuLinkInfo
                  pci += '      </pci_gpu_link_info>\n'


              pciBridgeChip = ''
              includeBridgeChip = False

              if (NVSMI_ALL in filter):
                  try:
                      bridgeHierarchy = nvmlDeviceGetBridgeChipInfo(handle)
                      bridge_type = ''
                      if bridgeHierarchy.bridgeChipInfo[0].type == 0:
                          bridge_type += 'PLX'
                      else:
                          bridge_type += 'BR04'
                      pciBridgeChip += '        <bridge_chip_type>' + bridge_type + '</bridge_chip_type>\n'

                      if bridgeHierarchy.bridgeChipInfo[0].fwVersion == 0:
                          strFwVersion = 'N/A'
                      else:
                          strFwVersion = '%08X' % (bridgeHierarchy.bridgeChipInfo[0].fwVersion)
                      pciBridgeChip += '        <bridge_chip_fw>%s</bridge_chip_fw>\n' % (strFwVersion)
                  except NVMLError as err:
                      pciBridgeChip += '        <bridge_chip_type>' + nvidia_smi.__handleError(err) + '</bridge_chip_type>\n'
                      pciBridgeChip += '        <bridge_chip_fw>' + nvidia_smi.__handleError(err) + '</bridge_chip_fw>\n'
                  includeBridgeChip = True

              if includeBridgeChip:
                  # Add additional code for hierarchy of bridges for Bug # 1382323
                  pci += '      <pci_bridge_chip>\n'
                  pci += pciBridgeChip
                  pci += '      </pci_bridge_chip>\n'

              if (NVSMI_ALL in filter):
                  try:
                      replay = nvmlDeviceGetPcieReplayCounter(handle)
                      pci += '      <replay_counter>' + nvidia_smi.__toString(replay) + '</replay_counter>'
                  except NVMLError as err:
                      pci += '      <replay_counter>' + nvidia_smi.__handleError(err) + '</replay_counter>'
                  includePci = True

              if (NVSMI_ALL in filter):
                  try:
                      tx_bytes = nvmlDeviceGetPcieThroughput(handle, NVML_PCIE_UTIL_TX_BYTES)
                      pci += '      <tx_util>' + nvidia_smi.__toString(tx_bytes) + ' KB/s' + '</tx_util>'
                  except NVMLError as err:
                      pci += '      <tx_util>' + nvidia_smi.__handleError(err) + '</tx_util>'
                  includePci = True

              if (NVSMI_ALL in filter):
                  try:
                      rx_bytes = nvmlDeviceGetPcieThroughput(handle, NVML_PCIE_UTIL_RX_BYTES)
                      pci += '      <rx_util>' + nvidia_smi.__toString(rx_bytes) + ' KB/s' + '</rx_util>'
                  except NVMLError as err:
                      pci += '      <rx_util>' + nvidia_smi.__handleError(err) + '</rx_util>'
                  includePci = True

              if includePci:
                  gpuInfo += '    <pci>\n'
                  gpuInfo += pci
                  gpuInfo += '    </pci>\n'

              if (NVSMI_ALL in filter or NVSMI_FAN_SPEED in filter):
                  try:
                      fan = nvidia_smi.__toString(nvmlDeviceGetFanSpeed(handle)) + ' %'
                  except NVMLError as err:
                      fan = nvidia_smi.__handleError(err)
                  gpuInfo += '    <fan_speed>' + fan + '</fan_speed>\n'

              if (NVSMI_ALL in filter or NVSMI_PSTATE in filter):
                  try:
                      perfState = nvmlDeviceGetPowerState(handle)
                      perfStateStr = 'P%s' % perfState
                  except NVMLError as err:
                      perfStateStr = nvidia_smi.__handleError(err)
                  gpuInfo += '    <performance_state>' + perfStateStr + '</performance_state>\n'

              if (NVSMI_ALL in filter or
                  NVSMI_CLOCK_THROTTLE_REASONS_SUPPORTED in filter or
                  NVSMI_CLOCK_THROTTLE_REASONS_ACTIVE in filter or
                  NVSMI_CLOCK_THROTTLE_REASONS_IDLE in filter or
                  NVSMI_CLOCK_THROTTLE_REASONS_APP_SETTING in filter or
                  NVSMI_CLOCK_THROTTLE_REASONS_SW_PWR_CAP in filter or
                  NVSMI_CLOCK_THROTTLE_REASONS_HW_SLOWDOWN in filter or
                  NVSMI_CLOCK_THROTTLE_REASONS_HW_THERMAL_SLOWDOWN in filter or
                  NVSMI_CLOCK_THROTTLE_REASONS_HW_PWR_BRAKE_SLOWDOWN in filter or
                  NVSMI_CLOCK_THROTTLE_REASONS_SW_THERMAL_SLOWDOWN in filter or
                  NVSMI_CLOCK_THROTTLE_REASONS_SYNC_BOOST in filter):
                  gpuInfo += nvidia_smi.__xmlGetClocksThrottleReasons(handle);

              fbMemoryUsage = ''
              includeMemoryUsage = False
              if (NVSMI_ALL in filter or
                  NVSMI_MEMORY_TOTAL in filter or
                  NVSMI_MEMORY_USED in filter or
                  NVSMI_MEMORY_FREE in filter):

                  includeMemoryUsage = True
                  try:
                      memInfo = nvmlDeviceGetMemoryInfo(handle)
                      mem_total = nvidia_smi.__toString(memInfo.total / 1024 / 1024) + ' MiB'
                      mem_used = nvidia_smi.__toString(memInfo.used / 1024 / 1024) + ' MiB'
                      mem_free = nvidia_smi.__toString(memInfo.total / 1024 / 1024 - memInfo.used / 1024 / 1024) + ' MiB'
                  except NVMLError as err:
                      error = nvidia_smi.__handleError(err)
                      mem_total = error
                      mem_used = error
                      mem_free = error

                  if (NVSMI_ALL in filter or NVSMI_MEMORY_TOTAL in filter):
                      fbMemoryUsage += '      <total>' + mem_total + '</total>\n'
                  if (NVSMI_ALL in filter or NVSMI_MEMORY_USED in filter):
                      fbMemoryUsage += '      <used>' + mem_used + '</used>\n'
                  if (NVSMI_ALL in filter or NVSMI_MEMORY_FREE in filter):
                      fbMemoryUsage += '      <free>' + mem_free + '</free>\n'

              if includeMemoryUsage:
                  gpuInfo += '    <fb_memory_usage>\n'
                  gpuInfo += fbMemoryUsage
                  gpuInfo += '    </fb_memory_usage>\n'

              if (NVSMI_ALL in filter or NVSMI_MEMORY_BAR1 in filter):
                  try:
                      memInfo = nvmlDeviceGetBAR1MemoryInfo(handle)
                      mem_total = nvidia_smi.__toString(memInfo.bar1Total / 1024 / 1024) + ' MiB'
                      mem_used = nvidia_smi.__toString(memInfo.bar1Used / 1024 / 1024) + ' MiB'
                      mem_free = nvidia_smi.__toString(memInfo.bar1Total / 1024 / 1024 - memInfo.bar1Used / 1024 / 1024) + ' MiB'
                  except NVMLError as err:
                      error = nvidia_smi.__handleError(err)
                      mem_total = error
                      mem_used = error
                      mem_free = error

                  gpuInfo += '    <bar1_memory_usage>\n'
                  gpuInfo += '      <total>' + mem_total + '</total>\n'
                  gpuInfo += '      <used>' + mem_used + '</used>\n'
                  gpuInfo += '      <free>' + mem_free + '</free>\n'
                  gpuInfo += '    </bar1_memory_usage>\n'

              if (NVSMI_ALL in filter or NVSMI_COMPUTE_MODE in filter):
                  try:
                      mode = nvmlDeviceGetComputeMode(handle)
                      if mode == NVML_COMPUTEMODE_DEFAULT:
                          modeStr = 'Default'
                      elif mode == NVML_COMPUTEMODE_EXCLUSIVE_THREAD:
                          modeStr = 'Exclusive Thread'
                      elif mode == NVML_COMPUTEMODE_PROHIBITED:
                          modeStr = 'Prohibited'
                      elif mode == NVML_COMPUTEMODE_EXCLUSIVE_PROCESS:
                          modeStr = 'Exclusive_Process'
                      else:
                          modeStr = 'Unknown'
                  except NVMLError as err:
                      modeStr = nvidia_smi.__handleError(err)

                  gpuInfo += '    <compute_mode>' + modeStr + '</compute_mode>\n'

              utilization = ''
              includeUtilization = False
              if (NVSMI_ALL in filter or
                  NVSMI_UTILIZATION_GPU in filter or
                  NVSMI_UTILIZATION_MEM in filter):
                  try:
                      util = nvmlDeviceGetUtilizationRates(handle)
                      gpu_util = nvidia_smi.__toString(util.gpu) + ' %'
                      mem_util = nvidia_smi.__toString(util.memory) + ' %'
                  except NVMLError as err:
                      error = nvidia_smi.__handleError(err)
                      gpu_util = error
                      mem_util = error

                  if (NVSMI_ALL in filter or NVSMI_UTILIZATION_GPU in filter):
                      utilization += '      <gpu_util>' + gpu_util + '</gpu_util>\n'

                  if (NVSMI_ALL in filter or NVSMI_UTILIZATION_MEM in filter):
                      utilization += '      <memory_util>' + mem_util + '</memory_util>\n'

                  includeUtilization = True

              if (NVSMI_ALL in filter or NVSMI_UTILIZATION_ENCODER in filter):
                  try:
                      (util_int, ssize) = nvmlDeviceGetEncoderUtilization(handle)
                      encoder_util = nvidia_smi.__toString(util_int) + ' %'
                  except NVMLError as err:
                      error = nvidia_smi.__handleError(err)
                      encoder_util = error

                  utilization += '      <encoder_util>' + encoder_util + '</encoder_util>\n'

                  includeUtilization = True

              if (NVSMI_ALL in filter or NVSMI_UTILIZATION_DECODER in filter):
                  try:
                      (util_int, ssize) = nvmlDeviceGetDecoderUtilization(handle)
                      decoder_util = nvidia_smi.__toString(util_int) + ' %'
                  except NVMLError as err:
                      error = nvidia_smi.__handleError(err)
                      decoder_util = error

                  utilization += '      <decoder_util>' + decoder_util + '</decoder_util>\n'

                  includeUtilization = True

              if includeUtilization:
                  gpuInfo += '    <utilization>\n'
                  gpuInfo += utilization
                  gpuInfo += '    </utilization>\n'

              if (NVSMI_ALL in filter or
                  NVSMI_ECC_MODE_CUR in filter or
                  NVSMI_ECC_MODE_PENDING in filter):
                  try:
                      (current, pending) = nvmlDeviceGetEccMode(handle)
                      curr_str = 'Enabled' if (current != 0) else 'Disabled'
                      pend_str = 'Enabled' if (pending != 0) else 'Disabled'
                  except NVMLError as err:
                      error = nvidia_smi.__handleError(err)
                      curr_str = error
                      pend_str = error

                  eccMode = ''
                  if (NVSMI_ALL in filter or NVSMI_ECC_MODE_CUR in filter):
                      eccMode += '      <current_ecc>' + curr_str + '</current_ecc>\n'
                  if (NVSMI_ALL in filter or NVSMI_ECC_MODE_PENDING in filter):
                      eccMode += '      <pending_ecc>' + pend_str + '</pending_ecc>\n'

                  gpuInfo += '    <ecc_mode>\n'
                  gpuInfo += eccMode
                  gpuInfo += '    </ecc_mode>\n'

              eccErrors,includeEccErrors = nvidia_smi.__xmlGetEcc(handle, filter)
              if includeEccErrors:
                  gpuInfo += '    <ecc_errors>\n'
                  gpuInfo += eccErrors
                  gpuInfo += '    </ecc_errors>\n'

              retiredPages, includeRetiredPages  = nvidia_smi.__xmlGetRetiredPages(handle, filter)
              if includeRetiredPages:
                  gpuInfo += '    <retired_pages>\n'
                  gpuInfo += retiredPages
                  gpuInfo += '    </retired_pages>\n'

              temperature = ''
              includeTemperature = False

              if (NVSMI_ALL in filter or NVSMI_TEMPERATURE_GPU in filter):
                  try:
                      temp = nvidia_smi.__toString(nvmlDeviceGetTemperature(handle, NVML_TEMPERATURE_GPU)) + ' C'
                  except NVMLError as err:
                      temp = nvidia_smi.__handleError(err)

                  temperature += '      <gpu_temp>' + temp + '</gpu_temp>\n'

                  try:
                      temp = nvidia_smi.__toString(nvmlDeviceGetTemperatureThreshold(handle, NVML_TEMPERATURE_THRESHOLD_SHUTDOWN)) + ' C'
                  except NVMLError as err:
                      temp = nvidia_smi.__handleError(err)

                  temperature += '      <gpu_temp_max_threshold>' + temp + '</gpu_temp_max_threshold>\n'
                  includeTemperature = True

                  try:
                      temp = nvidia_smi.__toString(nvmlDeviceGetTemperatureThreshold(handle, NVML_TEMPERATURE_THRESHOLD_SLOWDOWN)) + ' C'
                  except NVMLError as err:
                      temp = nvidia_smi.__handleError(err)

                  temperature += '      <gpu_temp_slow_threshold>' + temp + '</gpu_temp_slow_threshold>\n'
                  includeTemperature = True

              if includeTemperature:
                  gpuInfo += '    <temperature>\n'
                  gpuInfo + temperature
                  gpuInfo += '    </temperature>\n'

              power_readings = ''
              includePowerReadings = False

              if (NVSMI_ALL in filter or NVSMI_POWER_MGMT in filter):
                  try:
                      powMan = nvmlDeviceGetPowerManagementMode(handle)
                      powManStr = 'Supported' if powMan != 0 else 'N/A'
                  except NVMLError as err:
                      powManStr = nvidia_smi.__handleError(err)
                  power_readings += '      <power_management>' + powManStr + '</power_management>\n'
                  includePowerReadings = True

              if (NVSMI_ALL in filter or NVSMI_POWER_DRAW in filter):
                  try:
                      powDraw = (nvmlDeviceGetPowerUsage(handle) / 1000.0)
                      powDrawStr = '%.2f W' % powDraw
                  except NVMLError as err:
                      powDrawStr = nvidia_smi.__handleError(err)
                  power_readings += '      <power_draw>' + powDrawStr + '</power_draw>\n'
                  includePowerReadings = True

              if (NVSMI_ALL in filter or NVSMI_POWER_LIMIT in filter):
                  try:
                      powLimit = (nvmlDeviceGetPowerManagementLimit(handle) / 1000.0)
                      powLimitStr = '%.2f W' % powLimit
                  except NVMLError as err:
                      powLimitStr = nvidia_smi.__handleError(err)
                  power_readings += '      <power_limit>' + powLimitStr + '</power_limit>\n'
                  includePowerReadings = True

              if (NVSMI_ALL in filter or NVSMI_POWER_LIMIT_DEFAULT in filter):
                  try:
                      powLimit = (nvmlDeviceGetPowerManagementDefaultLimit(handle) / 1000.0)
                      powLimitStr = '%.2f W' % powLimit
                  except NVMLError as err:
                      powLimitStr = nvidia_smi.__handleError(err)
                  power_readings += '      <default_power_limit>' + powLimitStr + '</default_power_limit>\n'
                  includePowerReadings = True

              if (NVSMI_ALL in filter or NVSMI_POWER_LIMIT_ENFORCED in filter):
                  try:
                      enforcedPowLimit = (nvmlDeviceGetEnforcedPowerLimit(handle) / 1000.0)
                      enforcedPowLimitStr = '%.2f W' % enforcedPowLimit
                  except NVMLError as err:
                      enforcedPowLimitStr = nvidia_smi.__handleError(err)
                  power_readings += '      <enforced_power_limit>' + enforcedPowLimitStr + '</enforced_power_limit>\n'
                  includePowerReadings = True

              if (NVSMI_ALL in filter or
                  NVSMI_POWER_LIMIT_MIN in filter or
                  NVSMI_POWER_LIMIT_MAX in filter):
                  try:
                      powLimit = nvmlDeviceGetPowerManagementLimitConstraints(handle)
                      powLimitStrMin = '%.2f W' % (powLimit[0] / 1000.0)
                      powLimitStrMax = '%.2f W' % (powLimit[1] / 1000.0)
                  except NVMLError as err:
                      error = nvidia_smi.__handleError(err)
                      powLimitStrMin = error
                      powLimitStrMax = error
                  if (NVSMI_ALL in filter or NVSMI_POWER_LIMIT_MIN in filter):
                      power_readings += '      <min_power_limit>' + powLimitStrMin + '</min_power_limit>\n'
                  if (NVSMI_ALL in filter or NVSMI_POWER_LIMIT_MAX in filter):
                      power_readings += '      <max_power_limit>' + powLimitStrMax + '</max_power_limit>\n'
                  includePowerReadings = True

              if includePowerReadings:
                  gpuInfo += '    <power_readings>\n'
                  try:
                      perfState = 'P' + nvidia_smi.__toString(nvmlDeviceGetPowerState(handle))
                  except NVMLError as err:
                      perfState = nvidia_smi.__handleError(err)
                  gpuInfo += '      <power_state>%s</power_state>\n' % perfState
                  gpuInfo += power_readings
                  gpuInfo += '    </power_readings>\n'

              clocks = ''
              includeClocks = False
              if(NVSMI_ALL in filter or NVSMI_CLOCKS_GRAPHICS_CUR in filter):
                  try:
                      graphics = nvidia_smi.__toString(nvmlDeviceGetClockInfo(handle, NVML_CLOCK_GRAPHICS)) + ' MHz'
                  except NVMLError as err:
                      graphics = nvidia_smi.__handleError(err)
                  clocks += '      <graphics_clock>' +graphics + '</graphics_clock>\n'
                  includeClocks = True;

              if(NVSMI_ALL in filter or NVSMI_CLOCKS_GRAPHICS_CUR in filter):
                  try:
                      sm = nvidia_smi.__toString(nvmlDeviceGetClockInfo(handle, NVML_CLOCK_SM)) + ' MHz'
                  except NVMLError as err:
                      sm = nvidia_smi.__handleError(err)
                  clocks += '      <sm_clock>' + sm + '</sm_clock>\n'
                  includeClocks = True;

              if(NVSMI_ALL in filter or NVSMI_CLOCKS_MEMORY_CUR in filter):
                  try:
                      mem = nvidia_smi.__toString(nvmlDeviceGetClockInfo(handle, NVML_CLOCK_MEM)) + ' MHz'
                  except NVMLError as err:
                      mem = nvidia_smi.__handleError(err)
                  clocks += '      <mem_clock>' + mem + '</mem_clock>\n'
                  includeClocks = True;

              if includeClocks:
                  gpuInfo += '    <clocks>\n'
                  gpuInfo += clocks
                  gpuInfo += '    </clocks>\n'

              applicationClocks = ''
              includeAppClocks = False
              if(NVSMI_ALL in filter or NVSMI_CLOCKS_APPL_GRAPHICS in filter):
                  try:
                      graphics = nvidia_smi.__toString(nvmlDeviceGetApplicationsClock(handle, NVML_CLOCK_GRAPHICS)) + ' MHz'
                  except NVMLError as err:
                      graphics = nvidia_smi.__handleError(err)
                  applicationClocks += '      <graphics_clock>' +graphics + '</graphics_clock>\n'
                  includeAppClocks = True

              if(NVSMI_ALL in filter or NVSMI_CLOCKS_APPL_MEMORY in filter):
                  try:
                      mem = nvidia_smi.__toString(nvmlDeviceGetApplicationsClock(handle, NVML_CLOCK_MEM)) + ' MHz'
                  except NVMLError as err:
                      mem = nvidia_smi.__handleError(err)
                  applicationClocks += '      <mem_clock>' + mem + '</mem_clock>\n'
                  includeAppClocks = True

              if includeAppClocks:
                  gpuInfo += '    <applications_clocks>\n'
                  gpuInfo += applicationClocks
                  gpuInfo += '    </applications_clocks>\n'

              defaultApplicationClocks = ''
              includeDefaultAppClocks = False

              if(NVSMI_ALL in filter or NVSMI_CLOCKS_APPL_GRAPHICS_DEFAULT in filter):
                  try:
                      graphics = nvidia_smi.__toString(nvmlDeviceGetDefaultApplicationsClock(handle, NVML_CLOCK_GRAPHICS)) + ' MHz'
                  except NVMLError as err:
                      graphics = nvidia_smi.__handleError(err)
                  defaultApplicationClocks += '      <graphics_clock>' +graphics + '</graphics_clock>\n'
                  includeDefaultAppClocks = True

              if(NVSMI_ALL in filter or NVSMI_CLOCKS_APPL_MEMORY_DEFAULT in filter):
                  try:
                      mem = nvidia_smi.__toString(nvmlDeviceGetDefaultApplicationsClock(handle, NVML_CLOCK_MEM)) + ' MHz'
                  except NVMLError as err:
                      mem = nvidia_smi.__handleError(err)
                  defaultApplicationClocks += '      <mem_clock>' + mem + '</mem_clock>\n'
                  includeDefaultAppClocks = True

              if includeDefaultAppClocks:
                  gpuInfo += '    <default_applications_clocks>\n'
                  gpuInfo += defaultApplicationClocks
                  gpuInfo += '    </default_applications_clocks>\n'

              maxClocks = ''
              includeMaxClocks = False
              if(NVSMI_ALL in filter or NVSMI_CLOCKS_GRAPHICS_MAX in filter):
                  try:
                      graphics = nvidia_smi.__toString(nvmlDeviceGetMaxClockInfo(handle, NVML_CLOCK_GRAPHICS)) + ' MHz'
                  except NVMLError as err:
                      graphics = nvidia_smi.__handleError(err)
                  maxClocks += '      <graphics_clock>' + graphics + '</graphics_clock>\n'
                  includeMaxClocks = True

              if(NVSMI_ALL in filter or NVSMI_CLOCKS_SM_MAX in filter):
                  try:
                      sm = nvidia_smi.__toString(nvmlDeviceGetMaxClockInfo(handle, NVML_CLOCK_SM)) + ' MHz'
                  except NVMLError as err:
                      sm = nvidia_smi.__handleError(err)
                  maxClocks += '      <sm_clock>' + sm + '</sm_clock>\n'
                  includeMaxClocks = True

              if(NVSMI_ALL in filter or NVSMI_CLOCKS_MEMORY_MAX in filter):
                  try:
                      mem = nvidia_smi.__toString(nvmlDeviceGetMaxClockInfo(handle, NVML_CLOCK_MEM)) + ' MHz'
                  except NVMLError as err:
                      mem = nvidia_smi.__handleError(err)
                  maxClocks += '      <mem_clock>' + mem + '</mem_clock>\n'
                  includeMaxClocks = True

              if includeMaxClocks:
                  gpuInfo += '    <max_clocks>\n'
                  gpuInfo += maxClocks
                  gpuInfo += '    </max_clocks>\n'

              if (NVSMI_ALL in filter or NVSMI_CLOCKS_POLICY in filter):
                  gpuInfo += '    <clock_policy>\n'
                  try:
                      boostedState, boostedDefaultState = nvmlDeviceGetAutoBoostedClocksEnabled(handle)
                      if boostedState == NVML_FEATURE_DISABLED:
                          autoBoostStr = "Off"
                      else:
                          autoBoostStr = "On"

                      if boostedDefaultState == NVML_FEATURE_DISABLED:
                          autoBoostDefaultStr = "Off"
                      else:
                          autoBoostDefaultStr = "On"

                  except NVMLError_NotSupported:
                      autoBoostStr = "N/A"
                      autoBoostDefaultStr = "N/A"
                  except NVMLError as err:
                      autoBoostStr = nvidia_smi.__handleError(err)
                      autoBoostDefaultStr = nvidia_smi.__handleError(err)
                      pass
                  gpuInfo += '      <auto_boost>' + autoBoostStr + '</auto_boost>\n'
                  gpuInfo += '      <auto_boost_default>' + autoBoostDefaultStr + '</auto_boost_default>\n'
                  gpuInfo += '    </clock_policy>\n'

              if (NVSMI_ALL in filter or NVSMI_CLOCKS_SUPPORTED in filter):
                  try:
                      memClocks = nvmlDeviceGetSupportedMemoryClocks(handle)
                      gpuInfo += '    <supported_clocks>\n'

                      for m in memClocks:
                          gpuInfo += '      <supported_mem_clock>\n'
                          gpuInfo += '        <value>%d MHz</value>\n' % m
                          try:
                              clocks = nvmlDeviceGetSupportedGraphicsClocks(handle, m)
                              for c in clocks:
                                  gpuInfo += '        <supported_graphics_clock>%d MHz</supported_graphics_clock>\n' % c
                          except NVMLError as err:
                              gpuInfo += '        <supported_graphics_clock>%s</supported_graphics_clock>\n' % nvidia_smi.__handleError(err)
                          gpuInfo += '      </supported_mem_clock>\n'

                      gpuInfo += '    </supported_clocks>\n'
                  except NVMLError as err:
                      gpuInfo += '    <supported_clocks>' + nvidia_smi.__handleError(err) + '</supported_clocks>\n'

              if (NVSMI_ALL in filter or NVSMI_COMPUTE_APPS in filter):
                  try:
                      procs = nvmlDeviceGetComputeRunningProcesses(handle)
                      gpuInfo += '    <processes>\n'

                      for p in procs:
                          try:
                              name = nvidia_smi.__toString(nvmlSystemGetProcessName(p.pid))
                          except NVMLError as err:
                              if (err.value == NVML_ERROR_NOT_FOUND):
                                  # probably went away
                                  continue
                              else:
                                  name = nvidia_smi.__handleError(err)

                          gpuInfo += '    <process_info>\n'
                          gpuInfo += '      <pid>%d</pid>\n' % p.pid
                          gpuInfo += '      <process_name>' + name + '</process_name>\n'

                          if (p.usedGpuMemory == None):
                              mem = 'N/A'
                          else:
                              mem = '%d MiB' % (p.usedGpuMemory / 1024 / 1024)
                          gpuInfo += '      <used_memory>' + mem + '</used_memory>\n'
                          gpuInfo += '    </process_info>\n'

                      gpuInfo += '    </processes>\n'
                  except NVMLError as err:
                      gpuInfo += '    <processes>' + nvidia_smi.__handleError(err) + '</processes>\n'

              if (NVSMI_ALL in filter or NVSMI_ACCOUNTED_APPS in filter):
                  try:
                      pids = nvmlDeviceGetAccountingPids(handle)
                      gpuInfo += '    <accounted_processes>\n'

                      for pid in pids :
                          try:
                              stats = nvmlDeviceGetAccountingStats(handle, pid)
                              gpuUtilization = "%d %%" % stats.gpuUtilization
                              memoryUtilization = "%d %%" % stats.memoryUtilization
                              if (stats.maxMemoryUsage == None):
                                  maxMemoryUsage = 'N/A'
                              else:
                                  maxMemoryUsage = '%d MiB' % (stats.maxMemoryUsage / 1024 / 1024)
                              time = "%d ms" % stats.time
                              is_running = "%d" % stats.isRunning
                          except NVMLError as err:
                              if (err.value == NVML_ERROR_NOT_FOUND):
                                  # probably went away
                                  continue
                              err = nvidia_smi.__handleError(err)
                              gpuUtilization = err
                              memoryUtilization = err
                              maxMemoryUsage = err
                              time = err
                              is_running = err

                          gpuInfo += '    <accounted_process_info>\n'
                          gpuInfo += '      <pid>%d</pid>\n' % pid
                          gpuInfo += '      <gpu_util>' + gpuUtilization + '</gpu_util>\n'
                          gpuInfo += '      <memory_util>' + memoryUtilization + '</memory_util>\n'
                          gpuInfo += '      <max_memory_usage>' + maxMemoryUsage+ '</max_memory_usage>\n'
                          gpuInfo += '      <time>' + time + '</time>\n'
                          gpuInfo += '      <is_running>' + is_running + '</is_running>\n'
                          gpuInfo += '    </accounted_process_info>\n'

                      gpuInfo += '    </accounted_processes>\n'
                  except NVMLError as err:
                      gpuInfo += '    <accounted_processes>' + nvidia_smi.__handleError(err) + '</accounted_processes>\n'

              if len(gpuInfo) >0:
                  strResult += '  <gpu>'
                  strResult += gpuInfo
                  strResult += '  </gpu>\n'

          strResult += '</nvidia_smi>\n'

      except NVMLError as err:
          strResult += 'nvidia_smi.py: ' + err.__str__() + '\n'

      return strResult

  @classmethod
  def DeviceQuery(self, filter=None):
      '''
      Provides a Python interface to GPU management and monitoring functions.

      This is a wrapper around the NVML library.
      For information about the NVML library, see the NVML developer page
      http://developer.nvidia.com/nvidia-management-library-nvml

      Examples:
      ---------------------------------------------------------------------------
      For all elements as a list of dictionaries.  Similiar to nvisia-smi -q -x

      $ DeviceQuery()

      ---------------------------------------------------------------------------
      For a list of filtered dictionary elements by string name.
      Similiar ot nvidia-smi --query-gpu=pci.bus_id,memory.total,memory.free
      See help_query_gpu.txt or DeviceQuery("--help_query_gpu") for available filter elements

      $ DeviceQuery("pci.bus_id,memory.total,memory.free")

      ---------------------------------------------------------------------------
      For a list of filtered dictionary elements by enumeration value.
      See help_query_gpu.txt or DeviceQuery("--help_query_gpu") for available filter elements

      $ DeviceQuery([NVSMI_PCI_BUS_ID, NVSMI_MEMORY_TOTAL, NVSMI_MEMORY_FREE])

      '''

      if (filter is None):
          filter = [NVSMI_ALL]
      elif (isinstance(filter, str)):
          if (filter == "--help") or (filter == "-h"):
              return nvidia_smi.DeviceQuery.__doc__
          elif (filter == "--help-query-gpu"):
              with open("help_query_gpu.txt", 'r') as fin:
                  return fin.read()
          else:
              filter = nvidia_smi.__fromDeviceQueryString(filter)
      else:
          filter = list(filter)

      nvidia_smi_results = {}
      dictResult = []
      try:
          if (NVSMI_ALL in filter or NVSMI_TIMESTAMP in filter):
              nvidia_smi_results['timestamp']=nvidia_smi.__toString(datetime.date.today())
          if (NVSMI_ALL in filter or NVSMI_DRIVER_VERSION in filter):
              nvidia_smi_results['driver_version']=nvidia_smi.__toString(nvmlSystemGetDriverVersion())

          deviceCount = nvmlDeviceGetCount()
          if (NVSMI_ALL in filter or NVSMI_COUNT in filter):
              nvidia_smi_results['count']=deviceCount

          for i in range(0, deviceCount):
              gpuResults = {}
              handle = self.__handles[i]

              pciInfo = nvmlDeviceGetPciInfo(handle)

              if (NVSMI_ALL in filter or NVSMI_PCI_BUS_ID in filter):
                  gpuResults['id']=nvidia_smi.__toString(pciInfo.busId)
              if (NVSMI_ALL in filter or NVSMI_NAME in filter):
                  gpuResults['product_name']=nvidia_smi.__toString(nvmlDeviceGetName(handle))

                  try:
                      # if nvmlDeviceGetBrand() succeeds it is guaranteed to be in the dictionary
                      brandName = NVSMI_BRAND_NAMES[nvmlDeviceGetBrand(handle)]
                  except NVMLError as err:
                      brandName = nvidia_smi.__handleError(err)

                  gpuResults['product_brand']=brandName


              if (NVSMI_ALL in filter or NVSMI_DISPLAY_MODE in filter):
                  try:
                      state = ('Enabled' if (nvmlDeviceGetDisplayMode(handle) != 0) else 'Disabled')
                  except NVMLError as err:
                      state = nvidia_smi.__handleError(err)

                  gpuResults['display_mode']=state

              if (NVSMI_ALL in filter or NVSMI_DISPLAY_ACTIVE in filter):
                  try:
                      state = ('Enabled' if (nvmlDeviceGetDisplayActive(handle) != 0) else 'Disabled')
                  except NVMLError as err:
                      state = nvidia_smi.__handleError(err)

                  gpuResults['display_active']=state

              if (NVSMI_ALL in filter or NVSMI_PERSISTENCE_MODE in filter):
                  try:
                      mode = 'Enabled' if (nvmlDeviceGetPersistenceMode(handle) != 0) else 'Disabled'
                  except NVMLError as err:
                      mode = nvidia_smi.__handleError(err)

                  gpuResults['persistence_mode']=mode

              if (NVSMI_ALL in filter or NVSMI_ACCT_MODE in filter):
                  try:
                      mode = 'Enabled' if (nvmlDeviceGetAccountingMode(handle) != 0) else 'Disabled'
                  except NVMLError as err:
                      mode = nvidia_smi.__handleError(err)

                  gpuResults['accounting_mode']=mode

              if (NVSMI_ALL in filter or NVSMI_ACCT_BUFFER_SIZE in filter):
                  try:
                      bufferSize = nvidia_smi.__toString(nvmlDeviceGetAccountingBufferSize(handle))
                  except NVMLError as err:
                      bufferSize = nvidia_smi.__handleError(err)

                  gpuResults['accounting_mode_buffer_size']=bufferSize

              driverModel = {}
              includeDriverModel = False
              if (NVSMI_ALL in filter or NVSMI_DRIVER_MODEL_CUR in filter):
                  try:
                      current = 'WDDM' if (nvmlDeviceGetCurrentDriverModel(handle) == NVML_DRIVER_WDDM) else 'TCC'
                  except NVMLError as err:
                      current = nvidia_smi.__handleError(err)
                  driverModel['current_dm']=current
                  includeDriverModel = True

              if (NVSMI_ALL in filter or NVSMI_DRIVER_MODEL_PENDING in filter):
                  try:
                      pending = 'WDDM' if (nvmlDeviceGetPendingDriverModel(handle) == NVML_DRIVER_WDDM) else 'TCC'
                  except NVMLError as err:
                      pending = nvidia_smi.__handleError(err)

                  driverModel['pending_dm'] = pending
                  includeDriverModel = True

              if includeDriverModel:
                  gpuResults['driver_model'] = driverModel

              if (NVSMI_ALL in filter or NVSMI_SERIALNUMBER in filter):
                  try:
                      serial = nvmlDeviceGetSerial(handle)
                  except NVMLError as err:
                      serial = nvidia_smi.__handleError(err)

                  gpuResults['serial'] = nvidia_smi.__toString(serial)

              if (NVSMI_ALL in filter or NVSMI_UUID in filter):
                  try:
                      uuid = nvmlDeviceGetUUID(handle)
                  except NVMLError as err:
                      uuid = nvidia_smi.__handleError(err)

                  gpuResults['uuid'] = nvidia_smi.__toString(uuid)

              if (NVSMI_ALL in filter or NVSMI_INDEX in filter):
                  try:
                      minor_number = nvmlDeviceGetMinorNumber(handle)
                  except NVMLError as err:
                      minor_number = nvidia_smi.__handleError(err)

                  gpuResults['minor_number'] = nvidia_smi.__toString(minor_number)

              if (NVSMI_ALL in filter or NVSMI_VBIOS_VER in filter):
                  try:
                      vbios = nvmlDeviceGetVbiosVersion(handle)
                  except NVMLError as err:
                      vbios = nvidia_smi.__handleError(err)

                  gpuResults['vbios_version'] = nvidia_smi.__toString(vbios)

              if (NVSMI_ALL in filter or NVSMI_VBIOS_VER in filter):
                   try:
                        multiGpuBool = nvmlDeviceGetMultiGpuBoard(handle)
                   except NVMLError as err:
                        multiGpuBool = nvidia_smi.__handleError(err);

                   if multiGpuBool == "N/A":
                        gpuResults['multigpu_board'] = 'N/A'
                   elif multiGpuBool:
                        gpuResults['multigpu_board'] = 'Yes'
                   else:
                        gpuResults['multigpu_board'] = 'No'

              if (NVSMI_ALL in filter or NVSMI_BOARD_ID in filter):
                  try:
                      boardId = nvmlDeviceGetBoardId(handle)
                  except NVMLError as err:
                      boardId = nvidia_smi.__handleError(err)

                  try:
                      hexBID = "0x%x" % boardId
                  except:
                      hexBID = boardId

                  gpuResults['board_id'] = hexBID

              inforomVersion = {}
              includeInforom = False
              if (NVSMI_ALL in filter or NVSMI_INFOROM_IMG in filter):
                  try:
                      img = nvmlDeviceGetInforomImageVersion(handle)
                  except NVMLError as err:
                      img = nvidia_smi.__handleError(err)

                  inforomVersion['img_version'] = nvidia_smi.__toString(img)
                  includeInforom = True

              if (NVSMI_ALL in filter or NVSMI_INFOROM_OEM in filter):
                  try:
                      oem = nvmlDeviceGetInforomVersion(handle, NVML_INFOROM_OEM)
                  except NVMLError as err:
                      oem = nvidia_smi.__handleError(err)

                  inforomVersion['oem_object'] = nvidia_smi.__toString(oem)
                  includeInforom = True

              if (NVSMI_ALL in filter or NVSMI_INFOROM_ECC in filter):
                  try:
                      ecc = nvmlDeviceGetInforomVersion(handle, NVML_INFOROM_ECC)
                  except NVMLError as err:
                      ecc = nvidia_smi.__handleError(err)

                  inforomVersion['ecc_object'] = nvidia_smi.__toString(ecc)
                  includeInforom = True

              if (NVSMI_ALL in filter or NVSMI_INFOROM_PWR in filter):
                  try:
                      pwr = nvmlDeviceGetInforomVersion(handle, NVML_INFOROM_POWER)
                  except NVMLError as err:
                      pwr = nvidia_smi.__handleError(err)

                  inforomVersion['pwr_object'] = nvidia_smi.__toString(pwr)
                  includeInforom = True

              if includeInforom:
                  gpuResults['inforom_version'] = inforomVersion

              gpuOperationMode = {}
              includeGOM = False
              if (NVSMI_ALL in filter or NVSMI_INFOROM_PWR in filter):
                  try:
                      current = nvidia_smi.__toStrGOM(nvmlDeviceGetCurrentGpuOperationMode(handle))
                  except NVMLError as err:
                      current = nvidia_smi.__handleError(err)
                  gpuOperationMode['current_gom'] = nvidia_smi.__toString(current)
                  includeGOM = True

              if (NVSMI_ALL in filter or NVSMI_INFOROM_PWR in filter):
                  try:
                      pending = nvidia_smi.__toStrGOM(nvmlDeviceGetPendingGpuOperationMode(handle))
                  except NVMLError as err:
                      pending = nvidia_smi.__handleError(err)

                  gpuOperationMode['pending_gom'] = nvidia_smi.__toString(pending)
                  includeGOM = True

              if includeGOM:
                  gpuResults['gpu_operation_mode'] = gpuOperationMode

              pci = {}
              includePci = False

              if (NVSMI_ALL in filter or NVSMI_PCI_BUS in filter):
                  pci['pci_bus'] = '%02X' % pciInfo.bus
                  includePci = True

              if (NVSMI_ALL in filter or NVSMI_PCI_DEVICE in filter):
                  pci['pci_device'] = '%02X' % pciInfo.device
                  includePci = True

              if (NVSMI_ALL in filter or NVSMI_PCI_DOMAIN in filter):
                  pci['pci_domain'] = '%04X' % pciInfo.domain
                  includePci = True

              if (NVSMI_ALL in filter or NVSMI_PCI_DEVICE_ID in filter):
                  pci['pci_device_id'] = '%08X' % (pciInfo.pciDeviceId)
                  includePci = True

              if (NVSMI_ALL in filter or NVSMI_PCI_BUS_ID in filter):
                  pci['pci_bus_id'] = nvidia_smi.__toString(pciInfo.busId)
                  includePci = True

              if (NVSMI_ALL in filter or NVSMI_PCI_SUBDEVICE_ID in filter):
                  pci['pci_sub_system_id'] = '%08X' % (pciInfo.pciSubSystemId)
                  includePci = True

              pciGpuLinkInfo = {}
              includeLinkInfo = False
              pciGen = {}
              includeGen = False

              if (NVSMI_ALL in filter or NVSMI_PCI_LINK_GEN_MAX in filter):
                  try:
                      gen = nvidia_smi.__toString(nvmlDeviceGetMaxPcieLinkGeneration(handle))
                  except NVMLError as err:
                      gen = nvidia_smi.__handleError(err)

                  pciGen['max_link_gen'] = gen
                  includeGen = True

              if (NVSMI_ALL in filter or NVSMI_PCI_LINK_GEN_CUR in filter):
                  try:
                      gen = nvidia_smi.__toString(nvmlDeviceGetCurrPcieLinkGeneration(handle))
                  except NVMLError as err:
                      gen = nvidia_smi.__handleError(err)

                  pciGen['current_link_gen'] = gen
                  includeGen = True

              if includeGen:
                  pciGpuLinkInfo['pcie_gen'] = pciGen
                  includeLinkInfo = True

              pciLinkWidths = {}
              includeLinkWidths = False

              if (NVSMI_ALL in filter or NVSMI_PCI_LINK_WIDTH_MAX in filter):
                  try:
                      width = nvidia_smi.__toString(nvmlDeviceGetMaxPcieLinkWidth(handle)) + 'x'
                  except NVMLError as err:
                      width = nvidia_smi.__handleError(err)

                  pciLinkWidths['max_link_width'] = width
                  includeLinkWidths = True

              if (NVSMI_ALL in filter or NVSMI_PCI_LINK_WIDTH_CUR in filter):
                  try:
                      width = nvidia_smi.__toString(nvmlDeviceGetCurrPcieLinkWidth(handle)) + 'x'
                  except NVMLError as err:
                      width = nvidia_smi.__handleError(err)

                  pciLinkWidths['current_link_width'] = width
                  includeLinkWidths = True

              if includeLinkWidths:
                  pciGpuLinkInfo['link_widths'] = pciLinkWidths
                  includeLinkInfo = True

              if includeLinkInfo:
                  pci['pci_gpu_link_info'] = pciGpuLinkInfo
                  includePci = True

              pciBridgeChip = {}
              includeBridgeChip = False

              if (NVSMI_ALL in filter):
                  try:
                      bridgeHierarchy = nvmlDeviceGetBridgeChipInfo(handle)
                      bridge_type = ''
                      if bridgeHierarchy.bridgeChipInfo[0].type == 0:
                          bridge_type += 'PLX'
                      else:
                          bridge_type += 'BR04'
                      pciBridgeChip['bridge_chip_type'] = bridge_type

                      if bridgeHierarchy.bridgeChipInfo[0].fwVersion == 0:
                          strFwVersion = 'N/A'
                      else:
                          strFwVersion = '%08X' % (bridgeHierarchy.bridgeChipInfo[0].fwVersion)
                      pciBridgeChip['bridge_chip_fw'] = nvidia_smi.__toString(strFwVersion)
                  except NVMLError as err:
                      pciBridgeChip['bridge_chip_type'] = nvidia_smi.__handleError(err)
                      pciBridgeChip['bridge_chip_fw'] = nvidia_smi.__handleError(err)

                  includeBridgeChip = True

              if includeBridgeChip:
                  pci['pci_bridge_chip'] = pciBridgeChip
                  includePci = True

              if (NVSMI_ALL in filter):
                  try:
                      replay = nvmlDeviceGetPcieReplayCounter(handle)
                      pci['replay_counter'] = nvidia_smi.__toString(replay)
                  except NVMLError as err:
                      pci['replay_counter'] = nvidia_smi.__handleError(err)
                  includePci = True

              if (NVSMI_ALL in filter):
                  try:
                      tx_bytes = nvmlDeviceGetPcieThroughput(handle, NVML_PCIE_UTIL_TX_BYTES)
                      pci['tx_util'] = tx_bytes
                      pci['tx_util_unit'] = 'KB/s'
                  except NVMLError as err:
                      pci['tx_util'] = nvidia_smi.__handleError(err)
                  includePci = True

              if (NVSMI_ALL in filter):
                  try:
                      rx_bytes = nvmlDeviceGetPcieThroughput(handle, NVML_PCIE_UTIL_RX_BYTES)
                      pci['rx_util'] = rx_bytes
                      pci['rx_util_unit'] = 'KB/s'
                  except NVMLError as err:
                      pci['rx_util'] = nvidia_smi.__handleError(err)
                  includePci = True

              if includePci:
                  gpuResults['pci']= pci

              if (NVSMI_ALL in filter or NVSMI_FAN_SPEED in filter):
                  try:
                      fan = nvmlDeviceGetFanSpeed(handle)
                  except NVMLError as err:
                      fan = nvidia_smi.__handleError(err)
                  gpuResults['fan_speed']= fan
                  gpuResults['fan_speed_unit']= '%'

              if (NVSMI_ALL in filter or NVSMI_PSTATE in filter):
                  try:
                      perfState = nvmlDeviceGetPowerState(handle)
                      perfStateStr = 'P%s' % perfState
                  except NVMLError as err:
                      perfStateStr = nvidia_smi.__handleError(err)
                  gpuResults['performance_state']= perfStateStr


              if (NVSMI_ALL in filter or
                  NVSMI_CLOCK_THROTTLE_REASONS_SUPPORTED in filter or
                  NVSMI_CLOCK_THROTTLE_REASONS_ACTIVE in filter or
                  NVSMI_CLOCK_THROTTLE_REASONS_IDLE in filter or
                  NVSMI_CLOCK_THROTTLE_REASONS_APP_SETTING in filter or
                  NVSMI_CLOCK_THROTTLE_REASONS_SW_PWR_CAP in filter or
                  NVSMI_CLOCK_THROTTLE_REASONS_HW_SLOWDOWN in filter or
                  NVSMI_CLOCK_THROTTLE_REASONS_HW_THERMAL_SLOWDOWN in filter or
                  NVSMI_CLOCK_THROTTLE_REASONS_HW_PWR_BRAKE_SLOWDOWN in filter or
                  NVSMI_CLOCK_THROTTLE_REASONS_SW_THERMAL_SLOWDOWN in filter or
                  NVSMI_CLOCK_THROTTLE_REASONS_SYNC_BOOST in filter):
                  gpuResults['clocks_throttle']= nvidia_smi.__GetClocksThrottleReasons(handle)

              fbMemoryUsage = {}
              includeMemoryUsage = False
              if (NVSMI_ALL in filter or
                  NVSMI_MEMORY_TOTAL in filter or
                  NVSMI_MEMORY_USED in filter or
                  NVSMI_MEMORY_FREE in filter):

                  includeMemoryUsage = True
                  try:
                      memInfo = nvmlDeviceGetMemoryInfo(handle)
                      mem_total = memInfo.total / 1024 / 1024
                      mem_used = memInfo.used / 1024 / 1024
                      mem_free = memInfo.total / 1024 / 1024 - memInfo.used / 1024 / 1024
                  except NVMLError as err:
                      error = nvidia_smi.__handleError(err)
                      mem_total = error
                      mem_used = error
                      mem_free = error

                  if (NVSMI_ALL in filter or NVSMI_MEMORY_TOTAL in filter):
                      fbMemoryUsage['total']=mem_total

                  if (NVSMI_ALL in filter or NVSMI_MEMORY_USED in filter):
                      fbMemoryUsage['used']=mem_used

                  if (NVSMI_ALL in filter or NVSMI_MEMORY_FREE in filter):
                      fbMemoryUsage['free']=mem_free

              if includeMemoryUsage:
                  fbMemoryUsage['unit']='MiB'
                  gpuResults['fb_memory_usage'] = fbMemoryUsage

              if (NVSMI_ALL in filter or NVSMI_MEMORY_BAR1 in filter):
                  try:
                      memInfo = nvmlDeviceGetBAR1MemoryInfo(handle)
                      mem_total = memInfo.bar1Total / 1024 / 1024
                      mem_used = memInfo.bar1Used / 1024 / 1024
                      mem_free = memInfo.bar1Total / 1024 / 1024 - memInfo.bar1Used / 1024 / 1024
                  except NVMLError as err:
                      error = nvidia_smi.__handleError(err)
                      mem_total = error
                      mem_used = error
                      mem_free = error

                  bar1MemoryUsage = {}
                  bar1MemoryUsage['total']=mem_total
                  bar1MemoryUsage['used']=mem_used
                  bar1MemoryUsage['free']=mem_free
                  bar1MemoryUsage['unit']='MiB'
                  gpuResults['bar1_memory_usage'] = bar1MemoryUsage

              if (NVSMI_ALL in filter or NVSMI_COMPUTE_MODE in filter):
                  try:
                      mode = nvmlDeviceGetComputeMode(handle)
                      if mode == NVML_COMPUTEMODE_DEFAULT:
                          modeStr = 'Default'
                      elif mode == NVML_COMPUTEMODE_EXCLUSIVE_THREAD:
                          modeStr = 'Exclusive Thread'
                      elif mode == NVML_COMPUTEMODE_PROHIBITED:
                          modeStr = 'Prohibited'
                      elif mode == NVML_COMPUTEMODE_EXCLUSIVE_PROCESS:
                          modeStr = 'Exclusive_Process'
                      else:
                          modeStr = 'Unknown'
                  except NVMLError as err:
                      modeStr = nvidia_smi.__handleError(err)

                  gpuResults['compute_mode'] = modeStr

              utilization = {}
              includeUtilization = False
              if (NVSMI_ALL in filter or
                  NVSMI_UTILIZATION_GPU in filter or
                  NVSMI_UTILIZATION_MEM in filter):

                  try:
                      util = nvmlDeviceGetUtilizationRates(handle)
                      gpu_util = util.gpu
                      mem_util = util.memory
                  except NVMLError as err:
                      error = nvidia_smi.__handleError(err)
                      gpu_util = error
                      mem_util = error

                  if (NVSMI_ALL in filter or NVSMI_UTILIZATION_GPU in filter):
                      utilization['gpu_util'] = gpu_util

                  if (NVSMI_ALL in filter or NVSMI_UTILIZATION_MEM in filter):
                      utilization['memory_util'] = mem_util

                  includeUtilization = True

              if (NVSMI_ALL in filter or NVSMI_UTILIZATION_ENCODER in filter):
                  try:
                      (util_int, ssize) = nvmlDeviceGetEncoderUtilization(handle)
                      encoder_util = util_int
                  except NVMLError as err:
                      error = nvidia_smi.__handleError(err)
                      encoder_util = error

                  utilization['encoder_util'] = encoder_util
                  includeUtilization = True

              if (NVSMI_ALL in filter or NVSMI_UTILIZATION_DECODER in filter):
                  try:
                      (util_int, ssize) = nvmlDeviceGetDecoderUtilization(handle)
                      decoder_util = util_int
                  except NVMLError as err:
                      error = nvidia_smi.__handleError(err)
                      decoder_util = error

                  utilization['decoder_util'] = decoder_util
                  includeUtilization = True

              if includeUtilization:
                  utilization['unit'] = '%'
                  gpuResults['utilization'] = utilization

              if (NVSMI_ALL in filter or
                  NVSMI_ECC_MODE_CUR in filter or
                  NVSMI_ECC_MODE_PENDING in filter):
                  try:
                      (current, pending) = nvmlDeviceGetEccMode(handle)
                      curr_str = 'Enabled' if (current != 0) else 'Disabled'
                      pend_str = 'Enabled' if (pending != 0) else 'Disabled'
                  except NVMLError as err:
                      error = nvidia_smi.__handleError(err)
                      curr_str = error
                      pend_str = error

                  eccMode = {}
                  if (NVSMI_ALL in filter or NVSMI_ECC_MODE_CUR in filter):
                      eccMode['current_ecc'] = curr_str

                  if (NVSMI_ALL in filter or NVSMI_ECC_MODE_PENDING in filter):
                      eccMode['pending_ecc'] = pend_str

                  gpuResults['ecc_mode'] = eccMode


              eccErrors,includeEccErrors =   nvidia_smi.__GetEcc(handle, filter)
              if includeEccErrors:
                  gpuResults['ecc_errors'] = eccErrors

              retiredPages, includeRetiredPages = nvidia_smi.__GetRetiredPages(handle, filter)
              if includeRetiredPages:
                  gpuResults['retired_pages'] = retiredPages

              temperature = {}
              includeTemperature = False

              if (NVSMI_ALL in filter or NVSMI_TEMPERATURE_GPU in filter):
                  try:
                      temp = nvmlDeviceGetTemperature(handle, NVML_TEMPERATURE_GPU)
                  except NVMLError as err:
                      temp = nvidia_smi.__handleError(err)

                  temperature['gpu_temp'] = temp
                  includeTemperature = True

                  try:
                      temp = nvmlDeviceGetTemperatureThreshold(handle, NVML_TEMPERATURE_THRESHOLD_SHUTDOWN)
                  except NVMLError as err:
                      temp = nvidia_smi.__handleError(err)

                  temperature['gpu_temp_max_threshold'] = temp
                  includeTemperature = True

                  try:
                      temp = nvmlDeviceGetTemperatureThreshold(handle, NVML_TEMPERATURE_THRESHOLD_SLOWDOWN)
                  except NVMLError as err:
                      temp = nvidia_smi.__handleError(err)

                  temperature['gpu_temp_slow_threshold'] = temp
                  includeTemperature = True

              if includeTemperature:
                  temperature['unit'] = 'C'
                  gpuResults['temperature'] = temperature

              power_readings = {}
              includePowerReadings = False
              if (NVSMI_ALL in filter or NVSMI_POWER_MGMT in filter):
                  try:
                      powMan = nvmlDeviceGetPowerManagementMode(handle)
                      powManStr = 'Supported' if powMan != 0 else 'N/A'
                  except NVMLError as err:
                      powManStr = nvidia_smi.__handleError(err)
                  power_readings['power_management'] = powManStr
                  includePowerReadings = True

              if (NVSMI_ALL in filter or NVSMI_POWER_DRAW in filter):
                  try:
                      powDraw = nvmlDeviceGetPowerUsage(handle) / 1000.0
                      powDrawStr = powDraw
                  except NVMLError as err:
                      powDrawStr = nvidia_smi.__handleError(err)
                  power_readings['power_draw'] = powDrawStr
                  includePowerReadings = True

              if (NVSMI_ALL in filter or NVSMI_POWER_LIMIT in filter):
                  try:
                      powLimit = (nvmlDeviceGetPowerManagementLimit(handle) / 1000.0)
                      powLimitStr = powLimit
                  except NVMLError as err:
                      powLimitStr = nvidia_smi.__handleError(err)
                  power_readings['power_limit'] = powLimitStr
                  includePowerReadings = True

              if (NVSMI_ALL in filter or NVSMI_POWER_LIMIT_DEFAULT in filter):
                  try:
                      powLimit = (nvmlDeviceGetPowerManagementDefaultLimit(handle) / 1000.0)
                      powLimitStr = powLimit
                  except NVMLError as err:
                      powLimitStr = nvidia_smi.__handleError(err)
                  power_readings['default_power_limit'] = powLimitStr
                  includePowerReadings = True

              if (NVSMI_ALL in filter or NVSMI_POWER_LIMIT_ENFORCED in filter):
                  try:
                      enforcedPowLimit = (nvmlDeviceGetEnforcedPowerLimit(handle) / 1000.0)
                      enforcedPowLimitStr = enforcedPowLimit
                  except NVMLError as err:
                      enforcedPowLimitStr = nvidia_smi.__handleError(err)

                  power_readings['enforced_power_limit'] = enforcedPowLimitStr
                  includePowerReadings = True

              if (NVSMI_ALL in filter or
                  NVSMI_POWER_LIMIT_MIN in filter or
                  NVSMI_POWER_LIMIT_MAX in filter):
                  try:
                      powLimit = nvmlDeviceGetPowerManagementLimitConstraints(handle)
                      powLimitStrMin = powLimit[0] / 1000.0
                      powLimitStrMax = powLimit[1] / 1000.0
                  except NVMLError as err:
                      error = nvidia_smi.__handleError(err)
                      powLimitStrMin = error
                      powLimitStrMax = error

                  if (NVSMI_ALL in filter or NVSMI_POWER_LIMIT_MIN in filter):
                      power_readings['min_power_limit'] = powLimitStrMin
                  if (NVSMI_ALL in filter or NVSMI_POWER_LIMIT_MAX in filter):
                      power_readings['max_power_limit'] = powLimitStrMax
                  includePowerReadings = True

              if includePowerReadings:
                  try:
                      perfState = 'P' + nvidia_smi.__toString(nvmlDeviceGetPowerState(handle))
                  except NVMLError as err:
                      perfState = nvidia_smi.__handleError(err)
                  power_readings['power_state'] = perfState

                  power_readings['unit'] = 'W'
                  gpuResults['power_readings'] = power_readings

              clocks = {}
              includeClocks = False
              if(NVSMI_ALL in filter or NVSMI_CLOCKS_GRAPHICS_CUR in filter):
                  try:
                      graphics = nvmlDeviceGetClockInfo(handle, NVML_CLOCK_GRAPHICS)
                  except NVMLError as err:
                      graphics = nvidia_smi.__handleError(err)
                  clocks['graphics_clock'] = graphics
                  includeClocks = True;

              if(NVSMI_ALL in filter or NVSMI_CLOCKS_GRAPHICS_CUR in filter):
                  try:
                      sm = nvmlDeviceGetClockInfo(handle, NVML_CLOCK_SM)
                  except NVMLError as err:
                      sm = nvidia_smi.__handleError(err)
                  clocks['sm_clock'] = sm
                  includeClocks = True;

              if(NVSMI_ALL in filter or NVSMI_CLOCKS_MEMORY_CUR in filter):
                  try:
                      mem = nvmlDeviceGetClockInfo(handle, NVML_CLOCK_MEM)
                  except NVMLError as err:
                      mem = nvidia_smi.__handleError(err)
                  clocks['mem_clock'] = mem
                  includeClocks = True;

              if includeClocks:
                  clocks['unit'] = 'MHz'
                  gpuResults['clocks'] = clocks

              applicationClocks = {}
              includeAppClocks = False
              if(NVSMI_ALL in filter or NVSMI_CLOCKS_APPL_GRAPHICS in filter):
                  try:
                      graphics = nvmlDeviceGetApplicationsClock(handle, NVML_CLOCK_GRAPHICS)
                  except NVMLError as err:
                      graphics = nvidia_smi.__handleError(err)
                  applicationClocks['graphics_clock'] = graphics
                  includeAppClocks = True

              if(NVSMI_ALL in filter or NVSMI_CLOCKS_APPL_MEMORY in filter):
                  try:
                      mem = nvmlDeviceGetApplicationsClock(handle, NVML_CLOCK_MEM)
                  except NVMLError as err:
                      mem = nvidia_smi.__handleError(err)
                  applicationClocks['mem_clock'] = mem
                  includeAppClocks = True

              if includeAppClocks:
                  applicationClocks['unit'] = 'MHz'
                  gpuResults['applications_clocks'] = applicationClocks

              defaultApplicationClocks = {}
              includeDefaultAppClocks = False

              if(NVSMI_ALL in filter or NVSMI_CLOCKS_APPL_GRAPHICS_DEFAULT in filter):
                  try:
                      graphics = nvmlDeviceGetDefaultApplicationsClock(handle, NVML_CLOCK_GRAPHICS)
                  except NVMLError as err:
                      graphics = nvidia_smi.__handleError(err)
                  defaultApplicationClocks['graphics_clock'] = graphics
                  includeDefaultAppClocks = True

              if(NVSMI_ALL in filter or NVSMI_CLOCKS_APPL_MEMORY_DEFAULT in filter):
                  try:
                      mem = nvmlDeviceGetDefaultApplicationsClock(handle, NVML_CLOCK_MEM)
                  except NVMLError as err:
                      mem = nvidia_smi.__handleError(err)
                  defaultApplicationClocks['mem_clock'] = mem
                  includeDefaultAppClocks = True

              if includeDefaultAppClocks:
                  defaultApplicationClocks['unit'] = 'MHz'
                  gpuResults['default_applications_clocks'] = defaultApplicationClocks

              maxClocks = {}
              includeMaxClocks = False
              if(NVSMI_ALL in filter or NVSMI_CLOCKS_GRAPHICS_MAX in filter):
                  try:
                      graphics = nvmlDeviceGetMaxClockInfo(handle, NVML_CLOCK_GRAPHICS)
                  except NVMLError as err:
                      graphics = nvidia_smi.__handleError(err)
                  maxClocks['graphics_clock'] = graphics
                  includeMaxClocks = True

              if(NVSMI_ALL in filter or NVSMI_CLOCKS_SM_MAX in filter):
                  try:
                      sm = nvmlDeviceGetMaxClockInfo(handle, NVML_CLOCK_SM)
                  except NVMLError as err:
                      sm = nvidia_smi.__handleError(err)
                  maxClocks['sm_clock'] = sm
                  includeMaxClocks = True

              if(NVSMI_ALL in filter or NVSMI_CLOCKS_MEMORY_MAX in filter):
                  try:
                      mem = nvmlDeviceGetMaxClockInfo(handle, NVML_CLOCK_MEM)
                  except NVMLError as err:
                      mem = nvidia_smi.__handleError(err)
                  maxClocks['mem_clock'] = mem
                  includeMaxClocks = True

              if includeMaxClocks:
                  maxClocks['unit'] = 'MHz'
                  gpuResults['max_clocks'] = maxClocks

              if (NVSMI_ALL in filter or NVSMI_CLOCKS_POLICY in filter):
                  clockPolicy = {}
                  try:
                      boostedState, boostedDefaultState = nvmlDeviceGetAutoBoostedClocksEnabled(handle)
                      if boostedState == NVML_FEATURE_DISABLED:
                          autoBoostStr = "Off"
                      else:
                          autoBoostStr = "On"

                      if boostedDefaultState == NVML_FEATURE_DISABLED:
                          autoBoostDefaultStr = "Off"
                      else:
                          autoBoostDefaultStr = "On"

                  except NVMLError_NotSupported:
                      autoBoostStr = "N/A"
                      autoBoostDefaultStr = "N/A"
                  except NVMLError as err:
                      autoBoostStr = nvidia_smi.__handleError(err)
                      autoBoostDefaultStr = nvidia_smi.__handleError(err)

                  clockPolicy['auto_boost'] = autoBoostStr
                  clockPolicy['auto_boost_default'] = autoBoostDefaultStr
                  gpuResults['clock_policy'] = clockPolicy

              if (NVSMI_ALL in filter or NVSMI_CLOCKS_SUPPORTED in filter):
                  supportedClocks = []
                  try:
                      memClocks = nvmlDeviceGetSupportedMemoryClocks(handle)
 #                     jj = 1
                      for m in memClocks:
                          supportMemClock = {}
                          supportMemClock['current'] = m
                          supportMemClock['unit'] = 'MHz'

                          supportedGraphicsClocks = []
                          try:
                              clocks = nvmlDeviceGetSupportedGraphicsClocks(handle, m)
                              for c in clocks:
                                  supportedGraphicsClocks.append(c)
                          except NVMLError as err:
                              supportedGraphicsClocks = nvidia_smi.__handleError(err)

                          supportMemClock['supported_graphics_clock'] = supportedGraphicsClocks

                          supportedClocks.append( supportMemClock )
 #                         jj+=1

                  except NVMLError as err:
                      supportedClocks['Error'] = nvidia_smi.__handleError(err)

                  gpuResults['supported_clocks'] = supportedClocks if len(supportedClocks) > 0 else None

              if (NVSMI_ALL in filter or NVSMI_COMPUTE_APPS in filter):
                  processes = []
                  try:
                      procs = nvmlDeviceGetComputeRunningProcesses(handle)

 #                     ii = 1
                      for p in procs:
                          try:
                              name = nvidia_smi.__toString(nvmlSystemGetProcessName(p.pid))
                          except NVMLError as err:
                              if (err.value == NVML_ERROR_NOT_FOUND):
                                  # probably went away
                                  continue
                              else:
                                  name = nvidia_smi.__handleError(err)
                          processInfo = {}
                          processInfo['pid'] = p.pid
                          processInfo['process_name'] = name

                          if (p.usedGpuMemory == None):
                              mem = 0
                          else:
                              mem = int(p.usedGpuMemory / 1024 / 1024)
                          processInfo['used_memory'] = mem
                          processInfo['unit'] = 'MiB'
                          processes.append( processInfo )
#                          ii+=1

                  except NVMLError as err:
                      processes = nvidia_smi.__handleError(err)

                  gpuResults['processes'] = processes if len(processes) > 0 else None

              if (NVSMI_ALL in filter or NVSMI_ACCOUNTED_APPS in filter):
                  try:
                      pids = nvmlDeviceGetAccountingPids(handle)

                      accountProcess = []
#                      ii = 1
                      for pid in pids :
                          try:
                              stats = nvmlDeviceGetAccountingStats(handle, pid)
                              gpuUtilization = "%d %%" % stats.gpuUtilization
                              memoryUtilization = "%d %%" % stats.memoryUtilization
                              if (stats.maxMemoryUsage == None):
                                  maxMemoryUsage = 'N/A'
                              else:
                                  maxMemoryUsage = '%d MiB' % (stats.maxMemoryUsage / 1024 / 1024)
                              time = "%d ms" % stats.time
                              is_running = "%d" % stats.isRunning
                          except NVMLError as err:
                              if (err.value == NVML_ERROR_NOT_FOUND):
                                  # probably went away
                                  continue
                              err = nvidia_smi.__handleError(err)
                              gpuUtilization = err
                              memoryUtilization = err
                              maxMemoryUsage = err
                              time = err
                              is_running = err

                          accountProcessInfo = {}
                          accountProcessInfo['pid'] = '%d' % pid
                          accountProcessInfo['gpu_util'] = gpuUtilization
                          accountProcessInfo['memory_util'] = memoryUtilization
                          accountProcessInfo['max_memory_usage'] = maxMemoryUsage
                          accountProcessInfo['time'] = time
                          accountProcessInfo['is_running'] = is_running

                          accountProcess.append( accountProcessInfo )

                      gpuResults['accounted_processes'] = accountProcess if len(accountProcess) > 0 else None
#                          ii+=1
                  except NVMLError as err:
                      gpuResults['accounted_processes'] = nvidia_smi.__handleError(err)


              if (len(gpuResults) > 0):
                  dictResult.append(gpuResults)

          if (len(dictResult) > 0):
              nvidia_smi_results['gpu']=dictResult

      except NVMLError as err:
          print( 'nvidia_smi.py: ' + err.__str__() + '\n' )

      return nvidia_smi_results

  def __to_str_dictionary(self, value, indent):
      strResults = ''
      try:
          for key,val in value.items():
              if isinstance(val, collections.Mapping):
                  if len(val.values()) > 0:
                      strResults += ("%s%s:\n")%(indent,key)
                      strResults += self.__to_str_dictionary(val, '  '+indent)
                  else:
                      strResults += ("%s%s: %s\n")%(indent,key,"None")
              elif (type(val) is list) and (isinstance(val[0], collections.Mapping)):
                  for i in range(0,len(val)):
                      strResults += ("%s%s: [%d of %d]\n")%(indent,key, i+1,len(val))
                      strResults += self.__to_str_dictionary(val[i],'  '+indent)
              else:
                strResults += ("%s%s: %s\n")%(indent,key, str(val))

      except Exception as e:
          strResults += "\n[Error] " + str(e)

      return strResults

  def __to_str(self, results):
      strResults = ''
      indent = '  '
      for key,val in results.items():
          if type(val) is list:
              for i in range(0,len(val)):
                  strResults += ("%s%s: [%d of %d]\n")%(indent,key, i+1,len(val))
                  strResults += self.__to_str_dictionary(val[i],'  '+indent)
          else:
              strResults += ("%s%s: %s\n")%(indent,key, str(val))

      return strResults

  def format(self, results):

      if type(results) is str:
          return results

      return self.__to_str(results)

# this is not exectued when module is imported
if __name__ == "__main__":
    def main():
        #evaluate arguments
        as_xml = False
        query_gpu_args = None
        for i in range(1,len(sys.argv)):
            v = sys.argv[i]
            if (v.lower() == "xml"):
                as_xml = True
            else :
                query_gpu_args = v.lower()

        #execute device query
        nvsmi = nvidia_smi.getInstance()
        if (as_xml):
            results = nvsmi.XmlDeviceQuery(query_gpu_args)
        else:
            results = nvsmi.DeviceQuery(query_gpu_args)

        print(nvsmi.format(results))

    main()

