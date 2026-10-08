// Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
// All rights reserved.
//
// SPDX-License-Identifier: BSD-3-Clause

// Diagnostic only (do not merge). Minimal repro: does creating a Vulkan device make cuKernelSetAttribute reject
// MAX_DYNAMIC_SHARED_SIZE_BYTES = opt_in - static (the value NCCL requests)?
// No NCCL, no RTX/OVRTX, no PyTorch. Driver API + bare Vulkan only.
//
// usage: vk_smem_probe <vk:none|before|after> <setter:kernel|func> <delta> <image:nvrtc|ptx>
#include <cuda.h>
#include <dlfcn.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <signal.h>

// ---- Vulkan via official Khronos headers (v1.3.275, matching the system loader) ----
#include <vulkan/vulkan_core.h>

static int g_val_errors = 0, g_val_warnings = 0;
static VKAPI_ATTR VkBool32 VKAPI_CALL on_message(VkDebugUtilsMessageSeverityFlagBitsEXT sev,
    VkDebugUtilsMessageTypeFlagsEXT type, const VkDebugUtilsMessengerCallbackDataEXT* data, void* user) {
  (void)type; (void)user;
  if (sev & VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT) g_val_errors++;
  else g_val_warnings++;
  printf("VALIDATION %s %s\n", (sev & VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT) ? "ERROR" : "WARNING",
         data->pMessage);
  return VK_FALSE;
}

// VK_EXTS: comma-separated device extensions (or "all"). VALIDATE=1: Khronos validation layer + messenger.
// FEATURES=1: enable every supported Vulkan 1.2 / ray-tracing / ray-query feature, as a real RT app does.
static void create_vulkan_device(const unsigned char* cuda_uuid) {
  int validate = getenv("VALIDATE") != NULL;
  VkDebugUtilsMessengerCreateInfoEXT msg = {VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT};
  msg.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
  msg.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT;
  msg.pfnUserCallback = on_message;
  const char* layers[] = {"VK_LAYER_KHRONOS_validation"};
  const char* inst_exts[] = {VK_EXT_DEBUG_UTILS_EXTENSION_NAME};
  VkApplicationInfo app = {VK_STRUCTURE_TYPE_APPLICATION_INFO, NULL, "probe", 1, "probe", 1, VK_API_VERSION_1_3};
  VkInstanceCreateInfo ici = {VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO, validate ? &msg : NULL, 0, &app,
                              validate, layers, validate, inst_exts};
  VkInstance inst;
  VkResult r = vkCreateInstance(&ici, NULL, &inst);
  if (r) { fprintf(stderr, "vkCreateInstance=%d\n", r); exit(2); }
  if (validate) {
    PFN_vkCreateDebugUtilsMessengerEXT create_messenger =
        (PFN_vkCreateDebugUtilsMessengerEXT)vkGetInstanceProcAddr(inst, "vkCreateDebugUtilsMessengerEXT");
    VkDebugUtilsMessengerEXT messenger;
    if (!create_messenger || create_messenger(inst, &msg, NULL, &messenger)) { fprintf(stderr, "messenger failed\n"); exit(2); }
  }
  uint32_t n = 16;
  VkPhysicalDevice all[16], pd = VK_NULL_HANDLE;
  r = vkEnumeratePhysicalDevices(inst, &n, all);
  if (r < 0 || n < 1) { fprintf(stderr, "vkEnumeratePhysicalDevices=%d n=%u\n", r, n); exit(2); }
  for (uint32_t i = 0; i < n && pd == VK_NULL_HANDLE; i++) {
    VkPhysicalDeviceIDProperties id = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ID_PROPERTIES};
    VkPhysicalDeviceProperties2 p2 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2, &id};
    vkGetPhysicalDeviceProperties2(all[i], &p2);
    if (memcmp(id.deviceUUID, cuda_uuid, VK_UUID_SIZE) == 0) {
      pd = all[i];
      printf("vulkan: physical device %u '%s' matches the CUDA device UUID\n", i, p2.properties.deviceName);
    }
  }
  if (pd == VK_NULL_HANDLE) { fprintf(stderr, "no Vulkan device matches the CUDA device UUID\n"); exit(2); }

  uint32_t ne = 0; vkEnumerateDeviceExtensionProperties(pd, NULL, &ne, NULL);
  VkExtensionProperties* props = calloc(ne, sizeof *props);
  vkEnumerateDeviceExtensionProperties(pd, NULL, &ne, props);
  const char** names = calloc(ne, sizeof *names); uint32_t nn = 0;
  const char* want = getenv("VK_EXTS");
  for (uint32_t i = 0; want && i < ne; i++) {
    const char* e = props[i].extensionName; size_t L = strlen(e); const char* hit = strstr(want, e);
    if (strcmp(want, "all") == 0 || (hit && (hit == want || hit[-1] == ',') && (hit[L] == 0 || hit[L] == ',')))
      names[nn++] = e;
  }

  VkPhysicalDeviceRayQueryFeaturesKHR rq = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_QUERY_FEATURES_KHR};
  VkPhysicalDeviceRayTracingPipelineFeaturesKHR rt = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_FEATURES_KHR, &rq};
  VkPhysicalDeviceAccelerationStructureFeaturesKHR as = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_FEATURES_KHR, &rt};
  VkPhysicalDeviceVulkan12Features v12 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES, &as};
  VkPhysicalDeviceFeatures2 f2 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2, &v12};
  int features = getenv("FEATURES") != NULL;
  if (features) {
    vkGetPhysicalDeviceFeatures2(pd, &f2);  // query supported, then enable exactly what is supported
    printf("vulkan: features rayTracingPipeline=%u accelerationStructure=%u rayQuery=%u bufferDeviceAddress=%u\n",
           rt.rayTracingPipeline, as.accelerationStructure, rq.rayQuery, v12.bufferDeviceAddress);
  }
  printf("vulkan: enabling %u device extensions:", nn);
  for (uint32_t i = 0; i < nn; i++) printf(" %s", names[i]);
  printf("\n");

  float prio = 1.0f;
  VkDeviceQueueCreateInfo q = {VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO, NULL, 0, 0, 1, &prio};
  VkDeviceCreateInfo dci = {VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO, features ? &f2 : NULL, 0, 1, &q, 0, NULL, nn, names, NULL};
  VkDevice dev;
  r = vkCreateDevice(pd, &dci, NULL, &dev);
  if (r) { fprintf(stderr, "vkCreateDevice=%d\n", r); exit(2); }
  printf("vulkan: device created; validation errors=%d warnings=%d\n", g_val_errors, g_val_warnings);
  if (getenv("VK_DESTROY")) {  // lifetime check: tear the Vulkan device and instance down before the CUDA calls
    vkDestroyDevice(dev, NULL);
    vkDestroyInstance(inst, NULL);
    printf("vulkan: device and instance destroyed\n");
  }
}

#define CK(x) do { CUresult e_ = (x); if (e_) { const char* s_; cuGetErrorName(e_, &s_); \
  fprintf(stderr, "%s -> %s\n", #x, s_); exit(3); } } while (0)

static char kSrc[1024];
static void build_source(void) {  // STATIC_BYTES env: size of the kernel's static shared array (0 = none)
  int n = getenv("STATIC_BYTES") ? atoi(getenv("STATIC_BYTES")) : 36800;
  if (n > 0)
    snprintf(kSrc, sizeof kSrc,
      "extern \"C\" __global__ void k(int* o) {\n"
      "  __shared__ char buf[%d];\n  extern __shared__ char dyn[];\n"
      "  buf[threadIdx.x] = (char)threadIdx.x; dyn[threadIdx.x] = 1; __syncthreads();\n"
      "  if (o) o[threadIdx.x] = buf[(threadIdx.x + 1) %% %d] + dyn[0];\n}\n", n, n);
  else
    snprintf(kSrc, sizeof kSrc,
      "extern \"C\" __global__ void k(int* o) {\n  extern __shared__ char dyn[];\n"
      "  dyn[threadIdx.x] = 1; __syncthreads();\n  if (o) o[threadIdx.x] = dyn[0];\n}\n");
}

static void* nvrtc_cubin_sm120(void) {
  build_source();
  void* h = dlopen(getenv("NVRTC_LIB"), RTLD_NOW);
  if (!h) { fprintf(stderr, "dlopen nvrtc: %s\n", dlerror()); exit(4); }
  int (*create)(void**, const char*, const char*, int, const char**, const char**) = dlsym(h, "nvrtcCreateProgram");
  int (*compile)(void*, int, const char**) = dlsym(h, "nvrtcCompileProgram");
  int (*size)(void*, size_t*) = dlsym(h, "nvrtcGetCUBINSize");
  int (*get)(void*, char*) = dlsym(h, "nvrtcGetCUBIN");
  void* prog;
  char arch[32];  // NVRTC_ARCH: target SM, e.g. sm_121 on GB10
  snprintf(arch, sizeof arch, "-arch=%s", getenv("NVRTC_ARCH") ? getenv("NVRTC_ARCH") : "sm_120");
  const char* opts[] = {arch};
  if (create(&prog, kSrc, "k.cu", 0, NULL, NULL) || compile(prog, 1, opts)) { fprintf(stderr, "nvrtc failed\n"); exit(4); }
  size_t n; size(prog, &n);
  char* cubin = malloc(n); get(prog, cubin);
  return cubin;
}

static void* read_file(const char* path) {
  FILE* f = fopen(path, "rb"); if (!f) { perror(path); exit(4); }
  fseek(f, 0, SEEK_END); long n = ftell(f); fseek(f, 0, SEEK_SET);
  char* b = calloc(1, n + 1); fread(b, 1, n, f); fclose(f); return b;
}

int main(int argc, char** argv) {
  if (argc != 5) { fprintf(stderr, "usage: %s <vk:none|before|after> <setter:kernel|func> <delta> <image:nvrtc|ptx>\n", argv[0]); return 1; }
  const char *vk = argv[1], *setter = argv[2], *image_kind = argv[4];
  int delta = atoi(argv[3]);
  void* image = strcmp(image_kind, "nvrtc") == 0 ? nvrtc_cubin_sm120() : read_file(getenv("PTX_FILE"));

  CUdevice dev; CUcontext ctx;
  CK(cuInit(0)); CK(cuDeviceGet(&dev, 0));
  CK(cuDevicePrimaryCtxRetain(&ctx, dev)); CK(cuCtxSetCurrent(ctx));  // like the CUDA runtime / PyTorch
  if (getenv("PROBE_PAUSE")) raise(SIGSTOP);  // lets a parent debugger place offset breakpoints in libcuda
  CUuuid uuid; CK(cuDeviceGetUuid(&uuid, dev));
  if (strcmp(vk, "before") == 0) create_vulkan_device((const unsigned char*)uuid.bytes);

  CUlibrary lib; CUkernel kern; CUfunction fn;
  CK(cuLibraryLoadData(&lib, image, NULL, NULL, 0, NULL, NULL, 0));
  CK(cuLibraryGetKernel(&kern, lib, "k"));
  if (strcmp(vk, "after") == 0) create_vulkan_device((const unsigned char*)uuid.bytes);
  CK(cuKernelGetFunction(&fn, kern));

  int opt_in, reserved, stat, avail_ok = 0; size_t avail = 0;
  CK(cuDeviceGetAttribute(&opt_in, CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN, dev));
  CK(cuDeviceGetAttribute(&reserved, CU_DEVICE_ATTRIBUTE_RESERVED_SHARED_MEMORY_PER_BLOCK, dev));
  CK(cuKernelGetAttribute(&stat, CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES, kern, dev));
  avail_ok = cuOccupancyAvailableDynamicSMemPerBlock(&avail, fn, 1, 32) == CUDA_SUCCESS;
  if (getenv("BSEARCH")) {
    int lo = 0, hi = opt_in;  // invariant: lo accepted, hi+1 rejected
    while (lo < hi) {
      int mid = lo + (hi - lo + 1) / 2;
      if (cuKernelSetAttribute(CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, mid, kern, dev) == CUDA_SUCCESS) lo = mid;
      else hi = mid - 1;
    }
    printf("BSEARCH vk=%s max_accepted_dynamic=%d documented_max(opt_in-static)=%d difference=%d\n",
           vk, lo, opt_in - stat, (opt_in - stat) - lo);
  }
  int req = opt_in - stat - delta;
  printf("opt_in=%d reserved=%d static(reported)=%d driver_available_dynamic=%s%zu request=%d\n",
         opt_in, reserved, stat, avail_ok ? "" : "?", avail, req);

  CUresult r = strcmp(setter, "kernel") == 0
      ? cuKernelSetAttribute(CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, req, kern, dev)
      : cuFuncSetAttribute(fn, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, req);
  const char* name; cuGetErrorName(r, &name);
  printf("RESULT vk=%s setter=%s delta=%d image=%s -> %s\n", vk, setter, delta, image_kind, name);
  if (r == CUDA_SUCCESS) {  // prove the requested size is actually launchable in the CUDA context
    CUresult l = cuLaunchKernel(fn, 1, 1, 1, 32, 1, 1, req, NULL, (void*[]){&(CUdeviceptr){0}}, NULL);
    CUresult s = cuCtxSynchronize();
    const char *ln, *sn; cuGetErrorName(l, &ln); cuGetErrorName(s, &sn);
    printf("launch dynamic=%d -> %s / sync %s\n", req, ln, sn);
  }
  printf("VALIDATION_SUMMARY errors=%d warnings=%d\n", g_val_errors, g_val_warnings);
  return r == CUDA_SUCCESS ? 0 : 10;
}
