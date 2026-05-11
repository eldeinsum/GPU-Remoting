#include <nccl.h>

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static bool nccl_ok(ncclResult_t status, const char *call) {
  if (status == ncclSuccess) {
    return true;
  }
  fprintf(stderr, "%s failed with NCCL status %d\n", call, (int)status);
  return false;
}

static bool nccl_supported_status(ncclResult_t status, const char *call) {
  if (status == ncclSuccess || status == ncclInvalidArgument) {
    return true;
  }
  fprintf(stderr, "%s returned unexpected NCCL status %d\n", call,
          (int)status);
  return false;
}

static const char *find_key(const char **keys, int key_count,
                            const char *needle) {
  for (int i = 0; i < key_count; ++i) {
    if (keys[i] != NULL && strcmp(keys[i], needle) == 0) {
      return keys[i];
    }
  }
  return NULL;
}

int main(void) {
  setenv("NCCL_PARAM_DUMP_ALL", "true", 1);

  const char **keys = NULL;
  int key_count = 0;
  if (!nccl_ok(ncclParamGetAllParameterKeys(&keys, &key_count),
               "ncclParamGetAllParameterKeys")) {
    return EXIT_FAILURE;
  }
  if (keys == NULL || key_count <= 0) {
    fprintf(stderr, "NCCL parameter key table is empty\n");
    return EXIT_FAILURE;
  }

  const char *string_key = find_key(keys, key_count, "NCCL_NO_CACHE");
  const char *bool_key = find_key(keys, key_count, "NCCL_PARAM_DUMP_ALL");
  if (string_key == NULL || bool_key == NULL) {
    fprintf(stderr, "NCCL parameter key selection failed\n");
    return EXIT_FAILURE;
  }

  ncclParamHandle_t *handle = NULL;
  if (!nccl_ok(ncclParamBind(&handle, string_key), "ncclParamBind") ||
      handle == NULL) {
    return EXIT_FAILURE;
  }

  char raw_value[256];
  int raw_len = -1;
  if (!nccl_ok(ncclParamGet(handle, raw_value, (int)sizeof(raw_value),
                            &raw_len),
               "ncclParamGet")) {
    return EXIT_FAILURE;
  }
  if (raw_len < 0 || raw_len > (int)sizeof(raw_value)) {
    fprintf(stderr, "NCCL raw parameter length is invalid: %d\n", raw_len);
    return EXIT_FAILURE;
  }

  const char *str_value = NULL;
  if (!nccl_ok(ncclParamGetStr(handle, &str_value), "ncclParamGetStr")) {
    return EXIT_FAILURE;
  }

  const char *direct_value = NULL;
  int direct_len = -1;
  if (!nccl_ok(ncclParamGetParameter(string_key, &direct_value, &direct_len),
               "ncclParamGetParameter")) {
    return EXIT_FAILURE;
  }
  if (direct_len < 0) {
    fprintf(stderr, "NCCL direct parameter length is invalid: %d\n",
            direct_len);
    return EXIT_FAILURE;
  }

  ncclParamHandle_t *bool_handle = NULL;
  if (!nccl_ok(ncclParamBind(&bool_handle, bool_key), "ncclParamBind(bool)") ||
      bool_handle == NULL) {
    return EXIT_FAILURE;
  }

  int8_t i8_value = 0;
  int16_t i16_value = 0;
  int32_t i32_value = 0;
  int64_t i64_value = 0;
  uint8_t u8_value = 0;
  uint16_t u16_value = 0;
  uint32_t u32_value = 0;
  uint64_t u64_value = 0;
  if (!nccl_supported_status(ncclParamGetI8(bool_handle, &i8_value),
                             "ncclParamGetI8") ||
      !nccl_supported_status(ncclParamGetI16(bool_handle, &i16_value),
                             "ncclParamGetI16") ||
      !nccl_supported_status(ncclParamGetI32(bool_handle, &i32_value),
                             "ncclParamGetI32") ||
      !nccl_supported_status(ncclParamGetI64(bool_handle, &i64_value),
                             "ncclParamGetI64") ||
      !nccl_supported_status(ncclParamGetU8(bool_handle, &u8_value),
                             "ncclParamGetU8") ||
      !nccl_supported_status(ncclParamGetU16(bool_handle, &u16_value),
                             "ncclParamGetU16") ||
      !nccl_supported_status(ncclParamGetU32(bool_handle, &u32_value),
                             "ncclParamGetU32") ||
      !nccl_supported_status(ncclParamGetU64(bool_handle, &u64_value),
                             "ncclParamGetU64")) {
    return EXIT_FAILURE;
  }

  ncclParamDumpAll();
  printf("NCCL parameter coverage test passed\n");
  return EXIT_SUCCESS;
}
