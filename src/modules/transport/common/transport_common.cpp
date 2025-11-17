/*
 * Copyright (c) Nex-AGI. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
 *
 * See COPYRIGHT for license information
 */

#include "transport_common.h"
#include "non_abi/nvshmemx_error.h" // for NVSHMEMI_ERROR_PRINT, NVSHMEMX_E...
#include <stdint.h>                 // for uint64_t, uintptr_t
#include <stdlib.h>                 // for atoi, calloc, free, realloc

int nvshmemt_parse_hca_list(const char *string,
                            struct nvshmemt_hca_info *hca_list, int max_count,
                            int log_level) {
  if (!string)
    return 0;

  const char *ptr = string;
  // Ignore "^" name, will be detected outside of this function
  if (ptr[0] == '^')
    ptr++;

  int if_num = 0;
  int if_counter = 0;
  int segment_counter = 0;
  char c;
  do {
    c = *ptr;
    if (c == ':') {
      if (segment_counter == 0) {
        if (if_counter > 0) {
          hca_list[if_num].name[if_counter] = '\0';
          hca_list[if_num].port = atoi(ptr + 1);
          hca_list[if_num].found = 0;
          if_num++;
          if_counter = 0;
          segment_counter++;
        }
      } else {
        hca_list[if_num - 1].count = atoi(ptr + 1);
        segment_counter = 0;
      }
      c = *(ptr + 1);
      while (c != ',' && c != ':' && c != '\0') {
        ptr++;
        c = *(ptr + 1);
      }
    } else if (c == ',' || c == '\0') {
      if (if_counter > 0) {
        hca_list[if_num].name[if_counter] = '\0';
        hca_list[if_num].found = 0;
        if_num++;
        if_counter = 0;
      }
      segment_counter = 0;
    } else {
      if (if_counter == 0) {
        hca_list[if_num].port = -1;
        hca_list[if_num].count = 1;
      }
      hca_list[if_num].name[if_counter] = c;
      if_counter++;
    }
    ptr++;
  } while (if_num < max_count && c);

  INFO(log_level, "Begin - Parsed HCA list provided by user - ");
  for (int i = 0; i < if_num; i++) {
    INFO(log_level,
         "Parsed HCA list provided by user - i=%d (of %d), name=%s, port=%d, "
         "count=%d",
         i, if_num, hca_list[i].name, hca_list[i].port, hca_list[i].count);
  }
  INFO(log_level, "End - Parsed HCA list provided by user");

  return if_num;
}
