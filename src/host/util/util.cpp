/****
 * Copyright (c) 2016-2018, NVIDIA Corporation.  All rights reserved.
 *
 * See COPYRIGHT for license information
 ****/

#define __STDC_FORMAT_MACROS 1

#include "internal/host/util.h"                 // for getHostHash
#include "internal/host/debug.h"                // for INFO, NVSHMEM...
#include "internal/host/error_codes_internal.h" // for NVSHMEMI_SUCCESS
#include "non_abi/nvshmemx_error.h"             // for NVSHMEMI_ERRO...
#include <execinfo.h>                           // for backtrace
#include <inttypes.h>                           // for PRIu64
#include <sched.h>                              // for sched_getaffi...
#include <signal.h>                             // for signal, SIGSEGV
#include <stdint.h>                             // for uint64_t
#include <stdio.h>                              // for size_t, NULL
#include <stdlib.h>                             // for calloc, exit
#include <string.h>                             // for memcpy, memset
#include <string>                               // for string, basic...
#include <unistd.h>                             // for gethostname

/* Wrap 'str' to fit within 'wraplen' columns. Will not break a line of text
 * with no whitespace that exceeds the allowed length. After each line break,
 * insert 'indent' string (if provided).  Caller must free the returned buffer.
 */
char *nvshmemu_wrap(const char *str, const size_t wraplen, const char *indent,
                    const int strip_backticks) {
  const size_t indent_len = indent != NULL ? strlen(indent) : 0;
  size_t str_len = 0, line_len = 0, line_breaks = 0;
  char *str_s = NULL;

  /* Count characters and newlines */
  for (const char *s = str; *s != '\0'; s++, str_len++)
    if (*s == '\n')
      ++line_breaks;

  /* Worst case is wrapping at 1/2 wraplen plus explicit line breaks. Each
   * wrap adds an indent string. The newline is either already in the source
   * string or replaces a whitespace in the source string */
  const size_t out_len =
      str_len + 1 + (2 * (str_len / wraplen + 1) + line_breaks) * indent_len;
  char *out = (char *)calloc(out_len, sizeof(char));
  char *str_p = (char *)str;
  std::string statement = "";

  if (out == NULL) {
    fprintf(stderr, "%s:%d Unable to allocate output buffer\n", __FILE__,
            __LINE__);
    return NULL;
  }

  while (*str_p != '\0' &&
         /* avoid overflowing out */ statement.length() < (out_len - 1)) {
    /* Remember location of last space */
    if (*str_p == ' ') {
      str_s = str_p;
    }
    /* Wrap here if there is a newline */
    else if (*str_p == '\n') {
      str_s = str_p;
      statement += "\n"; /* Append newline and indent */
      if (indent) {
        statement += indent;
      }
      str_p++;
      str_s = NULL;
      line_len = 0;
      continue;
    }

    /* Remove backticks from the input string */
    else if (*str_p == '`' && strip_backticks) {
      str_p++;
      continue;
    }

    /* Reached end of line, try to wrap */
    if (line_len >= wraplen) {
      if (str_s != NULL) {
        str_p = str_s; /* Jump back to last space */
        size_t found = statement.find_last_of(
            " "); /* Find the last token, remove it from statement as
                     it will be appended subsequently */
        auto last_word = statement.substr(found + 1);
        statement.erase(found, found + 1 + last_word.length());
        statement += "\n"; /* Append newline and indent */
        if (indent) {
          statement += indent;
        }
        str_p++;
        str_s = NULL;
        line_len = 0;
        continue;
      }
    }
    statement += (*str_p);
    str_p++;
    line_len++;
  }

  memset(out, '\0', out_len);
  memcpy(out, statement.c_str(), statement.length());
  return out;
}

nvshmemResult_t nvshmemu_gethostname(char *hostname, int maxlen) {
  if (gethostname(hostname, maxlen) != 0) {
    strncpy(hostname, "unknown", maxlen);
    return NVSHMEMI_SYSTEM_ERROR;
  }
  int i = 0;
  while ((hostname[i] != '.') && (hostname[i] != '\0') && (i < maxlen - 1))
    i++;
  hostname[i] = '\0';
  return NVSHMEMI_SUCCESS;
}
