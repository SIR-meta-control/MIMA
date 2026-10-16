/**
 * @file dec2hex.cpp
 * @author 陈祈 (inorichen77@gmail.com)
 * @brief
 * @version 0.1
 * @date 2023-03-15
 *
 * @copyright Copyright (c) 2023
 *
 */
#include "dec2hex.h"

void Dec2Hex(uint8_t *lb, uint8_t *ub, int dec) {
  uint8_t a, b, c, d;
  a = dec / 4096;
  b = (dec % 4096) / 256;
  c = ((dec % 4096) % 256) / 16;
  d = dec % 16;
  *ub = a * 16 + b;
  *lb = c * 16 + d;
}
int Hex2Dec(char lb, char ub) {
  return static_cast<int>(lb) + static_cast<int>(ub) * 4096;
}
int Hex2Dec(char b0, char b1, char b2, char b3) {
  return static_cast<int>(b0) + static_cast<int>(b1) * 4096 +
         static_cast<int>(b2) * 65536 + static_cast<int>(b3) * 16777216;
}
