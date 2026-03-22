/**
 * @file dec2hex.hpp
 * @author 陈祈 (12332378@mail.sustech.edu.cn)
 * @brief
 * @version 0.1
 * @date 2024-03-20
 *
 * @copyright Copyright (c) 2024
 *
 */
#ifndef __DEC2HEX_H__
#define __DEC2HEX_H__

#include "ros/ros.h"
/**
 * @brief transmit dec to 2 byte hex
 *
 * @param ub dst upper byte
 * @param lb dst lower byte
 * @param dec src dec number
 */
void Dec2Hex(uint8_t *lb, uint8_t *ub, int dec) {
  uint8_t a, b, c, d;
  uint16_t src = dec >= 0 ? dec : 65536 + dec;
  a = src / 4096;
  b = (src % 4096) / 256;
  c = ((src % 4096) % 256) / 16;
  d = src % 16;
  *ub = a * 16 + b;
  *lb = c * 16 + d;
}
/**
 * @brief transmit dec to 4 byte hex
 *
 * @param b0 dst lower byte
 * @param b1
 * @param b2
 * @param b3 dst upper byte
 * @param dec src dec number
 */
void Dec2Hex(uint8_t *b0, uint8_t *b1, uint8_t *b2, uint8_t *b3, int32_t dec) {
  uint32_t src = dec >= 0 ? dec : dec - 0x100000000;
  *b0 = dec & 0xFF;          // 提取最低字节
  *b1 = (dec >> 8) & 0xFF;   // 提取次低字节
  *b2 = (dec >> 16) & 0xFF;  // 提取次高字节
  *b3 = dec >> 24;           // 提取最高字节
}
/**
 * @brief transmit 2byte hex to dec
 *
 * @param lb src lower byte
 * @param ub src upper byte
 * @return int dst dec number
 */
int16_t Hex2Dec(uint8_t lb, uint8_t ub) {
  int res = lb + ub * 256;
  return res > 20000 ? res - 65536 : res;
}

#endif