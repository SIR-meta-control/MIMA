#include <stdio.h>
#include <stdlib.h>

#define TTY_PATH "/dev/tty"
#define STTY_US "stty raw -echo -F "
#define STTY_DEF "stty -raw echo -F "

int get_char() {
  fd_set rfds;
  struct timeval tv;
  int ch = 0;

  FD_ZERO(&rfds);
  FD_SET(0, &rfds);
  tv.tv_sec = 0;
  tv.tv_usec = 10;  // 设置等待超时时间

  // 检测键盘是否有输入
  if (select(1, &rfds, NULL, NULL, &tv) > 0) {
    ch = getchar();
  }
  return ch;
}

// 键盘键值定义
// 移动相关
#define KEY_w 0x77
#define KEY_a 0x61
#define KEY_s 0x73
#define KEY_d 0x64
#define KEY_q 0x71
#define KEY_e 0x65

// 变形相关
#define KEY_TAB 0x09
#define KEY_0 0x30
#define KEY_1 0x31
#define KEY_2 0x32
#define KEY_3 0x33
#define KEY_4 0x34

// 功能
#define KEY_ESC 0x1B
#define KEY_WAVE 0x60
#define KEY_SPACE 0x20
#define KEY_ENTER 0x0A
#define KEY_BACKSLASH 0x5C
#define KEY_MINUS 0x2D
#define KEY_PLUS 0x2B
#define KEY_u 0x75
#define KEY_i 0x69
#define KEY_o 0x6F
#define KEY_p 0x70

// 未使用
#define KEY_r 0x72
#define KEY_t 0x74
#define KEY_y 0x79
#define KEY_f 0x66
#define KEY_g 0x67
#define KEY_h 0x68
#define KEY_v 0x76
#define KEY_b 0x62
#define KEY_n 0x6E
#define KEY_z 0x7A
#define KEY_c 0x63
#define KEY_j 0x6A
#define KEY_k 0x6B
#define KEY_5 0x35
#define KEY_6 0x36
#define KEY_7 0x37
#define KEY_8 0x38
#define KEY_9 0x39
#define KEY_BACKSPACE 0x7F
#include <termios.h>
#include <unistd.h>

#include <iostream>

class KeyboardCtrl {
  struct termios initial_settings, new_settings;
  int peek_character = -1;

 public:
  KeyboardCtrl() { init_keyboard(); };
  ~KeyboardCtrl() { close_keyboard(); };

  int get_keyboard_press_key() {
    kbhit();
    return readch();
    // printf("%02x \n", readch());
  };

 private:
  void init_keyboard() {
    tcgetattr(0, &initial_settings);
    new_settings = initial_settings;
    new_settings.c_lflag &= ~(ICANON | ECHO);
    new_settings.c_cc[VEOL] = 1;
    new_settings.c_cc[VEOF] = 2;
    tcsetattr(0, TCSANOW, &new_settings);
  }

  void close_keyboard() { tcsetattr(0, TCSANOW, &initial_settings); }

  int kbhit() {
    unsigned char ch;
    int nread;

    if (peek_character != -1) return 1;
    new_settings.c_cc[VMIN] = 0;
    tcsetattr(0, TCSANOW, &new_settings);
    nread = read(0, &ch, 1);
    new_settings.c_cc[VMIN] = 1;
    tcsetattr(0, TCSANOW, &new_settings);
    if (nread == 1) {
      peek_character = ch;
      return 1;
    }
    return 0;
  }

  int readch() {
    char ch;
    if (peek_character != -1) {
      ch = peek_character;
      peek_character = -1;
      return ch;
    }
    read(0, &ch, 1);
    return ch;
  }
};