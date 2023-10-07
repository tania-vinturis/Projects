#include <stdio.h>
#define SIZE 100

int main() {
  int n = 0;
  int i = 0;
  int j = 0;
  float v[SIZE];
  char op[SIZE];
  float rez = 0;
  scanf("%d", &n);
  for (i = 1; i <= n; i++) {
    scanf("%f", &v[i]);
  }
  for (i = 0; i < n; i++) {
    scanf("%c", &op[i]);
  }
  rez = v[1];
  for (j = 0; j < n; j++) {
    if (op[j] == '+') {
      rez = rez + v[j + 1];
    } else if (op[j] == '-') {
      rez = rez - v[j + 1];
    } else if (op[j] == '*') {
      rez = rez * v[j + 1];
    } else if (op[j] == '/') {
      rez = rez / v[j + 1];
    }
  }
  printf("%f\n", rez);
  return 0;
}