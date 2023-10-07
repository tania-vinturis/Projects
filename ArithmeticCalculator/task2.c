#include <stdio.h>
#include <stdlib.h>
#define SIZE 100

int main() {
  int n = 0;
  int i = 0;
  int j = 0;
  float v[SIZE];
  char op[SIZE];
  float rez = 0;
  scanf("%d", &n);
  for (i = 0; i < n; i++) {
    scanf("%f", &v[i]);
  }
  for (i = 0; i < n; i++) {
    scanf("%c", &op[i]);
  }
  for (i = 0; i < n - 1; i++) {
    op[i] = op[i + 1];
  }
  for (j = 0; j < n - 1; j++) {
    if (op[j] == '*') {
      v[j] = v[j] * v[j + 1];
      for (i = j + 1; i < n - 1; i++) {
        v[i] = v[i + 1];
        op[i - 1] = op[i];
      }
      n--;
      j--;
    }
    if (op[j] == '/') {
      v[j] = v[j] / v[j + 1];
      for (i = j + 1; i < n - 1; i++) {
        v[i] = v[i + 1];
        op[i - 1] = op[i];
      }
      n--;
      j--;
    }
  }
  rez = v[0];
  for (j = 0; j < n - 1; j++) {
    if (op[j] == '+') {
      rez = rez + v[j + 1];
    }
    if (op[j] == '-') {
      rez = rez - v[j + 1];
    }
  }
  printf("%f\n", rez);
  return 0;
}