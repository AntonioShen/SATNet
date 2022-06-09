//
// Created by Antonio Shen on 2022-06-09.
//
#include <stdio.h>
#include <limits.h>

void get() {
    char cwd[PATH_MAX];
    printf("Current working dir: %s\n", cwd);
    FILE* test_file;
    test_file = fopen("aaaaaaaaaaa.txt", "w");
    fclose(test_file);
}
