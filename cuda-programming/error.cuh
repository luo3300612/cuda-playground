#pragma once
#include<stdio.h>

#define CHECK(call) \
do \
{ \
    const cudaError_t error_code = call; \
    if (error_code != cudaSuccess){      \
        printf("CUDA Error:\n");         \
        printf("\tFile:\t%s\n",__FILE__);\
        printf("\tLine:\t%d\n",__LINE__);\
        printf("\tError code: %d\n", error_code); \
        printf("\tError text: %s\n", cudaGetErrorString(error_code)); \
        exit(1);\
    }               \
} while(0)