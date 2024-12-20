#!/bin/bash

/usr/local/opt/llvm/bin/clang -Wall -O2 -o gan gan.c -lm -lcurl -L/usr/local/opt/llvm/lib -fopenmp