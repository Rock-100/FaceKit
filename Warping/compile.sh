#!/usr/bin/en sh
echo "compile picture"
g++ -o picture picture.cpp IDW.cpp IDW.h -std=c++11 -O3 `pkg-config --cflags --libs opencv`
echo "done"
