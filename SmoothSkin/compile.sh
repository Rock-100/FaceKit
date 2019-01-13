#!/usr/bin/en sh
echo "compile picture"
g++ -o picture picture.cpp Common.h Common.cpp GuideFilter.h GuideFilter.cpp BilateralFilter.h BilateralFilter.cpp -std=c++11 -O3 `pkg-config --cflags --libs opencv`
echo "done"
