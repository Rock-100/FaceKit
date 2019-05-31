#!/usr/bin/en sh
echo "compile "$1
g++ -o $1 $1.cpp PCN.h libPCN.so -std=c++11 -O3 -D CPU_ONLY -I $CAFFEROOT/include/ -I $CAFFEROOT/.build_release/src/ -L $CAFFEROOT/build/lib/ -lcaffe -lglog -lboost_system -lprotobuf `pkg-config --cflags --libs opencv`
echo "done"
