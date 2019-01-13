#!/usr/bin/en sh
echo "compile libPCN.so"
g++ -fpic -shared -o libPCN.so PCN.cpp -O3 -D CPU_ONLY -I $CAFFEROOT/include/ -I $CAFFEROOT/.build_release/src/ -L $CAFFEROOT/build/lib/ -lcaffe -lglog -lboost_system -lprotobuf `pkg-config --cflags --libs opencv`
echo "done"
