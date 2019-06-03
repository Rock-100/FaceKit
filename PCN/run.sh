#!/usr/bin/en sh
CAFFEROOT=/usr/local/lib
export LD_LIBRARY_PATH=$CAFFEROOT/build/lib/:$LD_LIBRARY_PATH
./$1
