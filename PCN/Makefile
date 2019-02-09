CC=g++
CAFFEROOT=/usr/local/include
DEBUG=-O3 #-O0 -g #-O3 
CXX_FLAGS=-std=c++11 -DCPU_ONLY 
LD_FLAGS=-lopencv_core -lopencv_highgui -lopencv_imgcodecs -lopencv_imgproc -lopencv_video -lopencv_videoio -lcaffe -lglog -lboost_system -lprotobuf 
TARGETS=video picture crop fddb
LIB=libPCNApi.so
TARGET_DIR=./bin
.PHONY: clean model

all: $(TARGETS) $(LIB) model

${TARGET_DIR}/PCN.o: PCN.cpp 
	$(CC) -c -o $@ $< $(CXX_FLAGS) -fPIC ${DEBUG}

$(TARGETS): ${TARGET_DIR}/PCN.o 
	$(CC) -o ${TARGET_DIR}/$@ $@.cpp $< -std=c++11 -O3 -D CPU_ONLY ${LD_FLAGS} ${DEBUG}

$(LIB): ${TARGET_DIR}/PCN.o
	$(CC) -o ${TARGET_DIR}/$@ $< -std=c++11 -O3 -D CPU_ONLY ${LD_FLAGS} -shared -fPIC ${DEBUG}

model:
	cp -r model ${TARGET_DIR}/
clean:
	rm -rf ./bin/*

