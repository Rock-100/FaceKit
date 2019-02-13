CC=g++
CAFFEROOT=/usr/local/include
DEBUG=-O3 #-O0 -g #-O3 
CXX_FLAGS=-std=c++11 -DCPU_ONLY 
LD_FLAGS=-lopencv_core -lopencv_highgui -lopencv_imgcodecs -lopencv_imgproc -lopencv_video -lopencv_videoio -lcaffe -lglog -lboost_system -lprotobuf 
TARGETS=video picture crop fddb
LIB=libPCNApi.so
TARGET_DIR=.
SOURCE_DIR=./src
INSTALL_DIR=/usr/local/bin
LIB_DIR=/usr/local/lib
SHARE_DIR=/usr/local/share/pcn
.PHONY: clean model

all: $(TARGETS) $(LIB)

${TARGET_DIR}/PCN.o: ${SOURCE_DIR}/PCN.cpp 
	$(CC) -c -o $@ $< $(CXX_FLAGS) -fPIC ${DEBUG} -DMODEL_PATH=\"$(SHARE_DIR)\" #-DMODEL_PATH=\"./model\"

$(TARGETS): ${TARGET_DIR}/PCN.o 
	$(CC) -o ${TARGET_DIR}/$@ ${SOURCE_DIR}/$@.cpp $< -std=c++11 -O3 -D CPU_ONLY ${LD_FLAGS} ${DEBUG} 

$(LIB): ${TARGET_DIR}/PCN.o
	$(CC) -o ${TARGET_DIR}/$@ $< -std=c++11 -O3 -D CPU_ONLY ${LD_FLAGS} -shared -fPIC ${DEBUG}

install:
	mkdir -p $(SHARE_DIR)
	cp model/* $(SHARE_DIR)
	cp $(TARGETS) $(INSTALL_DIR)
	cp $(LIB) $(LIB_DIR)

clean:
	rm -rf $(TARGETS) $(LIB) *.o

