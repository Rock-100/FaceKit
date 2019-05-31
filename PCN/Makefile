CC=g++
CAFFEROOT=/usr/local/include
DEBUG=-O0 #-O0 -g #-O3 
CXX_FLAGS=-std=c++11 -DCPU_ONLY 
LD_FLAGS=-lopencv_core -lopencv_highgui -lopencv_imgcodecs -lopencv_imgproc -lopencv_video -lopencv_videoio -lcaffe -lglog -lboost_system -lprotobuf 
TARGETS=video picture fddb pcn_api_test
LIB=libPCN.so
INSTALL_DIR=/usr/local/bin
LIB_DIR=/usr/local/lib
SHARE_DIR=/usr/local/share/pcn
.PHONY: clean model

all: $(TARGETS)

$(TARGETS): $(LIB)
	$(CC) -o $@ $@.cpp $(LIB) -std=c++11 -O3 -D CPU_ONLY ${LD_FLAGS} ${DEBUG} 

$(LIB): PCN.cpp
	$(CC) -shared -fPIC -o $@ $^ ${CXX_FLAGS} ${LD_FLAGS}  ${DEBUG}

install:
	mkdir -p $(SHARE_DIR)
	cp model/* $(SHARE_DIR)
	cp $(TARGETS) $(INSTALL_DIR)
	cp $(LIB) $(LIB_DIR)

clean:
	rm -rf $(TARGETS) $(LIB) *.o

