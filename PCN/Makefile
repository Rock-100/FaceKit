CC=g++
DEBUG=-O3 
CXX_FLAGS=-std=c++11 -DCPU_ONLY 
LD_FLAGS=-lopencv_core -lopencv_highgui -lopencv_imgcodecs -lopencv_imgproc -lopencv_video -lopencv_videoio -lcaffe -lglog -lboost_system -lprotobuf 
TARGETS=video picture fddb pcn_api_test
LIB=libPCN.so
LIB_DIR=/usr/local/lib
SHARE_DIR=/usr/local/share/pcn
CAFFEROOT=/usr/local/lib

all: $(TARGETS)

$(LIB): PCN.cpp
	$(CC) -shared -fPIC -o $@ $^ ${CXX_FLAGS} ${LD_FLAGS} ${DEBUG} -I ${CAFFEROOT}/include/ -I ${CAFFEROOT}/.build_release/src/ -L ${CAFFEROOT}/build/lib/

$(TARGETS): $(LIB)
	$(CC) -o $@ $@.cpp $(LIB) ${CXX_FLAGS} ${LD_FLAGS} ${DEBUG} -I ${CAFFEROOT}/include/ -I ${CAFFEROOT}/.build_release/src/ -L ${CAFFEROOT}/build/lib/

install:
	mkdir -p $(SHARE_DIR)
	cp model/* $(SHARE_DIR)
	cp $(LIB) $(LIB_DIR)

clean:
	rm -rf $(TARGETS) $(LIB)
