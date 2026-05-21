CUDA_VER?=10.2
NVDS_VERSION?=5.1

CC:=g++
TARGET_LIB:=libnvdsinfer_custom_impl_Yolo.so
SRCFILES:=nvdsinfer_custombboxparser_yolov8.cpp

CFLAGS:= -Wall -std=c++11 -shared -fPIC -Wno-error=deprecated-declarations
CFLAGS+= -I/opt/nvidia/deepstream/deepstream-$(NVDS_VERSION)/sources/includes
CFLAGS+= -I/usr/local/cuda-$(CUDA_VER)/include

LIBS:= -lnvinfer_plugin -lnvinfer -lnvparsers
LIBS+= -L/usr/local/cuda-$(CUDA_VER)/lib64 -lcudart -lcublas -lstdc++

LFLAGS:= -shared -Wl,--start-group $(LIBS) -Wl,--end-group

all: $(TARGET_LIB)

$(TARGET_LIB): $(SRCFILES)
	$(CC) -o $@ $^ $(CFLAGS) $(LFLAGS)

clean:
	rm -f $(TARGET_LIB)
