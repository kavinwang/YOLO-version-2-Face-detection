GPU=0
CUDNN=0
OPENCV=1
DEBUG=0

ARCH= -gencode arch=compute_20,code=[sm_20,sm_21] \
      -gencode arch=compute_30,code=sm_30 \
      -gencode arch=compute_35,code=sm_35 \
      -gencode arch=compute_50,code=[sm_50,compute_50] \
      -gencode arch=compute_52,code=[sm_52,compute_52]

# This is what I use, uncomment if you know your arch and want to specify
# ARCH=  -gencode arch=compute_52,code=compute_52

VPATH=./src/
EXEC=darknet
OBJDIR=./obj/

CC=gcc
NVCC=nvcc 
OPTS=-Ofast
LDFLAGS= -lm -pthread 
COMMON= 
CFLAGS=-Wall -Wfatal-errors 

ifeq ($(DEBUG), 1) 
OPTS=-O0 -g
endif

CFLAGS+=$(OPTS)

ifeq ($(OPENCV), 1) 
COMMON+= -DOPENCV
CFLAGS+= -DOPENCV
# LDFLAGS+= `pkg-config --libs opencv` 
# COMMON+= `pkg-config --cflags opencv` 
LDFLAGS+=-Wl,-force_load /Users/evgenybaskakov/Projects/caffe2_cpp_tutorial/build/libcaffe2_cpp.a /usr/local/lib/libCaffe2_CPU.dylib /Users/evgenybaskakov/anaconda2/lib/libprotobuf.dylib /Users/evgenybaskakov/anaconda2/lib/libglog.dylib /Users/evgenybaskakov/anaconda2/lib/libgflags.dylib \
/Users/evgenybaskakov/anaconda2/lib/libopencv_stitching.3.2.0.dylib /Users/evgenybaskakov/anaconda2/lib/libopencv_superres.3.2.0.dylib /Users/evgenybaskakov/anaconda2/lib/libopencv_videostab.3.2.0.dylib /Users/evgenybaskakov/anaconda2/lib/libopencv_aruco.3.2.0.dylib /Users/evgenybaskakov/anaconda2/lib/libopencv_bgsegm.3.2.0.dylib /Users/evgenybaskakov/anaconda2/lib/libopencv_bioinspired.3.2.0.dylib /Users/evgenybaskakov/anaconda2/lib/libopencv_ccalib.3.2.0.dylib /Users/evgenybaskakov/anaconda2/lib/libopencv_dpm.3.2.0.dylib /Users/evgenybaskakov/anaconda2/lib/libopencv_freetype.3.2.0.dylib /Users/evgenybaskakov/anaconda2/lib/libopencv_fuzzy.3.2.0.dylib /Users/evgenybaskakov/anaconda2/lib/libopencv_hdf.3.2.0.dylib /Users/evgenybaskakov/anaconda2/lib/libopencv_line_descriptor.3.2.0.dylib /Users/evgenybaskakov/anaconda2/lib/libopencv_optflow.3.2.0.dylib /Users/evgenybaskakov/anaconda2/lib/libopencv_reg.3.2.0.dylib /Users/evgenybaskakov/anaconda2/lib/libopencv_saliency.3.2.0.dylib /Users/evgenybaskakov/anaconda2/lib/libopencv_stereo.3.2.0.dylib /Users/evgenybaskakov/anaconda2/lib/libopencv_structured_light.3.2.0.dylib /Users/evgenybaskakov/anaconda2/lib/libopencv_surface_matching.3.2.0.dylib /Users/evgenybaskakov/anaconda2/lib/libopencv_tracking.3.2.0.dylib /Users/evgenybaskakov/anaconda2/lib/libopencv_xfeatures2d.3.2.0.dylib /Users/evgenybaskakov/anaconda2/lib/libopencv_ximgproc.3.2.0.dylib /Users/evgenybaskakov/anaconda2/lib/libopencv_xobjdetect.3.2.0.dylib /Users/evgenybaskakov/anaconda2/lib/libopencv_xphoto.3.2.0.dylib /usr/lib/libcurl.dylib /Users/evgenybaskakov/anaconda2/lib/libopencv_shape.3.2.0.dylib /Users/evgenybaskakov/anaconda2/lib/libopencv_phase_unwrapping.3.2.0.dylib /Users/evgenybaskakov/anaconda2/lib/libopencv_rgbd.3.2.0.dylib /Users/evgenybaskakov/anaconda2/lib/libopencv_calib3d.3.2.0.dylib /Users/evgenybaskakov/anaconda2/lib/libopencv_video.3.2.0.dylib /Users/evgenybaskakov/anaconda2/lib/libopencv_datasets.3.2.0.dylib /Users/evgenybaskakov/anaconda2/lib/libopencv_dnn.3.2.0.dylib /Users/evgenybaskakov/anaconda2/lib/libopencv_face.3.2.0.dylib /Users/evgenybaskakov/anaconda2/lib/libopencv_plot.3.2.0.dylib /Users/evgenybaskakov/anaconda2/lib/libopencv_text.3.2.0.dylib /Users/evgenybaskakov/anaconda2/lib/libopencv_features2d.3.2.0.dylib /Users/evgenybaskakov/anaconda2/lib/libopencv_flann.3.2.0.dylib /Users/evgenybaskakov/anaconda2/lib/libopencv_objdetect.3.2.0.dylib /Users/evgenybaskakov/anaconda2/lib/libopencv_ml.3.2.0.dylib /Users/evgenybaskakov/anaconda2/lib/libopencv_highgui.3.2.0.dylib /Users/evgenybaskakov/anaconda2/lib/libopencv_photo.3.2.0.dylib /Users/evgenybaskakov/anaconda2/lib/libopencv_videoio.3.2.0.dylib /Users/evgenybaskakov/anaconda2/lib/libopencv_imgcodecs.3.2.0.dylib /Users/evgenybaskakov/anaconda2/lib/libopencv_imgproc.3.2.0.dylib /Users/evgenybaskakov/anaconda2/lib/libopencv_core.3.2.0.dylib \
 -Wl,-rpath,/usr/local/lib -Wl,-rpath,/Users/evgenybaskakov/anaconda2/lib
COMMON+=-I/Users/evgenybaskakov/anaconda2/include
endif

ifeq ($(GPU), 1) 
COMMON+= -DGPU -I/usr/local/cuda/include/
CFLAGS+= -DGPU
LDFLAGS+= -L/usr/local/cuda/lib64 -lcuda -lcudart -lcublas -lcurand
endif

ifeq ($(CUDNN), 1) 
COMMON+= -DCUDNN 
CFLAGS+= -DCUDNN
LDFLAGS+= -lcudnn
endif

OBJ=attribute_detections.o gemm.o utils.o cuda.o convolutional_layer.o list.o image.o activations.o im2col.o col2im.o blas.o crop_layer.o dropout_layer.o maxpool_layer.o softmax_layer.o data.o matrix.o network.o connected_layer.o cost_layer.o parser.o option_list.o darknet.o detection_layer.o captcha.o route_layer.o writing.o box.o nightmare.o normalization_layer.o avgpool_layer.o coco.o dice.o yolo.o detector.o layer.o compare.o classifier.o local_layer.o swag.o shortcut_layer.o activation_layer.o rnn_layer.o gru_layer.o rnn.o rnn_vid.o crnn_layer.o demo.o tag.o cifar.o go.o batchnorm_layer.o art.o region_layer.o reorg_layer.o super.o voxel.o tree.o
ifeq ($(GPU), 1) 
LDFLAGS+= -lstdc++ 
OBJ+=convolutional_kernels.o activation_kernels.o im2col_kernels.o col2im_kernels.o blas_kernels.o crop_layer_kernels.o dropout_layer_kernels.o maxpool_layer_kernels.o network_kernels.o avgpool_layer_kernels.o
endif

OBJS = $(addprefix $(OBJDIR), $(OBJ))
DEPS = $(wildcard src/*.h) Makefile

all: obj backup results $(EXEC)

$(EXEC): $(OBJS)
	g++ $(COMMON) $(CFLAGS) $^ -o $@ $(LDFLAGS)

$(OBJDIR)%.o: %.c $(DEPS)
	$(CC) $(COMMON) $(CFLAGS) -c $< -o $@

$(OBJDIR)%.o: %.cc $(DEPS)
	g++ -I/Users/evgenybaskakov/Projects/caffe2_cpp_tutorial/include -I/usr/local/include -I/usr/local/include/eigen3 -std=c++11 $(COMMON) $(CFLAGS) -c $< -o $@

$(OBJDIR)%.o: %.cu $(DEPS)
	$(NVCC) $(ARCH) $(COMMON) --compiler-options "$(CFLAGS)" -c $< -o $@

obj:
	mkdir -p obj
backup:
	mkdir -p backup
results:
	mkdir -p results

.PHONY: clean

clean:
	rm -rf $(OBJS) $(EXEC)

