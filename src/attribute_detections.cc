#include <assert.h>
#include <vector>

#include "caffe2/core/init.h"
#include "caffe2/core/predictor.h"
#include "caffe2/core/tensor.h"
#include "caffe2/utils/proto_utils.h"

#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include "res/imagenet_classes.h"

#ifdef OPENCV
#include "opencv2/core/fast_math.hpp"
#include "opencv2/videoio/videoio_c.h"
#include "opencv2/imgcodecs/imgcodecs_c.h"
#include "opencv2/highgui/highgui_c.h"
#include "opencv2/imgproc/imgproc_c.h"
#endif

#include "attribute_detections.h"

using namespace std;
using namespace caffe2;

CAFFE2_DEFINE_string(init_net, "/Users/evgenybaskakov/Dropbox (Personal)/caffe2_gender/init_net.pb",
                     "The given path to the init protobuffer.");
CAFFE2_DEFINE_string(predict_net, "/Users/evgenybaskakov/Dropbox (Personal)/caffe2_gender/predict_net.pb",
                     "The given path to the predict protobuffer.");
// CAFFE2_DEFINE_string(image_file, "/Users/evgenybaskakov/Dropbox (Personal)/people_portraits/male2.jpg", "The image file.");
CAFFE2_DEFINE_int(size_to_fit, 224, "The image file.");

caffe2::Predictor *predictor;

char *attribute_detections_internal(const cv::Mat &image);

#ifdef __cplusplus
extern "C" {
#endif
void attribute_detections(const image *im, float thresh, box *boxes, float **probs, int num,
                          box **out_boxes, char **out_labels, int *detections_num)
{
    int detected = 0;

    assert(im->src);
    cv::Mat m = cv::cvarrToMat(im->src);

    for(int i = 0; i < num; ++i){
        float prob = probs[i][0];

        if(prob > thresh){
            // printf("%d/%d: prob %f, thresh %f\n", i, num, prob, thresh);

            box *b = &boxes[i];

            int left  = (b->x-b->w/2.)*im->w;
            int right = (b->x+b->w/2.)*im->w;
            int top   = (b->y-b->h/2.)*im->h;
            int bot   = (b->y+b->h/2.)*im->h;

            if(left < 0) left = 0;
            if(right > im->w-1) right = im->w-1;
            if(top < 0) top = 0;
            if(bot > im->h-1) bot = im->h-1;

            auto r = cv::Rect(left, top, right - left + 1, bot - top + 1);
            printf("x %d, y %d, w %d, h %d\n", r.x, r.y, r.width, r.height);

            auto subimage = m(r);

            char *label = attribute_detections_internal(subimage);
            assert(label);

            out_labels[detected] = label;
            out_boxes[detected] = b;

            detected++;
        }
    }

    *detections_num = detected;
}
#ifdef __cplusplus
}
#endif

char *attribute_detections_internal(const cv::Mat &image)
{
  std::cout << "image size: " << image.size() << std::endl;

  // scale image to fit
  cv::Size scale(
      std::max(FLAGS_size_to_fit * image.cols / image.rows, FLAGS_size_to_fit),
      std::max(FLAGS_size_to_fit, FLAGS_size_to_fit * image.rows / image.cols));
  cv::Mat resized_image;
  cv::resize(image, resized_image, scale);
  std::cout << "scaled size: " << resized_image.size() << std::endl;

  // crop image to fit
  cv::Rect crop((resized_image.cols - FLAGS_size_to_fit) / 2,
                (resized_image.rows - FLAGS_size_to_fit) / 2, FLAGS_size_to_fit,
                FLAGS_size_to_fit);
  auto cropped_image = resized_image(crop);
  std::cout << "cropped size: " << cropped_image.size() << std::endl;

  // convert to float, normalize to mean 128
  cropped_image.convertTo(cropped_image, CV_32FC3, 1.0, -128);
  std::cout << "value range: ("
            << *std::min_element((float *)cropped_image.datastart,
                                 (float *)cropped_image.dataend)
            << ", "
            << *std::max_element((float *)cropped_image.datastart,
                                 (float *)cropped_image.dataend)
            << ")" << std::endl;

  // convert NHWC to NCHW
  vector<cv::Mat> channels(3);
  cv::split(cropped_image, channels);
  std::vector<float> data;
  for (auto &c : channels) {
    data.insert(data.end(), (float *)c.datastart, (float *)c.dataend);
  }
  std::vector<TIndex> dims({1, cropped_image.channels(), cropped_image.rows, cropped_image.cols});
  TensorCPU input(dims, data, NULL);

  // >>> results = p.run([img])
  Predictor::TensorVector inputVec({&input}), outputVec;
  predictor->run(inputVec, &outputVec);
  auto &output = *(outputVec[0]);

  // sort top results
  const auto &probs = output.data<float>();
  assert(output.size() == 2);
  const char *classes[] = {"Female", "Male"};

  int idx = (probs[0] > probs[1] ? 0 : 1);

  std::cout << classes[idx] << " (" << probs[idx] * 100 << "%)" << std::endl;

  return strdup(classes[idx]);
}

void init_attributes()
{
  if (!std::ifstream(FLAGS_init_net).good() ||
      !std::ifstream(FLAGS_predict_net).good()) {
    std::cerr << "error: Squeezenet model file missing: "
              << (std::ifstream(FLAGS_init_net).good() ? FLAGS_predict_net
                                                       : FLAGS_init_net)
              << std::endl;
    std::cerr << "Make sure to first run ./script/download_resource.sh"
              << std::endl;
    abort();
  }

  std::cout << "init_net: " << FLAGS_init_net << std::endl;
  std::cout << "predict_net: " << FLAGS_predict_net << std::endl;
  std::cout << "size_to_fit: " << FLAGS_size_to_fit << std::endl;

  std::cout << std::endl;

  // Load the model
  NetDef init_net, predict_net;

  // >>> with open(path_to_INIT_NET) as f:
  CAFFE_ENFORCE(ReadProtoFromFile(FLAGS_init_net, &init_net));

  // >>> with open(path_to_PREDICT_NET) as f:
  CAFFE_ENFORCE(ReadProtoFromFile(FLAGS_predict_net, &predict_net));

  // >>> p = workspace.Predictor(init_net, predict_net)
  predict_net.set_name("PredictNet");
  
  predictor = new Predictor(init_net, predict_net);

  std::cout << "Caffe2 model initialized" << std::endl;
}
