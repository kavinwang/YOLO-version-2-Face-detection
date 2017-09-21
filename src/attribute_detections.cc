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

CAFFE2_DEFINE_int(size_to_fit, 224, "The image file.");

caffe2::Predictor *gender_predictor;
caffe2::Predictor *age_predictor;

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

  const char *gender_names[] = {"Female", "Male"};
  int gender_class = 0;

  {
    // Predict gender.
    Predictor::TensorVector inputVec({&input}), outputVec;

    gender_predictor->run(inputVec, &outputVec);
    auto &gender_output = *(outputVec[0]);

    const auto &gender_probs = gender_output.data<float>();
    assert(gender_output.size() == 2);
    
    gender_class = (gender_probs[0] > gender_probs[1] ? 0 : 1);
    std::cout << gender_names[gender_class] << " (" << gender_probs[gender_class] * 100 << "%)" << std::endl;
  }

  char age_range_str[256];

  {
    // Predict age.
    Predictor::TensorVector inputVec({&input}), outputVec;

    age_predictor->run(inputVec, &outputVec);
    auto &age_output = *(outputVec[0]);

    const auto &age_probs = age_output.data<float>();

    float age_prob_threshold = 0.08;
    float age_max_prob = 0.0;
    int age_max_i = 0;
    for (auto i = 0; i < age_output.size(); i++) {
      if (age_probs[i] > age_max_prob) {
        age_max_prob = age_probs[i];
        age_max_i = i;
      }
    }

    const int age_range = 6;
    int age_left = age_max_i;
    int age_right = age_max_i;
    while (age_left > 0 && age_max_i - age_left < age_range/2 && fabs(age_probs[age_left-1] - age_max_prob) < age_prob_threshold)
      age_left--;
    while (age_right+1 < age_output.size() && age_right - age_max_i < age_range/2 && fabs(age_probs[age_right+1] - age_max_prob) < age_prob_threshold)
      age_right++;

    if (age_left == age_right)
      std::cout << age_max_i << " yo: " << age_max_prob * 100 << "%" << std::endl;
    else
      std::cout << age_left << "-" << age_right << " yo: " << age_max_prob * 100 << "%" << std::endl;

    if (age_left == age_right)
      sprintf(age_range_str, "%d", age_max_i);
    else
      sprintf(age_range_str, "%d-%d", age_left, age_right);
  }

  char buf[256];
  sprintf(buf, "%s, %s yo", gender_names[gender_class], age_range_str);
  return strdup(buf);
}

static void init_gender_net()
{
  std::cout << "Initializing Caffe2 gender prediction net" << std::endl;

  std::string name_init_net = "/Users/evgenybaskakov/Dropbox (Personal)/caffe2_gender/init_net.pb";
  std::string name_predict_net = "/Users/evgenybaskakov/Dropbox (Personal)/caffe2_gender/predict_net.pb";

  if (!std::ifstream(name_init_net).good() || !std::ifstream(name_predict_net).good()) {
    std::cerr << "error: model file missing: "
              << (std::ifstream(name_init_net).good() ? name_predict_net : name_init_net)
              << std::endl;
    abort();
  }

  std::cout << "init_net: " << name_init_net << std::endl;
  std::cout << "predict_net: " << name_predict_net << std::endl;
  std::cout << "size_to_fit: " << FLAGS_size_to_fit << std::endl;

  NetDef init_net, predict_net;

  CAFFE_ENFORCE(ReadProtoFromFile(name_init_net, &init_net));
  CAFFE_ENFORCE(ReadProtoFromFile(name_predict_net, &predict_net));

  predict_net.set_name("GenderPredictionNet");
  
  gender_predictor = new Predictor(init_net, predict_net);
}

static void init_age_net()
{
  std::cout << "Initializing Caffe2 age prediction net" << std::endl;

  std::string name_init_net = "/Users/evgenybaskakov/Dropbox (Personal)/caffe2_age/init_net.pb";
  std::string name_predict_net = "/Users/evgenybaskakov/Dropbox (Personal)/caffe2_age/predict_net.pb";

  if (!std::ifstream(name_init_net).good() || !std::ifstream(name_predict_net).good()) {
    std::cerr << "error: model file missing: "
              << (std::ifstream(name_init_net).good() ? name_predict_net : name_init_net)
              << std::endl;
    abort();
  }

  std::cout << "init_net: " << name_init_net << std::endl;
  std::cout << "predict_net: " << name_predict_net << std::endl;
  std::cout << "size_to_fit: " << FLAGS_size_to_fit << std::endl;

  NetDef init_net, predict_net;

  CAFFE_ENFORCE(ReadProtoFromFile(name_init_net, &init_net));
  CAFFE_ENFORCE(ReadProtoFromFile(name_predict_net, &predict_net));

  predict_net.set_name("AgePredictionNet");
  
  age_predictor = new Predictor(init_net, predict_net);
}

void init_attributes()
{
  init_gender_net();
  init_age_net();

  std::cout << "Caffe2 model initialized" << std::endl;
}
