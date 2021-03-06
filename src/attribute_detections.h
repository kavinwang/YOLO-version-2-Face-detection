#ifndef ATTRIBUTE_DETECTIONS
#define ATTRIBUTE_DETECTIONS

#include "box.h"
#include "image.h"

#ifdef __cplusplus
extern "C" {
#endif
void init_attributes();
void attribute_detections(const image *im, float thresh, box *boxes, float **probs, int num,
                          box **out_boxes, char **out_labels, int *detections_num);
#ifdef __cplusplus
}
#endif

#endif //ATTRIBUTE_DETECTIONS
