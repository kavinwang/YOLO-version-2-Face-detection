#ifndef ATTRIBUTE_DETECTIONS
#define ATTRIBUTE_DETECTIONS

#include "box.h"
#include "image.h"

#ifdef __cplusplus
extern "C" {
#endif
void attribute_detections(const image *im, box *boxes);
#ifdef __cplusplus
}
#endif

#endif //ATTRIBUTE_DETECTIONS
