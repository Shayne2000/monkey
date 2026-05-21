#include <algorithm>
#include <cmath>
#include <iostream>
#include <vector>

#include "nvdsinfer_custom_impl.h"

#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define MAX(a, b) ((a) > (b) ? (a) : (b))
#define CLIP(a, low, high) (MAX(MIN(a, high), low))

extern "C" bool NvDsInferParseYolo(
    std::vector<NvDsInferLayerInfo> const& outputLayersInfo,
    NvDsInferNetworkInfo const& networkInfo,
    NvDsInferParseDetectionParams const& detectionParams,
    std::vector<NvDsInferParseObjectInfo>& objectList)
{
    if (outputLayersInfo.empty()) {
        std::cerr << "Could not find YOLO output layer" << std::endl;
        return false;
    }

    const NvDsInferLayerInfo& output = outputLayersInfo[0];
    const int numClasses = detectionParams.numClassesConfigured > 0
        ? detectionParams.numClassesConfigured
        : 80;
    const int dimensions = 4 + numClasses;
    int numAnchors = output.inferDims.numElements / dimensions;
    if (numAnchors <= 0) {
        std::cerr << "Invalid YOLO output dimensions" << std::endl;
        return false;
    }

    float* outputData = static_cast<float*>(output.buffer);

    for (int i = 0; i < numAnchors; ++i) {
        float maxProb = 0.0f;
        int maxClass = -1;

        for (int c = 0; c < numClasses; ++c) {
            float prob = outputData[(4 + c) * numAnchors + i];
            if (prob > maxProb) {
                maxProb = prob;
                maxClass = c;
            }
        }

        if (maxClass < 0) {
            continue;
        }

        float threshold = detectionParams.perClassPreclusterThreshold[maxClass];
        if (maxProb < threshold) {
            continue;
        }

        float xc = outputData[0 * numAnchors + i];
        float yc = outputData[1 * numAnchors + i];
        float width = outputData[2 * numAnchors + i];
        float height = outputData[3 * numAnchors + i];
        float left = xc - width / 2.0f;
        float top = yc - height / 2.0f;

        NvDsInferParseObjectInfo obj;
        obj.classId = maxClass;
        obj.detectionConfidence = maxProb;
        obj.left = CLIP(left, 0, networkInfo.width - 1);
        obj.top = CLIP(top, 0, networkInfo.height - 1);
        obj.width = CLIP(width, 0, networkInfo.width - 1);
        obj.height = CLIP(height, 0, networkInfo.height - 1);
        objectList.push_back(obj);
    }

    return true;
}

extern "C" bool NvDsInferParseCustomYolo(
    std::vector<NvDsInferLayerInfo> const& outputLayersInfo,
    NvDsInferNetworkInfo const& networkInfo,
    NvDsInferParseDetectionParams const& detectionParams,
    std::vector<NvDsInferParseObjectInfo>& objectList)
{
    return NvDsInferParseYolo(outputLayersInfo, networkInfo, detectionParams, objectList);
}
