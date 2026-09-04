#include <iostream>
#include <vector>
#include <algorithm>
#include "nvdsinfer_custom_impl.h"

extern "C" bool NvDsInferParseYolo11(
    std::vector<NvDsInferLayerInfo> const& outputLayersInfo,
    NvDsInferNetworkInfo const& networkInfo,
    NvDsInferParseDetectionParams const& detectionParams,
    std::vector<NvDsInferParseObjectInfo>& objectList)
{
    if (outputLayersInfo.empty()) {
        std::cerr << "Could not find output layer" << std::endl;
        return false;
    }

    const NvDsInferLayerInfo& output = outputLayersInfo[0];
    const float* out_ptr = (const float*)output.buffer;

    int num_classes = 80;
    int num_anchors = 8400; 

    if (output.inferDims.numDims >= 2) {
        if (output.inferDims.d[0] == 84) {
            num_anchors = output.inferDims.d[1];
        } else if (output.inferDims.d[1] == 84) {
            // Unlikely, but if it is [8400, 84] format
            std::cerr << "Transposed tensor!" << std::endl;
            return false;
        }
    }

    float netW = networkInfo.width;
    float netH = networkInfo.height;

    for (int i = 0; i < num_anchors; ++i) {
        float max_prob = 0.0f;
        int max_index = -1;

        for (int c = 0; c < num_classes; ++c) {
            float prob = out_ptr[(4 + c) * num_anchors + i];
            if (prob > max_prob) {
                max_prob = prob;
                max_index = c;
            }
        }

        // YOLOv8/11 outputs class probabilities directly, not multiplied by objectness
        if (max_prob < detectionParams.perClassPreclusterThreshold[max_index]) {
            continue;
        }

        float cx = out_ptr[0 * num_anchors + i];
        float cy = out_ptr[1 * num_anchors + i];
        float w  = out_ptr[2 * num_anchors + i];
        float h  = out_ptr[3 * num_anchors + i];

        NvDsInferParseObjectInfo obj;
        obj.classId = max_index;
        obj.detectionConfidence = max_prob;

        float left = cx - w / 2.0f;
        float top = cy - h / 2.0f;
        
        obj.left = std::max(0.0f, left);
        obj.top = std::max(0.0f, top);
        obj.width = std::min(netW - obj.left, w);
        obj.height = std::min(netH - obj.top, h);

        objectList.push_back(obj);
    }

    return true;
}

CHECK_CUSTOM_PARSE_FUNC_PROTOTYPE(NvDsInferParseYolo11);
