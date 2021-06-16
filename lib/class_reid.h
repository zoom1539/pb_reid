#pragma once

#include "opencv2/opencv.hpp"

class ReID
{
public:
    explicit ReID();
    ~ReID();

    // 由于内部配置参数原因，目前只能为sbs_R50-ibn或sbs_R50
    bool init(const std::string &weights_path_, const std::string &engine_path_);

    bool extract_feature(const std::vector<cv::Mat> &imgs_, std::vector<cv::Mat> &features_);
    double compute_similarity(const cv::Mat &feature1_, const cv::Mat &feature2_);
    

private:
    ReID(const ReID &);
    const ReID &operator=(const ReID &);

    class Impl;
    Impl *_impl;
};
