#include "class_reid.h"
#include "class_reid_.h"

class ReID::Impl
{
public:
    _ReID _reid;
};

ReID::ReID() : _impl(new ReID::Impl())
{
}

ReID::~ReID()
{
    delete _impl;
    _impl = NULL;
}

bool ReID::init(const std::string &weights_path_, const std::string &engine_path_)
{
    return _impl->_reid.init(weights_path_, engine_path_);
}

bool ReID::extract_feature(const std::vector<cv::Mat> &imgs_, std::vector<cv::Mat> &features_)
{
    return _impl->_reid.extract_feature(imgs_, features_);
}

double ReID::compute_similarity(const cv::Mat &feature1_, const cv::Mat &feature2_)
{
    return _impl->_reid.compute_similarity(feature1_, feature2_);
}