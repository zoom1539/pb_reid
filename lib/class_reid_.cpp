#include "class_reid_.h"


using namespace fastrt;
using namespace nvinfer1;

static const int MAX_BATCH_SIZE = 4;
static const int INPUT_H = 384;
static const int INPUT_W = 128;
static const int OUTPUT_SIZE = 2048;
static const int DEVICE_ID = 0;

static const FastreidBackboneType BACKBONE = FastreidBackboneType::r50; 
static const FastreidHeadType HEAD = FastreidHeadType::EmbeddingHead;
static const FastreidPoolingType HEAD_POOLING = FastreidPoolingType::gempoolP;
static const int LAST_STRIDE = 1;
static const bool WITH_IBNA = true; 
static const bool WITH_NL = true;
static const int EMBEDDING_DIM = 0;

_ReID::_ReID() {}
_ReID::~_ReID() 
{
    if (_baseline)
    {
        delete _baseline;
        _baseline = nullptr;
    }
}

bool _ReID::init(const std::string &weights_path_, const std::string &engine_path_)
{
    trt::ModelConfig modelCfg { 
        weights_path_,
        MAX_BATCH_SIZE,
        INPUT_H,
        INPUT_W,
        OUTPUT_SIZE,
        DEVICE_ID};

    FastreidConfig reidCfg { 
        BACKBONE,
        HEAD,
        HEAD_POOLING,
        LAST_STRIDE,
        WITH_IBNA,
        WITH_NL,
        EMBEDDING_DIM};

    _baseline = new Baseline(modelCfg);
    
    if(!_baseline->deserializeEngine(engine_path_)) 
    {
        std::cout << "DeserializeEngine Failed." << std::endl;
        return false;
    }
    else
    {
        return true;
    }
}

bool _ReID::extract_feature(const std::vector<cv::Mat> &imgs_, 
                           std::vector<cv::Mat> &features_)
{
        for (size_t batch_start = 0; batch_start < imgs_.size(); batch_start+=MAX_BATCH_SIZE) 
        {
            std::vector<cv::Mat> input;

            int img_idx = 0;
            for (img_idx = 0; img_idx < MAX_BATCH_SIZE; ++img_idx) 
            {
                if ( (batch_start + img_idx) >= imgs_.size() ) 
                {
                    break; 
                }

                cv::Mat resizeImg(INPUT_H, INPUT_W, CV_8UC3);

                cv::resize(imgs_[batch_start + img_idx], resizeImg, resizeImg.size(), 0, 0, cv::INTER_CUBIC); /* cv::INTER_LINEAR */
                
                input.emplace_back(resizeImg);
            }

            if(!_baseline->inference(input)) 
            {
                std::cout << "Inference Failed." << std::endl;
                return false;
            }

            // output
            float* feat_embedding = _baseline->getOutput();

            TRTASSERT(feat_embedding);
            int feature_num = _baseline->getOutputSize();
            for (size_t i = 0; i < img_idx; ++i) 
            {
                int id_start = i * feature_num;
                
                cv::Mat feature;
                for (int dim = 0; dim < _baseline->getOutputSize(); ++dim) 
                {
                    feature.push_back(feat_embedding[id_start+dim]);
                }
                features_.push_back(feature);
            }

        }

        return true;
}

double _ReID::compute_similarity(const cv::Mat &feature1_, const cv::Mat &feature2_)
{
    double ab = feature1_.dot(feature2_);
    double aa = feature1_.dot(feature1_);
    double bb = feature2_.dot(feature2_);
    double sim = ab / (sqrt(aa * bb) + DBL_EPSILON);

    return sim;
}


