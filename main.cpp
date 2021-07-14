#include <sstream>
#include <iostream>
#include <fstream>
#include <string>
#include <chrono>
#include "class_reid.h"

int main()
{
    ReID reid;

    //
    std::string engine_path = "../lib/extra/fastrt/sbs_R50-ibn.engine";
    bool is_init = reid.init(engine_path);
    if(!is_init)
    {
        std::cout << "init fail\n";
        return 0;
    }

    //
    std::vector<cv::Mat> imgs;
    {
        cv::Mat img = cv::imread("../data/pose2.png");
        imgs.push_back(img);
    }
    {
        cv::Mat img = cv::imread("../data/pose11.png");
        imgs.push_back(img);
    }
    

    auto start1 = std::chrono::system_clock::now();


    std::vector<cv::Mat> features;
    bool is_extract = reid.extract_feature(imgs, features);

    if(!is_extract)
    {
        std::cout << "extract fail\n";
        return 0;
    }
    
    auto end1 = std::chrono::system_clock::now();
    std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(end1 - start1).count() << " ms" << std::endl;
        

    //
    double sim = reid.compute_similarity(features[0], features[1]);
    std::cout << sim << std::endl;
        
    
    std::cin.get();
    return 0;
}

