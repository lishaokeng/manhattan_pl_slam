#ifndef APP_BASED_ON_VO_VO_OPTICAL_FLOW_H
#define APP_BASED_ON_VO_VO_OPTICAL_FLOW_H

#include <vector>
#include "opencv2/opencv.hpp"
#include "Eigen/Core"
#include "Eigen/Dense"
#include "Eigen/Geometry"
#include <array>

#include "common/log_define.h"

namespace TTE {
namespace vo {

constexpr int kOpticalFlowMaxLevel = 3; // 金字塔层数,从0开始算,0表示第一层

// 用于光流跟踪的类(包含图像金字塔构建),比opencv快
class OpticalFlow{
public:
    OpticalFlow(int image_width, int image_height, bool is_resized);
    OpticalFlow(const OpticalFlow&) = delete;
    OpticalFlow&operator=(OpticalFlow&) = delete;

    // 将输入图片缩小一半, 并创建用于光流跟踪的后一个金字塔,返回金字塔第一张图像
    std::vector<cv::Mat> BuildPyramid(const cv::Mat &image);

    /**
     * @brief 光流跟踪
     * @param [in] last_distort_pts 上一帧的畸变后的特征点
     * @param [out] current_distort_pts 光流跟踪后的当前帧特征点
     * @param [out] status 跟踪后的每个特征点是否合格. true:合格; false:不合格
    */
    void OpticalFlowPyrLK(const std::vector<cv::Mat> &pyramid_old,
                          const std::vector<cv::Mat> &pyramid_new,
                          const std::vector<int> &last_levels,
                          const std::vector<cv::Point2f> &last_distort_pts, 
                          std::vector<cv::Point2f> &current_distort_pts,
                          std::vector<uchar> &status);
    
    // 单个特征点进行图像金字塔的光流跟踪
    bool TrackPoint(int target_level,
                    const std::vector<cv::Mat> &pyramid_old,
                    const cv::Point2f &last_distort_pt, 
                    const std::vector<cv::Mat> &pyramid_new,
                    cv::Point2f &current_distort_pt,
                    Eigen::Vector2f &translation,
                    int num_levels);

private:

    struct PyramidTmp{
        std::vector<ushort> data;
        int rows;
        int cols;
    };

    const int image_width_;
    const int image_height_;
    const bool is_resized_;
    std::vector<PyramidTmp> pyramid_tmp_;

    static constexpr int patch_size_ = 5; // 对patch_size_ X patch_size_像素块上的所有点进行光流跟踪
    static constexpr int patch_size_plus_1 = patch_size_ + 1;
    static constexpr int patch_size_plus_2 = patch_size_ + 2;
    static constexpr int patch_size_plus_3 = patch_size_ + 3;
    static constexpr int half_patch_size_ = patch_size_ / 2;
    static constexpr int patch_num_ = patch_size_ * patch_size_;
};




}
}

#endif