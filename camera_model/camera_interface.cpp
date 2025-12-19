#ifndef TTE_CAMERA_MODEL_CAMERA_INTERFACE_CPP
#define TTE_CAMERA_MODEL_CAMERA_INTERFACE_CPP
#include "camera_interface.h"
#include "common/interpolation.h"

namespace camera_model {

void CameraInterface::MakeAtlantaParams()
{
    int scale = pyr_down_ ? 2 : 1;
    float scale_height = (camera_idx_ == FRONT ? kAtlantaScaleHeightFront : kAtlantaScaleHeightOther);
    atlanta_undist_new_K_ << 310 / scale / kAtlantaScaleFxFy, 0, image_width_scaled_ / kAtlantaScaleFxFy, 
                             0, 310 / scale / kAtlantaScaleFxFy, image_height_scaled_ / scale_height, 
                             0, 0, 1;
    atlanta_undist_new_size_ = cv::Size(image_width_scaled_, image_height_scaled_ / kAtlantaScaleFxFy);

    atlanta_undist_table_.resize(atlanta_undist_new_size_.width * atlanta_undist_new_size_.height, cv::Point2f(-1, -1));
    for(int row = 0; row < atlanta_undist_new_size_.height; ++row)
    {
        auto cols = row * atlanta_undist_new_size_.width;
        for(int col = 0; col < atlanta_undist_new_size_.width; ++col)
        {
            auto x = (col - atlanta_undist_new_K_(0, 2)) / atlanta_undist_new_K_(0, 0);
            auto y = (row - atlanta_undist_new_K_(1, 2)) / atlanta_undist_new_K_(1, 1);
            cv::Point2f point = DistortInNormalizedPlaneAndProjectToPixelPlane(cv::Point2f(x, y));
            atlanta_undist_table_[cols + col] = point;
        }
    }
}

void CameraInterface::MakeMaskByReadOrReprojectError(const std::string& mask_path)
{
    if(!use_mask_)return;
    CHECK_IF(image_height_scaled_ > 0 && image_width_scaled_ > 0);
    if(mask_path != "")
        mask_ = cv::imread(mask_path, cv::IMREAD_GRAYSCALE);
    if(!mask_.empty() && mask_.rows == image_height_scaled_ && mask_.cols == image_width_scaled_)
    {
        #ifdef __DEBUG_TOOLS__
            cv::imshow("mask", mask_);
            cv::waitKey(1000);
            cv::destroyWindow("mask");
        #endif
        return;
    }
    mask_ = cv::Mat(image_height_scaled_, image_width_scaled_, CV_8UC1, cv::Scalar::all(0));
    const float threshold_err = 0.5;
    int left = image_width_scaled_;
    int right = 0;
    for(int row = 0; row < mask_.rows; ++row)
    {
        for(int col = 0; col < mask_.cols; ++col)
        {
            cv::Point2f p = DistortInNormalizedPlaneAndProjectToPixelPlane(PixelToNormalizedPlaneAndUndistort(cv::Point2f(col, row)));
            float err = std::sqrt((row - p.y) * (row - p.y) + (col - p.x) * (col - p.x));
            if(p.x >= 0 && p.x <= image_width_scaled_ - 1 && p.y >= 0 && p.y <= image_height_scaled_ - 1 && err < threshold_err)
            {
                mask_.at<uchar>(row, col) = 255;
                if(col < left) left = col;
                if(col > right) right = col;
            }
        }
    }
    int scale = pyr_down_? 2 : 1;
    if(model_type_ == ModelType::TTE || model_type_ == ModelType::KANNALA_BRANDT)
    {
        int border_up = std::ceil(50 / scale);
        int border_down = std::ceil(320 / scale);
        int border_left = std::ceil(120 / scale);
        int border_right = std::ceil(120 / scale);
        cv::rectangle(mask_, cv::Point2f(0, 0), cv::Point2f(mask_.cols, border_up),cv::Scalar::all(0),-1);
        cv::rectangle(mask_, cv::Point2f(0, mask_.rows - border_down), cv::Point2f(mask_.cols, mask_.rows),cv::Scalar::all(0),-1);
        cv::rectangle(mask_, cv::Point2f(0, 0), cv::Point2f(border_left, mask_.rows),cv::Scalar::all(0),-1);
        cv::rectangle(mask_, cv::Point2f(mask_.cols - border_right, 0), cv::Point2f(mask_.cols, mask_.rows),cv::Scalar::all(0),-1);
    }

    cv::circle(mask_, cv::Point(mask_.cols/2, mask_.rows/2), (right-left)/2 * 0.6, cv::Scalar(255), -1);
    // #ifdef __DEBUG_TOOLS__
    //     cv::imshow("mask", mask_);
    //     cv::waitKey(1000);
    //     cv::destroyWindow("mask");
    // #endif
}


void CameraInterface::AtlantaUndistortImage(cv::Mat src, cv::Mat &result, bool is_atlanta, const cv::Matx33d &new_K, 
                                            const cv::Size &new_size)const
{
    CHECK_IF(src.type() == CV_8UC1);
    cv::Matx33d K = new_K;
    cv::Size size = new_size;
    if(is_atlanta)
    {
        K = atlanta_undist_new_K_;
        size = atlanta_undist_new_size_;
    }
    CHECK_IF(result.size() == size);
    CHECK_IF(result.type() == CV_8UC1);
    result = cv::Mat(result.size(), CV_8UC1, cv::Scalar(0));
    if(is_atlanta)
    {
        for(int row = 0; row < result.rows; ++row)
        {
            int cols = row * result.cols;
            auto ptr = &result.data[cols];
            for(int col = 0; col < result.cols; ++col)
            {
                cv::Point2f point = atlanta_undist_table_[cols + col];
                bool in_border = border_size_ <= point.x && point.x < image_width_scaled_ - border_size_ && 
                                border_size_ <= point.y && point.y < image_height_scaled_ - border_size_;
                // 双线性插值更平滑但上位机i5速度慢0.6ms左右.不插值更粗糙,测试看到elsed不插值提取线段效果好像也还可以.
                if(in_border) ptr[col] = static_cast<uchar>(std::round(common::interpolate(src, point.x, point.y)));
                // if(in_border) ptr[col] = src.data[int(std::round(point.y) * src.cols + std::round(point.x))];
            }
        }
    }
    else
    {
        for(int row = 0; row < result.rows; ++row)
        {
            auto ptr = result.ptr<uchar>(row);
            for(int col = 0; col < result.cols; ++col)
            {
                auto x = (col - K(0, 2)) / K(0, 0);
                auto y = (row - K(1, 2)) / K(1, 1);
                cv::Point2f point = DistortInNormalizedPlaneAndProjectToPixelPlane(cv::Point2f(x, y));
                bool in_border = border_size_ <= point.x && point.x < image_width_scaled_ - border_size_ && 
                                border_size_ <= point.y && point.y < image_height_scaled_ - border_size_;
                if(in_border) result.at<uchar>(row, col) = static_cast<uchar>(std::round(common::interpolate(src, point.x, point.y)));
            }
        }
    } 
    return;
}


}



#endif
