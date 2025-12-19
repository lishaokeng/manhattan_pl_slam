#ifndef TTE_CAMERA_MODEL_TTE_CAMERA_H
#define TTE_CAMERA_MODEL_TTE_CAMERA_H


#include"camera_interface.h"

namespace camera_model {


class TTECamera: public CameraInterface {
public:
    TTECamera(const VOTTECamera &camera_intrinsic, bool pyr_down, const std::string& mask_path = "");
    TTECamera(const TTECamera&) = delete;
    TTECamera&operator=(TTECamera&) = delete;
    virtual ~TTECamera()override{}

    /**
     *@brief 从像素投影到归一化坐标系后并去畸变(畸变像素->投影到归一化坐标系->在归一化坐标系去畸变) 
    *@param[in] distort_pixel 像素坐标系畸变的点
    * @return 归一化坐标系去畸变后的点
    */
    virtual cv::Point2f PixelToNormalizedPlaneAndUndistort(const cv::Point2f& distort_pixel)const override;

    virtual cv::Point2f DistortInNormalizedPlaneAndProjectToPixelPlane(const cv::Point2f &point)const override;

    virtual double fx()const{return (pyr_down_ ? 310 / 2 : 310);}

private:

    // 畸变、去畸变五次多项式参数
    double distort_param_[6];
    double undistort_param_[6];
    double x_pixel_size_scaled_;
    double y_pixel_size_scaled_;
};

}


#endif