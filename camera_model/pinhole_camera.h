#ifndef TTE_CAMERA_MODEL_PINHOLE_CAMERA_H
#define TTE_CAMERA_MODEL_PINHOLE_CAMERA_H
#include"camera_interface.h"

namespace camera_model {


class PinholeCamera: public CameraInterface {
public:
    PinholeCamera(const VOPinholeCamera &camera_intrinsic, bool pyr_down, const std::string& mask_path = "");
    PinholeCamera(const PinholeCamera&) = delete;
    PinholeCamera&operator=(PinholeCamera&) = delete;
   virtual ~PinholeCamera()override{}

    /**
     *@brief 从像素投影到归一化坐标系后并去畸变(畸变像素->投影到归一化坐标系->在归一化坐标系去畸变) 
    *@param[in] distort_pixel 像素坐标系畸变的点
    * @return 归一化坐标系去畸变后的点
    */
    virtual cv::Point2f PixelToNormalizedPlaneAndUndistort(const cv::Point2f& distort_pixel)const override
    {
        return UnDistortInNormalizedPlane(PixelToNormalizedPlane(distort_pixel));
    }

    virtual cv::Point2f DistortInNormalizedPlaneAndProjectToPixelPlane(const cv::Point2f &point)const override
    {
        return NormalizedToPixelPlane(DistortInNormalizedPlane(point));
    }

    virtual double fx()const{return fx_;}

private:

    /**
     * @brief 从像素坐标系投影到归一化坐标系
     * @param[in] point 像素坐标系的点
     * @return 归一化坐标系的点
    */
    cv::Point2f PixelToNormalizedPlane(const cv::Point2f& point)const{
        return cv::Point2f((point.x - cx_) / fx_, (point.y - cy_) / fy_);}

    /**
     * @brief 归一化坐标系投影到像素坐标系
     * @param[in] point 归一化坐标系的点
     * @return 像素坐标系的点
    */
    inline cv::Point2f NormalizedToPixelPlane(const cv::Point2f& point)const{
        return cv::Point2f(fx_ * point.x + cx_, fy_ * point.y + cy_);}

    cv::Point2f DistortInNormalizedPlane(const cv::Point2f& point)const;
    cv::Point2f UnDistortInNormalizedPlane(const cv::Point2f& point)const;


    bool use_inverse_distortion_model_; // true:反畸变模型去畸变; false:迭代方式去畸变
    double fx_, fy_, cx_, cy_;          // 相机内参K
    double k1_, k2_, p1_, p2_;          // 畸变系数k1,k2,p1,p2
    cv::Vec4d cv_distort_;              // 畸变系数k1,k2,p1,p2
};

}



#endif