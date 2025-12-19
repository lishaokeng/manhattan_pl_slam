#ifndef TTE_CAMERA_MODEL_KANNALA_BRANDT_CAMERA_H_
#define TTE_CAMERA_MODEL_KANNALA_BRANDT_CAMERA_H_

#include"camera_interface.h"

namespace camera_model {


class KannalaBrandtCamera: public CameraInterface {
public:
    KannalaBrandtCamera(const VOEquidistantCamera &camera_intrinsic, bool pyr_down, const std::string& mask_path = "");
    KannalaBrandtCamera(const KannalaBrandtCamera&) = delete;
    KannalaBrandtCamera&operator=(KannalaBrandtCamera&) = delete;
    virtual ~KannalaBrandtCamera()override{}

    // virtual cv::Mat UndistortImage(cv::Mat src)const
    // {
    //     CHECK_IF(src.type() == CV_8UC1);
    //     cv::Mat undistort_img;
    //     cv::fisheye::undistortImage(src, undistort_img, K_, cv_distort_, undist_new_K_, undist_new_size_);
    //     return undistort_img;
    // }

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


    double fx_, fy_, cx_, cy_;   // 相机内参K
    double k1_, k2_, k3_, k4_;   // 畸变系数k1,k2,k3,k4
    cv::Vec4d cv_distort_;       // 畸变系数k1,k2,k3,k4
};

}


#endif