#ifndef TTE_CAMERA_MODEL_TTE_CAMERA_CPP
#define TTE_CAMERA_MODEL_TTE_CAMERA_CPP
#include"tte_camera.h"

namespace camera_model {

/**
 * @param[in] no_distortion 是否畸变
 * @param[in] use_mask 是否使用mask(判断边界/slam跟踪时使用)
*/
TTECamera::TTECamera(const VOTTECamera &camera_intrinsic, bool pyr_down, const std::string& mask_path)
    :CameraInterface(camera_intrinsic.camera_idx, ModelType::TTE, false, true, pyr_down)
{
    distort_param_[5] = camera_intrinsic.TYPE_AV[5];
    distort_param_[4] = camera_intrinsic.TYPE_AV[4];
    distort_param_[3] = camera_intrinsic.TYPE_AV[3];
    distort_param_[2] = camera_intrinsic.TYPE_AV[2];
    distort_param_[1] = camera_intrinsic.TYPE_AV[1];
    distort_param_[0] = camera_intrinsic.TYPE_AV[0];


    undistort_param_[5] = camera_intrinsic.TYPE_PH[5];
    undistort_param_[4] = camera_intrinsic.TYPE_PH[4];
    undistort_param_[3] = camera_intrinsic.TYPE_PH[3];
    undistort_param_[2] = camera_intrinsic.TYPE_PH[2];
    undistort_param_[1] = camera_intrinsic.TYPE_PH[1];
    undistort_param_[0] = camera_intrinsic.TYPE_PH[0];

    int scale = pyr_down_ ? 2 : 1;
    x_pixel_size_scaled_ = camera_intrinsic.XPixSize * scale;  // fx = f * x = 焦距 * pixel/mm
    y_pixel_size_scaled_= camera_intrinsic.YPixSize * scale;

    image_width_scaled_ = camera_intrinsic.image_width / scale;
    image_height_scaled_ = camera_intrinsic.image_height / scale;
    image_width_src_ = camera_intrinsic.image_width;
    image_height_src_ = camera_intrinsic.image_height;

    MakeAtlantaParams();
    MakeMaskByReadOrReprojectError(mask_path);
}

// void TTECamera::ReadParameters(const VisualOdometryParam& param, const std::string& mask_path)
// {
//     return;
//     int scale = pyr_down_ ? 2 : 1;
//     image_width_ = param.image_width / scale;
//     image_height_ = param.image_height / scale;
//     image_width_src_ = param.image_width;
//     image_height_src_ = param.image_height;
//     float x_pix_size = param.front_camera.f32HPixSize;
//     float y_pix_size = param.front_camera.f32VPixSize;
//     focal = param.focal; // 焦距
//     distort_param_[5] = param.front_camera.f64AV5;
//     distort_param_[4] = param.front_camera.f64AV4;
//     distort_param_[3] = param.front_camera.f64AV3;
//     distort_param_[2] = param.front_camera.f64AV2;
//     distort_param_[1] = param.front_camera.f64AV1;
//     distort_param_[0] = param.front_camera.f64AV0;

//     undistort_param_[5] = param.front_camera.f64PH5;
//     undistort_param_[4] = param.front_camera.f64PH4;
//     undistort_param_[3] = param.front_camera.f64PH3;
//     undistort_param_[2] = param.front_camera.f64PH2;
//     undistort_param_[1] = param.front_camera.f64PH1;
//     undistort_param_[0] = param.front_camera.f64PH0;

//     cx_ = image_width_ / 2;
//     cy_ = image_height_ / 2;
//     float m_fInImgPixelSizeX = x_pix_size * scale;
//     float m_fInImgPixelSizeY = y_pix_size * scale;
//     fx_ = focal / m_fInImgPixelSizeX; // fx = f * x = 焦距 * pixel/mm
//     fy_ = focal / m_fInImgPixelSizeY;
//     inv_K11_ = 1.0 / fx_;
//     inv_K13_ = -cx_ / fx_;
//     inv_K22_ = 1.0 / fy_;
//     inv_K23_ = -cy_ / fy_;
//     MakeMaskByReadOrReprojectError(mask_path);
// }

/**
 * @brief 解析相机内参并制作mask图,读参数方式不一样时继承此类并重写该函数,记得最后要制作mask图.
*/
// void TTECamera::ReadParameters(const std::string& param_file_path, const std::string& mask_path)
// {
//     cv::FileStorage fileread(param_file_path, cv::FileStorage::READ);
//     CHECK_IF(fileread.isOpened());

//     int scale = pyr_down_ ? 2 : 1;
//     int m_iOrigImgW = (int)fileread["originalimgW"];
//     int m_iOrigImgH = (int)fileread["originalimgH"];
//     image_width_scaled_ = (int)((int)fileread["inputimgW"] / scale);
//     image_height_scaled_ = (int)((int)fileread["inputimgH"] / scale);
//     image_width_src_ = (int)fileread["inputimgW"];
//     image_height_src_ = (int)fileread["inputimgH"];
//     float x_pix_size = (float)fileread["XPixSize"];
//     float y_pix_size = (float)fileread["YPixSize"];
//     focal = (float)fileread["focal"];
//     distort_param_[5] = (double)fileread["Timgdfav5"];
//     distort_param_[4] = (double)fileread["Timgdfav4"];
//     distort_param_[3] = (double)fileread["Timgdfav3"];
//     distort_param_[2] = (double)fileread["Timgdfav2"];
//     distort_param_[1] = (double)fileread["Timgdfav1"];
//     distort_param_[0] = (double)fileread["Timgdfav0"];

//     undistort_param_[5] = (double)fileread["Tvecdfav5"];
//     undistort_param_[4] = (double)fileread["Tvecdfav4"];
//     undistort_param_[3] = (double)fileread["Tvecdfav3"];
//     undistort_param_[2] = (double)fileread["Tvecdfav2"];
//     undistort_param_[1] = (double)fileread["Tvecdfav1"];
//     undistort_param_[0] = (double)fileread["Tvecdfav0"];

//     fileread.release();
//     cx_ = image_width_scaled_ / 2;
//     cy_ = image_height_scaled_ / 2;
//     float m_fInImgPixelSizeX = x_pix_size * m_iOrigImgW / image_width_scaled_;
//     float m_fInImgPixelSizeY = y_pix_size * m_iOrigImgH / image_height_scaled_;
//     fx_ = focal / m_fInImgPixelSizeX;
//     fy_ = focal / m_fInImgPixelSizeY;
//     inv_K11_ = 1.0 / fx_;
//     inv_K13_ = -cx_ / fx_;
//     inv_K22_ = 1.0 / fy_;
//     inv_K23_ = -cy_ / fy_;
//     MakeMaskByReadOrReprojectError(mask_path);
// }

/**
 * @brief 将归一化坐标系的点去除畸变
 * @param[in] point 归一化坐标系未去除畸变的点
 * @return 归一化坐标系去除畸变的点
*/
cv::Point2f TTECamera::PixelToNormalizedPlaneAndUndistort(const cv::Point2f& point)const
{
    double x = (point.x - image_width_scaled_ / 2) * x_pixel_size_scaled_;
    double y = (point.y - image_height_scaled_ / 2) * y_pixel_size_scaled_;
    double rd = sqrt(x * x + y * y);
    if (rd == 0.0 || no_distortion_) return point;

    double rd2 = rd * rd;
    double rd3 = rd2 * rd;
    double rd4 = rd3 * rd;
    double rd5 = rd4 * rd;
    double theta = undistort_param_[5] * rd5 + 
                   undistort_param_[4] * rd4 + 
                   undistort_param_[3] * rd3 +
                   undistort_param_[2] * rd2 + 
                   undistort_param_[1] * rd + 
                   undistort_param_[0];
    theta *= M_PI / 180;
    double tmp = fabs(tanf(theta)) / rd;
    return cv::Point2f(x * tmp, y * tmp);
}

/**
 * @brief 将归一化坐标系的点进行畸变
 * @param[in] point 归一化坐标系待畸变的点
 * @return 归一化坐标系畸变后的点
*/
cv::Point2f TTECamera::DistortInNormalizedPlaneAndProjectToPixelPlane(const cv::Point2f& point)const
{
    double r = sqrt(point.x * point.x + point.y * point.y);
    if (r == 0.0 || no_distortion_) return point;
    double theta = atanf(r);
    theta *= 180 / M_PI;
    double theta2 = theta * theta;
    double theta3 = theta2 * theta;
    double theta4 = theta3 * theta;
    double theta5 = theta4 * theta;
    double rd = distort_param_[5] * theta5 + 
                distort_param_[4] * theta4 + 
                distort_param_[3] * theta3 +
                distort_param_[2] * theta2 + 
                distort_param_[1] * theta + 
                distort_param_[0];
    double tmp = rd / r;
    return cv::Point2f(point.x * tmp / x_pixel_size_scaled_ + image_width_scaled_ / 2, 
                       point.y * tmp / y_pixel_size_scaled_ + image_height_scaled_ / 2);
}

}


#endif