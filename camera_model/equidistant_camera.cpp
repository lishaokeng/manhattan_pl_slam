#ifndef TTE_CAMERA_MODEL_KANNALA_BRANDT_CAMERA_CPP_
#define TTE_CAMERA_MODEL_KANNALA_BRANDT_CAMERA_CPP_

#include "equidistant_camera.h"

namespace camera_model {

/**
 * @param[in] no_distortion 是否畸变
 * @param[in] use_mask 是否使用mask(判断边界/slam跟踪时使用)
*/
KannalaBrandtCamera::KannalaBrandtCamera(const VOEquidistantCamera &camera_intrinsic, bool pyr_down, const std::string& mask_path)
    :CameraInterface(camera_intrinsic.camera_idx, ModelType::KANNALA_BRANDT, false, true, pyr_down),
     k1_(camera_intrinsic.k1), k2_(camera_intrinsic.k2), k3_(camera_intrinsic.k3), k4_(camera_intrinsic.k4), cv_distort_(k1_, k2_, k3_, k4_)
{
    CHECK_IF(!no_distortion_);
    CHECK_IF(use_mask_);

    int scale = pyr_down_ ? 2 : 1;
    image_width_scaled_ = camera_intrinsic.image_width / scale;
    image_height_scaled_ = camera_intrinsic.image_height / scale;
    image_width_src_ = camera_intrinsic.image_width;
    image_height_src_ = camera_intrinsic.image_height;

    fx_ = camera_intrinsic.fx / scale;
    fy_ = camera_intrinsic.fy / scale;
    cx_ = camera_intrinsic.cx / scale;
    cy_ = camera_intrinsic.cy / scale;

    MakeAtlantaParams();
    MakeMaskByReadOrReprojectError(mask_path);
}

/**
 * @brief 将归一化坐标系的点去除畸变
 * @param[in] point 归一化坐标系未去除畸变的点
 * @return 归一化坐标系去除畸变的点
*/
cv::Point2f KannalaBrandtCamera::UnDistortInNormalizedPlane(const cv::Point2f& pw)const
{
    // 照搬cv::undistortPoints()源码,它是用迭代的方式进行去畸变(耗时比较大但没办法)
    if(no_distortion_)return pw;

    double scale = 1.0;

    double theta_d = std::sqrt(pw.x*pw.x + pw.y*pw.y);

    // the current camera model is only valid up to 180 FOV
    // for larger FOV the loop below does not converge
    // clip values so we still get plausible results for super fisheye images > 180 grad
    theta_d = std::min(std::max(-CV_PI/2., theta_d), CV_PI/2.);

    if (theta_d > 1e-8)
    {
        // compensate distortion iteratively
        double theta = theta_d;

        const double EPS = 1e-8; // or std::numeric_limits<double>::epsilon();
        for (int j = 0; j < 10; j++)
        {
            double theta2 = theta*theta, theta4 = theta2*theta2, theta6 = theta4*theta2, theta8 = theta6*theta2;
            double k0_theta2 = k1_ * theta2, k1_theta4 = k2_ * theta4, k2_theta6 = k3_ * theta6, k3_theta8 = k4_ * theta8;
            /* new_theta = theta - theta_fix, theta_fix = f0(theta) / f0'(theta) */
            double theta_fix = (theta * (1 + k0_theta2 + k1_theta4 + k2_theta6 + k3_theta8) - theta_d) /
                                (1 + 3*k0_theta2 + 5*k1_theta4 + 7*k2_theta6 + 9*k3_theta8);
            theta = theta - theta_fix;
            if (fabs(theta_fix) < EPS)
                break;
        }

        scale = std::tan(theta) / theta_d;
    }

    return cv::Point2f(pw.x * scale, pw.y * scale); //undistorted point
}

/**
 * @brief 将归一化坐标系的点进行畸变
 * @param[in] point 归一化坐标系待畸变的点
 * @return 归一化坐标系畸变后的点
*/
cv::Point2f KannalaBrandtCamera::DistortInNormalizedPlane(const cv::Point2f& point)const
{
    if(no_distortion_)return point;
    double r = std::sqrt(point.x * point.x + point.y * point.y);
    double theta = std::atan(r);
    double theta_2 = theta * theta;
    double theta_3 = theta_2 * theta;
    double theta_5 = theta_3 * theta_2;
    double theta_7 = theta_5 * theta_2;
    double theta_9 = theta_7 * theta_2;
    double theta_d = theta + k1_ * theta_3 + k2_ * theta_5 + k3_ * theta_7 + k4_ * theta_9;
    return cv::Point2f(theta_d / r * point.x, theta_d / r * point.y);
}

}
#endif