#ifndef TTE_CAMERA_MODEL_PINHOLE_CAMERA_CPP
#define TTE_CAMERA_MODEL_PINHOLE_CAMERA_CPP
#include"pinhole_camera.h"

namespace camera_model {

/**
 * @param[in] no_distortion 是否畸变
 * @param[in] use_mask 是否使用mask(判断边界/slam跟踪时使用)
*/
PinholeCamera::PinholeCamera(const VOPinholeCamera &camera_intrinsic, bool pyr_down, const std::string& mask_path)
    :CameraInterface(camera_intrinsic.camera_idx, ModelType::PINHOLE, !camera_intrinsic.use_distort, true, pyr_down),
     k1_(camera_intrinsic.k1), k2_(camera_intrinsic.k2), p1_(camera_intrinsic.p1), p2_(camera_intrinsic.p2), cv_distort_(k1_, k2_, p1_, p2_)
{
    use_inverse_distortion_model_ = true;

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
cv::Point2f PinholeCamera::UnDistortInNormalizedPlane(const cv::Point2f& point)const
{
    if(no_distortion_)return point;
    double mx2_d, mxy_d, my2_d, mx_u, my_u;
    double rho2_d, rho4_d, radDist_d, Dx_d, Dy_d, inv_denom_d;

    // 使用畸变模型的逆变换来去畸变
    if(use_inverse_distortion_model_)
    {
        mx2_d = point.x * point.x;
        my2_d = point.y * point.y;
        mxy_d = point.x * point.y;
        rho2_d = mx2_d + my2_d;
        rho4_d = rho2_d * rho2_d;
        radDist_d = k1_ * rho2_d + k2_ * rho4_d;
        Dx_d = point.x * radDist_d + p2_ * (rho2_d + 2 * mx2_d) + 2 * p1_ * mxy_d;
        Dy_d = point.y * radDist_d + p1_ * (rho2_d + 2 * my2_d) + 2 * p2_ * mxy_d;
        inv_denom_d = 1 / (1 + 4 * k1_ * rho2_d + 
                           6 * k2_ * rho4_d + 8 * p1_ * point.y + 8 * p2_ * point.x);
        mx_u = point.x - inv_denom_d * Dx_d;
        my_u = point.y - inv_denom_d * Dy_d;
    }
    else
    {
        // 使用迭代方式去畸变
        int n = 8;
        cv::Point2f d_u = DistortInNormalizedPlane(point);
        // Approximate value
        mx_u = point.x - d_u.x;
        my_u = point.y - d_u.y;

        for (int i = 1; i < n; ++i)
        {
            d_u = DistortInNormalizedPlane(cv::Point2f(mx_u, my_u));
            mx_u = point.x - d_u.x;
            my_u = point.y - d_u.y;
        }
    }
    return cv::Point2f(mx_u, my_u);
}

/**
 * @brief 将归一化坐标系的点进行畸变
 * @param[in] point 归一化坐标系待畸变的点
 * @return 归一化坐标系畸变后的点
*/
cv::Point2f PinholeCamera::DistortInNormalizedPlane(const cv::Point2f& point)const
{
    if(no_distortion_)return point;
    double mx2_u, my2_u, mxy_u, rho2_u, rad_dist_u;

    mx2_u = point.x * point.x;
    my2_u = point.y * point.y;
    mxy_u = point.x * point.y;
    rho2_u = mx2_u + my2_u;
    rad_dist_u = k1_ * rho2_u + k2_ * rho2_u * rho2_u;
    return cv::Point2f(point.x * rad_dist_u + 2.0 * p1_ * mxy_u + p2_ * (rho2_u + 2.0 * mx2_u),
                       point.y * rad_dist_u + 2.0 * p2_ * mxy_u + p1_ * (rho2_u + 2.0 * my2_u));
}

}


#endif