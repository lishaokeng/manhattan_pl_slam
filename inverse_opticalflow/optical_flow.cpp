#ifndef APP_BASED_ON_VO_VO_OPTICAL_FLOW_CPP
#define APP_BASED_ON_VO_VO_OPTICAL_FLOW_CPP

#include "optical_flow.h"
#include "common/interpolation.h"

namespace TTE {
namespace vo {

// 单个特征点进行图像金字塔的光流跟踪
bool OpticalFlow::TrackPoint(int target_level,
                             const std::vector<cv::Mat> &pyramid_old,
                             const cv::Point2f &last_distort_pt, 
                             const std::vector<cv::Mat> &pyramid_new,
                             cv::Point2f &current_distort_pt,
                             Eigen::Vector2f &translation,
                             int num_levels)
{
    // 1.若待跟踪的特征点在边界则返回false
    const int bounder = ((half_patch_size_ + 2) << (num_levels - 1));
    if(unlikely(last_distort_pt.x < bounder || last_distort_pt.x + bounder > pyramid_old[0].cols || 
                last_distort_pt.y < bounder || last_distort_pt.y + bounder > pyramid_old[0].rows))
        return false;
    if(std::isnan(translation[0]) || std::isnan(translation[1])) return false;
    if(std::isnan(last_distort_pt.x) || std::isnan(last_distort_pt.y)) return false;
    else
    {
        // 2.遍历图像金字塔进行光流跟踪
        bool valid = true;
        for(int j = num_levels - 1; valid && j >= target_level; --j)
        {
            // 2.1 获取该层金字塔待跟踪特征点的坐标以及光流结果,需要对光流结果进行迭代更新
            const int scale = 1 << j;
            const int cols = pyramid_old[j].cols;
            float init_x = last_distort_pt.x / scale;
            float init_y = last_distort_pt.y / scale;
            translation /= scale;
            std::array<uchar*, patch_size_plus_3> old_patch_plus3_pixel;  // 老图像的像素块往外拓宽3(求梯度拓宽2, 亚像素再拓宽1)
            std::array<uchar*, patch_size_plus_1> new_patch_plus1_pixel;  // 新图像的像素块往外拓宽1(计算亚像素需要拓宽1)
            std::array<std::array<float, patch_size_plus_2>, patch_size_plus_2> old_patch_plus2_subpixel; // 老图像的像素块上的亚像素值往外拓宽2(求梯度拓宽2)

            // 2.2 获取老图像的像素块上往外拓宽3的像素值
            int i_x = int(init_x);
            int i_y = int(init_y);
            old_patch_plus3_pixel[0] = &pyramid_old[j].data[(i_y - half_patch_size_ - 1) * cols + (i_x - half_patch_size_ - 1)];
            for(int row = 1; row < patch_size_plus_3; ++row)
                old_patch_plus3_pixel[row] = old_patch_plus3_pixel[row-1] + cols;

            // 2.3 获取亚像素时需要双线性插值,提前计算双线性插值需要的几个数据(避免每个数据都计算一次会增加耗时)
            float dx = init_x - i_x;
            float dy = init_y - i_y;
            float ddx = 1.0f - dx;
            float ddy = 1.0f - dy;
            float ddxddy = ddx * ddy;
            float ddxdy = ddx * dy;
            float dxddy = dx * ddy;
            float dxdy = dx * dy;

            // 2.4 双线性插值预先计算亚像素值
            for(int row = 0; row < patch_size_plus_2; ++row)
            {
                for(int col = 0; col < patch_size_plus_2; ++col)
                {
                    old_patch_plus2_subpixel[row][col] = 
                        common::interpolate(old_patch_plus3_pixel[row][col], old_patch_plus3_pixel[row+1][col], 
                                            old_patch_plus3_pixel[row][col+1], old_patch_plus3_pixel[row+1][col+1], 
                                            ddxddy, ddxdy, dxddy, dxdy);
                }
            }

            // 2.5 逆向光流算法只需要计算一次雅可比和海塞矩阵(访问亚像素需要双线性插值)
            int idx = 0;
            Eigen::Matrix<float, patch_num_, 2> J;                    // 高斯牛顿迭代法的雅克比矩阵 JT * J * delta = -JT * r
            Eigen::Matrix<float, 2, patch_num_> H_inv_JT;             // 高斯牛顿迭代法的H逆乘J转置 = (JT * J).inverse() * JT
            Eigen::Matrix<float, patch_num_, 1> b;                    // 高斯牛顿迭代法的向量b
            for(int row = 0; row < patch_size_; ++row)
            {
                for(int col = 0; col < patch_size_; ++col)
                {
                    // 计算像素梯度
                    J.row(idx) << 0.5 * (old_patch_plus2_subpixel[row+1][col+2] - old_patch_plus2_subpixel[row+1][col]),
                                  0.5 * (old_patch_plus2_subpixel[row+2][col+1] - old_patch_plus2_subpixel[row][col+1]);
                    idx++;
                }
            }

            // 计算海塞矩阵
            Eigen::Matrix2f H = J.transpose() * J;
            Eigen::Matrix2f H_inv;
            H_inv.setIdentity();
            H.ldlt().solveInPlace(H_inv);
            H_inv_JT = H_inv * J.transpose();

            // 2.6 高斯牛顿法迭代计算delta(逆向光流每次迭代只需要更新残差,无需更新J和H)
            for(int iter = 0; iter < 5; ++iter)
            {
                if(unlikely(init_x + translation[0] < patch_size_plus_1 || init_x + translation[0] + patch_size_plus_1 > cols || 
                            init_y + translation[1] < patch_size_plus_1 || init_y + translation[1] + patch_size_plus_1 > pyramid_old[j].rows ||
                            std::isnan(translation[0]) || std::isnan(translation[1])))
                {
                    valid = false;
                    break;
                }
                else
                {
                    // 2.6.1 获取亚像素需要双线性插值,预先获取8x8网格的像素值
                    float x = init_x + translation[0];
                    float y = init_y + translation[1];
                    i_x = int(x);
                    i_y = int(y);
                    new_patch_plus1_pixel[0] = &pyramid_new[j].data[(i_y - half_patch_size_) * cols + (i_x - half_patch_size_)];
                    for(int row = 1; row < patch_size_plus_1; ++row)
                        new_patch_plus1_pixel[row] = new_patch_plus1_pixel[row-1] + cols;
                    // 2.6.2 预先计算双线性插值的中间变量
                    dx = x - i_x;
                    dy = y - i_y;
                    ddx = 1.0f - dx;
                    ddy = 1.0f - dy;
                    ddxddy = ddx * ddy;
                    ddxdy = ddx * dy;
                    dxddy = dx * ddy;
                    dxdy = dx * dy;
                    // 2.6.3 双线性插值计算亚像素值并计算残差
                    idx = 0;
                    for(int row = 0; row < patch_size_; ++row)
                    {
                        for(int col = 0; col < patch_size_; ++col)
                        {
                            b.row(idx) << common::interpolate(new_patch_plus1_pixel[row][col], new_patch_plus1_pixel[row+1][col], 
                                                              new_patch_plus1_pixel[row][col+1], new_patch_plus1_pixel[row+1][col+1], 
                                                              ddxddy, ddxdy, dxddy, dxdy) - old_patch_plus2_subpixel[row+1][col+1];
                            idx++;
                        }
                    }
                    // 2.6.4 计算delta并更新待优化变量
                    Eigen::Vector2f delta_translation = H_inv_JT * b;
                    translation -= delta_translation;
                    if(delta_translation.squaredNorm() < 0.03 * 0.03) break;
                }
            }
            translation *= scale;
        }
        current_distort_pt.x = translation[0] + last_distort_pt.x;
        current_distort_pt.y = translation[1] + last_distort_pt.y;
        return valid;
    }
}

OpticalFlow::OpticalFlow(int image_width, int image_height, bool is_resized)
:image_width_(image_width), image_height_(image_height), is_resized_(is_resized)
{
    pyramid_tmp_.resize(kOpticalFlowMaxLevel - 1);
    for(int i = 1, w = image_width_, h = image_height_; i < kOpticalFlowMaxLevel; ++i)
    {
        // 高斯核卷积进行加速的一个中间保存结果(高度为当前层图像高度,宽度为上一层图像宽度)
        h /= 2;
        pyramid_tmp_[i-1].rows = h;
        pyramid_tmp_[i-1].cols = w;
        pyramid_tmp_[i-1].data.resize(w * h);
        w /= 2;
    }
}

std::vector<cv::Mat> OpticalFlow::BuildPyramid(const cv::Mat &image)
{
    // 若降采样开关打开则进行图像降采样;否则原图拷贝
    std::vector<cv::Mat> result(kOpticalFlowMaxLevel);
    for(int i = 0, scale = 1; i < kOpticalFlowMaxLevel; ++i, scale *= 2)
        result[i] = cv::Mat(image_height_ / scale, image_width_ / scale, CV_8UC1, cv::Scalar::all(0));

    if(likely(is_resized_))
    {
        int rows = image_height_;
        int cols = image_width_;
        for (int row = 0; row < rows; ++row)
        {
            int row_ptr = row * cols;
            int row_ptr2 = (row * image.cols) << 1;
            int row_ptr2_2 = row_ptr2 + image.cols;
            for(int col = 0; col < cols; ++col)
            {
                // 2x2像素块取平均值
                auto ptr1 = &(image.data[row_ptr2 + (col << 1)]);
                auto ptr2 = &(image.data[row_ptr2_2 + (col << 1)]);
                ushort sum_u = ptr1[0] + ptr1[1] + ptr2[0] + ptr2[1];
                result[0].data[row_ptr + col] = static_cast<uchar>(sum_u >> 2);
            }
        }
    }
    else
    {
        memcpy(result[0].data, image.data, sizeof(uchar) * image.rows * image.cols);
    }

    // 构建图像金字塔
    constexpr ushort kernel[5] = {1, 4, 6, 4, 1};
    uchar val = 0;
    for(int i = 1; i < kOpticalFlowMaxLevel; ++i)
    {
        const auto &img_pre = result[i-1];
        auto &img_next = result[i];
        auto &img_tmp = pyramid_tmp_[i-1];
        int pre_cols = img_pre.cols;
        // 先做列卷积(复杂度从O(5^2)降到O(2x5))
        for (int row = 2; row < img_tmp.rows - 2; ++row)
        {
            int row_ptr = 2 * row * pre_cols;
            for(int col = 2; col < img_tmp.cols - 2; ++col)
            {
                // ushort sum_u = 0;
                // for(int i = -2; i <= 2; ++i)
                // {
                //     sum_u += kernel[i + 2] * img_pre.data[(2 * row + i) * img_pre.cols + col];
                // }
                ushort sum_u = img_pre.data[row_ptr - pre_cols - pre_cols + col]
                             + (img_pre.data[row_ptr - pre_cols + col] << 2)
                             + kernel[2] * img_pre.data[row_ptr + col]
                             + (img_pre.data[row_ptr + pre_cols + col] << 2)
                             + img_pre.data[row_ptr + pre_cols + pre_cols + col];
                img_tmp.data[row * img_tmp.cols + col] = sum_u;
            }
        }
        // 再做行卷积
        for (int row = 2; row < img_next.rows - 2; ++row)
        {
            int next_row_ptr = row * img_next.cols;
            int tmp_row_ptr = row*img_tmp.cols;
            for (int col = 2; col < img_next.cols - 2; ++col)
            {
                // int sum_i = 0;
                // for (int i = -2; i <= 2; i++)
                // {
                //     sum_i += kernel[i + 2] * img_tmp.at<ushort>(row, 2 * col + i);
                // }
                int col2 = 2 * col;
                int sum_i = img_tmp.data[tmp_row_ptr + col2 - 2]
                          + (img_tmp.data[tmp_row_ptr + col2 - 1] << 2)
                          + kernel[2] * img_tmp.data[tmp_row_ptr + col2]
                          + (img_tmp.data[tmp_row_ptr + col2 + 1] << 2)
                          + img_tmp.data[tmp_row_ptr + col2 + 2];
                val = static_cast<uchar>((sum_i + (1 << 7)) >> 8);
                img_next.data[next_row_ptr + col] = val;
            }
        }
    }
    return result;
}

void OpticalFlow::OpticalFlowPyrLK(const std::vector<cv::Mat> &pyramid_old,
                                   const std::vector<cv::Mat> &pyramid_new,
                                   const std::vector<int> &last_levels,
                                   const std::vector<cv::Point2f> &last_distort_pts, 
                                   std::vector<cv::Point2f> &current_distort_pts,
                                   std::vector<uchar> &status)
{
    CHECK_IF(pyramid_old.size() == kOpticalFlowMaxLevel);
    CHECK_IF(pyramid_new.size() == kOpticalFlowMaxLevel);
    constexpr float threshold = 0.09;
    current_distort_pts.clear();
    status.clear();
    current_distort_pts.resize(last_distort_pts.size());
    status.resize(last_distort_pts.size(), 0);

    for(int i = 0; i < last_distort_pts.size(); ++i)
    {
        Eigen::Vector2f translation = Eigen::Vector2f::Zero();
        bool valid = TrackPoint(last_levels[i], pyramid_old, last_distort_pts[i], pyramid_new, current_distort_pts[i], translation, kOpticalFlowMaxLevel);
        if(likely(valid))
        {
            cv::Point2f last_distort_pt = last_distort_pts[i];
            Eigen::Vector2f translation_recovered = -translation;
            valid = TrackPoint(last_levels[i], pyramid_new, current_distort_pts[i], pyramid_old, last_distort_pt, translation_recovered, kOpticalFlowMaxLevel);
            if(likely(valid))
            {
                float dx = last_distort_pt.x - last_distort_pts[i].x;
                float dy = last_distort_pt.y - last_distort_pts[i].y;
                if(dx*dx+dy*dy < threshold) 
                    status[i] = 1;
            }
        }
    }
}

}
}

#endif