#include <cstdlib>
#include "math_utils.h"
#include "matcher.h"
#include "vo/frame.h"
#include "vo/feature.h"
#include "vo/map_point.h"
#include "feature_alignment.h"
#include "common/interpolation.h"
#include "svo/patch_score.h"
#include "common/warp.h"


namespace TTE{
namespace svo {

namespace{
Eigen::Vector2d world2cam(const camera_model::CameraInterface& camera, const Eigen::Vector3d &pt)
{
    cv::Point2f cv_pt(pt[0]/pt[2], pt[1]/pt[2]);
    auto p = camera.DistortInNormalizedPlaneAndProjectToPixelPlane(cv_pt);
    return Eigen::Vector2d(p.x, p.y);
}
Eigen::Vector2d world2cam(const camera_model::CameraInterface& camera, const Eigen::Vector2d &pt)
{
    auto p = camera.DistortInNormalizedPlaneAndProjectToPixelPlane(cv::Point2f(pt.x(), pt.y()));
    return Eigen::Vector2d(p.x, p.y);
}

Eigen::Vector3d cam2world(const camera_model::CameraInterface& camera, const Eigen::Vector2d &pt)
{
    cv::Point2f cv_pt(pt.x(), pt.y());
    auto p = camera.PixelToNormalizedPlaneAndUndistort(cv_pt);
    return Eigen::Vector3d(p.x, p.y, 1).normalized();
}

/**
 * 三角化计算深度值
*/
bool depthFromTriangulation(
    const Sophus::SE3d& T_search_ref,
    const Vector3d& f_ref,
    const Vector3d& f_cur,
    double& depth)
{
    Matrix<double,3,2> A; A << T_search_ref.rotationMatrix() * f_ref, -f_cur;
    const Matrix2d AtA = A.transpose()*A;
    if(AtA.determinant() < 0.000001)
        return false;
    const Vector2d depth2 = - AtA.inverse()*A.transpose()*T_search_ref.translation();
    if(depth2[0] < 0.1 || depth2[1] < 0.1) return false;
    depth = depth2[0];
    return true;
}
}

void Matcher::createPatchFromPatchWithBorder()
{
    uint8_t* ref_patch_ptr = patch_;
    for(int y=1; y<patch_size_+1; ++y, ref_patch_ptr += patch_size_)
    {
        uint8_t* ref_patch_border_ptr = patch_with_border_ + y*(patch_size_+2) + 1;
        for(int x=0; x<patch_size_; ++x)
        ref_patch_ptr[x] = ref_patch_border_ptr[x];
    }
}

void Matcher::FindMatchDirect(vo::Frame *last_frame, vo::Frame *current_frame, std::unordered_set<vo::MapPoint*> &lost_map_points)
{
    std::cout<<"lost1:"<<lost_map_points.size()<<"/";
    using namespace vo;
    Eigen::Matrix3d R_Cur_Ref = (current_frame->q_Camera_World() * last_frame->q_World_Camera()).toRotationMatrix();
    Eigen::Vector3d t_Cur_Ref = current_frame->q_Camera_World() * last_frame->t_World_Camera() + current_frame->t_Camera_World();
    Eigen::Matrix3d R_CurCamera_World = current_frame->q_Camera_World().toRotationMatrix();
    Eigen::Vector3d t_CurCamera_World = current_frame->t_Camera_World();
    for(auto &feat : last_frame->features_)
    {
        auto map_pt = feat->map_point_.get();
        if(lost_map_points.find(map_pt) != lost_map_points.end())
        {
            Frame *origin_frame = nullptr;
            Feature *origin_feature = nullptr;
            double depth_inv = -1;
            std::tie(origin_frame, origin_feature, depth_inv) = map_pt->GetOriginFrameFeaturePtrDepth();
            CHECK_IF(origin_frame && origin_feature && depth_inv > 0);
            Eigen::Vector3d P_Origin = origin_feature->GetNormalizedXYVector3d() / depth_inv;
            Eigen::Vector3d P_World = origin_frame->q_World_Camera() * P_Origin + origin_frame->t_World_Camera();
            Eigen::Vector3d P_Target = R_CurCamera_World * P_World + t_CurCamera_World;
            P_Target /= P_Target[2];
            auto P_Cur_Pix = camera_->DistortInNormalizedPlaneAndProjectToPixelPlane(cv::Point2f(P_Target[0], P_Target[1]));
            if(!camera_->InBorder(P_Cur_Pix)) continue;

            // 计算仿射变换矩阵A_cur_ref_
            Eigen::Matrix2d A_Cur_Ref;
            Eigen::Vector2d P_Init = warp::getWarpMatrixAffine(*camera_, feat->GetDistortUv(), feat->GetUnitCircleXYZ(), 
                                        (P_World - last_frame->t_World_Camera()).norm(),
                                        R_Cur_Ref, t_Cur_Ref, feat->Level(), A_Cur_Ref);
            int search_level = warp::getBestSearchLevel(A_Cur_Ref, kOpticalFlowMaxLevel-1);
            if(!warp::warpAffine(A_Cur_Ref, last_frame->pyr_images_[feat->Level()], feat->GetDistortUvVector2d(),
                                 feat->Level(), search_level, halfpatch_size_+1, patch_with_border_))
                continue;
            // 获取patch像素块
            createPatchFromPatchWithBorder();

            // 考虑金字塔层级
            Eigen::Vector2d px_scaled(P_Init/(1<<search_level));
            bool success = false;
            success = svo::feature_alignment::align2D(current_frame->pyr_images_[search_level], patch_with_border_, patch_, 10, px_scaled);
            if(!success) continue;
            // 转回0层金字塔对应像素位置
            Eigen::Vector2d P_Update = px_scaled * (1<<search_level);
            cv::Point2f pt0(P_Update[0], P_Update[1]);

            map_pt->SetPointState(MapPointState::INITIALIZED);
            // auto score = shiTomasiScore(current_frame->pyr_images_[search_level], std::round(px_scaled[0]), std::round(px_scaled[1]));
            Feature::PtLevelScore pt_level_score(pt0, search_level, 0);
            if(map_pt->observations_by_frames_.find(current_frame->ID()) == map_pt->observations_by_frames_.end())
            {
                auto feature = std::make_unique<Feature>(origin_feature->map_point_, pt_level_score, 
                    camera_->PixelToNormalizedPlaneAndUndistort(pt0));
                current_frame->features_.push_back(std::move(feature));
                map_pt->AddPointObservation(current_frame->ID(), current_frame, current_frame->features_.back().get());
            }
            else
            {
                auto frame_feat = map_pt->observations_by_frames_[current_frame->ID()];
                std::get<1>(frame_feat)->Reset(origin_feature->map_point_, pt_level_score, 
                    camera_->PixelToNormalizedPlaneAndUndistort(pt0));
            }
            lost_map_points.erase(map_pt);
        }
    }
    int c1 = 0, c2 = 0;
    for(auto &it : lost_map_points)
    {
        if(it->observations_by_frames_.find(current_frame->ID()) != it->observations_by_frames_.end())
            c2++;
        else c1++;
    }
    std::cout<<c1<<" "<<c2<<std::endl;
    lost_map_points.clear();
    return;
}

/**
 * 极线约束，计算种子点在当前帧上的极线段，进行块匹配，找到最佳匹配点，然后用该点三角化计算深度
*/
bool Matcher::findEpipolarMatchDirect(
    const vo::Frame& ref_frame,
    const vo::Frame& cur_frame,
    const vo::Feature& ref_ftr,
    const double d_estimate,  // 种子点估计深度
    const double d_min,       // 种子点最小深度
    const double d_max,       // 种子点最大深度
    double& depth)
{

    if(images_.find(cur_frame.ID()) == images_.end())
    {
        cv::Mat img;
        cv::cvtColor(cur_frame.pyr_images_[0], img, cv::COLOR_GRAY2BGR);
        images_[cur_frame.ID()] = img;
    }


    Sophus::SE3d T_cur_ref(cur_frame.q_Camera_World() * ref_frame.q_World_Camera(), 
                           cur_frame.q_Camera_World() * ref_frame.t_World_Camera() + cur_frame.t_Camera_World());
    int zmssd_best = PatchScore::threshold();
    Vector2d uv_best;

    Vector2d P_Estimate = vk::project2d(T_cur_ref * (ref_ftr.GetUnitCircleXYZ() * d_estimate));
    Vector2d P_pix_est = world2cam(*camera_, P_Estimate);
    cv::circle(images_[cur_frame.ID()], cv::Point2f(P_pix_est[0], P_pix_est[1]), 2, cv::Scalar(0,0,0), -1);


    // Compute start and end of epipolar line in old_kf for match search, on unit plane!
    // 种子点最小深度、最大深度对应线段，投影到当前帧相机平面上的线段向量，极线候选搜索区域
    Vector2d A = vk::project2d(T_cur_ref * (ref_ftr.GetUnitCircleXYZ() * d_min));
    Vector2d B = vk::project2d(T_cur_ref * (ref_ftr.GetUnitCircleXYZ() * d_max));
    epi_dir_ = A - B;

    // Compute affine warp matrix
    warp::getWarpMatrixAffine(*camera_, ref_ftr.GetDistortUv(), ref_ftr.GetUnitCircleXYZ(), d_estimate, 
                              T_cur_ref.rotationMatrix(), T_cur_ref.translation(), ref_ftr.Level(), A_cur_ref_);

    reject_ = false;
    search_level_ = warp::getBestSearchLevel(A_cur_ref_, vo::kOpticalFlowMaxLevel-1);

    // Find length of search range on epipolar line
    // 极线段投影到像素平面，计算搜索长度
    Vector2d px_A(world2cam(*camera_, A));
    Vector2d px_B(world2cam(*camera_, B));
    epi_length_ = (px_A-px_B).norm() / (1<<search_level_);

    // Warp reference patch at ref_level
    warp::warpAffine(A_cur_ref_, ref_frame.pyr_images_[ref_ftr.Level()], ref_ftr.GetDistortUvVector2d(),
                     ref_ftr.Level(), search_level_, halfpatch_size_+1, patch_with_border_);

    createPatchFromPatchWithBorder();

    // 如果极线段区域只剩一个像素，说明种子点的深度值已经比较确定了，再进行特征点配准，得到更加精确的像素坐标，然后三角化计算深度
    if(epi_length_ < 2.0)
    {
        px_cur_ = (px_A+px_B)/2.0;
        
        cv::circle(images_[cur_frame.ID()], cv::Point2f(px_cur_[0], px_cur_[1]), 1, cv::Scalar(0,255,0), -1);

        Vector2d px_scaled(px_cur_/(1<<search_level_));
        bool res;
        if(options_.align_1d)
            res = feature_alignment::align1D(
                  cur_frame.pyr_images_[search_level_], (px_A-px_B).cast<float>().normalized(),
                  patch_with_border_, patch_, options_.align_max_iter, px_scaled, h_inv_);
        else
            res = feature_alignment::align2D(
                  cur_frame.pyr_images_[search_level_], patch_with_border_, patch_,
                  options_.align_max_iter, px_scaled);
        if(res)
        {
            px_cur_ = px_scaled*(1<<search_level_);
            // 三角化计算深度值
            if(depthFromTriangulation(T_cur_ref, ref_ftr.GetUnitCircleXYZ(), cam2world(*camera_, px_cur_), depth))
                return true;
        }
        return false;
    }

    size_t n_steps = epi_length_/0.3; // one step per pixel
    Vector2d step = epi_dir_/n_steps;

    if(n_steps > options_.max_epi_search_steps)
    {
        printf("WARNING: skip epipolar search: %zu evaluations, px_lenght=%f, d_min=%f, d_max=%f.\n", n_steps, epi_length_, d_min, d_max);
        return false;
    }

    // for matching, precompute sum and sum2 of warped reference patch
    int pixel_sum = 0;
    int pixel_sum_square = 0;
    PatchScore patch_score(patch_);

    // now we sample along the epipolar line
    Vector2d uv = B-step;
    Vector2i last_checked_pxi(0,0);
    ++n_steps;
    // 沿着极线取块，取匹配程度最高的块，对应像素点位置
    const int cols = cur_frame.pyr_images_[search_level_].cols;
    Vector2i best_pxi;
    for(size_t i=0; i<n_steps; ++i, uv+=step)
    {
        Vector2d px(world2cam(*camera_, uv));
        Vector2i pxi(px[0]/(1<<search_level_)+0.5,
                     px[1]/(1<<search_level_)+0.5); // +0.5 to round to closest int

        images_[cur_frame.ID()].at<cv::Vec3b>(std::round(px[1]), std::round(px[0])) = cv::Vec3b(0,0,255);


        if(pxi == last_checked_pxi) continue;
        last_checked_pxi = pxi;

        // check if the patch is full within the new frame
        if(!camera_->InBorder(cv::Point2f(px[0], px[1]))) continue;

        // TODO interpolation would probably be a good idea
        uint8_t* cur_patch_ptr = cur_frame.pyr_images_[search_level_].data
                                + (pxi[1]-halfpatch_size_)*cols
                                + (pxi[0]-halfpatch_size_);
        int zmssd = patch_score.computeScore(cur_patch_ptr, cols);

        if(zmssd < zmssd_best)
        {
            zmssd_best = zmssd;
            uv_best = uv;
            best_pxi = pxi;
        }
    }

    // 由于位姿误差导致极线误差,所以在匹配点3x3范围内再搜索一遍ssd
    if(zmssd_best < PatchScore::threshold() && !options_.subpix_refinement)
    {
        for(int row = -1; row <= 1; ++row)
        {
            for(int col = -1; col <= 1; ++col)
            {
                Vector2i pxi(best_pxi[0]+col, best_pxi[1]+row);
                uint8_t* cur_patch_ptr = cur_frame.pyr_images_[search_level_].data
                                    + (pxi[1]-halfpatch_size_)*cols
                                    + (pxi[0]-halfpatch_size_);
                int zmssd = patch_score.computeScore(cur_patch_ptr, cols);

                if(zmssd < zmssd_best)
                {
                    zmssd_best = zmssd;
                    auto p = camera_->PixelToNormalizedPlaneAndUndistort(cv::Point2f(pxi[0], pxi[1])*(1<<search_level_));
                    uv_best[0] = p.x;
                    uv_best[1] = p.y;
                }
            }
        }
    }
    Vector2d p = world2cam(*camera_, uv_best);
    cv::circle(images_[cur_frame.ID()], cv::Point2f(p[0], p[1]), 1, cv::Scalar(0,255,0), -1);

    // 匹配较好，匹配到的像素点进一步配准优化，之后在三角化计算深度值
    if(zmssd_best < PatchScore::threshold())
    {
        if(options_.subpix_refinement)
        {
            px_cur_ = world2cam(*camera_, uv_best);
            Vector2d px_scaled(px_cur_/(1<<search_level_));
            bool res;
            if(options_.align_1d)
                res = feature_alignment::align1D(
                    cur_frame.pyr_images_[search_level_], (px_A-px_B).cast<float>().normalized(),
                    patch_with_border_, patch_, options_.align_max_iter, px_scaled, h_inv_);
            else
                res = feature_alignment::align2D(
                    cur_frame.pyr_images_[search_level_], patch_with_border_, patch_,
                    options_.align_max_iter, px_scaled);
            if(res)
            {
                px_cur_ = px_scaled*(1<<search_level_);
                if(depthFromTriangulation(T_cur_ref, ref_ftr.GetUnitCircleXYZ(), cam2world(*camera_, px_cur_), depth))
                return true;
            }
            return false;
        }
        px_cur_ = world2cam(*camera_, uv_best);
        if(depthFromTriangulation(T_cur_ref, ref_ftr.GetUnitCircleXYZ(), vk::unproject2d(uv_best).normalized(), depth))
        return true;
    }
    return false;
}

} // namespace svo
}