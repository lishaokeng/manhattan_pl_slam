#include <algorithm>
#include <svo/sparse_img_align.h>
// #include <vikit/vision.h>
// #include <vikit/math_utils.h>

namespace svo {

SparseImgAlign::SparseImgAlign(
    camera_model::CameraInterface *const camera,
    int max_level, int min_level, int n_iter,
    Method method, bool display, bool verbose) :
        NLLSSolver(),
        camera_(camera),
        display_(display),
        max_level_(max_level),
        min_level_(min_level)
{
  n_iter_ = n_iter;
  n_iter_init_ = n_iter_;
  method_ = method;
  verbose_ = verbose;
  eps_ = 0.000001;
}

/**
 * 当前帧与前一帧最小化测量误差，LM求解最优位姿
 * 1、从金字塔分辨率最低层，遍历到分辨率最高层，优化位姿传递，coarse-to-fine过程
 * 2、构建Jacobian、Residual，LM迭代求解位姿
 * 3、更新cur_frame的位姿
 * @param ref_frame 前一帧，已知特征点、深度
 * @param cur_frame 当前帧，只有图像金字塔
*/
size_t SparseImgAlign::run(FramePtr ref_frame, FramePtr cur_frame)
{
  reset();

  if(ref_frame->features_.empty())
  {
    printf_log("SparseImgAlign: no features to track!");
    return 0;
  }

  ref_frame_ = ref_frame;
  cur_frame_ = cur_frame;
  ref_features_.clear();
  for(auto &it : ref_frame_->features_)
  {
    if(it->map_point_->GetPointState() == ::TTE::vo::MapPointState::INITIALIZED)
    {
      Eigen::Vector3d xyz;
      auto frame_feat_depth = it->map_point_->GetOriginFrameFeaturePtrDepth();
      auto frame = std::get<0>(frame_feat_depth);
      auto feature = std::get<1>(frame_feat_depth);
      auto depth_inv = std::get<2>(frame_feat_depth);
      if(frame == nullptr || feature == nullptr || depth_inv <= 0) continue;
      CHECK_IF(frame);
      CHECK_IF(feature);
      auto pc_norm = feature->GetNormalizedXYVector3d();
      Eigen::Vector3d P_Camera = pc_norm / depth_inv;
      xyz = frame->q_World_Camera() * P_Camera + frame->t_World_Camera();
      ref_features_.push_back(it.get());
      map_points_.push_back(xyz);
    }
  }
  Sophus::SE3d cur_T_Camera_World(cur_frame->q_Camera_World(), cur_frame->t_Camera_World());
  Sophus::SE3d ref_T_Camera_World(ref_frame_->q_Camera_World(), ref_frame_->t_Camera_World());

  // 大小(n,16), patch块大小是4x4
  ref_patch_cache_ = cv::Mat(ref_features_.size(), patch_area_, CV_32F);
  // 大小(6,n,16)
  jacobian_cache_.resize(Eigen::NoChange, ref_patch_cache_.rows*patch_area_);
  visible_fts_.resize(ref_patch_cache_.rows, false); // TODO: should it be reset at each level?

  Sophus::SE3d T_cur_from_ref(cur_T_Camera_World * ref_T_Camera_World.inverse());

  // 从金字塔分辨率最低层，遍历到分辨率最高层，coarse-to-fine过程
  for(level_=max_level_; level_>=min_level_; --level_)
  {
    mu_ = 0.1;
    // 线性化点位置变了，重新计算Jacobian
    jacobian_cache_.setZero();
    have_ref_patch_cache_ = false;
    if(verbose_)
      printf("\nPYRAMID LEVEL %i\n---------------\n", level_);
    /**
     * 调用过程
     * 1、computeResiduals。对所有特征点patch，计算测量误差（Photometric Error）对位姿的Jacobian，以及像素值残差
     * 2、solve。计算delta_x
     * 3、update。更新位姿
    */
    optimize(T_cur_from_ref);
  }
  // 更新当前帧位姿
  auto DT = T_cur_from_ref * ref_T_Camera_World * cur_T_Camera_World.inverse();
  std::cout<<DT.so3().log().norm()<<" "<<DT.translation().norm()<<std::endl;
  if(DT.so3().log().norm() < 0.01 && DT.translation().norm() < 0.1) // 光度误差优化的位姿和BA重投影误差优化的结果不能相差太大(角度<0.5° && 平移<0.1)
  {
      Sophus::SE3d T_World_Camera = (T_cur_from_ref * ref_T_Camera_World).inverse();
      cur_frame->SetPose(T_World_Camera.rotationMatrix(), T_World_Camera.translation());
  }
  return n_meas_/patch_area_;
}

Eigen::Matrix<double, 6, 6> SparseImgAlign::getFisherInformation()
{
  double sigma_i_sq = 5e-4*255*255; // image noise
  Eigen::Matrix<double,6,6> I = H_/sigma_i_sq;
  return I;
}

/**
 * 计算测量误差（Photometric Error）对位姿的Jacobian，存jacobian_cache_
 * https://www.cnblogs.com/gaoxiang12/p/5689927.html
*/
void SparseImgAlign::precomputeReferencePatches()
{
  // patch是2x2，外层再添加1个像素的边界
  const int border = patch_halfsize_+1;
  const cv::Mat& ref_img = ref_frame_->pyr_images_.at(level_);
  const int stride = ref_img.cols;
  const float scale = 1.0f/(1<<level_);
  const Eigen::Vector3d ref_pos = ref_frame_->t_World_Camera();
  const double focal_length = (camera_->fx()); // 后端优化的特征点重投影尺度:像素
  size_t feature_counter = 0;
  std::vector<bool>::iterator visiblity_it = visible_fts_.begin();
  // 遍历参考帧的特征点
  for(auto it=ref_features_.begin(), ite=ref_features_.end();
      it!=ite; ++it, ++feature_counter, ++visiblity_it)
  {
    // check if reference with patch size is within image
    // 判断patch块是否超出图像边界
    auto uv = (*it)->GetDistortUv();
    const float u_ref = uv.x*scale;
    const float v_ref = uv.y*scale;
    const int u_ref_i = floorf(u_ref);
    const int v_ref_i = floorf(v_ref);
    if((*it)->map_point_ == nullptr || u_ref_i-border < 0 || v_ref_i-border < 0 || u_ref_i+border >= ref_img.cols || v_ref_i+border >= ref_img.rows)
      continue;
    *visiblity_it = true;

    // cannot just take the 3d points coordinate because of the reprojection errors in the reference image!!!
    // 点的深度值
    const double depth((map_points_[feature_counter] - ref_pos).norm());
    // 相机坐标点，注意不用世界坐标去计算Jacobian，世界坐标与像素之间本身存在投影误差，世界坐标是不准确的，用像素坐标对应的相机坐标去计算
    const Eigen::Vector3d xyz_ref((*it)->GetUnitCircleXYZ() *depth);

    // evaluate projection jacobian
    // 重投影像素误差相对于相机位姿的Jacobian，在当前点位置展开
    Eigen::Matrix<double,2,6> frame_jac;
    const double x = xyz_ref[0];
    const double y = xyz_ref[1];
    const double z_inv = 1./xyz_ref[2];
    const double z_inv_2 = z_inv*z_inv;

    frame_jac(0,0) = -z_inv;              // -1/z
    frame_jac(0,1) = 0.0;                 // 0
    frame_jac(0,2) = x*z_inv_2;           // x/z^2
    frame_jac(0,3) = y*frame_jac(0,2);            // x*y/z^2
    frame_jac(0,4) = -(1.0 + x*frame_jac(0,2));   // -(1.0 + x^2/z^2)
    frame_jac(0,5) = y*z_inv;             // y/z

    frame_jac(1,0) = 0.0;                 // 0
    frame_jac(1,1) = -z_inv;              // -1/z
    frame_jac(1,2) = y*z_inv_2;           // y/z^2
    frame_jac(1,3) = 1.0 + y*frame_jac(1,2);      // 1.0 + y^2/z^2
    frame_jac(1,4) = -frame_jac(0,3);             // -x*y/z^2
    frame_jac(1,5) = -x*z_inv;            // x/z

    // compute bilateral interpolation weights for reference image
    // 双线性插值，4个权重
    const float subpix_u_ref = u_ref-u_ref_i;
    const float subpix_v_ref = v_ref-v_ref_i;
    const float w_ref_tl = (1.0-subpix_u_ref) * (1.0-subpix_v_ref);
    const float w_ref_tr = subpix_u_ref * (1.0-subpix_v_ref);
    const float w_ref_bl = (1.0-subpix_u_ref) * subpix_v_ref;
    const float w_ref_br = subpix_u_ref * subpix_v_ref;
    size_t pixel_counter = 0;
    float* cache_ptr = reinterpret_cast<float*>(ref_patch_cache_.data) + patch_area_*feature_counter;
    // 按行遍历patch块
    for(int y=0; y<patch_size_; ++y)
    {
      // 第一行
      uint8_t* ref_img_ptr = (uint8_t*) ref_img.data + (v_ref_i+y-patch_halfsize_)*stride + (u_ref_i-patch_halfsize_);
      for(int x=0; x<patch_size_; ++x, ++ref_img_ptr, ++cache_ptr, ++pixel_counter)
      {
        // precompute interpolated reference patch color
        // 当前点用相邻右下角共4个点，插值计算当前点的像素值
        *cache_ptr = w_ref_tl*ref_img_ptr[0] + w_ref_tr*ref_img_ptr[1] + w_ref_bl*ref_img_ptr[stride] + w_ref_br*ref_img_ptr[stride+1];

        // we use the inverse compositional: thereby we can take the gradient always at the same position
        // get gradient of warped image (~gradient at warped position)
        // 用当前点前后、上下点分别计算x、y方向上的像素梯度
        float dx = 0.5f * ((w_ref_tl*ref_img_ptr[1] + w_ref_tr*ref_img_ptr[2] + w_ref_bl*ref_img_ptr[stride+1] + w_ref_br*ref_img_ptr[stride+2])
                          -(w_ref_tl*ref_img_ptr[-1] + w_ref_tr*ref_img_ptr[0] + w_ref_bl*ref_img_ptr[stride-1] + w_ref_br*ref_img_ptr[stride]));
        float dy = 0.5f * ((w_ref_tl*ref_img_ptr[stride] + w_ref_tr*ref_img_ptr[1+stride] + w_ref_bl*ref_img_ptr[stride*2] + w_ref_br*ref_img_ptr[stride*2+1])
                          -(w_ref_tl*ref_img_ptr[-stride] + w_ref_tr*ref_img_ptr[1-stride] + w_ref_bl*ref_img_ptr[0] + w_ref_br*ref_img_ptr[1]));

        // cache the jacobian
        // 最终的测量误差对位姿的Jacobian
        jacobian_cache_.col(feature_counter*patch_area_ + pixel_counter) =
            (dx*frame_jac.row(0) + dy*frame_jac.row(1))*(focal_length / (1<<level_));
      }
    }
  }
  have_ref_patch_cache_ = true;
}

/**
 * 对所有特征点patch，计算测量误差（Photometric Error）对位姿的Jacobian，以及像素值残差
 * J.t() * J * delta_x = -J.t() * r
*/
double SparseImgAlign::computeResiduals(
    const Sophus::SE3d& T_cur_from_ref,
    bool linearize_system,
    bool compute_weight_scale)
{
  // Warp the (cur)rent image such that it aligns with the (ref)erence image
  // 当前帧level_层金字塔图像
  const cv::Mat& cur_img = cur_frame_->pyr_images_.at(level_);

  if(linearize_system && display_)
    resimg_ = cv::Mat(cur_img.size(), CV_32F, cv::Scalar(0));

  // 计算测量误差（Photometric Error）对位姿的Jacobian，存jacobian_cache_
  if(have_ref_patch_cache_ == false)
    precomputeReferencePatches();

  // compute the weights on the first iteration
  std::vector<float> errors;
  if(compute_weight_scale)
    errors.reserve(visible_fts_.size());
  const int stride = cur_img.cols;
  const int border = patch_halfsize_+1;
  const float scale = 1.0f/(1<<level_);
  const Eigen::Vector3d ref_pos(ref_frame_->t_World_Camera());
  float chi2 = 0.0;
  size_t feature_counter = 0; // is used to compute the index of the cached jacobian
  // 遍历特征点的patch，计算残差
  std::vector<bool>::iterator visiblity_it = visible_fts_.begin();
  for(auto it=ref_features_.begin(); it!=ref_features_.end();
      ++it, ++feature_counter, ++visiblity_it)
  {
    // check if feature is within image
    if(!*visiblity_it)
      continue;

    // compute pixel location in cur img
    const double depth = (map_points_[feature_counter] - ref_pos).norm();
    const Eigen::Vector3d xyz_ref((*it)->GetUnitCircleXYZ() *depth);
    const Eigen::Vector3d xyz_cur(T_cur_from_ref * xyz_ref);
    cv::Point2f norm_pt(xyz_cur[0]/xyz_cur[2], xyz_cur[1]/xyz_cur[2]);
    cv::Point2f pixel_pt = camera_->DistortInNormalizedPlaneAndProjectToPixelPlane(norm_pt);
    const Eigen::Vector2f uv_cur_pyr(Eigen::Vector2f(pixel_pt.x, pixel_pt.y) * scale);
    const float u_cur = uv_cur_pyr[0];
    const float v_cur = uv_cur_pyr[1];
    const int u_cur_i = floorf(u_cur);
    const int v_cur_i = floorf(v_cur);

    // check if projection is within the image
    if(u_cur_i < 0 || v_cur_i < 0 || u_cur_i-border < 0 || v_cur_i-border < 0 || u_cur_i+border >= cur_img.cols || v_cur_i+border >= cur_img.rows)
      continue;

    // compute bilateral interpolation weights for the current image
    const float subpix_u_cur = u_cur-u_cur_i;
    const float subpix_v_cur = v_cur-v_cur_i;
    const float w_cur_tl = (1.0-subpix_u_cur) * (1.0-subpix_v_cur);
    const float w_cur_tr = subpix_u_cur * (1.0-subpix_v_cur);
    const float w_cur_bl = (1.0-subpix_u_cur) * subpix_v_cur;
    const float w_cur_br = subpix_u_cur * subpix_v_cur;
    float* ref_patch_cache_ptr = reinterpret_cast<float*>(ref_patch_cache_.data) + patch_area_*feature_counter;
    size_t pixel_counter = 0; // is used to compute the index of the cached jacobian
    for(int y=0; y<patch_size_; ++y)
    {
      uint8_t* cur_img_ptr = (uint8_t*) cur_img.data + (v_cur_i+y-patch_halfsize_)*stride + (u_cur_i-patch_halfsize_);

      for(int x=0; x<patch_size_; ++x, ++pixel_counter, ++cur_img_ptr, ++ref_patch_cache_ptr)
      {
        // compute residual
        // 计算像素值残差
        const float intensity_cur = w_cur_tl*cur_img_ptr[0] + w_cur_tr*cur_img_ptr[1] + w_cur_bl*cur_img_ptr[stride] + w_cur_br*cur_img_ptr[stride+1];
        const float res = intensity_cur - (*ref_patch_cache_ptr);

        // used to compute scale for robust cost
        if(compute_weight_scale)
          errors.push_back(fabsf(res));

        // robustification
        float weight = 1.0;
        if(use_weights_) {
          weight = weight_function_->value(res/scale_);
        }

        // 总残差
        chi2 += res*res*weight;
        n_meas_++;

        if(linearize_system)
        {
          // compute Jacobian, weighted Hessian and weighted "steepest descend images" (times error)
          // J.t() * J * delta_x = -J.t() * r
          const Eigen::Matrix<double, 6, 1> J(jacobian_cache_.col(feature_counter*patch_area_ + pixel_counter));
          H_.noalias() += J*J.transpose()*weight;
          Jres_.noalias() -= J*res*weight;
          if(display_)
            resimg_.at<float>((int) v_cur+y-patch_halfsize_, (int) u_cur+x-patch_halfsize_) = res/255.0;
        }
      }
    }
  }

  // compute the weights on the first iteration
  if(compute_weight_scale && iter_ == 0)
    scale_ = scale_estimator_->compute(errors);

  return chi2/n_meas_;
}

/**
 * 计算delta_x
*/
int SparseImgAlign::solve()
{
  x_ = H_.ldlt().solve(Jres_);
  if((bool) std::isnan((double) x_[0]))
    return 0;
  return 1;
}

/**
 * 更新位姿
*/
void SparseImgAlign::update(
    const ModelType& T_curold_from_ref,
    ModelType& T_curnew_from_ref)
{
  T_curnew_from_ref =  T_curold_from_ref * Sophus::SE3d::exp(-x_);
}

void SparseImgAlign::startIteration()
{}

void SparseImgAlign::finishIteration()
{
  if(display_)
  {
    cv::namedWindow("residuals");
    cv::imshow("residuals", resimg_*10);
    cv::waitKey(0);
  }
}

} // namespace svo

