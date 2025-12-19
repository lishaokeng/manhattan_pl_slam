#ifndef SVO_SPARSE_IMG_ALIGN_H_
#define SVO_SPARSE_IMG_ALIGN_H_

#include "nlls_solver.h"
#include "vo/frame.h"
#include "common/log_define.h"

#include "opencv2/opencv.hpp"

namespace vk {
class AbstractCamera;
}

namespace svo {

class Feature;

/// Optimize the pose of the frame by minimizing the photometric error of feature patches.
class SparseImgAlign : public vk::NLLSSolver<6, Sophus::SE3d>
{
  static const int patch_halfsize_ = 2;
  static const int patch_size_ = 2*patch_halfsize_;
  static const int patch_area_ = patch_size_*patch_size_;
  using FramePtr = ::TTE::vo::Frame*;
  using FeaturePtr = ::TTE::vo::Feature*;
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  cv::Mat resimg_;

  SparseImgAlign(
      camera_model::CameraInterface *const camera,
      int n_levels,
      int min_level,
      int n_iter,
      Method method,
      bool display,
      bool verbose);

  size_t run(
      FramePtr ref_frame,
      FramePtr cur_frame);

  /// Return fisher information matrix, i.e. the Hessian of the log-likelihood
  /// at the converged state.
  Eigen::Matrix<double, 6, 6> getFisherInformation();

protected:
  camera_model::CameraInterface *const camera_;
  FramePtr ref_frame_;            //!< reference frame, has depth for gradient pixels.
  std::vector<FeaturePtr> ref_features_;
  std::vector<Eigen::Vector3d> map_points_;
  FramePtr cur_frame_;            //!< only the image is known!
  int level_;                     //!< current pyramid level on which the optimization runs.
  bool display_;                  //!< display residual image.
  int max_level_;                 //!< coarsest pyramid level for the alignment.
  int min_level_;                 //!< finest pyramid level for the alignment.

  // cache:
  Eigen::Matrix<double, 6, Eigen::Dynamic, Eigen::ColMajor> jacobian_cache_;
  bool have_ref_patch_cache_;
  cv::Mat ref_patch_cache_;
  std::vector<bool> visible_fts_;

  void precomputeReferencePatches();
  virtual double computeResiduals(const Sophus::SE3d& model, bool linearize_system, bool compute_weight_scale = false);
  virtual int solve();
  virtual void update (const ModelType& old_model, ModelType& new_model);
  virtual void startIteration();
  virtual void finishIteration();
};

} // namespace svo

#endif // SVO_SPARSE_IMG_ALIGN_H_
