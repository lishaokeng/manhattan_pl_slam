#ifndef SVO_MATCHER_H_
#define SVO_MATCHER_H_

#include "Eigen/Core"
#include "sophus/se3.hpp"
#include "opencv2/opencv.hpp"
#include <unordered_set>

#include "camera_model/camera_interface.h"
#include "patch_score.h"

namespace TTE{
namespace vo{

  class Frame;
  class Feature;
  class MapPoint;

}

namespace svo {

using namespace Eigen;

/// Patch-matcher for reprojection-matching and epipolar search in triangulation.
class Matcher
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    std::map<unsigned long, cv::Mat> images_;

    static const int halfpatch_size_ = 4;
    static const int patch_size_ = 8;
    typedef vk::patch_score::ZMSSD<halfpatch_size_> PatchScore;

    struct Options
    {
        bool align_1d;              //!< in epipolar search: align patch 1D along epipolar line
        int align_max_iter;         //!< number of iterations for aligning the feature patches in gauss newton
        double max_epi_length_optim;//!< max length of epipolar line to skip epipolar search and directly go to img align
        size_t max_epi_search_steps;//!< max number of evaluations along epipolar line
        bool subpix_refinement;     //!< do gauss newton feature patch alignment after epipolar search
        bool epi_search_edgelet_filtering;
        double epi_search_edgelet_max_angle;
        Options() :
        align_1d(false),
        align_max_iter(10),
        max_epi_length_optim(2.0),
        max_epi_search_steps(1000),
        subpix_refinement(true),
        epi_search_edgelet_filtering(true),
        epi_search_edgelet_max_angle(0.7)
        {}
    } options_;

    uint8_t patch_[patch_size_*patch_size_] __attribute__ ((aligned (16)));
    uint8_t patch_with_border_[(patch_size_+2)*(patch_size_+2)] __attribute__ ((aligned (16)));
    Eigen::Matrix2d A_cur_ref_;   //!< affine warp matrix
    Eigen::Vector2d epi_dir_;
    double epi_length_;           //!< length of epipolar line segment in pixels (only used for epipolar search)
    double h_inv_;                //!< hessian of 1d image alignment along epipolar line
    int search_level_;
    bool reject_;
    vo::Feature* ref_ftr_;
    Eigen::Vector2d px_cur_;

    camera_model::CameraInterface *const camera_;

    Matcher(camera_model::CameraInterface *const camera):camera_(camera){};
    ~Matcher() = default;

    /// Find a match by directly applying subpix refinement.
    /// IMPORTANT! This function assumes that px_cur is already set to an estimate that is within ~2-3 pixel of the final result!
    void FindMatchDirect(vo::Frame *last_frame, vo::Frame *current_frame, std::unordered_set<vo::MapPoint*> &lost_map_points);


    /// Find a match by searching along the epipolar line without using any features.
    bool findEpipolarMatchDirect(
        const vo::Frame& ref_frame,
        const vo::Frame& cur_frame,
        const vo::Feature& ref_ftr,
        const double d_estimate,
        const double d_min,
        const double d_max,
        double& depth);

    void createPatchFromPatchWithBorder();
};

} // namespace svo
}

#endif // SVO_MATCHER_H_
