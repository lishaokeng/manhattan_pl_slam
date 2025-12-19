#ifndef LM_SOLVER_H_
#define LM_SOLVER_H_

#include <stdint.h>
#include <stdio.h>
#include <iostream>
#include <math.h>
#include "Eigen/Core"
#include "Eigen/StdVector"
#include "math_utils.h"
#include "robust_cost.h"

namespace vk {

using namespace std;
using namespace Eigen;

/**
 * \brief Abstract Class for solving nonlinear least-squares (NLLS) problems.
 *
 * The function implements two algorithms: Levenberg Marquardt and Gauss Newton
 *
 * Example implementations of this function can be found in the rpl_examples
 * package: img_align_2d.cpp, img_align_3d.cpp
 *
 * Template Parameters:
 * D  : dimension of the residual
 * T  : type of the model, e.g. SE2, SE3
 */

template <int D, typename T>
class NLLSSolver {

public:
  typedef T ModelType;
  enum Method{GaussNewton, LevenbergMarquardt};
  enum ScaleEstimatorType{UnitScale, TDistScale, MADScale, NormalScale};
  enum WeightFunctionType{UnitWeight, TDistWeight, TukeyWeight, HuberWeight};

protected:
  Eigen::Matrix<double, D, D>  H_;       //!< Hessian approximation
  Eigen::Matrix<double, D, 1>  Jres_;    //!< Jacobian x Residual
  Eigen::Matrix<double, D, 1>  x_;       //!< update step
  bool                  have_prior_;
  ModelType prior_;
  Eigen::Matrix<double, D, D>  I_prior_; //!< Prior information matrix (inverse covariance)
  double                chi2_;
  double                rho_;
  Method                method_;

  /// If the flag linearize_system is set, the function must also compute the
  /// Jacobian and set the member variables H_, Jres_
  virtual double
  computeResiduals      (const ModelType& model,
                         bool linearize_system,
                         bool compute_weight_scale) = 0;

  /// Solve the linear system H*x = Jres. This function must set the update
  /// step in the member variable x_. Must return true if the system could be
  /// solved and false if it was singular.
  virtual int
  solve                 () = 0;

  virtual void
  update                (const ModelType& old_model, ModelType& new_model) = 0;

  virtual void
  applyPrior            (const ModelType& current_model) { }

  virtual void
  startIteration        () { }

  virtual void
  finishIteration       () { }

  virtual void
  finishTrial           () { }

public:

  /// Damping parameter. If mu > 0, coefficient matrix is positive definite, this
  /// ensures that x is a descent direction. If mu is large, x is a short step in
  /// the steepest direction. This is good if the current iterate is far from the
  /// solution. If mu is small, LM approximates gauss newton iteration and we
  /// have (almost) quadratic convergence in the final stages.
  double                mu_init_, mu_;
  double                nu_init_, nu_;          //!< Increase factor of mu after fail
  size_t                n_iter_init_, n_iter_;  //!< Number of Iterations
  size_t                n_trials_;              //!< Number of trials
  size_t                n_trials_max_;          //!< Max number of trials
  size_t                n_meas_;                //!< Number of measurements
  bool                  stop_;                  //!< Stop flag
  bool                  verbose_;               //!< Output Statistics
  double                eps_;                   //!< Stop if update norm is smaller than eps
  size_t                iter_;                  //!< Current Iteration

  // robust least squares
  bool                  use_weights_;
  float                 scale_;
  robust_cost::ScaleEstimatorPtr scale_estimator_;
  robust_cost::WeightFunctionPtr weight_function_;

  NLLSSolver() :
    have_prior_(false),
    method_(LevenbergMarquardt),
    mu_init_(0.01f),
    mu_(mu_init_),
    nu_init_(2.0),
    nu_(nu_init_),
    n_iter_init_(15),
    n_iter_(n_iter_init_),
    n_trials_(0),
    n_trials_max_(5),
    n_meas_(0),
    stop_(false),
    verbose_(true),
    eps_(0.0000000001),
    iter_(0),
    use_weights_(false),
    scale_(0.0),
    scale_estimator_(NULL),
    weight_function_(NULL)
  { }

  virtual ~NLLSSolver() {}

  /// Gauss Newton optimization strategy
  /********************************
   * @ function: GN求解反向组合算法
   * 
   * @ param:  待优化变量
   * 
   * @ note: GN迭代过程，并update，并且判断停止条件
   *******************************/
  void optimizeGaussNewton(ModelType& model)
  {
    // Compute weight scale
    if(use_weights_)
      computeResiduals(model, false, true);

    // Save the old model to rollback in case of unsuccessful update
    ModelType old_model(model);

    // perform iterative estimation
    for (iter_ = 0; iter_<n_iter_; ++iter_)
    {
      rho_ = 0;
      startIteration();

      H_.setZero();
      Jres_.setZero();

      // compute initial error
      n_meas_ = 0;
      double new_chi2 = computeResiduals(model, true, false);

      // add prior
      if(have_prior_)
        applyPrior(model);

      // solve the linear system
      if(!solve())
      {
        // matrix was singular and could not be computed
        std::cout << "Matrix is close to singular! Stop Optimizing." << std::endl;
        std::cout << "H = " << H_ << std::endl;
        std::cout << "Jres = " << Jres_ << std::endl;
        stop_ = true;
      }

      // check if error increased since last optimization
      if((iter_ > 0 && new_chi2 > chi2_) || stop_)
      {
        if(verbose_)
        {
          std::cout << "It. " << iter_
                    << "\t Failure"
                    << "\t new_chi2 = " << new_chi2
                    << "\t Error increased. Stop optimizing."
                    << std::endl;
        }
        model = old_model; // rollback
        break;
      }

      // update the model
      ModelType new_model;
      update(model, new_model);
      old_model = model;
      model = new_model;

      chi2_ = new_chi2;

      if(verbose_)
      {
        std::cout << "It. " << iter_
                  << "\t Success"
                  << "\t new_chi2 = " << new_chi2
                  << "\t n_meas = " << n_meas_
                  << "\t x_norm = " << vk::norm_max(x_)
                  << std::endl;
      }

      finishIteration();

      // stop when converged, i.e. update step too small
      if(vk::norm_max(x_)<=eps_)
        break;
    }
  }

  /// Levenberg Marquardt optimization strategy
  void optimizeLevenbergMarquardt(ModelType& model)
  {
    // Compute weight scale
    if(use_weights_)
      computeResiduals(model, false, true);

    // compute the initial error
    chi2_ = computeResiduals(model, true, false);

    if(verbose_)
      cout << "init chi2 = " << chi2_
          << "\t n_meas = " << n_meas_
          << endl;

    // TODO: compute initial lambda
    // Hartley and Zisserman: "A typical init value of lambda is 10^-3 times the
    // average of the diagonal elements of J'J"

    // Compute Initial Lambda
    if(mu_ < 0)
    {
      double H_max_diag = 0;
      double tau = 1e-4;
      //6*6
      for(size_t j=0; j<D; ++j)
        H_max_diag = max(H_max_diag, fabs(H_(j,j)));
      mu_ = tau*H_max_diag;
    }

    // perform iterative estimation
    for (iter_ = 0; iter_<n_iter_; ++iter_)
    {
      rho_ = 0;
      startIteration();

      // try to compute and update, if it fails, try with increased mu
      n_trials_ = 0;
      do
      {
        // init variables
        ModelType new_model;
        double new_chi2 = -1;
        H_.setZero();
        //H_ = mu_ * Matrix<double,D,D>::Identity(D,D);
        Jres_.setZero();

        // compute initial error
        n_meas_ = 0;
        computeResiduals(model, true, false); //这样重复计算了好多次啊。。。

        // add damping term:
        H_ += (H_.diagonal()*mu_).asDiagonal();

        // add prior
        if(have_prior_)
          applyPrior(model);

        // solve the linear system
        if(solve())
        {
          // update the model
          update(model, new_model);

          // compute error with new model and compare to old error
          n_meas_ = 0;
          new_chi2 = computeResiduals(new_model, false, false); //只是为了计算chi2
          rho_ = chi2_-new_chi2;
        }
        else
        {
          // matrix was singular and could not be computed
          cout << "Matrix is close to singular!" << endl;
          cout << "H = " << H_ << endl;
          cout << "Jres = " << Jres_ << endl;
          rho_ = -1;
        }

        if(rho_>0)
        {
          //求解成功则减小mu_
          // update decrased the error -> success
          model = new_model;
          chi2_ = new_chi2;
          stop_ = vk::norm_max(x_)<=eps_;
          mu_ *= max(1./3., min(1.-pow(2*rho_-1,3), 2./3.));
          nu_ = 2.;
          if(verbose_)
          {
            cout << "It. " << iter_
                << "\t Trial " << n_trials_
                << "\t Success"
                << "\t n_meas = " << n_meas_
                << "\t new_chi2 = " << new_chi2
                << "\t mu = " << mu_
                << "\t nu = " << nu_
                << endl;
          }
        }
        else
        {
          //求解失败则扩大mu_, 尝试使H_变得非奇异
          // update increased the error -> fail
          mu_ *= nu_;
          nu_ *= 2.;
          ++n_trials_;
          //大于最大尝试次数则退出
          if (n_trials_ >= n_trials_max_) 
            stop_ = true;

          if(verbose_)
          {
            cout << "It. " << iter_
                << "\t Trial " << n_trials_
                << "\t Failure"
                << "\t n_meas = " << n_meas_
                << "\t new_chi2 = " << new_chi2
                << "\t mu = " << mu_
                << "\t nu = " << nu_
                << endl;
          }
        }

        finishTrial();

      } while(!(rho_>0 || stop_)); //为了trials设置的循环
      if (stop_)
        break;

      finishIteration();
    }
  }



  /// Calls the GaussNewton or LevenbergMarquardt optimization strategy
/********************************
 * @ function: 优化, 可选LM或者GN
 * 
 * @ param:  ModelType = T 待优化的量
 *           D 优化变量的维数
 * 
 * @ note:
 *******************************/
  void optimize(ModelType& model)
  {
    if(method_ == GaussNewton)
      optimizeGaussNewton(model);
    else if(method_ == LevenbergMarquardt)
      optimizeLevenbergMarquardt(model);
  }

  /// Specify the robust cost that should be used and the appropriate scale estimator
  void setRobustCostFunction(
      ScaleEstimatorType scale_estimator,
      WeightFunctionType weight_function)
  {
    switch(scale_estimator)
    {
      case TDistScale:
        if(verbose_)
          printf("Using TDistribution Scale Estimator\n");
        scale_estimator_.reset(new robust_cost::TDistributionScaleEstimator());
        use_weights_=true;
        break;
      case MADScale:
        if(verbose_)
          printf("Using MAD Scale Estimator\n");
        scale_estimator_.reset(new robust_cost::MADScaleEstimator());
        use_weights_=true;
      break;
      case NormalScale:
        if(verbose_)
          printf("Using Normal Scale Estimator\n");
        scale_estimator_.reset(new robust_cost::NormalDistributionScaleEstimator());
        use_weights_=true;
        break;
      default:
        if(verbose_)
          printf("Using Unit Scale Estimator\n");
        scale_estimator_.reset(new robust_cost::UnitScaleEstimator());
        use_weights_=false;
    }

    switch(weight_function)
    {
      case TDistWeight:
        if(verbose_)
          printf("Using TDistribution Weight Function\n");
        weight_function_.reset(new robust_cost::TDistributionWeightFunction());
        break;
      case TukeyWeight:
        if(verbose_)
          printf("Using Tukey Weight Function\n");
        weight_function_.reset(new robust_cost::TukeyWeightFunction());
        break;
      case HuberWeight:
        if(verbose_)
          printf("Using Huber Weight Function\n");
        weight_function_.reset(new robust_cost::HuberWeightFunction());
        break;
      default:
        if(verbose_)
          printf("Using Unit Weight Function\n");
        weight_function_.reset(new robust_cost::UnitWeightFunction());
    }
  }


  /// Add prior to optimization.
  void setPrior(
      const ModelType&  prior,
      const Eigen::Matrix<double, D, D>&  Information)
  {
    have_prior_ = true;
    prior_ = prior;
    I_prior_ = Information;
  }


  /// Reset all parameters to restart the optimization
  void reset()
  {
    have_prior_ = false;
    chi2_ = 1e10;
    mu_ = mu_init_;
    nu_ = nu_init_;
    n_meas_ = 0;
    n_iter_ = n_iter_init_;
    iter_ = 0;
    stop_ = false;
  }

  /// Get the squared error
  const double& getChi2() const
  {
    return chi2_;
  }

  /// The Information matrix is equal to the inverse covariance matrix.
  const Eigen::Matrix<double, D, D>& getInformationMatrix() const
  {
    return H_;  //原来你叫informationMatrix
  }
};

} // end namespace vk

#endif /* LM_SOLVER_H_ */
