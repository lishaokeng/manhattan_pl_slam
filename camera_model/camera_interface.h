/*
1 成像模型:
    1.1 Pinhole 针孔模型
    1.2 Omnidirectional Camera Model 全向模型(主要两种方式: 1.Dioptric[一组鱼眼镜头]; 2.Catadioptric[一个标准镜头+反射面镜])
        2.1 Unified model for catadioptric cameras 反射式相机统一模型
        2.2 Extended Unified model for catadioptric cameras(EUCM)
        2.3 Omnidirectional Camera Model By Scaramuzza
2 畸变模型:
    2.1 Equidistant(EQUI)等距畸变模型(也叫kannala_brandt模型)
    2.2 Radtan切向径向畸变,Brown模型
    2.3 FOV视野畸变模型(用得少,dso才用到)
    2.4 TTE独有五次多项式拟合

*********************************************

slam常用相机模型组合:
类型     名称                     组合                 组合中文名             fov      描述
针孔   Pinhole             Pinhole +  Radtan        针孔 + Radtan          <90      
鱼眼   Equidistant         Pinhole +  Equidistant   针孔 + kannala_brandt  90-170 opencv默认的鱼眼类型
鱼眼   Mei(catadioptric)   Omni    +  Radtan        全向 + Radtan          >90    Mei模型专门针对catadioptric相机提出
鱼眼   Davide Scaramuzza    
TTE   TTE独有五次多项式模型  Pinhole + TTE五次多项式                                  该模型不通用,本公司特色...

*/

#ifndef TTE_CAMERA_MODEL_CAMERA_INTERFACE_H
#define TTE_CAMERA_MODEL_CAMERA_INTERFACE_H

#include <opencv2/opencv.hpp>
#include "common/log_define.h"
#include "vo_based_on_front_camera.h"


namespace camera_model {

constexpr float kAtlantaScaleFxFy = 2; 
constexpr float kAtlantaScaleHeightFront = 2.5; // 前视相机安装位置较矮,需要调整视野看高点
constexpr float kAtlantaScaleHeightOther = 3; 

class CameraInterface{
public:
    enum ModelType
    {
        KANNALA_BRANDT, // KANNALA_BRANDT鱼眼模型
        MEI,            // MEI鱼眼模型
        PINHOLE,        // 针孔模型
        SCARAMUZZA,     // 广角模型
        TTE             // TTE独有5次多项式
    };

    CameraInterface(int camera_idx, ModelType model_type, bool no_distortion, bool use_mask, int pyr_down)
    :camera_idx_(camera_idx), model_type_(model_type), no_distortion_(no_distortion), use_mask_(use_mask), pyr_down_(pyr_down){}
    CameraInterface(const CameraInterface&) = delete;
    CameraInterface&operator=(CameraInterface&) = delete;
    virtual ~CameraInterface(){}

    // 返回相机号
    inline int CameraIndex()const{return camera_idx_;}
    // 返回经过缩放后的图像宽度
    inline int WidthScaled()const{return image_width_scaled_;}
    // 返回经过缩放后的图像高度
    inline int HeightScaled()const{return image_height_scaled_;}
    // 返回原始图像宽度
    inline int WidthSrc()const{return image_width_src_;}
    // 返回原始图像高度
    inline int HeightSrc()const{return image_height_src_;}
    // 返回是否使用mask
    inline bool UseMask()const{return use_mask_;}
    // 返回是否缩放
    inline bool IsPyrDown()const{return pyr_down_;}
    // 返回mask图深拷贝,slam特征点跟踪时需要用到
    cv::Mat MaskDeepCopy()const{return mask_.clone();}
    // 返回mask图浅拷贝,提高mask使用效率.慎用,一个相机只有一张初始mask,避免修改该mask
    cv::Mat MaskShallowCopy()const{return mask_;}
    // 返回亚特兰大世界去畸变图像使用的K
    inline cv::Matx33d AtlantaNewK()const{return atlanta_undist_new_K_;}
    // 返回亚特兰大世界去畸变图像的大小
    inline cv::Size AtlantaNewSize()const{return atlanta_undist_new_size_;}
    double fxAtlanta()const{return atlanta_undist_new_K_(0, 0);}

    // 是否在图像有效范围内(边界的点不靠谱不跟踪)
    inline bool InBorder(const cv::Point2f& point)const
    {
        bool in_border = border_size_ <= point.x && point.x < image_width_scaled_ - border_size_ && 
                         border_size_ <= point.y && point.y < image_height_scaled_ - border_size_;
        // 分支预测(更可能在边界内)
        if(__builtin_expect((!in_border || !use_mask_), 0)) return in_border;
        else return mask_.at<uchar>(round(point.y), round(point.x)) != 0;
    }


    /**
     *@brief 从像素投影到归一化坐标系后并去畸变(畸变像素->投影到归一化坐标系->在归一化坐标系去畸变) 
    *@param[in] distort_pixel 像素坐标系畸变的点
    * @return 归一化坐标系去畸变后的点
    */
    virtual cv::Point2f PixelToNormalizedPlaneAndUndistort(const cv::Point2f& distort_pixel)const = 0;

    /**
     *@brief 在归一化坐标系进行畸变后投影到像素平面(在归一化坐标系进行畸变->投影到像素平面) 
    *@param[in] point 归一化系去畸变点
    * @return 像素平面畸变后的点
    */
    virtual inline cv::Point2f DistortInNormalizedPlaneAndProjectToPixelPlane(const cv::Point2f &point)const = 0;

    /**
     * @brief 对整个图像去畸变,得到去畸变后的图像.is_atlanta=true时使用亚特兰大内参和图像大小,否则使用new_K和new_size
    */
    void AtlantaUndistortImage(cv::Mat src, cv::Mat &result, bool is_atlanta, const cv::Matx33d &new_K=cv::Matx33d(), 
                               const cv::Size &new_size=cv::Size())const;

    // 对于去畸变图像使用的相机内参undist_new_K_,从像素平面转到归一化平面
    cv::Point3d AtlantaPixelToNormalized(const cv::Point2f& pt) const
    {
        return cv::Point3d((pt.x - atlanta_undist_new_K_(0, 2)) / atlanta_undist_new_K_(0, 0), 
                           (pt.y - atlanta_undist_new_K_(1, 2)) / atlanta_undist_new_K_(1, 1), 
                           1);
    }

    // 对于去畸变图像使用的相机内参undist_new_K_,从归一化平面转到像素平面
    cv::Point2f AtlantaNormalizedToPixel(const cv::Point3d& pt) const
    {
        return cv::Point2f(pt.x * atlanta_undist_new_K_(0, 0) + atlanta_undist_new_K_(0, 2), 
                           pt.y * atlanta_undist_new_K_(1, 1) + atlanta_undist_new_K_(1, 2));
    }
    cv::Point2f AtlantaNormalizedToPixel(double x, double y, double z) const
    {
        return cv::Point2f(x / z * atlanta_undist_new_K_(0, 0) + atlanta_undist_new_K_(0, 2), 
                           y / z * atlanta_undist_new_K_(1, 1) + atlanta_undist_new_K_(1, 2));
    }

    virtual double fx()const = 0;

    const ModelType model_type_;

protected:

    /**
     * @brief 读取mask图,读取失败时通过畸变、去畸变后的像素点投影误差来制作mask图.仅当使用use_mask_ == true且图像宽高读取后才能使用该函数.
     * @return 计算后给成员变量mask_赋值
    */
    void MakeMaskByReadOrReprojectError(const std::string& mask_path = "");

    // 制作亚特兰大去畸变图片大小和查找表
    void MakeAtlantaParams();

    const int camera_idx_;
    int image_width_scaled_ = 0;  // 缩放后的图像宽
    int image_height_scaled_ = 0; // 缩放后的图像高
    int image_width_src_ = 0;     // 原始图像宽
    int image_height_src_ = 0;    // 原始图像高
    const bool no_distortion_;    // true: 没有畸变; false: 有畸变
    const bool use_mask_;         // true: 使用mask; false: 不使用mask
    const bool pyr_down_;         // true: 图像降采样; false: 图像不降采样
    cv::Mat mask_;                // mask图
    static constexpr int border_size_ = 1; // 判断点是否在图像边界的阈值
    cv::Matx33d atlanta_undist_new_K_;    // 亚特兰大模型需要在去畸变图像上提取elsed线段,这个去畸变图像内参和原图不一样,内参(决定线段提取的图像范围)
    cv::Size atlanta_undist_new_size_;    // 亚特兰大模型需要在去畸变图像上提取elsed线段,这个去畸变图像大小和原图不一样,去畸变图像的大小(决定线段提取的图像范围)
    std::vector<cv::Point2f> atlanta_undist_table_; // 亚特兰大图片去畸变查找表(去畸变太耗时,空间换时间)
};


}


#endif