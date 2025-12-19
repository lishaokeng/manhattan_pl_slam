#ifndef TTE_CAMERA_MODEL_CAMERA_FACTORY_H
#define TTE_CAMERA_MODEL_CAMERA_FACTORY_H

#include "equidistant_camera.h"
#include "pinhole_camera.h"
#include "tte_camera.h"

namespace camera_model {
namespace CameraFactory{

static std::unique_ptr<CameraInterface> MakeCameraModel(const VOEquidistantCamera &camera_intrinsic, bool pyr_down)
{
    return std::make_unique<KannalaBrandtCamera>(camera_intrinsic, pyr_down);
}

static std::unique_ptr<CameraInterface> MakeCameraModel(const VOPinholeCamera &camera_intrinsic, bool pyr_down)
{
    return std::make_unique<PinholeCamera>(camera_intrinsic, pyr_down);
}

static std::unique_ptr<CameraInterface> MakeCameraModel(const VOTTECamera &camera_intrinsic, bool pyr_down)
{
    return std::make_unique<TTECamera>(camera_intrinsic, pyr_down);
}


}

}
#endif