#include <ros/ros.h>
#include <sensor_msgs/image_encodings.h>
#include <image_geometry/stereo_camera_model.h>
#include <cv_bridge/cv_bridge.h>

#include <fovis_ros/FovisInfo.h>

#include <libfovis/stereo_depth.hpp>
#include <libfovis/stereo_calibration.hpp>

#include "stereo_processor.hpp"
#include "odometer_base.hpp"
#include "visualization.hpp"

#include <memory>

#include <fovis_ros/StereoConfig.h>
#include <dynamic_reconfigure/server.h>

namespace fovis_ros
{

class StereoOdometer : public StereoProcessor, OdometerBase
{

private:

  std::shared_ptr<fovis::StereoDepth> stereo_depth_;
  ros::Time _lastTime;
  dynamic_reconfigure::Server<StereoConfig> params_server_;

public:

  StereoOdometer(const std::string& transport) :
    StereoProcessor(transport), _lastTime( 0 )
  {
    dynamic_reconfigure::Server<StereoConfig>::CallbackType cb;
    cb = boost::bind(&StereoOdometer::reconfigureCallback, this, _1, _2);
    params_server_.setCallback(cb);
  }

  void reconfigureCallback( StereoConfig& config, uint32_t level )
  {
    reset();

    fovis::VisualOdometryOptions& opts = OdometerBase::getOptions();
    
    // NOTE Need window size to be odd
    opts["feature-window-size"] = std::to_string( 2*config.feature_window_size + 1 );
    opts["max-pyramid-level"] = std::to_string( config.max_pyramid_level );
    if( config.min_pyramid_level > config.max_pyramid_level )
    {
      ROS_WARN_STREAM( "Min pyramid level greater than max! Tweaking...");
      config.min_pyramid_level = config.max_pyramid_level;
    }
    opts["min-pyramid-level"] = std::to_string( config.min_pyramid_level );

    opts["target-pixels-per-feature"] = std::to_string( config.target_pixels_per_feature );
    opts["fast-threshold"] = std::to_string( config.fast_threshold );
    opts["use-adaptive-threshold"] = config.use_adaptive_threshold ? "true" : "false";
    opts["fast-threshold-adaptive-gain"] = std::to_string( config.fast_threshold_adaptive_gain );
    opts["use-homography-initialization"] = config.use_homography_initialization ? "true" : "false";
    opts["ref-frame-change-threshold"] = std::to_string( config.ref_frame_change_threshold );

    opts["use-bucketing"] = config.use_bucketing? "true" : "false";
    opts["bucket-width"] = std::to_string( config.bucket_width );
    opts["bucket-height"] = std::to_string( config.bucket_height );
    opts["max-keypoints-per-bucket"] = std::to_string( config.max_keypoints_per_bucket );
    opts["use-image-normalization"] = config.use_image_normalization ? "true" : "false";

    opts["inlier-max-reprojection-error"] = std::to_string( config.inlier_max_reprojection_error );
    opts["clique-inlier-threshold"] = std::to_string( config.clique_inlier_threshold );
    opts["min-features-for-estimate"] = std::to_string( config.min_features_for_estimate );
    opts["max-mean-reprojection-error"] = std::to_string( config.max_mean_reprojection_error );
    opts["use-subpixel-refinement"] = config.use_subpixel_refinement ? "true" : "false";
    
    // NOTE Need window size to be odd
    opts["feature-search-window"] = std::to_string( 2*config.feature_search_window + 1 );
    opts["update-target-features-with-refined"] = config.update_target_features_with_refined ? "true" : "false";

    opts["stereo-require-mutual-match"] = config.stereo_require_mutual_match ? "true" : "false";
    opts["stereo-max-dist-epipolar-line"] = std::to_string( config.stereo_max_dist_epipolar_line );
    opts["stereo-max-refinement-displacement"] = std::to_string( config.stereo_max_refinement_displacement );
    opts["stereo-max-disparity"] = std::to_string( config.stereo_max_disparity );
  }

protected:

  void reset()
  {
    stereo_depth_.reset();
    StereoProcessor::reset();
    OdometerBase::reset();
    _lastTime = ros::Time(0);
  }

  void initStereoDepth(
      const sensor_msgs::CameraInfoConstPtr& l_info_msg,
      const sensor_msgs::CameraInfoConstPtr& r_info_msg)
  {
    // read calibration info from camera info message
    // to fill remaining parameters
    image_geometry::StereoCameraModel model;
    model.fromCameraInfo(*l_info_msg, *r_info_msg);

    // initialize left camera parameters
    fovis::CameraIntrinsicsParameters left_parameters;
    rosToFovis(model.left(), left_parameters);
    left_parameters.height = l_info_msg->height;
    left_parameters.width = l_info_msg->width;
    // initialize right camera parameters
    fovis::CameraIntrinsicsParameters right_parameters;
    rosToFovis(model.right(), right_parameters);
    right_parameters.height = r_info_msg->height;
    right_parameters.width = r_info_msg->width;

    // as we use rectified images, rotation is identity
    // and translation is baseline only
    fovis::StereoCalibrationParameters stereo_parameters;
    stereo_parameters.left_parameters = left_parameters;
    stereo_parameters.right_parameters = right_parameters;
    stereo_parameters.right_to_left_rotation[0] = 1.0;
    stereo_parameters.right_to_left_rotation[1] = 0.0;
    stereo_parameters.right_to_left_rotation[2] = 0.0;
    stereo_parameters.right_to_left_rotation[3] = 0.0;
    stereo_parameters.right_to_left_translation[0] = -model.baseline();
    stereo_parameters.right_to_left_translation[1] = 0.0;
    stereo_parameters.right_to_left_translation[2] = 0.0;

	  // NOTE This is left raw since fovis::StereoDepth takes ownership
    fovis::StereoCalibration* stereo_calibration =
      new fovis::StereoCalibration(stereo_parameters);

    //return new fovis::StereoDepth(stereo_calibration, getOptions());
	  stereo_depth_ = std::make_shared<fovis::StereoDepth>(stereo_calibration,
	                                                       getOptions());
  }

  void imageCallback(
      const sensor_msgs::ImageConstPtr& l_image_msg,
      const sensor_msgs::ImageConstPtr& r_image_msg,
      const sensor_msgs::CameraInfoConstPtr& l_info_msg,
      const sensor_msgs::CameraInfoConstPtr& r_info_msg)
  {
    const ros::Time& currTime = l_image_msg->header.stamp;
    
    if( _lastTime.isZero() )
    {
      _lastTime = currTime;
    }
    double dt = (currTime - _lastTime).toSec();
    _lastTime = currTime;
    if( dt < 0.0 )
    {
      ROS_INFO_STREAM( "Negative dt detected in stereo image stream. Resetting..." );
      reset();
    }

    if (!stereo_depth_)
    {
      initStereoDepth(l_info_msg, r_info_msg);
      setDepthSource(stereo_depth_);
    }

    // convert image if necessary
    uint8_t *r_image_data;
    int r_step;
    cv_bridge::CvImageConstPtr r_cv_ptr;
    r_cv_ptr = cv_bridge::toCvShare(r_image_msg, sensor_msgs::image_encodings::MONO8);
    r_image_data = r_cv_ptr->image.data;
    r_step = r_cv_ptr->image.step[0];

    ROS_ASSERT(r_step == static_cast<int>(r_image_msg->width));
    ROS_ASSERT(l_image_msg->width == r_image_msg->width);
    ROS_ASSERT(l_image_msg->height == r_image_msg->height);

    // pass image to depth source
    stereo_depth_->setRightImage(r_image_data);

    // call base implementation
    process(l_image_msg, l_info_msg);
  }
};

} // end of namespace


int main(int argc, char **argv)
{
  ros::init(argc, argv, "stereo_odometer");
  if (ros::names::remap("stereo") == "stereo") {
    ROS_WARN("'stereo' has not been remapped! Example command-line usage:\n"
             "\t$ rosrun fovis_ros stereo_odometer stereo:=narrow_stereo image:=image_rect");
  }
  if (ros::names::remap("image").find("rect") == std::string::npos) {
    ROS_WARN("stereo_odometer needs rectified input images. The used image "
             "topic is '%s'. Are you sure the images are rectified?",
             ros::names::remap("image").c_str());
  }

  std::string transport = argc > 1 ? argv[1] : "raw";
  fovis_ros::StereoOdometer odometer(transport);

  ros::spin();
  return 0;
}

