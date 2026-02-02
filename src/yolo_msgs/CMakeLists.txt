cmake_minimum_required(VERSION 3.8)
project(yolo_msgs)

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(std_msgs REQUIRED)

# Generate C++ and Python interfaces for messages and services
rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/BoundingBox2D.msg"
  "msg/BoundingBox3D.msg"
  "msg/Detection.msg"
  "msg/DetectionArray.msg"
  "msg/KeyPoint2D.msg"
  "msg/KeyPoint2DArray.msg"
  "msg/KeyPoint3D.msg"
  "msg/KeyPoint3DArray.msg"
  "msg/Mask.msg"
  "msg/Point2D.msg"
  "msg/Pose2D.msg"
  "msg/Vector2.msg"
  "srv/SetClasses.srv"
  DEPENDENCIES geometry_msgs std_msgs
)

# Install rules
install(
  DIRECTORY "msg"
  DESTINATION "share/${PROJECT_NAME}"
)
install(
  DIRECTORY "srv"
  DESTINATION "share/${PROJECT_NAME}"
)

ament_package()
