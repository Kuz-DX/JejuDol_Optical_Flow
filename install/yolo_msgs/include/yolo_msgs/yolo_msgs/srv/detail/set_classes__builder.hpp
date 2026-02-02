// generated from rosidl_generator_cpp/resource/idl__builder.hpp.em
// with input from yolo_msgs:srv/SetClasses.idl
// generated code does not contain a copyright notice

#ifndef YOLO_MSGS__SRV__DETAIL__SET_CLASSES__BUILDER_HPP_
#define YOLO_MSGS__SRV__DETAIL__SET_CLASSES__BUILDER_HPP_

#include <algorithm>
#include <utility>

#include "yolo_msgs/srv/detail/set_classes__struct.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


namespace yolo_msgs
{

namespace srv
{

namespace builder
{

class Init_SetClasses_Request_classes
{
public:
  Init_SetClasses_Request_classes()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  ::yolo_msgs::srv::SetClasses_Request classes(::yolo_msgs::srv::SetClasses_Request::_classes_type arg)
  {
    msg_.classes = std::move(arg);
    return std::move(msg_);
  }

private:
  ::yolo_msgs::srv::SetClasses_Request msg_;
};

}  // namespace builder

}  // namespace srv

template<typename MessageType>
auto build();

template<>
inline
auto build<::yolo_msgs::srv::SetClasses_Request>()
{
  return yolo_msgs::srv::builder::Init_SetClasses_Request_classes();
}

}  // namespace yolo_msgs


namespace yolo_msgs
{

namespace srv
{


}  // namespace srv

template<typename MessageType>
auto build();

template<>
inline
auto build<::yolo_msgs::srv::SetClasses_Response>()
{
  return ::yolo_msgs::srv::SetClasses_Response(rosidl_runtime_cpp::MessageInitialization::ZERO);
}

}  // namespace yolo_msgs

#endif  // YOLO_MSGS__SRV__DETAIL__SET_CLASSES__BUILDER_HPP_
