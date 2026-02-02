from setuptools import setup

package_name = 'yolo_ros_ros1'

setup(
    name=package_name,
    version='0.0.0',
    # 'yolo_ros_ros1' 폴더가 실제로 없으므로 빼고, 'scripts'만 넣습니다.
    packages=['scripts'], 
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='kuzdx',
    description='ROS 2 package for YOLO tracking',
    license='TODO',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            # scripts 패키지 안의 yolo_track_node.py 내 main 함수 호출
            'yolo_track_node = scripts.yolo_track_node:main',
        ],
    },
)