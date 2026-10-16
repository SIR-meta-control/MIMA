cmake_minimum_required(VERSION 2.8.3)

project(crimson_stand_legInit_forSimOnly)

find_package(catkin REQUIRED)

catkin_package()

find_package(roslaunch)

foreach(dir config launch meshes urdf)
	install(DIRECTORY ${dir}/
		DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/${dir})
endforeach(dir)

# Keep the MJCF and its mesh directory together for installed simulation nodes.
install(DIRECTORY ../mjcf
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION})
