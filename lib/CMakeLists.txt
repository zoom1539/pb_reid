cmake_minimum_required(VERSION 3.0.0)

#
aux_source_directory(. LIB_SRC)
add_library(reid SHARED ${LIB_SRC})

target_link_libraries(reid ${OpenCV_LIBS} ${FASTRTENGINE_LIB})

# cd build 
# make install
install(FILES class_reid.h DESTINATION include)
install(TARGETS reid LIBRARY DESTINATION lib)
install(FILES extra/fastrt/libFastRTEngine.so.0 DESTINATION lib)
install(FILES extra/fastrt/sbs_R50-ibn.engine DESTINATION data)
# install(FILES extra/fastrt/sbs_R50-ibn.wts DESTINATION data)
