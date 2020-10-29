all:
	g++ -std=c++11 -DDEBUG toonify.cpp -o toonify -lopencv_videostab -lopencv_photo -lopencv_objdetect -lopencv_video -lopencv_ml -lopencv_calib3d -lopencv_features2d -lopencv_highgui -lopencv_flann -lopencv_imgproc -lopencv_dnn -lopencv_imgcodecs -lopencv_core -I /usr/local/Cellar/opencv/4.5.0_2/include/opencv4
