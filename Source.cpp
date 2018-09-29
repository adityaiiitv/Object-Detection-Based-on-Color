#include "opencv2/objdetect.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include <iostream>
#include <stdio.h>
#include <math.h>

using namespace std;
using namespace cv;

// Declare structure to be used to pass data from C++ to Mono.
struct Circle
{
	Circle(int x, int y, int radius) : X(x), Y(y), Radius(radius) {}
	int X, Y, Radius;
};

CascadeClassifier _faceCascade;
String _windowName = "Unity OpenCV Interop Sample";
VideoCapture _capture;
int _scale = 1;
// Set to Blue
int iLowH = 130;
int iHighH = 160;
int iLowS = 50;
int iHighS = 255;
int iLowV = 50;
int iHighV = 255;
int iLastX = -1;
int iLastY = -1;
extern "C" int __declspec(dllexport) __stdcall  Init(int& outCameraWidth, int& outCameraHeight)
{
	// Load LBP face cascade.
	if (!_faceCascade.load("lbpcascade_frontalface.xml"))
		return -1;

	// Open the stream.
	_capture.open(0);
	if (!_capture.isOpened())
		return -2;

	outCameraWidth = _capture.get(CAP_PROP_FRAME_WIDTH);
	outCameraHeight = _capture.get(CAP_PROP_FRAME_HEIGHT);

	return 0;
}

extern "C" void __declspec(dllexport) __stdcall  Close()
{
	destroyAllWindows();
	_capture.release();
}

extern "C" void __declspec(dllexport) __stdcall SetScale(int scale)
{
	_scale = scale;
}

extern "C" void __declspec(dllexport) __stdcall Detect(Circle* outFaces, int maxOutFacesCount, int& outDetectedFacesCount, int& cR, int& cG, int& cB)
{
	Mat frame;
	_capture >> frame;
	if (frame.empty())
		return;

	Mat imgHSV;
	cvtColor(frame, imgHSV, COLOR_BGR2HSV); //BGR to HSV
	Mat imgThresholded;
	inRange(imgHSV, Scalar(iLowH, iLowS, iLowV), Scalar(iHighH, iHighS, iHighV), imgThresholded); // Threshold
	//morphological opening (removes small objects from the foreground)
	erode(imgThresholded, imgThresholded, getStructuringElement(MORPH_ELLIPSE, Size(5, 5)));
	dilate(imgThresholded, imgThresholded, getStructuringElement(MORPH_ELLIPSE, Size(5, 5)));
	//morphological closing (removes small holes from the foreground)
	dilate(imgThresholded, imgThresholded, getStructuringElement(MORPH_ELLIPSE, Size(5, 5)));
	erode(imgThresholded, imgThresholded, getStructuringElement(MORPH_ELLIPSE, Size(5, 5)));
	//Calculate the moments of the thresholded image
	Moments oMoments = moments(imgThresholded);
	double dM01 = oMoments.m01;
	double dM10 = oMoments.m10;
	double dArea = oMoments.m00;
	
	// if area <= 10000, no objects 
	if (dArea > 10000)
	{
		// calculate the position of object
		int posX = dM10 / dArea;
		int posY = dM01 / dArea;
		int radius = sqrt( (dArea/ 130) / (22 / 7));
		if (iLastX >= 0 && iLastY >= 0 && posX >= 0 && posY >= 0)
		{
			// Set color
			Vec3b colour = frame.at<Vec3b>(Point(posX, posY));
			cR = colour.val[0];
			cG = colour.val[1];
			cB = colour.val[2];
			// Draw center
			line(frame, Point(posX, posY), Point(iLastX, iLastY), Scalar(0, 0, 255), 2);
			// Draw Boundary
			circle(frame, Point(posX, posY), radius, 1, 8, 0);
			maxOutFacesCount = 1;
			outDetectedFacesCount = 1;
			outFaces[0] = Circle(posX, posY, radius);
		}
		iLastX = posX;
		iLastY = posY;
	}
	imshow(_windowName, frame);
}

