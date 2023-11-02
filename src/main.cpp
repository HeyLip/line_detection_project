#include <iostream>
#include <string>
#include <cmath>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include "opencv2/opencv.hpp"

using namespace std;
using namespace cv;

const Point p1(0, 400), p2(260, 400);
const Point p3(380, 400), p4(640, 400);

vector<Point2f> find_edges(const Mat& img, const String& direction);
void drawCross(Mat& img, Point pt, Scalar color);
void writeText(Mat& img, vector<Point2f> pt, Point base, const String& direction);

int main()
{
	Mat frame, frame_ycrcb, gray, dst;
	vector<Mat> ycrcb_planes;
	vector<Point2f> pts1, pts2;
	VideoCapture cap;

	if(cap.open("C:\\workspace\\line_detection_project\\resource\\Sub_project.avi") == false){
		cerr << "Video load failed!" << endl;
		return -1;
	}

	int delay = 1000 / cap.get(CAP_PROP_FPS);
	
	int width = cap.get(CAP_PROP_FRAME_WIDTH);
	int height = cap.get(CAP_PROP_FRAME_HEIGHT);

	cout << "width: " << width << "  hegiht: " << height << endl;
	
	while(true){
		cap >> frame;

		if (frame.empty()) {
			cout << "end of video" << endl;
			break;
		}

		// histogram equalization
		// cvtColor(frame, frame_ycrcb, COLOR_RGB2YCrCb);
		// split(frame_ycrcb, ycrcb_planes);

		// equalizeHist(ycrcb_planes[0], ycrcb_planes[0]);
		// merge(ycrcb_planes, frame_ycrcb);
		// cvtColor(frame_ycrcb, frame, COLOR_YCrCb2BGR);

		// find left position, right position
		cvtColor(frame, gray, COLOR_BGR2GRAY);

		Rect rc1(p1 + Point(0, -10), p2 + Point(0, 10));
		Rect rc2(p3 + Point(0, -10), p4 + Point(0, 10));

		pts1 = find_edges(gray(rc1), "LEFT");
		pts2 = find_edges(gray(rc2), "RIGHT");

		dst = frame.clone();
		line(dst, p1, p4, Scalar(0, 255, 128), 1, LINE_AA);

		// Left position
		drawCross(dst, Point(cvRound(p1.x + pts1[0].x), p1.y), Scalar(255, 0, 0));
		drawCross(dst, Point(cvRound(p1.x + pts1[1].x), p1.y), Scalar(0, 0, 255));
		writeText(dst, pts1, p1, "LEFT");
		

		// right position
		drawCross(dst, Point(cvRound(p3.x + pts2[0].x), p3.y), Scalar(255, 0, 0));
		drawCross(dst, Point(cvRound(p3.x + pts2[1].x), p3.y), Scalar(0, 0, 255));
		writeText(dst, pts2, p3, "RIGHT");
		
		imshow("dst", dst);
		if(waitKey(delay) == 27) {
			break;
		}
	}
	destroyAllWindows();
	
}


vector<Point2f> find_edges(const Mat& img, const String& direction)
{
	
	Mat fimg, binary_img, blr, dx, dx_binary;
	float X, Y, Z, maxloc_float, minloc_float;
	img.convertTo(fimg, CV_32F);

	GaussianBlur(fimg, blr, Size(), 1.);

	threshold(blr, binary_img, 165, 255, THRESH_BINARY_INV);

	Sobel(binary_img, dx_binary, CV_32F, 1, 0);

	double minv, maxv;
	Point minloc, maxloc;

	int y2 = img.rows / 2;
	Mat roi = dx_binary.row(y2);
	minMaxLoc(roi, &minv, &maxv, &minloc, &maxloc);

	imshow("img", img);
	imshow("binary_img", binary_img);
	

	vector<Point2f> pts;

	// find max location in float
	if(fabs(maxv - 0.f) < 0.001) {
		if(direction == "LEFT"){
			maxloc_float = 0.0f;
		}else{
			maxloc_float = 260.f;
		}
	}
	else {
		X = roi.at<float>(0, maxloc.x - 1);
		Y = roi.at<float>(0, maxloc.x); // maxv
		Z = roi.at<float>(0, maxloc.x + 1);

		maxloc_float = static_cast<float>((X - Z) / (2*X-4*Y+2*Z) + maxloc.x);
	}
	
	cout << "maxloc_float: " << maxloc_float << endl;

	// find min location in float
	if(fabs(minv - 0.f) < 0.001) {
		if(direction == "LEFT"){
			minloc_float = 0.0f;
		}else{
			minloc_float = 260.f;
		}
	}
	else {
		X = roi.at<float>(0, minloc.x - 1);
		Y = roi.at<float>(0, minloc.x); // minv
		Z = roi.at<float>(0, minloc.x + 1);

		minloc_float = static_cast<float>((X - Z) / (2*X-4*Y+2*Z) + minloc.x);
	}

	cout << "minloc_float: " << minloc_float << endl;

	pts.push_back(Point2f(maxloc_float, y2));
	pts.push_back(Point2f(minloc_float, y2));

	return pts;
}

void drawCross(Mat& img, Point pt, Scalar color)
{
	int span = 5;
	line(img, pt + Point(-span, -span), pt + Point(span, span), color, 1, LINE_AA);
	line(img, pt + Point(-span, span), pt + Point(span, -span), color, 1, LINE_AA);
}

void writeText(Mat& img, vector<Point2f> pt, Point base, const String& direction)
{
	float number = 0.0f;

	if(direction == "RIGHT") {
		number = 260.f;
	}

	// position left
	if(fabs(pt[0].x - number) > 0.0001){
		putText(img, format("(%4.3f, %d)", base.x + pt[0].x, base.y),
		Point(base.x + pt[0].x - 50, base.y - 20),
		FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255, 0, 0), 1, LINE_AA);
	}
	
	
	// position right
	if(fabs(pt[1].x - number) > 0.0001){
		putText(img, format("(%4.3f, %d)", base.x + pt[1].x, base.y),
		Point(base.x + pt[1].x - 20, base.y + 30),
		FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 255), 1, LINE_AA);
	}
}