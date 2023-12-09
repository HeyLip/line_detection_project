#include <opencv2/opencv.hpp>
#include <eigen3/Eigen/Dense>
#include <iostream>
#include <numeric>

using namespace cv;
using namespace std;
using namespace Eigen;

// 다항식 피팅 함수
VectorXd fitPoly(const vector<Point>& points, int degree) {
    // 행렬 초기화
    MatrixXd A(points.size(), degree + 1);
    VectorXd b(points.size());

    // 행렬 A와 벡터 b에 데이터 대입
    for (size_t i = 0; i < points.size(); ++i) {
        for (int j = 0; j <= degree; ++j) {
            A(i, j) = pow(points[i].x, j);
        }
        b(i) = points[i].y;
    }

    // 최소 자승법을 사용하여 다항식 계수 계산
    VectorXd coefficients = A.jacobiSvd(ComputeThinU | ComputeThinV).solve(b);

    return coefficients;
}

// 슬라이딩 윈도우를 사용하여 차선 검출하는 함수
void slidingWindow(Mat& binary_warped, Mat& out_img, vector<double>& left_fit, vector<double>& right_fit) {
    // 이미지의 하단 부분에서 시작
    int bottom_half = binary_warped.rows / 2;

    // 히스토그램 생성
    Mat histogram = Mat::zeros(1, binary_warped.cols, CV_32SC1);
    reduce(binary_warped, histogram, 0, REDUCE_SUM, CV_32SC1);

    // 왼쪽과 오른쪽 차선의 시작 위치 결정
    int midpoint = histogram.cols / 2;
    int leftx_base, rightx_base;
    minMaxIdx(histogram.colRange(0, midpoint), NULL, NULL, NULL, &leftx_base);
    minMaxIdx(histogram.colRange(midpoint, histogram.cols), NULL, NULL, NULL, &rightx_base);
    rightx_base += midpoint;

    // 슬라이딩 윈도우 설정
    int nwindows = 9;
    int window_height = binary_warped.rows / nwindows;
    int margin = 100;
    int minpix = 50;

    // 차선 좌표 추적할 변수 초기화
    vector<int> leftx, rightx;

    // 윈도우 이동 및 차선 검출
    for (int window = 0; window < nwindows; ++window) {
        int win_y_low = binary_warped.rows - (window + 1) * window_height;
        int win_y_high = binary_warped.rows - window * window_height;

        int win_xleft_low = leftx_base - margin;
        int win_xleft_high = leftx_base + margin;
        int win_xright_low = rightx_base - margin;
        int win_xright_high = rightx_base + margin;

        // 윈도우 내에서 차선 좌표 식별
        vector<int> good_left_inds, good_right_inds;

        for (int i = 0; i < binary_warped.rows; ++i) {
            if ((i >= win_y_low) && (i < win_y_high) && (win_xleft_low >= 0) && (win_xleft_high < binary_warped.cols)) {
                if (binary_warped.at<uint8_t>(i, win_xleft_low) == 1)
                    good_left_inds.push_back(i * binary_warped.cols + win_xleft_low);
            }

            if ((i >= win_y_low) && (i < win_y_high) && (win_xright_low >= 0) && (win_xright_high < binary_warped.cols)) {
                if (binary_warped.at<uint8_t>(i, win_xright_low) == 1)
                    good_right_inds.push_back(i * binary_warped.cols + win_xright_low);
            }
        }

        // 좌표 저장
        for (int i : good_left_inds)
            leftx.push_back(i % binary_warped.cols);
        for (int i : good_right_inds)
            rightx.push_back(i % binary_warped.cols);

        // 다음 윈도우의 시작 위치 업데이트
        if (!good_left_inds.empty())
            leftx_base = static_cast<int>(accumulate(good_left_inds.begin(), good_left_inds.end(), 0) / good_left_inds.size());
        if (!good_right_inds.empty())
            rightx_base = static_cast<int>(accumulate(good_right_inds.begin(), good_right_inds.end(), 0) / good_right_inds.size());
    }

    // 왼쪽 차선의 2차 다항식 피팅
    vector<Point> left_points;
    for (size_t i = 0; i < leftx.size(); ++i) {
        left_points.push_back(Point(leftx[i], i));
    }
    if (!left_points.empty()) {
        VectorXd left_fit_coefficients = fitPoly(left_points, 2);
        left_fit = {left_fit_coefficients(2), left_fit_coefficients(1), left_fit_coefficients(0)};
    }

    // 오른쪽 차선의 2차 다항식 피팅
    vector<Point> right_points;
    for (size_t i = 0; i < rightx.size(); ++i) {
        right_points.push_back(Point(rightx[i], i));
    }
    if (!right_points.empty()) {
        VectorXd right_fit_coefficients = fitPoly(right_points, 2);
        right_fit = {right_fit_coefficients(2), right_fit_coefficients(1), right_fit_coefficients(0)};
    }
}

int main() {
    // 비디오 캡처 초기화
    VideoCapture cap("/home/harim/workspace/line_detection_project/resource/Sub_project.avi");  // 비디오 파일 경로를 설정하세요

    if (!cap.isOpened()) {
        cout << "Error opening video stream or file" << endl;
        return -1;
    }

    while (1) {
        Mat frame;
        // 비디오에서 프레임 읽기
        cap >> frame;

        // 비디오 끝에 도달하면 루프 종료
        if (frame.empty())
            break;

        // 이미지를 그레이스케일로 변환
        Mat gray;
        cvtColor(frame, gray, COLOR_BGR2GRAY);

        // 가우시안 블러 적용
        GaussianBlur(gray, gray, Size(5, 5), 0);

        // 엣지 감지 (Canny 알고리즘 사용)
        Mat edges;
        Canny(gray, edges, 50, 150, 3);

        // Perspective Transform을 사용하여 원근 변환 적용

        // 슬라이딩 윈도우 방식으로 차선 검출
        vector<double> left_fit, right_fit;
        slidingWindow(edges, frame, left_fit, right_fit);

        // 차선 그리기
        if(!left_fit.empty()){
            for (int i = 0; i < frame.rows; ++i) {
                double left_x = left_fit[0] * pow(i, 2) + left_fit[1] * i + left_fit[2];

                if (left_x >= 0 && left_x < frame.cols) {
                    // 왼쪽 차선은 빨간색
                    frame.at<Vec3b>(i, static_cast<int>(left_x)) = Vec3b(0, 0, 255);
                    circle(frame, Point(i, left_x), 50, Scalar(0, 0, 255), 1, 8, 0);
                }
            }
        }

        if(!right_fit.empty()){
            for (int i = 0; i < frame.rows; ++i) {
                double right_x = right_fit[0] * pow(i, 2) + right_fit[1] * i + right_fit[2];

                if (right_x >= 0 && right_x < frame.cols) {
                    // 오른쪽 차선은 녹색
                    frame.at<Vec3b>(i, static_cast<int>(right_x)) = Vec3b(0, 255, 0);
                    circle(frame, Point(i, right_x), 50, Scalar(0, 255, 0), 1, 8, 0);
                }
            }
        }
        

        // 결과 출력
        imshow("Lane Detection", frame);

        // 'ESC' 키를 누르면 루프 종료
        if (waitKey(30) == 27)
            break;
    }

    // 비디오 캡처 해제 및 창 닫기
    cap.release();
    destroyAllWindows();

    return 0;
}
