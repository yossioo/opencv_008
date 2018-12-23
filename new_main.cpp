#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <X11/Xlib.h>
#include <X11/Xutil.h>

#include <cstdint>
#include <cstring>
#include <vector>
#include <stack>
#include <ctime>


using namespace cv;
using namespace std;


/// Global variables

int X = 454;
int Y = 1311;
int Height = 800;
int Width = 480;
int dilation_size = 1;
int resize_ratio = 25;
Mat src, src_gray;
int thresh = 200;
int max_thresh = 255;
char *source_window = "Source image";
char *corners_window = "Corners detected";

struct ScreenShot {
    ScreenShot(int x, int y, int width, int height) : x(x), y(y), width(width), height(height) {
        display = XOpenDisplay(nullptr);
        root = DefaultRootWindow(display);

        init = true;
    }

    void operator()(Mat &cvImg) {
        if (init == true) {
            init = false;
        }
        else
            XDestroyImage(img);

        img = XGetImage(display, root, x, y, width, height, AllPlanes, ZPixmap);

        cvImg = Mat(height, width, CV_8UC4, img->data);
    }

    ~ScreenShot() {
        if (init == false)
            XDestroyImage(img);

        XCloseDisplay(display);
    }

    Display *display;
    Window root;
    int x, y, width, height;
    XImage *img;

    bool init;
};

std::stack<clock_t> tictoc_stack;

void tic() {
    tictoc_stack.push(clock());
}

double toc() {
    double dt = ((double) (clock() - tictoc_stack.top())) / CLOCKS_PER_SEC;
    tictoc_stack.pop();
    return dt;
}

void cornerHarris_loop(int thresh) {
    Mat dst, dst_norm, dst_norm_scaled;
    dst = Mat::zeros(src.size(), CV_32FC1);

    /// Detector parameters
    int blockSize = 2;
    int apertureSize = 3;
    double k = 0.04;

    /// Detecting corners
    cornerHarris(src_gray, dst, blockSize, apertureSize, k, BORDER_DEFAULT);
    dilate(dst, dst, getStructuringElement(MORPH_RECT, Size(2 * dilation_size + 1, 2 * dilation_size + 1),
                                                       Point(dilation_size, dilation_size)));
    /// Normalizing
    normalize(dst, dst_norm, 0, 255, NORM_MINMAX, CV_32FC1, Mat());
    convertScaleAbs(dst_norm, dst_norm_scaled);

    /// Drawing a circle around corners
    for (int j = 0; j < dst_norm.rows; j++) {
        for (int i = 0; i < dst_norm.cols; i++) {
            if ((int) dst_norm.at<float>(j, i) > thresh) {
                circle(dst_norm_scaled, Point(i, j), 5, Scalar(0), 2, 8, 0);
            }
        }
    }
    /// Showing the result
    namedWindow(corners_window, CV_WINDOW_AUTOSIZE);
    imshow(corners_window, dst_norm_scaled);
}

int main(int, char **) {

    Mat prev_img;
    Mat prev_img_gray;
    Mat img;
    Mat img_gray;
    Mat diff, diff_gray;

    prev_img.setTo(0);


    auto captured_image_window_name = "captured_image";
    auto captured_image_grayscale_window_name = "captured_image grayscale";
    auto difference_window_name = "difference";
    auto grayscale_difference_window_name = "grayscale_difference";
    namedWindow(captured_image_window_name, CV_WINDOW_AUTOSIZE);
    namedWindow(captured_image_grayscale_window_name, CV_WINDOW_AUTOSIZE);
    namedWindow(difference_window_name, CV_WINDOW_AUTOSIZE);
    namedWindow(grayscale_difference_window_name, CV_WINDOW_AUTOSIZE);


    namedWindow(source_window, CV_WINDOW_AUTOSIZE);
    createTrackbar("Threshold: ", source_window, &thresh, max_thresh);
    createTrackbar("dilation_size: ", source_window, &dilation_size, 50);
    createTrackbar("resize ratio: ", source_window, &resize_ratio, 100);


    char str[200];

    for (;;) {
        tic();

        ScreenShot screen(X, Y, Width, Height);
        screen(img);
        cv::cvtColor(img, img_gray, CV_BGR2GRAY);
        // src_gray = img_gray;
        if (prev_img.empty()) {
            prev_img = img.clone();
        }
        if (prev_img_gray.empty()) {
            prev_img_gray = img_gray.clone();
        }


        absdiff(img, prev_img, diff);
        absdiff(img_gray, prev_img_gray, diff_gray);
        prev_img = img.clone();
        prev_img_gray = img_gray.clone();

        cv::resize(diff_gray, diff_gray, cv::Size(), (float)resize_ratio/100.0, (float)resize_ratio/100.0);


        src_gray = diff_gray;

        if (img.size().width > 0 && img.size().height > 0) {
            imshow(captured_image_window_name, img);
        }
        if (img_gray.size().width > 0 && img_gray.size().height > 0) {
            imshow(captured_image_grayscale_window_name, img_gray);
        }
        if (diff_gray.size().width > 0 && diff_gray.size().height > 0) {
            imshow(grayscale_difference_window_name, diff_gray);
        }
        if (diff.size().width > 0 && diff.size().height > 0) {
            imshow(difference_window_name, diff);
        }


        cornerHarris_loop(thresh);

        auto dt = toc();
        auto FPS = 1.0 / dt;
        sprintf(str, "%f  dt, FPS: %.2f", dt, FPS);
        putText(img_gray, String(str), Point2i(10, 50), FONT_HERSHEY_PLAIN, 2, Scalar(0, 0, 255, 255));
        imshow(source_window, img_gray);

        auto k = waitKey(25);
        if (k == 'q') {
            std::cout << "Exiting." << std::endl;
            break;
        }
    }
    return 0;
}