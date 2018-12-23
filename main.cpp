#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <X11/Xlib.h>
#include <X11/Xutil.h>

#include <cstdint>
#include <cstring>
#include <vector>
#include <stack>
#include <ctime>
#include <boost/algorithm/string.hpp>


using namespace cv;
using namespace std;


/// Global variables

int X = 347;
int Y = 1185;
int Height = 800;
int Width = 480;
int dilation_size = 1;
int resize_ratio = 100;
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

std::string exec(const char *cmd) {
    std::array<char, 128> buffer;
    std::string result;
    std::unique_ptr<FILE, decltype(&pclose)> pipe(popen(cmd, "r"), pclose);
    if (!pipe) {
        throw std::runtime_error("popen() failed!");
    }
    while (fgets(buffer.data(), buffer.size(), pipe.get()) != nullptr) {
        result += buffer.data();
    }
    return result;
}

Point2i getXY_enmulator() {
    int window_id;
    auto resp = exec("xwininfo -tree -root | grep \"Android Emulator\"");
    // cout << "Android emulator found:\n" << resp << endl;

    std::vector<std::string> results;
    boost::split(results, resp, [](char c) { return c == '+'; });
    return Point2i(stoi(results.at(3)), stoi(results.at(4)));
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

Mat CannyThreshold(Mat &src_gray) {

    const int lowThreshold = 1;
    const int ratio = 3;
    const int kernel_size = 3;
    const char *window_name = "Edge Map";
    /// Reduce noise with a kernel 3x3
    Mat dst, detected_edges;
    dst = Mat(Size(src_gray.size()), src_gray.type());
    blur(src_gray, detected_edges, Size(3, 3));

    /// Canny detector
    Canny(detected_edges, detected_edges, lowThreshold, lowThreshold * ratio, kernel_size);

    /// Using Canny's output as a mask, we display our result
    dst = Scalar::all(0);

    // src.copyTo(dst, detected_edges);

    return detected_edges;
}

int main(int, char **) {
    Point2i XY = getXY_enmulator();
    // cout << XY << endl;
    Mat prev_img;
    Mat prev_img_gray;
    Mat img;
    Mat img_gray;
    Mat diff, diff_gray;
    Mat matrix_a, prev_matrix_a;
    Mat test_image;
    prev_img.setTo(0);


    auto captured_image_window_name = "captured_image";
    auto captured_image_grayscale_window_name = "captured_image grayscale";
    auto difference_window_name = "difference";
    // auto grayscale_difference_window_name = "grayscale_difference";
    auto test_image_window_name = "test image";
    auto output_window_name = "OUTPUT";
    // namedWindow(captured_image_window_name, CV_WINDOW_AUTOSIZE);
    // namedWindow(captured_image_grayscale_window_name, CV_WINDOW_AUTOSIZE);
    // namedWindow(difference_window_name, CV_WINDOW_AUTOSIZE);
    // namedWindow(grayscale_difference_window_name, CV_WINDOW_AUTOSIZE);
    namedWindow(test_image_window_name, CV_WINDOW_AUTOSIZE);
    namedWindow(output_window_name, CV_WINDOW_AUTOSIZE);


    // namedWindow(source_window, CV_WINDOW_AUTOSIZE);
    // createTrackbar("Threshold: ", source_window, &thresh, max_thresh);
    // createTrackbar("dilation_size: ", source_window, &dilation_size, 50);
    // createTrackbar("resize ratio: ", source_window, &resize_ratio, 100);


    char str[200];
    Mat kernelHORIZONTAL = (Mat_<float>(3, 3) << -1, -1, -1, 2, 2, 2, -1, -1, -1);
    Mat kernelTR = (Mat_<float>(3, 3) << -1, -1, 2, -1, 2, -1, 2, -1, -1);
    Mat kernelTL = (Mat_<float>(3, 3) << 2, -1, -1, -1, 2, -1, -1, -1, 2);
    Point anchor = Point(-1, -1);
    double delta = 0;
    int ddepth = -1;
    int max_ddepth = 100;
    int threshold = 1;
    int max_thresholdh = 100;
    int ROItop = 190;
    int ROIheight = 470;
    createTrackbar("ddepth: ", test_image_window_name, &ddepth, max_ddepth);
    createTrackbar("threshold: ", output_window_name, &threshold, max_thresholdh);
    createTrackbar("ROI top: ", output_window_name, &ROItop, 800);
    createTrackbar("ROI height: ", output_window_name, &ROIheight, 800);
    std::vector<KeyPoint> keypoints;
    Mat output;

    SimpleBlobDetector detector;

    Mat horizontal;
    Mat TRdiagonal, TLdiagonal;

    for (;;) {
        tic();

        ScreenShot screen(XY.x, XY.y, Width, Height);
        screen(img);
        if (ROItop + ROIheight > img.rows) {
            ROIheight = img.rows - ROItop - 1;
            puts("ROI out of image.");
        }
        cv::Rect myROI(0, ROItop, 480, ROIheight);
        img = img(myROI);
        cv::cvtColor(img, img_gray, CV_BGR2GRAY);

        Mat EdgeMap = CannyThreshold(img_gray);
        Mat not_vertical_EdgeMap = EdgeMap.clone();
        filter2D(EdgeMap, not_vertical_EdgeMap, ddepth, kernelHORIZONTAL, anchor, delta, BORDER_DEFAULT);
        absdiff(EdgeMap, not_vertical_EdgeMap, matrix_a);
        dilate(matrix_a, matrix_a, getStructuringElement(MORPH_RECT, Size(2 * dilation_size + 1, 2 * dilation_size + 1),
                                               Point(dilation_size, dilation_size)));
        if (prev_matrix_a.size().empty()) {
            prev_matrix_a = matrix_a.clone();
        }
        else {
            // adaptiveThreshold(diff, diff, 255, CV_ADAPTIVE_THRESH_GAUSSIAN_C, CV_THRESH_BINARY, 11, -2);
            Mat temp_diff;
            absdiff(prev_matrix_a, matrix_a, temp_diff);

            imshow("temp_diff", temp_diff);
            prev_matrix_a = matrix_a;
        }
        // Laplacian(img_gray, test_image, ddepth);
        filter2D(img_gray, horizontal, ddepth, kernelHORIZONTAL, anchor, delta, BORDER_DEFAULT);
        filter2D(img_gray, TRdiagonal, ddepth, kernelTR, anchor, delta, BORDER_DEFAULT);
        filter2D(img_gray, TLdiagonal, ddepth, kernelTL, anchor, delta, BORDER_DEFAULT);
        // Sobel(img_gray, horizontal, ddepth, 1, 0);
        // Sobel(img_gray, TRdiagonal, ddepth, 1, 1);
        // Sobel(img_gray, TLdiagonal, ddepth, 0, 1);

        std::vector<cv::Mat> images(3);
        images.at(0) = horizontal; //for blue channel
        images.at(1) = TRdiagonal;   //for green channel
        images.at(2) = TLdiagonal;  //for red channel

        // cv::Mat colorImage;
        cv::merge(images, test_image);

        FAST(img_gray, keypoints, threshold);
        cvtColor(img_gray, output, CV_GRAY2BGR);
        drawKeypoints(output, keypoints, output, Scalar(0, 255, 255), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

        // for (auto &kp: keypoints) {
        //     circle(output, kp.pt, 4, Scalar(0, 255, 255));
        // }
        // detector.detect(img_gray, keypoints);



        imshow(test_image_window_name, test_image);
        imshow(output_window_name, output);

        // cornerHarris_loop(thresh);

        auto dt = toc();
        auto FPS = 1.0 / dt;
        sprintf(str, "%f  dt, FPS: %.2f", dt, FPS);
        putText(img_gray, String(str), Point2i(10, 50), FONT_HERSHEY_PLAIN, 2, Scalar(0, 0, 255, 255));
        // imshow(source_window, img_gray);

        auto k = waitKey(25);
        if (k == 'q') {
            std::cout << "Exiting." << std::endl;
            break;
        }
    }
    return 0;
}