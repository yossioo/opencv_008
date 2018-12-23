//
// Created by yossi on 10/15/18.
//


#include <stdio.h>
#include <iostream>
#include <fstream>
#include <stdlib.h>
#include <opencv2/opencv.hpp>
#include <chrono>
#include "opencv2/core.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/features2d.hpp"
#include <opencv2/opencv.hpp>
#include "opencv2/xfeatures2d.hpp"

#define MANUAL_POINTS 1
#define SSTR(x) static_cast< std::ostringstream & >( \
    ( std::ostringstream() << std::fixed << std::setprecision(2)  << std::dec << x ) ).str()

using namespace cv;
using namespace std;
using namespace cv::xfeatures2d;

int calib = 0;
bool new_point = false;

void mouseCallback1(int evt, int x, int y, int flags, void *param) {

    if (evt == CV_EVENT_LBUTTONDOWN) {
        cv::Point2f *ptPtr = (cv::Point2f *) param;
        ptPtr->x = x;
        ptPtr->y = y;
        if (calib > 0)new_point = true;
    }
}

void mouseCallback2(int evt, int x, int y, int flags, void *param) {

    if (evt == CV_EVENT_LBUTTONDOWN) {
        cv::Point2f *ptPtr = (cv::Point2f *) param;
        ptPtr->x = x;
        ptPtr->y = y;
        if (calib > 0) new_point = true;
    }
}

void save_calibration_data(Mat img1_temp, Mat img2_temp, Mat H) {
    const char *ANGRY_HOME = getenv("ANGRY_HOME");
    if (ANGRY_HOME == NULL) {
        cout << "ANGRY_HOME is empty." << endl;
        return;
    }
    auto folder_for_save = string(ANGRY_HOME) + "/";
    if (calib == 0) {
        char i;
        cout << "Do you want to save calibration data? [y/n] ";
        i = getchar();
        // cin >> i;
        cout << "Got char: " << i << endl;

        if (i == 'y') {
            cout << "OK" << endl;
            cout << "Saving file to: " << folder_for_save << endl;
            cout << "Press input filename: ";
            string filename;
            cin >> filename;
            cout << "Got filename: " << filename << endl;
            auto filename_full = folder_for_save + string(filename) + ".yaml";

            ofstream myfile;
            myfile.open (filename_full);
            myfile << H.reshape(9) << endl;
            myfile.close();

            cout << "Done writing to: " << filename_full << endl;
        } else {
            cout << "Nevermind..." << endl;
        }
    }
}

bool find_points(Mat img, Size cs_size, vector<Point2f> &pointBuf) {
    Mat gray;
    cvtColor(img, gray, COLOR_BGR2GRAY);

    int chessBoardFlags = CALIB_CB_ADAPTIVE_THRESH | CALIB_CB_NORMALIZE_IMAGE;
    bool found = findChessboardCorners(gray, cs_size, pointBuf, chessBoardFlags);
    if (found) {

        cornerSubPix(gray, pointBuf, Size(11, 11),
                     Size(-1, -1), TermCriteria(TermCriteria::EPS + TermCriteria::COUNT, 30, 0.1));
        drawChessboardCorners(img, Size(5, 7), Mat(pointBuf), found);
        return true;
    }
    return false;
}


int main(int argc, char **argv) {
    //    VideoCapture cap1("v4l2src device=/dev/video2 ! video/x-raw, width=1920, height=1080, framerate=30/1, format=I420 ! videoconvert ! appsink"); // open the default camera
    int d1=0,d2=1;

    if (argc==3) {
        d1=atoi(argv[1]);
        d2=atoi(argv[2]);
    }
    VideoCapture cap1(d1);
    VideoCapture cap2(d2);
    cap1.set(CV_CAP_PROP_FRAME_WIDTH, 1920); // 640
    cap1.set(CV_CAP_PROP_FRAME_HEIGHT, 1080); //480
    cap1.set(CV_CAP_PROP_FPS, 30);
    cap1.set(CV_CAP_PROP_FOURCC, VideoWriter::fourcc('M', 'J', 'P', 'G'));
    cap2.set(CV_CAP_PROP_FRAME_WIDTH, 1920); // 640
    cap2.set(CV_CAP_PROP_FRAME_HEIGHT, 1080); //480
    cap2.set(CV_CAP_PROP_FPS, 30);
    cap2.set(CV_CAP_PROP_FOURCC, VideoWriter::fourcc('M', 'J', 'P', 'G'));


    if (!cap1.isOpened())  // check if we succeeded
    {
        cout << "Can't open camera 1" << endl;
        return -1;
    }
    if (!cap2.isOpened())  // check if we succeeded
    {
        cout << "Can't open camera 1" << endl;
        return -1;
    }

    Mat img1;
    Mat img2;

    namedWindow("NarrowCam", CV_WINDOW_KEEPRATIO);
    namedWindow("WideCam", CV_WINDOW_KEEPRATIO);
    Point2f pt1(0, 0), pt2(0, 0);
    cv::setMouseCallback("NarrowCam", mouseCallback1, (void *) &pt1);
    cv::setMouseCallback("WideCam", mouseCallback2, (void *) &pt2);
    /*Mat H = (Mat_<double>(3, 3) << 0.4896453431792935, -0.00547482441962388, 520.2560257556257,
            0.000761784422577072, 0.4910246493190614, 258.7433130382033,
            -3.954812311339419e-06, 2.497195667382972e-06, 1);*/

    // Updated matrix: 2 cameras, right-long-wide 11/11/2018, tested inside
    /*    Mat H = (Mat_<double>(3, 3) << 0.4897424647953454, -0.0001680104714870678, 440.0890477400966,
                -0.002704668756315309, 0.4909507216972098, 308.9967748651658,
                -2.333020454631999e-06,  -1.548543077907819e-06, 1);*/

    // Updated matrix: 2 cameras, right-long-wide 11/11/2018, tested outside
    Mat H = (Mat_<double>(3, 3) << 0.5139829966779133, 0.007821475945632927, 448.3906592767852,
            0.004526074087187859, 0.5052877535883971, 305.9255367367809,
            1.602895333815502e-05, 4.747723507680972e-06, 1.0);

    bool first_ok = true;
    vector<Point2f> points1, points2;
    for (;;) {
        auto start = chrono::steady_clock::now();
        // Acquire images
        cap1 >> img1;
        cap2 >> img2;
        bool found1 = false, found2 = false;

        if (calib == 0 && MANUAL_POINTS == 0) {
            points1.clear();
            points2.clear();
            found1 = find_points(img1, Size(5, 7), points1);
            found2 = find_points(img2, Size(5, 7), points2);
        } else {
            if (calib > 0 && calib <= 4 && new_point) {
                cout << pt1 << endl;
                points1.push_back(pt1);
                calib++;
                new_point = false;
            } else if (calib > 4 && calib <= 8 && new_point) {
                cout << pt2 << endl;
                points2.push_back(pt2);
                calib++;
                new_point = false;
            } else if (calib == 9) {
                calib = 0;
                puts("done");
                new_point = false;
                found1 = true;
                found2 = true;
            }


        }
        if (found1 && found2) {
            if (MANUAL_POINTS == 0) {
                //  H = findHomography(points1, points2, CV_RANSAC);
                vector<Point2f> _points1, _points2;
                _points1.push_back(points1[0]);
                _points1.push_back(points1[4]);
                _points1.push_back(points1[30]);
                _points1.push_back(points1[34]);

                _points2.push_back(points2[0]);
                _points2.push_back(points2[4]);
                _points2.push_back(points2[30]);
                _points2.push_back(points2[34]);
                H = getPerspectiveTransform(_points1, _points2);
            } else {
                H = getPerspectiveTransform(points1, points2);
            }
            first_ok = true;
            cout << H << endl;
            puts("Done calibrating.\n");

        }

        if (first_ok) {
            std::vector<Point2f> wide(4);
            std::vector<Point2f> narrow(4);

            narrow[0] = Point2f(0, 0);
            narrow[1] = Point2f(img1.cols, 0);
            narrow[2] = Point2f(img1.cols, img1.rows);
            narrow[3] = Point2f(0, img1.rows);
            //Point2f p=points1[0];
            narrow.push_back(pt1);
            perspectiveTransform(narrow, wide, H);

            line(img2, wide[0], wide[1], Scalar(0, 255, 0), 4);
            line(img2, wide[1], wide[2], Scalar(0, 255, 0), 4);
            line(img2, wide[2], wide[3], Scalar(0, 255, 0), 4);
            line(img2, wide[3], wide[0], Scalar(0, 255, 0), 4);


            circle(img1, pt1, 8, Scalar(0, 255, 255), 2);
            circle(img2, wide[4], 8, Scalar(0, 255, 255), 2);

        }
        /*
        int minHessian = 400;
        Ptr<SIFT> detector = SIFT::create( minHessian );
        std::vector<KeyPoint> keypoints1, keypoints2;
        Mat descriptors1, descriptors2;
        detector->detectAndCompute( img1, noArray(), keypoints1, descriptors1 );
        detector->detectAndCompute( img2, noArray(), keypoints2, descriptors2 );
        //-- Step 2: Matching descriptor vectors with a FLANN based matcher
        // Since SURF is a floating-point descriptor NORM_L2 is used
        Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create(DescriptorMatcher::FLANNBASED);
        std::vector< std::vector<DMatch> > knn_matches;
        matcher->knnMatch( descriptors1, descriptors2, knn_matches, 2 );
        //-- Filter matches using the Lowe's ratio test
        const float ratio_thresh = 0.9f;
        std::vector<DMatch> good_matches;
        for (size_t i = 0; i < knn_matches.size(); i++)
        {
            if (knn_matches[i][0].distance < ratio_thresh * knn_matches[i][1].distance)
            {
                good_matches.push_back(knn_matches[i][0]);
            }
        }
        //-- Draw matches
        Mat img_matches;
        drawMatches( img1, keypoints1, img2, keypoints2, good_matches, img_matches, Scalar::all(-1),
                     Scalar::all(-1), std::vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );
        //-- Show detected matches



        //-- Localize the object
        std::vector<Point2f> obj;
        std::vector<Point2f> scene;
 Mat H = findHomography( obj, scene, CV_RANSAC );

        for( int i = 0; i < good_matches.size(); i++ )
        {
            //-- Get the keypoints from the good matches
            obj.push_back( keypoints1[ good_matches[i].queryIdx ].pt );
            scene.push_back( keypoints2[ good_matches[i].trainIdx ].pt );
        }

        Mat H = findHomography( obj, scene, CV_RANSAC );

        //-- Get the corners from the image_1 ( the object to be "detected" )
        std::vector<Point2f> obj_corners(4);
        obj_corners[0] = cvPoint(0,0); obj_corners[1] = cvPoint( img1.cols, 0 );
        obj_corners[2] = cvPoint( img1.cols, img1.rows ); obj_corners[3] = cvPoint( 0, img1.rows );
        std::vector<Point2f> scene_corners(4);

        perspectiveTransform( obj_corners, scene_corners, H);

        //-- Draw lines between the corners (the mapped object in the scene - image_2 )
        line( img_matches, scene_corners[0] + Point2f( img1.cols, 0), scene_corners[1] + Point2f( img1.cols, 0), Scalar(0, 255, 0), 4 );
        line( img_matches, scene_corners[1] + Point2f( img1.cols, 0), scene_corners[2] + Point2f( img1.cols, 0), Scalar( 0, 255, 0), 4 );
        line( img_matches, scene_corners[2] + Point2f( img1.cols, 0), scene_corners[3] + Point2f( img1.cols, 0), Scalar( 0, 255, 0), 4 );
        line( img_matches, scene_corners[3] + Point2f( img1.cols, 0), scene_corners[0] + Point2f( img1.cols, 0), Scalar( 0, 255, 0), 4 );

        cv::Rect r = cv::boundingRect(scene_corners);
        Mat im_cropped=img2(r);

        //-- Show detected matches


        // Calculate the framerate and display results:
        auto end = chrono::steady_clock::now();
        auto diff = end - start;
        double fps = 1000.0 / (chrono::duration<double, milli>(diff).count());
        putText(img_matches, "FPS: " + SSTR(int(fps)), Point(30, 30 * 1 * 2), 1, 5,
                Scalar(50, 170, 50), 2);
        imshow( "Good Matches & Object detection", img_matches );

        resize(im_cropped,im_cropped,img1.size());
        imshow("crop",im_cropped);
        imshow("original",img1);
*/
        if (calib == 0) {
            imshow("NarrowCam", img1);
            imshow("WideCam", img2);
        }


        switch (waitKey(1)) {
            case 'c':
                calib = 1;
                points1.clear();
                points2.clear();
                break;
            case 's':
                save_calibration_data(img1, img2, H);
                break;
            case 'q':
                return 0;
                break;


        }


    }
    return 0;
}