#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <X11/Xlib.h>
#include <X11/Xutil.h>

#include <cstdint>
#include <cstring>
#include <vector>

using namespace cv;

struct ScreenShot
{
    ScreenShot(int x, int y, int width, int height):
            x(x),
            y(y),
            width(width),
            height(height)
    {
        display = XOpenDisplay(nullptr);
        root = DefaultRootWindow(display);

        init = true;
    }

    void operator() (Mat& cvImg)
    {
        if(init == true)
            init = false;
        else
            XDestroyImage(img);

        img = XGetImage(display, root, x, y, width, height, AllPlanes, ZPixmap);

        cvImg = Mat(height, width, CV_8UC4, img->data);
    }

    ~ScreenShot()
    {
        if(init == false)
            XDestroyImage(img);

        XCloseDisplay(display);
    }

    Display* display;
    Window root;
    int x,y,width,height;
    XImage* img;

    bool init;
};

int main(int, char**)
{
    for(;;){
        ScreenShot screen(0,0,1366,768);

        Mat img;
        screen(img);

        imshow("img", img);
        if(waitKey(30) >= 0) break;
    }
    return 0;
}