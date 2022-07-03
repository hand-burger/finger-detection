#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/bgsegm.hpp>

using namespace std;
using namespace cv;
using namespace cv::bgsegm;

// Arguments on track function are for the trackbars
void track(int, void *);
Mat imgOriginal, fgMask;
Mat imgGray, imgCrop, imgCanny, imgMirror;

// Trackbar values
int thresh = 140, maxVal = 255;
int type = 2, value = 8;

int main()
{
    // Background subtractor
    Ptr<BackgroundSubtractor> pMOG2 = createBackgroundSubtractorMOG2();

    // Use for cropping images and rectangles
    Rect myRoi(288, 12, 288, 288);
    VideoCapture cap(0);

    // Main loop
    while (true)
    {
        cap.read(imgOriginal);

        // Use mirrored image (it's much more intuitive)
        flip(imgOriginal, imgMirror, 1);
        rectangle(imgMirror, myRoi, Scalar(0, 0, 255));

        // Crop image to the defined frame for image processing
        imgCrop = imgMirror(myRoi);
        // Pre-process image
        cvtColor(imgCrop, imgGray, COLOR_BGR2GRAY);
        GaussianBlur(imgGray, imgGray, Size(23, 23), 0);

        // Create trackbars
        namedWindow("Set");
        createTrackbar("Thresh", "Set", &thresh, 250, track);
        createTrackbar("Maximum", "Set", &maxVal, 255, track);
        createTrackbar("Thresh Type", "Set", &type, 4, track);
        createTrackbar("Kernel", "Set", &value, 100, track);

        // Subtract the background
        pMOG2->apply(imgCrop, fgMask);
        rectangle(fgMask, myRoi, Scalar(0, 0, 255));

        // Process image
        track(0, 0);

        // Show images
        imshow("Original img", imgMirror);
        imshow("Foreground mask", fgMask);
        imshow("Gray img", imgGray);

        waitKey(1);
    }

    return 0;
}

void track(int, void *)
{
    // Finger count and string for number of fingers
    int count = 0;
    string a;

    // Contour storage
    vector<vector<Point>> contours;
    vector<Vec4i> hierarchy;

    // Pre-process and find the contours of the foreground mask
    GaussianBlur(fgMask, fgMask, Size(27, 27), 3.5, 3.5);
    threshold(fgMask, fgMask, thresh, maxVal, type);
    Canny(fgMask, imgCanny, value, value * 2, 3);
    findContours(fgMask, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE, Point(0, 0));

    // Initialize empty mat (contours get drawn on it)
    Mat drawing = Mat::zeros(imgCanny.size(), CV_8UC3);

    if (contours.size() > 0)
    {
        // Get the biggest contour
        int indexOfBiggestContour = -1;
        long sizeOfBiggestContour = 0;
        for (int i = 0; i < contours.size(); i++)
        {
            if (contours[i].size() > sizeOfBiggestContour)
            {
                sizeOfBiggestContour = contours[i].size();
                indexOfBiggestContour = i;
            }
        }

        // Image processing storage
        vector<vector<int>> hull(contours.size());
        vector<vector<Point>> hullPoint(contours.size());
        vector<vector<Vec4i>> defects(contours.size());
        vector<vector<Point>> defectPoint(contours.size());
        vector<vector<Point>> conPoly(contours.size());
        Point2f rect_point[4];
        vector<RotatedRect> minRect(contours.size());
        vector<Rect> boundRect(contours.size());

        // Looping through each contour
        for (int i = 0; i < contours.size(); i++)
        {
            int area = contourArea(contours[i]);
            // Filter for small defects
            if (area > 5000)
            {
                // Get convex hulls and convexity defects of hand (fingers)
                convexHull(contours[i], hull[i], true);
                convexityDefects(contours[i], hull[i], defects[i]);
                // When on the biggest contour
                if (indexOfBiggestContour == i)
                {
                    // Get area of the rectangle of the biggest contour
                    minRect[i] = minAreaRect(contours[i]);
                    for (int j = 0; j < hull[i].size(); j++)
                    {
                        // Get hull point
                        int ind = hull[i][j];
                        hullPoint[i].push_back(contours[i][ind]);
                    }
                    // Reset finger count
                    count = 0;

                    // Loop through each defect and get the number of fingers
                    for (int j = 0; j < defects[i].size(); j++)
                    {
                        if (defects[i][j][3] > 13 * 256)
                        {
                            int p_end = defects[i][j][1];
                            int p_far = defects[i][j][2];
                            defectPoint[i].push_back(contours[i][p_far]);
                            circle(imgGray, contours[i][p_end], 3, Scalar(0, 255, 0), 2);
                            count++;
                        }
                    }

                    // Handle output
                    // Get count number and determine the number of fingers
                    a = to_string(count - 1);
                    if (count > 5 && count < 8)
                    {
                        a = "5";
                    }
                    else if (count >= 8)
                    {
                        a = "Show hand in red square";
                    }
                    else if (count == 0)
                    {
                        a = "0";
                    }

                    // Put the estimated number of fingers on the image
                    putText(imgMirror, a, Point(75, 420), FONT_HERSHEY_SIMPLEX, 3, Scalar(0, 255, 0), 3, 8, false);

                    // Draw the contours of the image and the convex hulls on the drawing image and gray
                    drawContours(drawing, contours, i, Scalar(0, 255, 0), 2, 8, vector<Vec4i>(), 0, Point());
                    drawContours(drawing, hullPoint, i, Scalar(0, 255, 255), 1, 8, vector<Vec4i>(), 0, Point());
                    drawContours(imgGray, hullPoint, i, Scalar(0, 0, 255), 2, 8, vector<Vec4i>(), 0, Point());
                    // Approximate the polygon shape
                    approxPolyDP(contours[i], conPoly[i], 3, false);
                    // Get bounding reactangle of the polygon for drawing the rectangle
                    boundRect[i] = boundingRect(conPoly[i]);
                    rectangle(imgGray, boundRect[i].tl(), boundRect[i].br(), Scalar(255, 0, 0), 2, 8, 0);
                    minRect[i].points(rect_point);
                    // Draw the lines of the hand
                    for (int j = 0; j < 4; j++)
                    {
                        line(imgGray, rect_point[j], rect_point[(j + 1) % 4], Scalar(0, 255, 0), 2, 8);
                    }
                }
            }
        }
    }
    imshow("Drawing", drawing);
}
