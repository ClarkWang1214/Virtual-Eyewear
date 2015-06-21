#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <iostream>
#include <queue>
#include <stdio.h>
#include <math.h>

#include "constants.h"
#include "findEyeCenter.h"
#include "findEyeCorner.h"
#include "helpers.h"

using namespace cv;
// Constants
#define PI (22/7.0)

// Function Headers
bool detectAndDisplay( cv::Mat frame,float theta );

// Global variables
cv::String face_cascade_name = "/Users/andrewekhalel/Downloads/opencv-2.4.10/data/haarcascades/haarcascade_frontalface_alt.xml";
cv::CascadeClassifier face_cascade;
String glassesPath = "/Users/andrewekhalel/Developer/OpenCV test/OpenCV test/g1.png";
cv::Mat glasses;
std::string main_window_name = "TryIt";
cv::RNG rng(12345);
cv::Mat debugImage;
cv::Mat skinCrCbHist = cv::Mat::zeros(cv::Size(256, 256), CV_8UC1);
cv::Point leftPupil;
cv::Point rightPupil;

String glassesArr [6] = { "/Users/andrewekhalel/Developer/OpenCV test/OpenCV test/g1.png","/Users/andrewekhalel/Developer/OpenCV test/OpenCV test/g2.png","/Users/andrewekhalel/Developer/OpenCV test/OpenCV test/g3.png","/Users/andrewekhalel/Developer/OpenCV test/OpenCV test/g4.png","/Users/andrewekhalel/Developer/OpenCV test/OpenCV test/g5.png","/Users/andrewekhalel/Developer/OpenCV test/OpenCV test/g6.png"};
int glassesX=0;

int main( int argc, const char** argv ) {
    CvCapture* capture;
    cv::Mat frame;
    // Load the cascades
    if( !face_cascade.load( face_cascade_name ) ){ printf("--(!)Error loading face cascade, please change face_cascade_name in source code.\n"); return -1; };
    
    cv::namedWindow(main_window_name,CV_WINDOW_NORMAL);
    
   /* createCornerKernels();
    ellipse(skinCrCbHist, cv::Point(113, 155.6), cv::Size(23.4, 15.2),
           43.0, 0.0, 360.0, cv::Scalar(255, 255, 255), -1);*/
    
    // Read the video stream
    capture = cvCaptureFromCAM( -1 );
    if( capture ) {
        while( true ) {
            glasses = imread(glassesPath,-1);
            frame = cvQueryFrame( capture );
            // mirror it
            cv::flip(frame, frame, 1);
            frame.copyTo(debugImage);
            
            // Apply the classifier to the frame
            if( !frame.empty() ) {
                float fraction = 0.15;
                for(float theta = -PI*fraction;theta<=PI*fraction+0.01;theta+=PI*fraction)
                {
                    // try different orientations
                    cv::Mat dst;
                    rotate(frame,(int)((theta*180)/PI),dst,cv::Point2f(frame.cols/2,frame.rows/2));
                    if(detectAndDisplay( dst ,theta))
                        break;
                }
            }
            else {
                printf(" --(!) No captured frame -- Break!");
                break;
            }
            
            imshow(main_window_name,debugImage);
            
            int c = cv::waitKey(10);
            if( (char)c == 'q' ) { break; }
            if( (char)c == 's' ) {
                imwrite("/Users/andrewekhalel/Desktop/frame.png",debugImage);
            }
            if( (char)c == 'n')
            {
                if(glassesX == 5)
                    glassesX =0;
                else
                    glassesX++;
                glassesPath = glassesArr[glassesX];
            }
            
        }
    }
    
    releaseCornerKernels();
    
    return 0;
}

void findEyes(cv::Mat frame_gray, cv::Rect face,float theta) {
    cv::Mat faceROI = frame_gray(face);
    cv::Mat debugFace = faceROI;
    
    if (kSmoothFaceImage) {
        double sigma = kSmoothFaceFactor * face.width;
        GaussianBlur( faceROI, faceROI, cv::Size( 0, 0 ), sigma);
    }
    //-- Find eye regions and draw them
    int eye_region_width = face.width * (kEyePercentWidth/100.0);
    int eye_region_height = face.width * (kEyePercentHeight/100.0);
    int eye_region_top = face.height * (kEyePercentTop/100.0);
    cv::Rect leftEyeRegion;
    cv::Rect rightEyeRegion;
    leftEyeRegion = Rect(face.width*(kEyePercentSide/100.0),
                           eye_region_top,eye_region_width,eye_region_height);
    rightEyeRegion = Rect(face.width - eye_region_width - face.width*(kEyePercentSide/100.0),
                            eye_region_top,eye_region_width,eye_region_height);
    
    //-- Find Eye Centers
    leftPupil = findEyeCenter(faceROI,leftEyeRegion,"Left Eye");
    rightPupil = findEyeCenter(faceROI,rightEyeRegion,"Right Eye");
    
    
    // change eye centers to face coordinates
    rightPupil.x += rightEyeRegion.x + face.x;
    rightPupil.y += rightEyeRegion.y + face.y;
    leftPupil.x += leftEyeRegion.x + face.x;
    leftPupil.y += leftEyeRegion.y + face.y;

    //reverse the rotation
    //x' = x*cos(alpha) - y*sin(alpha)
    //y' = x*sin(alpha) + y*cos(alpha)
    cv::Point ImageCenter (debugImage.cols/2,debugImage.rows/2);
    
    rightPupil.x-=ImageCenter.x;
    rightPupil.y-=ImageCenter.y;
    int x = rightPupil.x,y = rightPupil.y;
    rightPupil.x = x*cos(theta) - y*sin(theta);
    rightPupil.y = x*sin(theta) + y*cos(theta);
    rightPupil.x+=ImageCenter.x;
    rightPupil.y+=ImageCenter.y;
    
    leftPupil.x-=ImageCenter.x;
    leftPupil.y-=ImageCenter.y;
    x = leftPupil.x,y = leftPupil.y;
    leftPupil.x = x*cos(theta) - y*sin(theta);
    leftPupil.y = x*sin(theta) + y*cos(theta);
    leftPupil.x+=ImageCenter.x;
    leftPupil.y+=ImageCenter.y;
    
    
    
}

bool detectAndDisplay( cv::Mat frame, float theta ) {
    std::vector<cv::Rect> faces;
    //cv::Mat frame_gray;
    
    std::vector<cv::Mat> rgbChannels(3);
    cv::split(frame, rgbChannels);
    cv::Mat frame_gray = rgbChannels[2];
    
    //-- Detect faces
    face_cascade.detectMultiScale( frame_gray, faces, 1.1, 2, 0|CV_HAAR_SCALE_IMAGE|CV_HAAR_FIND_BIGGEST_OBJECT, cv::Size(150, 150) );
    //-- Show what you got
    if (faces.size() > 0) {
        findEyes(frame_gray, faces[0],theta);
        //line(debugImage,leftPupil,rightPupil,255,5);
        int lineLength = ceil(sqrt(pow(rightPupil.x-leftPupil.x,2)+pow(rightPupil.y-leftPupil.y,2)));
        float rate =2;
        Size ss (lineLength*rate,glasses.rows * lineLength*rate / glasses.cols);
        Mat newglasses;
        resize(glasses, newglasses,ss);
        float alpha = atan((rightPupil.y-leftPupil.y)*1.0/(rightPupil.x-leftPupil.x));
        int dim =  sqrt(newglasses.cols * newglasses.cols + newglasses.rows * newglasses.rows);
        Mat toRotate = Mat::zeros(2*dim,2*dim,newglasses.type());
        newglasses.copyTo(toRotate(Rect(dim,dim,newglasses.cols,newglasses.rows)));
        rotate(toRotate, -alpha*180/PI, toRotate,Point2f(dim,dim));
        
        float scale = newglasses.cols*1.0/(glasses.cols>glasses.rows?glasses.cols:glasses.rows);
        float z,r;
        if(glassesX != 5){
         z = atan((glasses.rows*scale/2)/(glasses.cols*scale/4));
         r = sqrt((glasses.rows*scale/2)*(glasses.rows*scale/2) + (glasses.cols*scale/4)*(glasses.cols*scale/4));
            Point2f match = Point2f(r*cos(z+alpha),r*sin(z+alpha));
            overlayImage(debugImage, toRotate, debugImage, Point2f(leftPupil.x - dim - match.x, leftPupil.y - dim -match.y));
            
        }
        else
        {
            z = atan((glasses.rows*scale*0.68)/(glasses.cols*scale/4));
            r = sqrt((glasses.rows*scale*0.68)*(glasses.rows*scale*0.68) + (glasses.cols*scale/4)*(glasses.cols*scale/4));
            Point2f match = Point2f(r*cos(z+alpha),r*sin(z+alpha));
            overlayImage(debugImage, toRotate, debugImage, Point2f(leftPupil.x - dim - match.x-dim/16, leftPupil.y - dim -match.y-dim/4));
            
        }
        return true;
    }
    return false;
}
