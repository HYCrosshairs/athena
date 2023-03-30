#include <opencv2/opencv.hpp>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <iostream>
#include <string>

#include "Socket.hpp"

constexpr uint16_t PORT = 5000; // Port number to connect to
constexpr uint16_t WIDTH = 640; // Width of the video stream
constexpr uint16_t HEIGHT = 480; // Height of the video stream

using namespace service::system::network;
//using namespace cv;

int main()
{
    Socket clientSocket;
    clientSocket.connect("192.168.1.67", PORT);
    std::cout << "Connected to server!\n";

    // Loop to receive frames from the server and display them
    while (true)
    {
        cv::Mat frame(HEIGHT, WIDTH, CV_8UC3);
        int size = frame.total() * frame.elemSize();
        int received = 0;
        while (received < size)
        {
            int bytes = clientSocket.recv(reinterpret_cast<char*>(frame.data + received), size - received, 0);
            if (bytes == -1)
            {
                std::cerr << "Unable to receive the frame data" << std::endl;
                break;
            }
            received += bytes;
        }

        // Check if the connection has been closed
        if (received == 0)
        {
            std::cerr << "Connection closed by the server" << std::endl;
            break;
        }

        // Convert to grayscale
        cv::Mat gray;
        cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);

        // Apply edge detection
        cv::Mat edges;
        cv::Canny(gray, edges, 25, 250, 3);

        // Find contours
        std::vector<std::vector<cv::Point>> contours;
        cv::findContours(edges, contours, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE);

        // Approximate contours to polygons and draw rectangles
        for (size_t i = 0; i < contours.size(); i++) 
        {
            std::vector<cv::Point> approx;
            cv::approxPolyDP(contours[i], approx, cv::arcLength(contours[i], true) * 0.02, true);
            
            if (approx.size() == 4 && cv::isContourConvex(approx)) 
            {
                cv::rectangle(frame, approx[0], approx[2], cv::Scalar(0, 255, 0), 3);
            }
        }

        /*
        // Apply thresholding
        cv::Mat thresh;
        cv::threshold(gray, thresh, 128, 255, cv::THRESH_BINARY);

        // Find contours
        std::vector<std::vector<cv::Point>> contours;
        cv::findContours(thresh, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

        // Iterate over contours
        for (auto& contour : contours) 
        {
            // Approximate contour as a polygon
            std::vector<cv::Point> polygon;
            cv::approxPolyDP(contour, polygon, 10, true);

            // If polygon has four sides, it's likely to be a rectangle
            if (polygon.size() == 4) 
            {
                // Draw rectangle on frame
                cv::polylines(frame, polygon, true, cv::Scalar(0, 255, 0), 2);
            }
        }
        */
        // Display the received frame
        cv::imshow("Video Stream", frame);

        // Wait for a key press to exit the program
        if (cv::waitKey(1) == 27)
            break;
    }
    return 0;
}
