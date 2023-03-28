#include <opencv2/opencv.hpp>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <iostream>
#include <string>

#include "Socket.hpp"

#define PORT 5000 // Port number to connect to
#define WIDTH 640 // Width of the video stream
#define HEIGHT 480 // Height of the video stream

using namespace service::system::network;
using namespace cv;

int main()
{
    Socket clientSocket;
    clientSocket.connect("192.168.1.67", 5000);
    std::cout << "Connected to server!\n";

    // Loop to receive frames from the server and display them
    while (true)
    {
        Mat frame(HEIGHT, WIDTH, CV_8UC3);
        int size = frame.total() * frame.elemSize();
        int received = 0;
        while (received < size)
        {
            int bytes = clientSocket.recv(sockfd, frame.data + received, size - received, 0);
            if (bytes == -1)
            {
                cerr << "Unable to receive the frame data" << endl;
                break;
            }
            received += bytes;
        }

        // Check if the connection has been closed
        if (received == 0)
        {
            cerr << "Connection closed by the server" << endl;
            break;
        }

        // Convert to grayscale
        cv::Mat gray;
        cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);

        // Apply thresholding
        cv::Mat thresh;
        cv::threshold(gray, thresh, 128, 255, cv::THRESH_BINARY);

        // Find contours
        std::vector<std::vector<cv::Point>> contours;
        cv::findContours(thresh, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

        // Iterate over contours
        for (auto& contour : contours) {
            // Approximate contour as a polygon
            std::vector<cv::Point> polygon;
            cv::approxPolyDP(contour, polygon, 10, true);

            // If polygon has four sides, it's likely to be a rectangle
            if (polygon.size() == 4) {
                // Draw rectangle on frame
                cv::polylines(frame, polygon, true, cv::Scalar(0, 255, 0), 2);
            }
        }

        // Display the received frame
        imshow("Video Stream", frame);

        // Wait for a key press to exit the program
        if (waitKey(1) == 27)
            break;
    }

/*
    Socket serverSocket;
    serverSocket.bind(5000);
    serverSocket.listen();

    std::vector<Socket> clients;
    while (true) 
    {
        Socket client;
        int client_fd = serverSocket.accept();
        client.setSockFd(client_fd);
        clients.push_back(client);

        thread t(client_handler, client);
        t.detach();
    }
*/
    

    return 0;
}
