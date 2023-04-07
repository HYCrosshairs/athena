#include <opencv2/opencv.hpp>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <iostream>
#include <string>
#include <getopt.h>
#include <vector>
#include <cuda_runtime.h>
#include "Socket.hpp"

#include "Vectorization.cuh"

constexpr uint16_t PORT = 5000; // Port number to connect to
constexpr uint16_t WIDTH = 640; // Width of the video stream
constexpr uint16_t HEIGHT = 480; // Height of the video stream

using namespace service::system::network;


std::string preListName;
/*
void showUsage(const char *appName)
{
    std::cout << "Usage: --<command_name> <command_parameters>" << appName << std::endl;
    std::cout << "\t--ip-server|-i: Insert the server IP" << std::endl;
    std::cout << "\t--port|-p: Choose communication port" << std::endl;
}


int parse_command_line(int argc, char **argv)
{
    int ret{0};
    int opt;

    const option long_options[] = {
        {"ip-server",       no_argument, 0, 'i'},
        {"port",  no_argument, 0, 'p'},
        {"help",        no_argument, 0, 'h'},
        {0, 0, 0, 0}
    };

    while ((opt = getopt_long(argc, argv, "ip:h", long_options, NULL)) != -1) {
        switch (opt)
        {
        case 'i':
            break;
        case 'p':
            preListName = optarg;
            break;
        case 'h':
        default:
            showUsage(argv[0]);
            ret = -EINVAL;
            break;
        }
    }

    return ret;
}*/

int main()
{
    Vectorization();
    return 0;
}

/*
int main(int argc, const char *argv[])
{
    if (argc <= 0)
    {
    }
    
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

        // Display the received frame
        cv::imshow("Video Stream", frame);

        // Wait for a key press to exit the program
        if (cv::waitKey(1) == 27)
            break;
    }
    return 0;
}
*/