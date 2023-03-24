#include <iostream>
#include <string>

#include "Socket.hpp"

using namespace service::system::network;

int main()
{
    Socket clientSocket;
    clientSocket.connect("192.168.1.67", 5000);
    std::cout << "Connected to server!\n";

    //std::string message;
    while (true)
    {
        std::cout << "Message to send : \n";
        
        clientSocket.send("Okay", 4);
        char buf[1024] = {0};
        int n = clientSocket.recv(buf, sizeof(buf));
        std::cout << "Received message: " << buf << "\n";
    }

    return 0;
}
