#include "Socket.hpp"

#include <iostream>

using namespace system::service::network;

int main(int argc, char **argv)
{
    Socket client;

    client.connect("192.168.1.67", 5000);

    while (true)
    {
        int received{0};
        char* data = nullptr;
        while (received < 0)
        {
            int bytes = client.recv(data + received, sizeof(char));

            if(bytes == -1)
            {
                std::cerr <<  "Unable to receive the data" << std::endl;
                break;
            }
            received += bytes;
        }

        if (received == 0)
        {
            std::cerr << "Connection closed by the server" << std::endl;
            break;
        }
    }    
    return 0;
}