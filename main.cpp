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
        char sendBuf[1024];
        std::cout << "Message to send : \n";
        std::cin.getline(sendBuf, sizeof(sendBuf));

        clientSocket.send(sendBuf, sizeof(sendBuf));
        char buf[1024] = {0};
        int n = clientSocket.recv(buf, sizeof(buf));
        std::cout << "Received message: " << buf << "\n";
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
