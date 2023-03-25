#pragma once

#include <iostream>
#include <string>
#include <vector>

namespace service::system::network
{
class Socket 
{
public:
    Socket();

    ~Socket();

    void bind(unsigned short port);

    void listen(int backlog = 5);

    int accept();

    void connect(const std::string& hostname, unsigned short port);

    int send(const char* buf, size_t len, int flags = 0);

    int recv(char* buf, size_t len, int flags = 0);

    void handleClient(Socket& client);

    void receiveMessages(Socket& server);

    int getSockFd() const;

    void setSockFd(int sockfd);

private:
    int sockfd;
    std::vector<Socket> clients;
};    
} // namespace service::system::network
