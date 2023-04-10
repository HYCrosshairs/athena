/*#include "Socket.hpp"

#include <cstring>
#include <cstdlib>
#include <unistd.h>
#include <arpa/inet.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <netdb.h>

using namespace service::system::network;

Socket::Socket() 
{
    sockfd = socket(AF_INET, SOCK_STREAM, 0);
    if (sockfd < 0)
    {
        std::cerr << "Error creating socket\n";
        exit(EXIT_FAILURE);
    }
}

Socket::~Socket()
{
    close(sockfd);
}

void Socket::bind(unsigned short port)
{
    struct sockaddr_in addr = {0};
    addr.sin_family = AF_INET;
    addr.sin_port = htons(port);
    addr.sin_addr.s_addr = INADDR_ANY;

    int ret = ::bind(sockfd, (struct sockaddr*)&addr, sizeof(addr));
    if (ret < 0)
    {
        std::cerr << "Error binding socket\n";
        exit(EXIT_FAILURE);
    }
}

void Socket::listen(int backlog) 
{
    int ret = ::listen(sockfd, backlog);
    if (ret < 0)
    {
        std::cerr << "Error listening on socket\n";
        exit(EXIT_FAILURE);
    }
}

int Socket::accept()
{
    struct sockaddr_in addr = {0};
    socklen_t addrlen = sizeof(addr);
    int clientfd = ::accept(sockfd, (struct sockaddr*)&addr, &addrlen);
    if (clientfd < 0)
    {
        std::cerr << "Error accepting connection\n";
        exit(EXIT_FAILURE);
    }
    //clients.push_back(clientfd);
    return clientfd;
}

void Socket::connect(const std::string& hostname, unsigned short port) 
{
    struct addrinfo hints = {0};
    hints.ai_family = AF_INET;
    hints.ai_socktype = SOCK_STREAM;
    struct addrinfo* res;
    int ret = getaddrinfo(hostname.c_str(), std::to_string(port).c_str(), &hints, &res);
    
    if (ret not_eq 0)
    {
        std::cerr << "Error getting address info: " << gai_strerror(ret) << std::endl;
        exit(EXIT_FAILURE);
    }
    for (struct addrinfo* p = res; p != NULL; p = p->ai_next)
    {
        int fd = socket(p->ai_family, p->ai_socktype, p->ai_protocol);
        if (fd < 0)
        {
            continue;
        }
        if (::connect(fd, p->ai_addr, p->ai_addrlen) == 0)
        {
            sockfd = fd;
            break;
        }
        close(fd);
    }
    freeaddrinfo(res);
    if (sockfd < 0)
    {
        std::cerr << "Error connecting to host\n";
        exit(EXIT_FAILURE);
    }
}

int Socket::send(const char* buf, size_t len, int flags)
{
    return ::send(sockfd, buf, len, flags);
}

int Socket::recv(char* buf, size_t len, int flags)
{
    return ::recv(sockfd, buf, len, flags);
}

void Socket::handleClient(Socket& client)
{
    char buffer[1024];
    int bytesRecv;

    while (true)
    {
        bytesRecv = client.recv(buffer, sizeof(buffer));
        if (bytesRecv < 1)
        {
            break;
        }

        for (auto& clt : clients)
        {
            if (clt.getSockFd() == client.getSockFd())
            {
                continue;
            }
            clt.send(buffer, bytesRecv);
        }
    }
    //clients.erase(remove(clients.begin(), clients.end(), client), clients.end());
}

void Socket::receiveMessages(Socket& server)
{
    char buffer[1024];
    int bytesRecv;

    while (true)
    {
        bytesRecv = server.recv(buffer, sizeof(buffer));
        if (bytesRecv < 1)
        {
            break;
        }
        std::cout << buffer << std::endl;
    }
}

int Socket::getSockFd() const
{
    return sockfd;
}

void Socket::setSockFd(int sockfd)
{
    this->sockfd = sockfd;
}
*/