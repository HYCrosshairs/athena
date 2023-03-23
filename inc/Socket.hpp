#include <iostream>
#include <string>

namespace system::service::network
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

private:
    int sockfd;
};    
} // namespace system::service::network
