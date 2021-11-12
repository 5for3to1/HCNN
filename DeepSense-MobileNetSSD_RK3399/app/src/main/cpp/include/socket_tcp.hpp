//
// Created by George on 2019/3/15.
//

#ifndef DEEPSENSE_MOBILENETSSD_3399_SOCKET_TCP_HPP
#define DEEPSENSE_MOBILENETSSD_3399_SOCKET_TCP_HPP
//socket
#include <netinet/in.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <sys/ioctl.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <iostream>
#include <string>
#include <cstdio>
#include <cstdlib>
#include <cstring>

//LOGI
#include <android/log.h>
#define LOG_TAG "hellojni"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO,LOG_TAG,__VA_ARGS__)

using namespace std;


#define UNIX_DOMAIN "/data/UNIX2.domain"

//
class tcpServer{

private:
    sockaddr_in server_addr;
    socklen_t server_addr_len;
    int connect_fd;
    int data_fd;

public:
    tcpServer(int port);
    ~tcpServer();
    void Bind();
    void Listen();
    void Accept();
    void Recv();
    void Send();
    void Close();
};

//
class tcpClient{

private:
    sockaddr_in server_addr;
    socklen_t server_addr_len;
    int data_fd;

public:
    tcpClient(string ip,int port);
    ~tcpClient();
    void Connect();
    void Send();
    void Recv();
};


//Server
tcpServer::tcpServer(int port)
{
    //清0
    bzero(&server_addr, sizeof(server_addr));
    //IPv4
    server_addr.sin_family = AF_INET;
    //将本机IP地址转换成网络字节序，INADDR_ANY代表本机所有IP地址
    server_addr.sin_addr.s_addr = htons(INADDR_ANY);
    server_addr.sin_port = htons(port);

    //获取套接字描述符
    connect_fd = socket(PF_INET, SOCK_STREAM, 0);
    if(connect_fd < 0)
    {
        LOGI("Server Create Socket Failed!");
    }
}

tcpServer::~tcpServer()
{
    close(connect_fd);
}

void tcpServer::Bind()
{
    int optval=1;
    if(0 > setsockopt(connect_fd,SOL_SOCKET,SO_REUSEADDR,&optval,sizeof(optval)))
    {
        LOGI("Failed to set address reuse!");
    }
    if(-1 == (bind(connect_fd, (struct sockaddr*)&server_addr, sizeof(server_addr))))
    {
        LOGI("Server Bind Failed!");
    }
}

void tcpServer::Listen()
{
    //5为tcp连接的客户端最大数量
    if(-1 == listen(connect_fd, 5))
    {
        LOGI("Server Listen Failed!");
    }
}

void tcpServer::Accept()
{
    //客户端信息
    struct sockaddr_in client_addr;
    socklen_t client_addr_len = sizeof(client_addr);

    //如果连接上一个新的客户端则返回一个新的fd
    data_fd = accept(connect_fd, (struct sockaddr*)&client_addr, &client_addr_len);
    if(0 > data_fd)
    {
        LOGI("Server Accept Failed!");
    } else
    {
        LOGI("Server Accept Success!");
    }
}

void tcpServer::Recv()
{
    int ret;
    char * data = new char[1];
    ret = recv(data_fd, data, sizeof(char), 0);
    if(-1 == ret)
    {
        LOGI("recv error!");
    }
    else
    {
        LOGI("recv success %c \n",data[0]);
    }
    delete[] data;
}

void tcpServer::Send()
{
    int ret;
    char * data = new char[1];
    data[0] = 'Q';
    ret = send(data_fd, data, sizeof(char), 0);
    if(-1 == ret)
    {
        LOGI("send error!");
    } else{
        LOGI("send success %c",data[0]);
    }
    delete[] data;
}

void tcpServer::Close()
{
    close(data_fd);
}

//client
tcpClient::tcpClient(string ip, int port)
{
    //从&servaddr指针所指的地址位置开始,将sizeof(sevaddr)字节置为0
    bzero(&server_addr, sizeof(server_addr));
    server_addr.sin_family = AF_INET;
    //将点分十进制IP转换为二进制整数，同时默认为网络字节序
    if(inet_pton(AF_INET, ip.c_str(),&server_addr.sin_addr) == 0)
    {
        LOGI("Server IP Address error!");
    }
    //将主机字节序的端口号转换为网络字节序
    server_addr.sin_port = htons(port);


    server_addr_len = sizeof(server_addr);

    data_fd = socket(AF_INET, SOCK_STREAM, 0);
    if(0 > data_fd)
    {
        LOGI("Client Create socket Failed!");
    }
}

tcpClient::~tcpClient()
{
    close(data_fd);
}

void tcpClient::Connect()
{
    //将客户端套接字连接服务器
    if(0 > connect(data_fd, (struct sockaddr*)&server_addr, server_addr_len))
    {
        LOGI("Can not Connect to Server IP!");
    } else{
        LOGI("Connect Success");
    }
}

//发送任务操作请求
void tcpClient::Send()
{
    int ret;
    char * data = new char[1];
    data[0]='L';
    ret = send(data_fd, data, sizeof(char), 0);
    if(-1 == ret)
    {
        LOGI("send error!");
    }
    else{
        LOGI("send success %c",data[0]);
    }
    delete[] data;
}

//接收确认数据
void tcpClient::Recv()
{
    int ret;
    char * data = new char[1];
    ret = recv(data_fd, data, sizeof(char), 0);
    if(-1 == ret)
    {
        LOGI("recv error!");
    } else
    {
        LOGI("recv success %c \n",data[0]);
    }
    delete[] data;
}



#endif //DEEPSENSE_MOBILENETSSD_3399_SOCKET_TCP_HPP
