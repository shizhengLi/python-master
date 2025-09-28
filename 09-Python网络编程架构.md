# Python网络编程架构：从套接字到分布式系统的深度解析

## 引言：网络编程的哲学思考

网络编程是现代软件开发的基石，它体现了"连接与通信"的深刻哲学。在数字时代，几乎所有的应用程序都需要在网络上进行通信，理解网络编程的底层原理和架构设计，对于构建高性能、可靠的分布式系统至关重要。

从哲学角度来看，网络编程反映了"分布式思维"——将计算和存储分布在不同的节点上，通过网络协议进行协调和通信。这种思维模式要求我们从单机的、同步的编程范式，转向分布式的、异步的编程范式。

## 网络编程基础架构

### 1. 套接字编程基础

```python
import socket
import threading
import select
import json
import time
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from enum import Enum

class SocketProgramming:
    """套接字编程基础"""

    def demonstrate_tcp_server(self):
        """演示TCP服务器"""
        print("=== TCP服务器演示 ===")

        def start_tcp_server(host='localhost', port=8888):
            """启动TCP服务器"""
            server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            server_socket.bind((host, port))
            server_socket.listen(5)
            print(f"TCP服务器启动在 {host}:{port}")

            try:
                while True:
                    client_socket, client_address = server_socket.accept()
                    print(f"接受来自 {client_address} 的连接")

                    client_thread = threading.Thread(
                        target=self.handle_tcp_client,
                        args=(client_socket, client_address)
                    )
                    client_thread.start()

            except KeyboardInterrupt:
                print("TCP服务器关闭")
            finally:
                server_socket.close()

        def handle_tcp_client(self, client_socket, client_address):
            """处理TCP客户端"""
            try:
                while True:
                    data = client_socket.recv(1024)
                    if not data:
                        break

                    message = data.decode('utf-8')
                    print(f"来自 {client_address} 的消息: {message}")

                    response = f"服务器回复: {message}"
                    client_socket.send(response.encode('utf-8'))

            except Exception as e:
                print(f"处理客户端 {client_address} 时出错: {e}")
            finally:
                client_socket.close()
                print(f"客户端 {client_address} 连接关闭")

        return start_tcp_server

    def demonstrate_tcp_client(self):
        """演示TCP客户端"""
        print("\n=== TCP客户端演示 ===")

        def tcp_client_example(host='localhost', port=8888):
            """TCP客户端示例"""
            client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

            try:
                client_socket.connect((host, port))
                print(f"连接到服务器 {host}:{port}")

                messages = ["Hello", "World", "Python", "Network"]
                for message in messages:
                    client_socket.send(message.encode('utf-8'))
                    response = client_socket.recv(1024)
                    print(f"服务器回复: {response.decode('utf-8')}")
                    time.sleep(1)

            except Exception as e:
                print(f"TCP客户端错误: {e}")
            finally:
                client_socket.close()

        return tcp_client_example

    def demonstrate_udp_communication(self):
        """演示UDP通信"""
        print("\n=== UDP通信演示 ===")

        def udp_server(host='localhost', port=8889):
            """UDP服务器"""
            server_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            server_socket.bind((host, port))
            print(f"UDP服务器启动在 {host}:{port}")

            try:
                while True:
                    data, client_address = server_socket.recvfrom(1024)
                    message = data.decode('utf-8')
                    print(f"来自 {client_address} 的UDP消息: {message}")

                    response = f"UDP服务器回复: {message}"
                    server_socket.sendto(response.encode('utf-8'), client_address)

            except KeyboardInterrupt:
                print("UDP服务器关闭")
            finally:
                server_socket.close()

        def udp_client(host='localhost', port=8889):
            """UDP客户端"""
            client_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

            try:
                messages = ["UDP Message 1", "UDP Message 2", "UDP Message 3"]
                for message in messages:
                    client_socket.sendto(message.encode('utf-8'), (host, port))
                    data, _ = client_socket.recvfrom(1024)
                    print(f"UDP服务器回复: {data.decode('utf-8')}")
                    time.sleep(0.5)

            except Exception as e:
                print(f"UDP客户端错误: {e}")
            finally:
                client_socket.close()

        return udp_server, udp_client

    def demonstrate_select_multiplexing(self):
        """演示select多路复用"""
        print("\n=== Select多路复用演示 ===")

        def select_server(host='localhost', port=8890):
            """使用select的服务器"""
            server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            server_socket.bind((host, port))
            server_socket.listen(5)
            server_socket.setblocking(False)

            inputs = [server_socket]
            outputs = []
            message_queues = {}

            print(f"Select服务器启动在 {host}:{port}")

            try:
                while inputs:
                    readable, writable, exceptional = select.select(inputs, outputs, inputs, 1)

                    for s in readable:
                        if s is server_socket:
                            client_socket, client_address = s.accept()
                            print(f"新连接: {client_address}")
                            client_socket.setblocking(False)
                            inputs.append(client_socket)
                            message_queues[client_socket] = []
                        else:
                            data = s.recv(1024)
                            if data:
                                message = data.decode('utf-8')
                                print(f"收到消息: {message}")
                                message_queues[s].append(data)
                                if s not in outputs:
                                    outputs.append(s)
                            else:
                                print(f"连接关闭: {s.getpeername()}")
                                if s in outputs:
                                    outputs.remove(s)
                                inputs.remove(s)
                                s.close()
                                del message_queues[s]

                    for s in writable:
                        if message_queues[s]:
                            next_msg = message_queues[s].pop(0)
                            s.send(next_msg)
                        else:
                            outputs.remove(s)

                    for s in exceptional:
                        print(f"异常连接: {s.getpeername()}")
                        inputs.remove(s)
                        if s in outputs:
                            outputs.remove(s)
                        s.close()
                        del message_queues[s]

            except KeyboardInterrupt:
                print("Select服务器关闭")
            finally:
                server_socket.close()

        return select_server

# 运行套接字编程演示
socket_demo = SocketProgramming()

# 注意：这些函数需要单独运行，不能在同一个脚本中同时运行多个服务器
# tcp_server = socket_demo.demonstrate_tcp_server()
# tcp_client = socket_demo.demonstrate_tcp_client()
# udp_server, udp_client = socket_demo.demonstrate_udp_communication()
# select_server = socket_demo.demonstrate_select_multiplexing()
```

### 2. 高级网络协议实现

```python
import asyncio
import ssl
import http.client
import xmlrpc.client
import xmlrpc.server
import json
import base64
import hashlib
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from abc import ABC, abstractmethod

class AdvancedProtocols:
    """高级网络协议"""

    def demonstrate_http_client(self):
        """演示HTTP客户端"""
        print("=== HTTP客户端演示 ===")

        # 使用http.client
        def http_client_example():
            """HTTP客户端示例"""
            try:
                conn = http.client.HTTPSConnection("httpbin.org")
                conn.request("GET", "/get")

                response = conn.getresponse()
                print(f"状态码: {response.status}")
                print(f"响应头: {response.getheaders()}")

                data = response.read().decode('utf-8')
                print(f"响应数据: {data[:200]}...")

                conn.close()

            except Exception as e:
                print(f"HTTP客户端错误: {e}")

        # 使用XML-RPC
        def xmlrpc_client_example():
            """XML-RPC客户端示例"""
            try:
                proxy = xmlrpc.client.ServerProxy("http://localhost:8000/")

                # 调用远程方法
                try:
                    result = proxy.add(2, 3)
                    print(f"XML-RPC结果: {result}")
                except:
                    print("XML-RPC服务器不可用")

            except Exception as e:
                print(f"XML-RPC客户端错误: {e}")

        def xmlrpc_server_example():
            """XML-RPC服务器示例"""
            try:
                class MathFunctions:
                    def add(self, x, y):
                        return x + y

                    def subtract(self, x, y):
                        return x - y

                    def multiply(self, x, y):
                        return x * y

                    def divide(self, x, y):
                        if y == 0:
                            raise ZeroDivisionError("除数不能为零")
                        return x / y

                server = xmlrpc.server.SimpleXMLRPCServer(("localhost", 8000))
                print("XML-RPC服务器启动在 localhost:8000")

                server.register_function(MathFunctions.add, "add")
                server.register_function(MathFunctions.subtract, "subtract")
                server.register_function(MathFunctions.multiply, "multiply")
                server.register_function(MathFunctions.divide, "divide")

                server.serve_forever()

            except Exception as e:
                print(f"XML-RPC服务器错误: {e}")

        print("HTTP客户端示例:")
        http_client_example()

        print("\nXML-RPC示例:")
        xmlrpc_client_example()

        return xmlrpc_server_example

    def demonstrate_custom_protocol(self):
        """演示自定义协议"""
        print("\n=== 自定义协议演示 ===")

        @dataclass
class Message:
            """消息结构"""
            type: str
            payload: Dict[str, Any]
            timestamp: float
            checksum: str

        class CustomProtocol:
            """自定义协议处理器"""

            def __init__(self):
                self.handlers = {}

            def register_handler(self, message_type: str, handler):
                """注册消息处理器"""
                self.handlers[message_type] = handler

            def create_message(self, message_type: str, payload: Dict[str, Any]) -> str:
                """创建消息"""
                message = Message(
                    type=message_type,
                    payload=payload,
                    timestamp=time.time(),
                    checksum=self.calculate_checksum(payload)
                )
                return self.serialize_message(message)

            def parse_message(self, data: bytes) -> Message:
                """解析消息"""
                message_dict = json.loads(data.decode('utf-8'))
                return Message(**message_dict)

            def serialize_message(self, message: Message) -> str:
                """序列化消息"""
                return json.dumps({
                    'type': message.type,
                    'payload': message.payload,
                    'timestamp': message.timestamp,
                    'checksum': message.checksum
                })

            def calculate_checksum(self, payload: Dict[str, Any]) -> str:
                """计算校验和"""
                payload_str = json.dumps(payload, sort_keys=True)
                return hashlib.md5(payload_str.encode()).hexdigest()

            def validate_message(self, message: Message) -> bool:
                """验证消息"""
                calculated_checksum = self.calculate_checksum(message.payload)
                return calculated_checksum == message.checksum

        # 使用自定义协议
        protocol = CustomProtocol()

        # 注册处理器
        def handle_echo(message: Message):
            """处理echo消息"""
            print(f"Echo消息: {message.payload}")
            return f"Echo: {message.payload.get('text', '')}"

        def handle_calculation(message: Message):
            """处理计算消息"""
            operation = message.payload.get('operation')
            a = message.payload.get('a', 0)
            b = message.payload.get('b', 0)

            if operation == 'add':
                result = a + b
            elif operation == 'subtract':
                result = a - b
            elif operation == 'multiply':
                result = a * b
            elif operation == 'divide':
                result = a / b if b != 0 else 0
            else:
                result = 0

            print(f"计算结果: {result}")
            return result

        protocol.register_handler('echo', handle_echo)
        protocol.register_handler('calculation', handle_calculation)

        # 创建消息
        echo_message = protocol.create_message('echo', {'text': 'Hello Custom Protocol!'})
        calc_message = protocol.create_message('calculation', {
            'operation': 'add',
            'a': 10,
            'b': 20
        })

        print("自定义协议消息:")
        print(f"Echo消息: {echo_message}")
        print(f"计算消息: {calc_message}")

        # 解析和验证消息
        echo_parsed = protocol.parse_message(echo_message.encode())
        calc_parsed = protocol.parse_message(calc_message.encode())

        print(f"Echo消息验证: {protocol.validate_message(echo_parsed)}")
        print(f"计算消息验证: {protocol.validate_message(calc_parsed)}")

        return protocol

    def demonstrate_websocket_communication(self):
        """演示WebSocket通信"""
        print("\n=== WebSocket通信演示 ===")

        import websockets
        import asyncio

        async def websocket_server():
            """WebSocket服务器"""
            async def handle_connection(websocket, path):
                print(f"WebSocket连接来自: {websocket.remote_address}")

                try:
                    async for message in websocket:
                        print(f"收到消息: {message}")

                        # 处理消息
                        if message.startswith('echo:'):
                            response = message[5:]
                        elif message.startswith('calc:'):
                            try:
                                expr = message[5:]
                                result = eval(expr)
                                response = f"结果: {result}"
                            except:
                                response = "计算错误"
                        else:
                            response = f"服务器回复: {message}"

                        await websocket.send(response)

                except websockets.exceptions.ConnectionClosed:
                    print("WebSocket连接关闭")

            server = await websockets.serve(handle_connection, "localhost", 8765)
            print("WebSocket服务器启动在 localhost:8765")
            await server.wait_closed()

        async def websocket_client():
            """WebSocket客户端"""
            try:
                async with websockets.connect("ws://localhost:8765") as websocket:
                    messages = [
                        "echo: Hello WebSocket!",
                        "calc: 2 + 3",
                        "calc: 10 * 5",
                        "General message"
                    ]

                    for message in messages:
                        await websocket.send(message)
                        response = await websocket.recv()
                        print(f"客户端收到: {response}")
                        await asyncio.sleep(0.5)

            except Exception as e:
                print(f"WebSocket客户端错误: {e}")

        def run_websocket_demo():
            """运行WebSocket演示"""
            async def demo():
                # 启动服务器
                server_task = asyncio.create_task(websocket_server())

                # 等待服务器启动
                await asyncio.sleep(1)

                # 运行客户端
                await websocket_client()

                # 关闭服务器
                server_task.cancel()

            asyncio.run(demo())

        return run_websocket_demo

# 运行高级协议演示
advanced_protocols = AdvancedProtocols()
xmlrpc_server = advanced_protocols.demonstrate_http_client()
custom_protocol = advanced_protocols.demonstrate_custom_protocol()
websocket_demo = advanced_protocols.demonstrate_websocket_communication()
```

## 异步网络编程架构

### 1. asyncio网络编程

```python
import asyncio
import aiohttp
import aiofiles
import logging
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass
from abc import ABC, abstractmethod

class AsyncNetworkArchitecture:
    """异步网络编程架构"""

    def demonstrate_asyncio_tcp(self):
        """演示asyncio TCP服务器"""
        print("=== Asyncio TCP服务器演示 ===")

        class AsyncTCPServer:
            """异步TCP服务器"""

            def __init__(self, host='localhost', port=9000):
                self.host = host
                self.port = port
                self.clients = set()
                self.message_history = []

            async def handle_client(self, reader, writer):
                """处理客户端连接"""
                client_addr = writer.get_extra_info('peername')
                print(f"新客户端连接: {client_addr}")
                self.clients.add(writer)

                try:
                    while True:
                        data = await reader.read(1024)
                        if not data:
                            break

                        message = data.decode('utf-8')
                        print(f"来自 {client_addr} 的消息: {message}")

                        # 广播消息给所有客户端
                        await self.broadcast_message(f"{client_addr}: {message}")

                        # 存储消息历史
                        self.message_history.append({
                            'client': client_addr,
                            'message': message,
                            'timestamp': time.time()
                        })

                except Exception as e:
                    print(f"客户端 {client_addr} 错误: {e}")
                finally:
                    print(f"客户端 {client_addr} 断开连接")
                    self.clients.remove(writer)
                    writer.close()
                    await writer.wait_closed()

            async def broadcast_message(self, message: str):
                """广播消息"""
                for client in self.clients:
                    try:
                        client.write(message.encode('utf-8'))
                        await client.drain()
                    except:
                        # 移除断开的客户端
                        self.clients.discard(client)

            async def start_server(self):
                """启动服务器"""
                server = await asyncio.start_server(
                    self.handle_client,
                    self.host,
                    self.port
                )
                print(f"异步TCP服务器启动在 {self.host}:{self.port}")

                async with server:
                    await server.serve_forever()

        class AsyncTCPClient:
            """异步TCP客户端"""

            def __init__(self, host='localhost', port=9000):
                self.host = host
                self.port = port

            async def connect(self):
                """连接到服务器"""
                reader, writer = await asyncio.open_connection(self.host, self.port)
                print(f"连接到服务器 {self.host}:{self.port}")

                try:
                    # 发送消息
                    messages = ["Hello", "Asyncio", "TCP", "Server"]
                    for message in messages:
                        writer.write(message.encode('utf-8'))
                        await writer.drain()

                        # 等待响应
                        response = await reader.read(1024)
                        print(f"服务器回复: {response.decode('utf-8')}")

                        await asyncio.sleep(1)

                except Exception as e:
                    print(f"客户端错误: {e}")
                finally:
                    writer.close()
                    await writer.wait_closed()

        return AsyncTCPServer, AsyncTCPClient

    def demonstrate_async_http_client(self):
        """演示异步HTTP客户端"""
        print("\n=== 异步HTTP客户端演示 ===")

        class AsyncHTTPClient:
            """异步HTTP客户端"""

            def __init__(self):
                self.session = None

            async def __aenter__(self):
                self.session = aiohttp.ClientSession()
                return self

            async def __aexit__(self, exc_type, exc_val, exc_tb):
                if self.session:
                    await self.session.close()

            async def get(self, url: str, **kwargs) -> Dict[str, Any]:
                """GET请求"""
                async with self.session.get(url, **kwargs) as response:
                    return {
                        'status': response.status,
                        'headers': dict(response.headers),
                        'data': await response.text(),
                        'json': await response.json() if response.content_type == 'application/json' else None
                    }

            async def post(self, url: str, data: Dict = None, json_data: Dict = None, **kwargs) -> Dict[str, Any]:
                """POST请求"""
                async with self.session.post(url, data=data, json=json_data, **kwargs) as response:
                    return {
                        'status': response.status,
                        'headers': dict(response.headers),
                        'data': await response.text(),
                        'json': await response.json() if response.content_type == 'application/json' else None
                    }

            async def download_file(self, url: str, filename: str):
                """下载文件"""
                async with self.session.get(url) as response:
                    response.raise_for_status()
                    async with aiofiles.open(filename, 'wb') as f:
                        async for chunk in response.content.iter_chunked(1024):
                            await f.write(chunk)

        async def http_client_demo():
            """HTTP客户端演示"""
            async with AsyncHTTPClient() as client:
                try:
                    # GET请求
                    print("发送GET请求...")
                    result = await client.get('https://httpbin.org/get')
                    print(f"状态码: {result['status']}")
                    print(f"响应数据: {result['json']}")

                    # POST请求
                    print("\n发送POST请求...")
                    post_data = {'name': 'Alice', 'age': 25}
                    result = await client.post('https://httpbin.org/post', json_data=post_data)
                    print(f"状态码: {result['status']}")
                    print(f"响应数据: {result['json']}")

                    # 下载文件
                    print("\n下载文件...")
                    await client.download_file('https://httpbin.org/image/png', 'downloaded_image.png')
                    print("文件下载完成")

                except Exception as e:
                    print(f"HTTP客户端错误: {e}")

        return http_client_demo

    def demonstrate_async_websocket(self):
        """演示异步WebSocket"""
        print("\n=== 异步WebSocket演示 ===")

        class AsyncWebSocketServer:
            """异步WebSocket服务器"""

            def __init__(self, host='localhost', port=9001):
                self.host = host
                self.port = port
                self.clients = set()

            async def handle_client(self, websocket, path):
                """处理WebSocket客户端"""
                client_addr = websocket.remote_address
                print(f"WebSocket客户端连接: {client_addr}")
                self.clients.add(websocket)

                try:
                    async for message in websocket:
                        print(f"收到WebSocket消息: {message}")

                        # 处理不同类型的消息
                        if message.startswith('broadcast:'):
                            # 广播消息
                            broadcast_msg = message[10:]
                            await self.broadcast(broadcast_msg)
                        elif message.startswith('echo:'):
                            # 回显消息
                            await websocket.send(f"Echo: {message[5:]}")
                        else:
                            # 默认处理
                            await websocket.send(f"Server received: {message}")

                except Exception as e:
                    print(f"WebSocket客户端错误: {e}")
                finally:
                    print(f"WebSocket客户端断开: {client_addr}")
                    self.clients.remove(websocket)

            async def broadcast(self, message: str):
                """广播消息"""
                if self.clients:
                    await asyncio.gather(
                        *[client.send(message) for client in self.clients],
                        return_exceptions=True
                    )

            async def start_server(self):
                """启动WebSocket服务器"""
                import websockets

                server = await websockets.serve(
                    self.handle_client,
                    self.host,
                    self.port
                )
                print(f"异步WebSocket服务器启动在 {self.host}:{self.port}")

                await server.wait_closed()

        class AsyncWebSocketClient:
            """异步WebSocket客户端"""

            def __init__(self, host='localhost', port=9001):
                self.host = host
                self.port = port
                self.uri = f"ws://{self.host}:{self.port}"

            async def connect(self):
                """连接到WebSocket服务器"""
                import websockets

                try:
                    async with websockets.connect(self.uri) as websocket:
                        print(f"连接到WebSocket服务器: {self.uri}")

                        messages = [
                            "echo: Hello WebSocket!",
                            "broadcast: Hello everyone!",
                            "General message"
                        ]

                        for message in messages:
                            await websocket.send(message)
                            response = await websocket.recv()
                            print(f"服务器回复: {response}")
                            await asyncio.sleep(0.5)

                except Exception as e:
                    print(f"WebSocket客户端错误: {e}")

        return AsyncWebSocketServer, AsyncWebSocketClient

# 运行异步网络架构演示
async_network = AsyncNetworkArchitecture()
async_tcp_server, async_tcp_client = async_network.demonstrate_asyncio_tcp()
async_http_demo = async_network.demonstrate_async_http_client()
async_ws_server, async_ws_client = async_network.demonstrate_async_websocket()
```

### 2. 高并发网络架构

```python
import asyncio
import uvloop
import aiohttp
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass
from abc import ABC, abstractmethod
import concurrent.futures
import multiprocessing

class HighConcurrencyArchitecture:
    """高并发网络架构"""

    def demonstrate_connection_pooling(self):
        """演示连接池"""
        print("=== 连接池演示 ===")

        class ConnectionPool:
            """连接池"""

            def __init__(self, max_connections: int = 10):
                self.max_connections = max_connections
                self.connections = []
                self.available = asyncio.Queue()
                self.lock = asyncio.Lock()
                self.total_connections = 0

            async def get_connection(self):
                """获取连接"""
                if not self.available.empty():
                    return await self.available.get()

                async with self.lock:
                    if self.total_connections < self.max_connections:
                        connection = await self.create_connection()
                        self.total_connections += 1
                        return connection

                # 等待可用连接
                return await self.available.get()

            async def release_connection(self, connection):
                """释放连接"""
                await self.available.put(connection)

            async def create_connection(self):
                """创建新连接"""
                # 模拟连接创建
                await asyncio.sleep(0.1)
                return {"id": self.total_connections, "created": time.time()}

            async def close_all(self):
                """关闭所有连接"""
                while not self.available.empty():
                    connection = await self.available.get()
                    await self.close_connection(connection)

            async def close_connection(self, connection):
                """关闭连接"""
                print(f"关闭连接 {connection['id']}")

        async def connection_pool_demo():
            """连接池演示"""
            pool = ConnectionPool(max_connections=5)

            async def worker(worker_id: int):
                """工作线程"""
                for i in range(3):
                    connection = await pool.get_connection()
                    print(f"Worker {worker_id} 获得连接 {connection['id']}")
                    await asyncio.sleep(0.1)
                    await pool.release_connection(connection)

            # 创建多个工作线程
            workers = [worker(i) for i in range(10)]
            await asyncio.gather(*workers)

            await pool.close_all()

        return connection_pool_demo

    def demonstrate_load_balancing(self):
        """演示负载均衡"""
        print("\n=== 负载均衡演示 ===")

        @dataclass
class Server:
            """服务器节点"""
            host: str
            port: int
            weight: int = 1
            current_connections: int = 0

        class LoadBalancer:
            """负载均衡器"""

            def __init__(self):
                self.servers: List[Server] = []
                self.algorithm = "round_robin"
                self.current_index = 0

            def add_server(self, server: Server):
                """添加服务器"""
                self.servers.append(server)

            def get_next_server(self) -> Optional[Server]:
                """获取下一个服务器"""
                if not self.servers:
                    return None

                if self.algorithm == "round_robin":
                    server = self.servers[self.current_index]
                    self.current_index = (self.current_index + 1) % len(self.servers)
                    return server

                elif self.algorithm == "least_connections":
                    return min(self.servers, key=lambda s: s.current_connections)

                elif self.algorithm == "weighted":
                    # 加权轮询
                    total_weight = sum(s.weight for s in self.servers)
                    if total_weight == 0:
                        return None

                    # 简化的加权选择
                    import random
                    r = random.randint(1, total_weight)
                    current_weight = 0
                    for server in self.servers:
                        current_weight += server.weight
                        if r <= current_weight:
                            return server

                    return self.servers[0]

                return None

            def set_algorithm(self, algorithm: str):
                """设置负载均衡算法"""
                self.algorithm = algorithm

        async def load_balancer_demo():
            """负载均衡演示"""
            lb = LoadBalancer()

            # 添加服务器
            lb.add_server(Server("server1", 8001, weight=3))
            lb.add_server(Server("server2", 8002, weight=2))
            lb.add_server(Server("server3", 8003, weight=1))

            algorithms = ["round_robin", "least_connections", "weighted"]

            for algorithm in algorithms:
                print(f"\n使用 {algorithm} 算法:")
                lb.set_algorithm(algorithm)

                for i in range(10):
                    server = lb.get_next_server()
                    if server:
                        print(f"  请求 {i+1} -> {server.host}:{server.port}")
                        server.current_connections += 1

                # 重置连接数
                for server in lb.servers:
                    server.current_connections = 0

        return load_balancer_demo

    def demonstrate_rate_limiting(self):
        """演示限流"""
        print("\n=== 限流演示 ===")

        class RateLimiter:
            """限流器"""

            def __init__(self, max_requests: int, time_window: float):
                self.max_requests = max_requests
                self.time_window = time_window
                self.requests = []
                self.lock = asyncio.Lock()

            async def acquire(self):
                """获取许可"""
                async with self.lock:
                    now = time.time()

                    # 清理过期的请求
                    self.requests = [req_time for req_time in self.requests
                                   if now - req_time < self.time_window]

                    if len(self.requests) >= self.max_requests:
                        wait_time = self.time_window - (now - self.requests[0])
                        if wait_time > 0:
                            await asyncio.sleep(wait_time)

                    self.requests.append(now)

        class TokenBucket:
            """令牌桶限流"""

            def __init__(self, capacity: int, refill_rate: float):
                self.capacity = capacity
                self.refill_rate = refill_rate
                self.tokens = capacity
                self.last_refill = time.time()
                self.lock = asyncio.Lock()

            async def acquire(self):
                """获取令牌"""
                async with self.lock:
                    now = time.time()

                    # 补充令牌
                    elapsed = now - self.last_refill
                    tokens_to_add = elapsed * self.refill_rate
                    self.tokens = min(self.capacity, self.tokens + tokens_to_add)
                    self.last_refill = now

                    if self.tokens >= 1:
                        self.tokens -= 1
                        return True
                    else:
                        # 计算需要等待的时间
                        wait_time = (1 - self.tokens) / self.refill_rate
                        await asyncio.sleep(wait_time)
                        self.tokens = 0
                        return True

        async def rate_limiting_demo():
            """限流演示"""
            # 固定窗口限流
            fixed_limiter = RateLimiter(max_requests=5, time_window=1.0)

            print("固定窗口限流:")
            for i in range(10):
                start_time = time.time()
                await fixed_limiter.acquire()
                elapsed = time.time() - start_time
                print(f"  请求 {i+1} 耗时: {elapsed:.3f}s")

            # 令牌桶限流
            token_bucket = TokenBucket(capacity=5, refill_rate=2.0)  # 2个令牌/秒

            print("\n令牌桶限流:")
            for i in range(10):
                start_time = time.time()
                await token_bucket.acquire()
                elapsed = time.time() - start_time
                print(f"  请求 {i+1} 耗时: {elapsed:.3f}s")

        return rate_limiting_demo

# 运行高并发架构演示
high_concurrency = HighConcurrencyArchitecture()
connection_pool_demo = high_concurrency.demonstrate_connection_pooling()
load_balancer_demo = high_concurrency.demonstrate_load_balancing()
rate_limiting_demo = high_concurrency.demonstrate_rate_limiting()
```

## 网络安全与性能优化

### 1. 网络安全架构

```python
import ssl
import hashlib
import hmac
import json
import base64
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import asyncio

class NetworkSecurity:
    """网络安全架构"""

    def demonstrate_ssl_tls(self):
        """演示SSL/TLS"""
        print("=== SSL/TLS安全通信演示 ===")

        def create_ssl_context():
            """创建SSL上下文"""
            context = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
            context.load_cert_chain(certfile="server.crt", keyfile="server.key")
            context.load_verify_locations(cafile="ca.crt")
            context.verify_mode = ssl.CERT_REQUIRED
            return context

        def ssl_server_example():
            """SSL服务器示例"""
            context = create_ssl_context()

            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                sock.bind(('localhost', 9443))
                sock.listen(5)
                print("SSL服务器启动在 localhost:9443")

                with context.wrap_socket(sock, server_side=True) as sslsock:
                    while True:
                        client_socket, client_address = sslsock.accept()
                        print(f"SSL连接来自: {client_address}")

                        try:
                            data = client_socket.recv(1024)
                            if data:
                                print(f"收到加密数据: {data}")
                                response = b"SSL Server Response"
                                client_socket.send(response)
                        except Exception as e:
                            print(f"SSL处理错误: {e}")
                        finally:
                            client_socket.close()

        def ssl_client_example():
            """SSL客户端示例"""
            context = ssl.create_default_context()
            context.check_hostname = False
            context.verify_mode = ssl.CERT_NONE

            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                with context.wrap_socket(sock, server_hostname="localhost") as ssock:
                    ssock.connect(('localhost', 9443))
                    ssock.send(b"Hello SSL Server")
                    response = ssock.recv(1024)
                    print(f"SSL服务器回复: {response}")

        return ssl_server_example, ssl_client_example

    def demonstrate_encryption_decryption(self):
        """演示加密解密"""
        print("\n=== 加密解密演示 ===")

        class EncryptionManager:
            """加密管理器"""

            def __init__(self):
                self.key = Fernet.generate_key()
                self.cipher_suite = Fernet(self.key)

            def encrypt_data(self, data: str) -> bytes:
                """加密数据"""
                return self.cipher_suite.encrypt(data.encode())

            def decrypt_data(self, encrypted_data: bytes) -> str:
                """解密数据"""
                return self.cipher_suite.decrypt(encrypted_data).decode()

            def generate_key_from_password(self, password: str, salt: bytes = None) -> bytes:
                """从密码生成密钥"""
                if salt is None:
                    salt = b'salt_'  # 在实际应用中应该使用随机盐

                kdf = PBKDF2HMAC(
                    algorithm=hashes.SHA256(),
                    length=32,
                    salt=salt,
                    iterations=100000,
                )
                return base64.urlsafe_b64encode(kdf.derive(password.encode()))

            def create_hmac(self, data: str, key: bytes) -> str:
                """创建HMAC"""
                return hmac.new(key, data.encode(), hashlib.sha256).hexdigest()

            def verify_hmac(self, data: str, key: bytes, hmac_value: str) -> bool:
                """验证HMAC"""
                expected_hmac = self.create_hmac(data, key)
                return hmac.compare_digest(expected_hmac, hmac_value)

        # 使用加密管理器
        crypto_manager = EncryptionManager()

        # 加密解密示例
        original_data = "This is a secret message"
        encrypted_data = crypto_manager.encrypt_data(original_data)
        decrypted_data = crypto_manager.decrypt_data(encrypted_data)

        print(f"原始数据: {original_data}")
        print(f"加密数据: {encrypted_data}")
        print(f"解密数据: {decrypted_data}")

        # HMAC示例
        secret_key = b'secret_key_123'
        message = "Important message"
        hmac_value = crypto_manager.create_hmac(message, secret_key)

        print(f"\n消息: {message}")
        print(f"HMAC值: {hmac_value}")
        print(f"HMAC验证: {crypto_manager.verify_hmac(message, secret_key, hmac_value)}")

        return crypto_manager

    def demonstrate_authentication_authorization(self):
        """演示认证授权"""
        print("\n=== 认证授权演示 ===")

        @dataclass
class User:
            """用户"""
            username: str
            password_hash: str
            roles: List[str]
            permissions: Dict[str, bool]

        class AuthManager:
            """认证管理器"""

            def __init__(self):
                self.users: Dict[str, User] = {}
                self.sessions: Dict[str, Dict] = {}
                self.jwt_secret = "jwt_secret_key"

            def register_user(self, username: str, password: str, roles: List[str] = None):
                """注册用户"""
                password_hash = self.hash_password(password)
                user = User(
                    username=username,
                    password_hash=password_hash,
                    roles=roles or ['user'],
                    permissions={'read': True, 'write': False}
                )
                self.users[username] = user

            def hash_password(self, password: str) -> str:
                """哈希密码"""
                import hashlib
                return hashlib.sha256(password.encode()).hexdigest()

            def verify_password(self, username: str, password: str) -> bool:
                """验证密码"""
                user = self.users.get(username)
                if not user:
                    return False
                return user.password_hash == self.hash_password(password)

            def login(self, username: str, password: str) -> Optional[str]:
                """用户登录"""
                if not self.verify_password(username, password):
                    return None

                # 生成会话token
                import uuid
                session_id = str(uuid.uuid4())
                self.sessions[session_id] = {
                    'username': username,
                    'expires': time.time() + 3600  # 1小时过期
                }

                return session_id

            def verify_session(self, session_id: str) -> Optional[str]:
                """验证会话"""
                session = self.sessions.get(session_id)
                if not session or session['expires'] < time.time():
                    return None
                return session['username']

            def check_permission(self, username: str, permission: str) -> bool:
                """检查权限"""
                user = self.users.get(username)
                if not user:
                    return False
                return user.permissions.get(permission, False)

            def create_jwt(self, username: str) -> str:
                """创建JWT令牌"""
                import jwt
                payload = {
                    'username': username,
                    'exp': time.time() + 3600,
                    'iat': time.time()
                }
                return jwt.encode(payload, self.jwt_secret, algorithm='HS256')

            def verify_jwt(self, token: str) -> Optional[str]:
                """验证JWT令牌"""
                try:
                    import jwt
                    payload = jwt.decode(token, self.jwt_secret, algorithms=['HS256'])
                    return payload['username']
                except:
                    return None

        # 使用认证管理器
        auth_manager = AuthManager()

        # 注册用户
        auth_manager.register_user('admin', 'admin123', ['admin'])
        auth_manager.register_user('user1', 'user123', ['user'])

        # 登录测试
        session_id = auth_manager.login('admin', 'admin123')
        print(f"管理员登录会话ID: {session_id}")

        if session_id:
            username = auth_manager.verify_session(session_id)
            print(f"会话验证用户: {username}")

            # 权限检查
            print(f"管理员读权限: {auth_manager.check_permission('admin', 'read')}")
            print(f"管理员写权限: {auth_manager.check_permission('admin', 'write')}")

        # JWT测试
        jwt_token = auth_manager.create_jwt('user1')
        print(f"\nJWT令牌: {jwt_token}")
        jwt_username = auth_manager.verify_jwt(jwt_token)
        print(f"JWT验证用户: {jwt_username}")

        return auth_manager

# 运行网络安全演示
network_security = NetworkSecurity()
ssl_server, ssl_client = network_security.demonstrate_ssl_tls()
crypto_manager = network_security.demonstrate_encryption_decryption()
auth_manager = network_security.demonstrate_authentication_authorization()
```

### 2. 性能优化策略

```python
import asyncio
import time
import psutil
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import uvloop

class NetworkPerformanceOptimization:
    """网络性能优化"""

    def demonstrate_event_loop_optimization(self):
        """演示事件循环优化"""
        print("=== 事件循环优化演示 ===")

        def benchmark_event_loops():
            """基准测试不同事件循环"""
            import asyncio

            async def async_task():
                await asyncio.sleep(0.1)
                return "Task completed"

            async def run_tasks(num_tasks: int):
                tasks = [async_task() for _ in range(num_tasks)]
                await asyncio.gather(*tasks)

            # 默认事件循环
            print("使用默认事件循环:")
            start_time = time.time()
            asyncio.run(run_tasks(100))
            default_time = time.time() - start_time
            print(f"执行时间: {default_time:.3f}s")

            # UVLoop事件循环
            try:
                uvloop.install()
                print("\n使用UVLoop事件循环:")
                start_time = time.time()
                asyncio.run(run_tasks(100))
                uvloop_time = time.time() - start_time
                print(f"执行时间: {uvloop_time:.3f}s")
                print(f"性能提升: {default_time / uvloop_time:.2f}x")
            except ImportError:
                print("\nUVLoop未安装，跳过UVLoop测试")

        return benchmark_event_loops

    def demonstrate_connection_reuse(self):
        """演示连接复用"""
        print("\n=== 连接复用演示 ===")

        class ConnectionManager:
            """连接管理器"""

            def __init__(self):
                self.connections: Dict[str, Any] = {}
                self.connection_pools: Dict[str, List] = {}

            def get_connection(self, host: str, port: int):
                """获取连接"""
                key = f"{host}:{port}"
                if key in self.connections:
                    return self.connections[key]

                # 创建新连接
                import socket
                conn = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                conn.connect((host, port))
                self.connections[key] = conn
                return conn

            def release_connection(self, host: str, port: int):
                """释放连接"""
                key = f"{host}:{port}"
                if key in self.connections:
                    conn = self.connections[key]
                    conn.close()
                    del self.connections[key]

            def create_connection_pool(self, host: str, port: int, size: int = 5):
                """创建连接池"""
                key = f"{host}:{port}"
                if key not in self.connection_pools:
                    self.connection_pools[key] = []
                    for _ in range(size):
                        import socket
                        conn = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                        conn.connect((host, port))
                        self.connection_pools[key].append(conn)

            def get_pooled_connection(self, host: str, port: int):
                """获取池化连接"""
                key = f"{host}:{port}"
                if key in self.connection_pools and self.connection_pools[key]:
                    return self.connection_pools[key].pop()
                return None

            def return_pooled_connection(self, host: str, port: int, conn):
                """返回池化连接"""
                key = f"{host}:{port}"
                if key not in self.connection_pools:
                    self.connection_pools[key] = []
                self.connection_pools[key].append(conn)

        async def connection_reuse_demo():
            """连接复用演示"""
            manager = ConnectionManager()

            # 模拟HTTP服务器
            class MockHTTPServer:
                def __init__(self, port):
                    self.port = port
                    self.running = False

                async def start(self):
                    """启动模拟服务器"""
                    import asyncio

                    async def handle_client(reader, writer):
                        data = await reader.read(1024)
                        if data:
                            response = b"HTTP/1.1 200 OK\r\nContent-Length: 13\r\n\r\nHello, World!"
                            writer.write(response)
                            await writer.drain()
                        writer.close()

                    server = await asyncio.start_server(handle_client, 'localhost', self.port)
                    self.running = True
                    print(f"模拟HTTP服务器启动在端口 {self.port}")
                    async with server:
                        while self.running:
                            await asyncio.sleep(1)

            # 启动模拟服务器
            server = MockHTTPServer(8000)
            server_task = asyncio.create_task(server.start())

            # 等待服务器启动
            await asyncio.sleep(0.1)

            # 测试连接复用
            print("测试连接复用:")
            start_time = time.time()

            for i in range(10):
                conn = manager.get_connection('localhost', 8000)
                conn.send(b"GET / HTTP/1.1\r\nHost: localhost\r\n\r\n")
                response = conn.recv(1024)
                # 不关闭连接，供下次使用

            reuse_time = time.time() - start_time
            print(f"连接复用时间: {reuse_time:.3f}s")

            # 测试新建连接
            print("\n测试新建连接:")
            start_time = time.time()

            for i in range(10):
                import socket
                conn = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                conn.connect(('localhost', 8000))
                conn.send(b"GET / HTTP/1.1\r\nHost: localhost\r\n\r\n")
                response = conn.recv(1024)
                conn.close()

            new_conn_time = time.time() - start_time
            print(f"新建连接时间: {new_conn_time:.3f}s")
            print(f"性能提升: {new_conn_time / reuse_time:.2f}x")

            # 停止服务器
            server.running = False
            server_task.cancel()

        return connection_reuse_demo

    def demonstrate_caching_strategies(self):
        """演示缓存策略"""
        print("\n=== 缓存策略演示 ===")

        class CacheManager:
            """缓存管理器"""

            def __init__(self):
                self.memory_cache: Dict[str, Dict] = {}
                self.cache_ttl: Dict[str, float] = {}

            def set_memory_cache(self, key: str, value: Any, ttl: float = 3600):
                """设置内存缓存"""
                self.memory_cache[key] = {'value': value, 'timestamp': time.time()}
                self.cache_ttl[key] = ttl

            def get_memory_cache(self, key: str) -> Optional[Any]:
                """获取内存缓存"""
                if key in self.memory_cache:
                    cache_entry = self.memory_cache[key]
                    if time.time() - cache_entry['timestamp'] < self.cache_ttl.get(key, 3600):
                        return cache_entry['value']
                    else:
                        # 缓存过期
                        del self.memory_cache[key]
                        if key in self.cache_ttl:
                            del self.cache_ttl[key]
                return None

            def clear_expired_cache(self):
                """清理过期缓存"""
                current_time = time.time()
                expired_keys = []

                for key, cache_entry in self.memory_cache.items():
                    if current_time - cache_entry['timestamp'] > self.cache_ttl.get(key, 3600):
                        expired_keys.append(key)

                for key in expired_keys:
                    del self.memory_cache[key]
                    if key in self.cache_ttl:
                        del self.cache_ttl[key]

        async def caching_demo():
            """缓存演示"""
            cache_manager = CacheManager()

            # 模拟API调用
            async def fetch_data(api_url: str) -> Dict:
                """获取数据"""
                # 检查缓存
                cached_data = cache_manager.get_memory_cache(api_url)
                if cached_data:
                    print(f"缓存命中: {api_url}")
                    return cached_data

                # 模拟网络请求
                print(f"缓存未命中，请求: {api_url}")
                await asyncio.sleep(0.1)  # 模拟网络延迟

                # 模拟响应数据
                data = {'url': api_url, 'data': f'Response for {api_url}', 'timestamp': time.time()}

                # 设置缓存
                cache_manager.set_memory_cache(api_url, data, ttl=5.0)  # 5秒TTL

                return data

            # 测试缓存
            urls = [
                "https://api.example.com/users",
                "https://api.example.com/products",
                "https://api.example.com/orders"
            ]

            print("测试缓存效果:")
            start_time = time.time()

            # 第一次请求（缓存未命中）
            for url in urls:
                await fetch_data(url)

            first_request_time = time.time() - start_time

            # 第二次请求（缓存命中）
            start_time = time.time()
            for url in urls:
                await fetch_data(url)

            second_request_time = time.time() - start_time

            print(f"\n第一次请求时间: {first_request_time:.3f}s")
            print(f"第二次请求时间: {second_request_time:.3f}s")
            print(f"缓存加速比: {first_request_time / second_request_time:.2f}x")

            # 等待缓存过期
            print("\n等待缓存过期...")
            await asyncio.sleep(6)

            # 第三次请求（缓存已过期）
            start_time = time.time()
            for url in urls:
                await fetch_data(url)

            third_request_time = time.time() - start_time
            print(f"缓存过期后请求时间: {third_request_time:.3f}s")

        return caching_demo

    def demonstrate_monitoring_metrics(self):
        """演示监控指标"""
        print("\n=== 监控指标演示 ===")

        class NetworkMonitor:
            """网络监控器"""

            def __init__(self):
                self.metrics = {
                    'connections': 0,
                    'requests': 0,
                    'response_times': [],
                    'errors': 0,
                    'bandwidth': 0
                }
                self.start_time = time.time()

            def record_connection(self):
                """记录连接"""
                self.metrics['connections'] += 1

            def record_request(self, response_time: float):
                """记录请求"""
                self.metrics['requests'] += 1
                self.metrics['response_times'].append(response_time)

            def record_error(self):
                """记录错误"""
                self.metrics['errors'] += 1

            def record_bandwidth(self, bytes_sent: int, bytes_received: int):
                """记录带宽"""
                self.metrics['bandwidth'] += bytes_sent + bytes_received

            def get_metrics(self) -> Dict:
                """获取指标"""
                response_times = self.metrics['response_times']
                avg_response_time = sum(response_times) / len(response_times) if response_times else 0

                uptime = time.time() - self.start_time
                requests_per_second = self.metrics['requests'] / uptime if uptime > 0 else 0

                return {
                    'connections': self.metrics['connections'],
                    'requests': self.metrics['requests'],
                    'avg_response_time': avg_response_time,
                    'errors': self.metrics['errors'],
                    'error_rate': self.metrics['errors'] / self.metrics['requests'] if self.metrics['requests'] > 0 else 0,
                    'bandwidth': self.metrics['bandwidth'],
                    'uptime': uptime,
                    'requests_per_second': requests_per_second
                }

        async def monitoring_demo():
            """监控演示"""
            monitor = NetworkMonitor()

            # 模拟网络请求
            async def simulated_request():
                """模拟请求"""
                monitor.record_connection()
                monitor.record_request(0.1)  # 模拟响应时间
                monitor.record_bandwidth(100, 200)  # 模拟带宽

            # 模拟一些错误请求
            async def simulated_error_request():
                """模拟错误请求"""
                monitor.record_connection()
                monitor.record_error()
                monitor.record_bandwidth(50, 0)

            # 运行模拟
            tasks = []
            for i in range(20):
                if i % 5 == 0:  # 每5个请求模拟一个错误
                    tasks.append(simulated_error_request())
                else:
                    tasks.append(simulated_request())

            await asyncio.gather(*tasks)

            # 显示监控指标
            metrics = monitor.get_metrics()
            print("网络监控指标:")
            for key, value in metrics.items():
                print(f"  {key}: {value}")

        return monitoring_demo

# 运行性能优化演示
performance_opt = NetworkPerformanceOptimization()
event_loop_benchmark = performance_opt.demonstrate_event_loop_optimization()
connection_reuse_demo = performance_opt.demonstrate_connection_reuse()
caching_demo = performance_opt.demonstrate_caching_strategies()
monitoring_demo = performance_opt.demonstrate_monitoring_metrics()
```

## 结论：网络编程的架构智慧

网络编程架构体现了"分布式思维"和"系统设计"的深刻智慧。通过理解网络编程的底层原理和架构模式，我们能够构建出高性能、高可用、安全的分布式系统。

### 核心架构原则：

1. **分层设计**：将网络功能按层次分离，便于维护和扩展
2. **异步优先**：使用异步编程提高并发性能
3. **连接管理**：合理使用连接池和连接复用
4. **安全第一**：在网络层面和应用层面都要考虑安全性

### 性能优化策略：

1. **事件循环优化**：使用高效的事件循环实现
2. **连接复用**：减少连接建立的开销
3. **缓存策略**：减少重复的网络请求
4. **负载均衡**：合理分配请求到不同服务器

### 安全考虑：

1. **加密通信**：使用SSL/TLS保护数据传输
2. **认证授权**：实现完整的身份验证和权限控制
3. **数据完整性**：使用数字签名和HMAC验证数据
4. **监控告警**：实时监控网络状态和安全事件

网络编程不仅是一门技术，更是一种思维方式。通过理解网络编程的架构原理，我们能够构建出更加可靠、高效和安全的分布式系统。

---

*这篇文章深入探讨了Python网络编程的各个方面，从基础套接字编程到高级网络架构，从安全实现到性能优化。希望通过这篇文章，你能够真正理解网络编程的架构智慧，并在实际项目中合理地运用这些技术。*