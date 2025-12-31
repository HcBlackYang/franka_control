# import socket
# import json
# import struct
# import cv2
# import numpy as np
# import logging

# class TCPClientPolicy:
#     """
#     一个即插即用的客户端，用于替换 WebsocketClientPolicy。
#     负责将图片压缩并发送给 GPU 服务器。
#     """
#     def __init__(self, host, port):
#         self.host = host
#         self.port = int(port)
#         self.sock = None
#         self.connect()

#     def connect(self):
#         try:
#             self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
#             self.sock.setsockopt(socket.SOL_TCP, socket.TCP_NODELAY, 1) # 禁用 Nagle 算法降低延迟
#             self.sock.connect((self.host, self.port))
#             logging.info(f"✅ TCP连接成功: {self.host}:{self.port}")
#         except Exception as e:
#             logging.error(f"❌ TCP连接失败: {e}")

#     def infer(self, element):
#         """
#         参数 element: 也就是 robot_policy_system.py 里的那个字典
#         """
#         # 1. 提取数据
#         # 注意：这里需要确保 robot_policy_system 传入了关节角度
#         # 如果 element 里没有 'qpos'，我们尝试从 'observation/state' 猜
#         if 'qpos' in element:
#             qpos = element['qpos']
#         else:
#             # 假设 state 的前7位是关节角度 (你需要确认这一点!)
#             # 你的 RDT 模型是用 关节角度 训练的
#             qpos = element['observation/state'][:7].tolist()

#         image = element['observation/image'] # BGR numpy array
        
#         # 2. 图像压缩 (关键！传输原始 1280x720 图片太慢了)
#         # 压缩为 JPEG，质量 90 (肉眼几乎无损，体积减小 10 倍)
#         encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 90]
#         _, img_encoded = cv2.imencode('.jpg', image, encode_param)
#         img_bytes = img_encoded.tobytes()
        
#         # 3. 构造包头
#         header = {
#             "qpos": qpos,
#             "img_size": len(img_bytes)
#         }
#         header_bytes = json.dumps(header).encode('utf-8')
        
#         try:
#             # 4. 发送: [头长度] + [头] + [图片体]
#             self.sock.sendall(struct.pack('>I', len(header_bytes)))
#             self.sock.sendall(header_bytes)
#             self.sock.sendall(img_bytes)
            
#             # 5. 接收响应
#             len_bytes = self.recv_all(4)
#             if not len_bytes: return self._empty_response()
            
#             resp_len = struct.unpack('>I', len_bytes)[0]
#             resp_bytes = self.recv_all(resp_len)
            
#             response = json.loads(resp_bytes.decode('utf-8'))
#             return response
            
#         except Exception as e:
#             logging.error(f"推理通信错误: {e}")
#             self.connect() # 尝试重连
#             return self._empty_response()

#     def recv_all(self, n):
#         data = b''
#         while len(data) < n:
#             packet = self.sock.recv(n - len(data))
#             if not packet: return None
#             data += packet
#         return data
        
#     def _empty_response(self):
#         # 返回空动作以防报错，让机器人停在原地
#         return {"actions": [[0.0]*7], "trajectory": None}

# import socket
# import json
# import struct
# import cv2
# import numpy as np
# import logging

# class TCPClientPolicy:
#     """
#     一个即插即用的客户端，用于替换 WebsocketClientPolicy。
#     负责将图片压缩并发送给 GPU 服务器。
#     """
#     def __init__(self, host, port):
#         self.host = host
#         self.port = int(port)
#         self.sock = None
#         self.connect()

#     def connect(self):
#         try:
#             self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
#             self.sock.setsockopt(socket.SOL_TCP, socket.TCP_NODELAY, 1) 
#             self.sock.connect((self.host, self.port))
#             logging.info(f"✅ TCP连接成功: {self.host}:{self.port}")
#         except Exception as e:
#             logging.error(f"❌ TCP连接失败: {e}")

#     def infer(self, element):
#         """
#         参数 element: robot_policy_system.py 里的字典
#         """
#         # 1. 提取关节状态
#         if 'qpos' in element:
#             qpos = element['qpos']
#         else:
#             qpos = element['observation/state'][:7].tolist()

#         # === 核心修复：优先使用 Wrist Camera ===
#         # 训练时使用的是 robot0_eye_in_hand_image (Wrist)
#         if 'observation/wrist_image' in element:
#             image = element['observation/wrist_image']
#         elif 'observation/image' in element:
#             image = element['observation/image']
#             logging.warning("⚠️ Warning: Wrist image not found, using Main image (Out of Distribution!)")
#         else:
#             logging.error("❌ No image found in element!")
#             return self._empty_response()
        
#         # 2. 图像压缩 
#         encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 90]
#         _, img_encoded = cv2.imencode('.jpg', image, encode_param)
#         img_bytes = img_encoded.tobytes()
        
#         # 3. 构造包头
#         header = {
#             "qpos": qpos,
#             "img_size": len(img_bytes)
#         }
#         header_bytes = json.dumps(header).encode('utf-8')
        
#         try:
#             # 4. 发送
#             self.sock.sendall(struct.pack('>I', len(header_bytes)))
#             self.sock.sendall(header_bytes)
#             self.sock.sendall(img_bytes)
            
#             # 5. 接收响应
#             len_bytes = self.recv_all(4)
#             if not len_bytes: return self._empty_response()
            
#             resp_len = struct.unpack('>I', len_bytes)[0]
#             resp_bytes = self.recv_all(resp_len)
            
#             response = json.loads(resp_bytes.decode('utf-8'))
#             return response
            
#         except Exception as e:
#             logging.error(f"推理通信错误: {e}")
#             self.connect() # 尝试重连
#             return self._empty_response()

#     def recv_all(self, n):
#         data = b''
#         while len(data) < n:
#             packet = self.sock.recv(n - len(data))
#             if not packet: return None
#             data += packet
#         return data
        
#     def _empty_response(self):
#         return {"actions": [[0.0]*7], "trajectory": None}

import socket
import json
import struct
import cv2
import numpy as np
import logging

class TCPClientPolicy:
    def __init__(self, host, port):
        self.host = host
        self.port = int(port)
        self.sock = None
        self.connect()

    def connect(self):
        try:
            if self.sock: self.sock.close()
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.sock.setsockopt(socket.SOL_TCP, socket.TCP_NODELAY, 1) 
            
            # [修改点] 延长超时到 60秒，防止 compile 导致的超时断连
            self.sock.settimeout(180.0) 
            
            self.sock.connect((self.host, self.port))
            logging.info(f"✅ TCP连接成功: {self.host}:{self.port}")
        except Exception as e:
            logging.error(f"❌ TCP连接失败: {e}")
            self.sock = None

    def infer(self, element):
        if self.sock is None:
            self.connect()
            if self.sock is None: return self._empty_response()

        # 1. 提取
        if 'qpos' in element:
            qpos = element['qpos']
        else:
            qpos = element['observation/state'][:7].tolist()

        if 'observation/wrist_image' in element:
            image = element['observation/wrist_image']
        elif 'observation/image' in element:
            image = element['observation/image']
        else:
            return self._empty_response()
        
        # 2. 压缩
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 90]
        _, img_encoded = cv2.imencode('.jpg', image, encode_param)
        img_bytes = img_encoded.tobytes()
        
        # 3. 构造 Header
        header = {"qpos": qpos, "img_size": len(img_bytes)}
        header_bytes = json.dumps(header).encode('utf-8')
        
        try:
            # 4. 发送
            self.sock.sendall(struct.pack('>I', len(header_bytes)))
            self.sock.sendall(header_bytes)
            self.sock.sendall(img_bytes)
            
            # 5. 接收
            len_bytes = self.recv_all(4)
            if not len_bytes: 
                logging.warning("⚠️ Server closed connection (EOF).")
                self.sock.close() 
                self.sock = None
                return self._empty_response()
            
            resp_len = struct.unpack('>I', len_bytes)[0]
            resp_bytes = self.recv_all(resp_len)
            if not resp_bytes: return self._empty_response()
            
            response = json.loads(resp_bytes.decode('utf-8'))
            return response
            
        except socket.timeout:
            logging.error("⏰ 推理超时 (60s Timeout).")
            # 超时后连接可能已脏，建议重置
            if self.sock: self.sock.close()
            self.sock = None
            return self._empty_response()
        except Exception as e:
            logging.error(f"💥 通信异常: {e}")
            if self.sock: self.sock.close()
            self.sock = None
            return self._empty_response()

    def recv_all(self, n):
        data = b''
        try:
            while len(data) < n:
                chunk = self.sock.recv(n - len(data))
                if not chunk: return None
                data += chunk
            return data
        except:
            return None
        
    def _empty_response(self):
        # 返回全0动作，但现在 robot_policy_system 会拦截它
        return {
            "actions": [ [[0.0] * 8] ], 
            "trajectory": None
        }