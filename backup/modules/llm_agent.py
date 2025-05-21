import os
import json
import logging
import requests
from app.config import Config

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class DeepSeekAgent:
    """DeepSeek API 客户端代理"""
    
    def __init__(self, api_key=None, api_base=None):
        """
        初始化DeepSeek客户端代理
        
        Args:
            api_key (str, optional): DeepSeek API密钥
            api_base (str, optional): DeepSeek API基础URL
        """
        self.api_key = api_key or Config.DEEPSEEK_API_KEY
        self.api_base = api_base or Config.DEEPSEEK_API_BASE
        self.chat_endpoint = f"{self.api_base}/v1//chat/completions"
        self.upload_endpoint = f"{self.api_base}/upload"
        self.parse_endpoint = f"{self.api_base}/parse"
        
        # 基本请求头
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        # 设置日志
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger("DeepSeekAgent")
    
    def chat_completion(self, messages, model=None, temperature=0.7, max_tokens=None, 
                        top_p=1.0, stream=False, **kwargs):
        """
        发送聊天完成请求到DeepSeek API
        
        Args:
            messages (list): 消息列表，每个消息为包含role和content的字典
            model (str, optional): 模型名称
            temperature (float, optional): 采样温度
            max_tokens (int, optional): 最大生成token数
            top_p (float, optional): 核采样概率
            stream (bool, optional): 是否流式响应
            **kwargs: 其他API参数
            
        Returns:
            dict: API响应结果
        """
        model = model or Config.DEEPSEEK_MODEL
        
        # 构建请求体
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "top_p": top_p,
            "stream": stream
        }
        
        # 添加可选参数
        if max_tokens:
            payload["max_tokens"] = max_tokens
            
        # 添加其他参数
        for key, value in kwargs.items():
            payload[key] = value
            
        self.logger.info(f"调用DeepSeek API，模型: {model}")
        
        try:
            # 发送请求
            response = requests.post(
                self.chat_endpoint,
                headers=self.headers,
                json=payload,
                timeout=60
            )
            
            # 检查响应状态
            response.raise_for_status()
            
            # 解析响应
            result = response.json()
            
            if stream:
                # 流式响应处理逻辑
                return result
            else:
                # 常规响应处理
                self.logger.info(f"DeepSeek API调用成功: {len(result['choices'][0]['message']['content'])} 字符")
                return result
                
        except requests.exceptions.RequestException as e:
            self.logger.error(f"DeepSeek API请求失败: {str(e)}")
            if hasattr(e, 'response') and e.response:
                self.logger.error(f"状态码: {e.response.status_code}")
                self.logger.error(f"响应内容: {e.response.text}")
            raise
        except json.JSONDecodeError:
            self.logger.error(f"无法解析API响应为JSON: {response.text}")
            raise
        except Exception as e:
            self.logger.error(f"调用DeepSeek API时发生未知错误: {str(e)}")
            raise
    
    def embedding(self, text, model=None):
        """
        获取文本的嵌入向量
        
        Args:
            text (str): 输入文本
            model (str, optional): 模型名称
            
        Returns:
            list: 文本的嵌入向量
        """
        model = model or Config.DEEPSEEK_EMBEDDING_MODEL
        embedding_endpoint = f"{self.api_base}/v1/embeddings"
        
        payload = {
            "model": model,
            "input": text
        }
        
        try:
            response = requests.post(
                embedding_endpoint,
                headers=self.headers,
                json=payload
            )
            
            response.raise_for_status()
            result = response.json()
            
            return result["data"][0]["embedding"]
            
        except Exception as e:
            self.logger.error(f"获取嵌入向量时出错: {str(e)}")
            raise
    
    def upload_file(self, file_path):
        """
        上传文件到DeepSeek API
        
        Args:
            file_path (str): 要上传的文件路径
            
        Returns:
            dict: 包含file_id的响应
        """
        self.logger.info(f"开始上传文件: {file_path}")
        
        try:
            # 确保文件存在
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"文件不存在: {file_path}")
            
            # 准备上传请求
            with open(file_path, 'rb') as f:
                files = {'file': (os.path.basename(file_path), f)}
                
                # 移除Content-Type头，由requests自动设置
                headers = {
                    "Authorization": f"Bearer {self.api_key}"
                }
                
                # 发送上传请求
                response = requests.post(
                    self.upload_endpoint,
                    headers=headers,
                    files=files
                )
                
                # 检查响应状态
                response.raise_for_status()
                
                # 解析响应
                result = response.json()
                file_id = result.get('file_id')
                
                if not file_id:
                    raise ValueError("上传响应中没有file_id")
                
                self.logger.info(f"文件上传成功，file_id: {file_id}")
                return result
                
        except requests.exceptions.RequestException as e:
            self.logger.error(f"文件上传请求失败: {str(e)}")
            if hasattr(e, 'response') and e.response:
                self.logger.error(f"状态码: {e.response.status_code}")
                self.logger.error(f"响应内容: {e.response.text}")
            raise
        except Exception as e:
            self.logger.error(f"文件上传时发生错误: {str(e)}")
            raise
    
    def parse_file(self, file_id):
        """
        请求DeepSeek API解析已上传的文件
        
        Args:
            file_id (str): 上传文件返回的文件ID
            
        Returns:
            dict: 解析结果
        """
        self.logger.info(f"开始解析文件，file_id: {file_id}")
        
        try:
            # 构建解析端点URL
            parse_url = f"{self.parse_endpoint}/{file_id}"
            
            # 发送解析请求
            response = requests.get(
                parse_url,
                headers=self.headers
            )
            
            # 检查响应状态
            response.raise_for_status()
            
            # 解析响应
            result = response.json()
            
            self.logger.info(f"文件解析成功")
            return result
            
        except requests.exceptions.RequestException as e:
            self.logger.error(f"文件解析请求失败: {str(e)}")
            if hasattr(e, 'response') and e.response:
                self.logger.error(f"状态码: {e.response.status_code}")
                self.logger.error(f"响应内容: {e.response.text}")
            raise
        except Exception as e:
            self.logger.error(f"文件解析时发生错误: {str(e)}")
            raise

# 创建单例实例
deepseek_agent = DeepSeekAgent() 