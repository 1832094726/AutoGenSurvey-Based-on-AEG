import os
import json
import logging
import time
import requests
from openai import OpenAI
from app.config import Config

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("LLM_Interface")

class DeepSeekAgent:
    """DeepSeek API客户端包装类"""
    
    def __init__(self, api_key=None, api_base=None):
        """
        初始化DeepSeek API客户端
        
        Args:
            api_key (str, optional): API密钥
            api_base (str, optional): API基础URL
        """
        self.api_key = api_key or Config.DEEPSEEK_API_KEY
        self.api_base = api_base or Config.DEEPSEEK_BASE_URL
        self.logger = logging.getLogger("deepseek_agent")
        
        # 设置API端点
        self.chat_endpoint = f"{self.api_base}/v1/chat/completions"
        self.embedding_endpoint = f"{self.api_base}/v1/embeddings"
        self.upload_endpoint = f"{self.api_base}/v1/files"
        self.parse_endpoint = f"{self.api_base}/v1/files/retrieve"
        
        # 设置HTTP头
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
    
    def chat_completion(self, messages, model=None, temperature=0.7, max_tokens=None, 
                        top_p=1.0, stream=False, **kwargs):
        """
        发送聊天请求
        
        Args:
            messages (list): 消息列表
            model (str, optional): 模型名称
            temperature (float): 温度参数
            max_tokens (int, optional): 最大token数量，默认为None表示不限制
            top_p (float): Top-p采样参数
            stream (bool): 是否使用流式响应
            **kwargs: 其他参数
            
        Returns:
            dict: API响应
        """
        self.logger.info("发送聊天请求")
        
        try:
            # 构建请求体
            payload = {
                "model": model or "deepseek-chat",
                "messages": messages,
                "temperature": temperature,
                "top_p": top_p,
                "stream": stream
            }
            
            # 仅当max_tokens不为None时添加该参数
            if max_tokens is not None:
                payload["max_tokens"] = max_tokens
            
            # 添加其他参数
            payload.update(kwargs)
            
            # 发送请求
            response = requests.post(
                self.chat_endpoint,
                headers=self.headers,
                json=payload,
                stream=stream
            )
            
            # 检查响应状态
            response.raise_for_status()
            
            if stream:
                return response.iter_lines()
            else:
                # 解析响应
                return response.json()
                
        except requests.exceptions.RequestException as e:
            self.logger.error(f"聊天请求失败: {str(e)}")
            if hasattr(e, 'response') and e.response:
                self.logger.error(f"状态码: {e.response.status_code}")
                self.logger.error(f"响应内容: {e.response.text}")
            raise
        except Exception as e:
            self.logger.error(f"聊天请求时发生错误: {str(e)}")
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
        解析已上传的文件
        
        Args:
            file_id (str): 文件ID，从upload_file方法获取
            
        Returns:
            dict: 解析结果
        """
        self.logger.info(f"开始解析文件，file_id: {file_id}")
        
        try:
            # 构建请求体
            payload = {
                "file_id": file_id
            }
            
            # 发送解析请求
            response = requests.post(
                self.parse_endpoint,
                headers=self.headers,
                json=payload
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

def call_openai_api(system_prompt, user_prompt, model="gpt-3.5-turbo", temperature=0.7, max_tokens=None):
    """
    调用OpenAI API获取回复
    
    Args:
        system_prompt (str): 系统提示词
        user_prompt (str): 用户提示词
        model (str): 使用的模型名称
        temperature (float): 温度参数，控制输出的随机性
        max_tokens (int, optional): 生成的最大token数量，设置为None则不限制
        
    Returns:
        str: API的回复内容
    """
    try:
        # 优先使用环境变量中的API Key，若无则使用配置文件中的Key
        api_key = os.environ.get("OPENAI_API_KEY", Config.OPENAI_API_KEY)
        if not api_key:
            raise ValueError("未设置OpenAI API Key")
            
        # 创建OpenAI客户端
        client = OpenAI(api_key=api_key)
        logger.info(f"调用OpenAI API, 模型: {model}")
        
        # 记录开始时间
        start_time = time.time()
        
        # 构建消息列表
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        # 准备API调用参数
        params = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
        }
        
        # 仅当max_tokens不为None时添加该参数
        if max_tokens is not None:
            params["max_tokens"] = max_tokens
        
        # 调用API
        response = client.chat.completions.create(**params)
        
        # 记录结束时间
        elapsed_time = time.time() - start_time
        logger.info(f"OpenAI API调用完成，耗时: {elapsed_time:.2f}秒")
        
        # 提取回复内容
        content = response.choices[0].message.content
        
        return content
        
    except Exception as e:
        logger.error(f"调用OpenAI API时出错: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return None

def call_qwen_api(system_prompt, user_prompt, file_id=None, model="qwen-long", temperature=0.7, max_tokens=None):
    """
    调用千问API获取回复
    
    Args:
        system_prompt (str): 系统提示词
        user_prompt (str): 用户提示词
        file_id (str, optional): 文件ID，用于引用上传的文件
        model (str): 使用的模型名称
        temperature (float): 温度参数，控制输出的随机性
        max_tokens (int, optional): 生成的最大token数量，设置为None则不限制
        
    Returns:
        str: API的回复内容
    """
    try:
        # 创建千问客户端
        client = OpenAI(
            api_key=Config.QWEN_API_KEY,
            base_url=Config.QWEN_BASE_URL
        )
        logger.info(f"调用千问API, 模型: {model}")
        
        # 构建消息列表
        messages = [{"role": "system", "content": system_prompt}]
        
        # 如果提供了文件ID，添加文件引用
        if file_id:
            messages.append({"role": "system", "content": f"fileid://{file_id}"})
            
        messages.append({"role": "user", "content": user_prompt})
        
        # 记录开始时间
        start_time = time.time()
        
        # 准备API调用参数
        params = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "stream": True
        }
        
        # 仅当max_tokens不为None时添加该参数
        if max_tokens is not None:
            params["max_tokens"] = max_tokens
        
        # 调用API
        completion = client.chat.completions.create(**params)
        
        # 收集流式响应
        response_content = ""
        chunk_count = 0
        
        for chunk in completion:
            chunk_count += 1
            if chunk.choices and chunk.choices[0].delta.content:
                content_piece = chunk.choices[0].delta.content
                response_content += content_piece
                if chunk_count % 10 == 0:  # 每10个块记录一次
                    logger.info(f"响应块 #{chunk_count}，当前响应长度: {len(response_content)}")
        
        # 记录结束时间
        elapsed_time = time.time() - start_time
        logger.info(f"千问API调用完成，耗时: {elapsed_time:.2f}秒，响应长度: {len(response_content)}字符")
        
        return response_content
        
    except Exception as e:
        logger.error(f"调用千问API时出错: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return None

# 默认代理实例
deepseek_agent = DeepSeekAgent() 