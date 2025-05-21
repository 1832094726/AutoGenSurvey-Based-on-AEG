import os
import json
import logging
import time
from openai import OpenAI
from app.config import Config

def call_openai_api(system_prompt, user_prompt, model="gpt-3.5-turbo", temperature=0.7, max_tokens=2000):
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
        logging.info(f"调用OpenAI API, 模型: {model}")
        
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
        logging.info(f"OpenAI API调用完成，耗时: {elapsed_time:.2f}秒")
        
        # 提取回复内容
        content = response.choices[0].message.content
        
        return content
        
    except Exception as e:
        logging.error(f"调用OpenAI API时出错: {str(e)}")
        import traceback
        logging.error(traceback.format_exc())
        return None

def call_qwen_api(system_prompt, user_prompt, file_id=None, model="qwen-long", temperature=0.7, max_tokens=2000):
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
        logging.info(f"调用千问API, 模型: {model}")
        
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
                    logging.info(f"响应块 #{chunk_count}，当前响应长度: {len(response_content)}")
        
        # 记录结束时间
        elapsed_time = time.time() - start_time
        logging.info(f"千问API调用完成，耗时: {elapsed_time:.2f}秒，响应长度: {len(response_content)}字符")
        
        return response_content
        
    except Exception as e:
        logging.error(f"调用千问API时出错: {str(e)}")
        import traceback
        logging.error(traceback.format_exc())
        return None 