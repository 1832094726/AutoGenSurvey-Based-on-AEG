# 论文分析工具模块
# 文件：app/modules/paper_analysis_utils.py

import os
import json
import base64
import logging
import re
from typing import Optional, Dict, Any
from openai import OpenAI
from app.config import Config

def call_llm_with_prompt(prompt: str, model_name: str = None, pdf_path: Optional[str] = None) -> str:
    """
    通用大模型调用函数，基于现有Config配置
    
    Args:
        prompt: 提示词
        model_name: 模型名称（如果为None，使用默认模型）
        pdf_path: PDF文件路径（可选）
    
    Returns:
        str: 模型响应内容
    """
    # 如果没有指定模型，使用默认模型
    if model_name is None:
        model_name = Config.DEFAULT_MODEL
    
    try:
        # 根据模型名称选择配置
        client, actual_model = _get_model_client_from_config(model_name)
        
        # 构建消息
        messages = [
            {"role": "system", "content": "You are a professional academic analysis assistant."}
        ]
        
        # 如果有PDF文件，添加文件内容
        if pdf_path and os.path.exists(pdf_path):
            messages.append(_build_pdf_message_from_config(prompt, pdf_path, model_name))
        else:
            # 纯文本输入
            messages.append({"role": "user", "content": prompt})
        
        # 调用API（使用现有配置的重试机制）
        logging.info(f"调用{model_name}模型进行分析")
        
        for attempt in range(Config.MAX_RETRY_ATTEMPTS):
            try:
                response = client.chat.completions.create(
                    model=actual_model,
                    messages=messages,
                    temperature=0.1,
                    stream=True
                )
                
                # 收集响应
                content = ""
                chunk_count = 0
                for chunk in response:
                    chunk_count += 1
                    if chunk.choices and chunk.choices[0].delta.content:
                        content += chunk.choices[0].delta.content
                    
                    # 每100个块记录一次进度
                    if chunk_count % 100 == 0:
                        logging.info(f"收到响应块 #{chunk_count}，当前响应长度: {len(content)} 字符")
                
                logging.info(f"响应接收完成，共 {chunk_count} 个响应块，总长度: {len(content)} 字符")
                return content
                
            except Exception as e:
                logging.warning(f"第 {attempt + 1} 次调用失败: {str(e)}")
                if attempt < Config.MAX_RETRY_ATTEMPTS - 1:
                    import time
                    time.sleep(Config.RETRY_DELAY)
                else:
                    raise
        
    except Exception as e:
        logging.error(f"大模型调用失败: {str(e)}")
        raise

def _get_model_client_from_config(model_name: str) -> tuple:
    """基于现有Config获取模型客户端"""
    
    if model_name == "gemini-2.0-flash":
        client = OpenAI(
            api_key=Config.GEMINI_API_KEY,
            base_url=Config.GEMINI_BASE_URL
        )
        return client, "gemini-2.0-flash"
    
    elif model_name == "qwen-long" or model_name == Config.DEFAULT_MODEL:
        client = OpenAI(
            api_key=Config.QWEN_API_KEY,
            base_url=Config.QWEN_BASE_URL
        )
        return client, Config.QWEN_MODEL
    
    elif model_name == "claude-3-7-sonnet-20250219":
        client = OpenAI(
            api_key=Config.ANTHROPIC_API_KEY,
            base_url=Config.ANTHROPIC_BASE_URL
        )
        return client, "claude-3-7-sonnet-20250219"
    
    elif model_name in ["gpt-3.5-turbo", "gpt-4.1-mini"]:
        client = OpenAI(
            api_key=Config.OPENAI_API_KEY,
            base_url=Config.OPENAI_BASE_URL
        )
        return client, Config.OPENAI_MODEL
    
    elif model_name == "deepseek-v3":
        client = OpenAI(
            api_key=Config.DEEPSEEK_API_KEY,
            base_url=Config.DEEPSEEK_BASE_URL
        )
        return client, "deepseek-v3"
    
    else:
        # 默认使用千问模型
        logging.warning(f"未知模型 {model_name}，使用默认模型 {Config.DEFAULT_MODEL}")
        client = OpenAI(
            api_key=Config.QWEN_API_KEY,
            base_url=Config.QWEN_BASE_URL
        )
        return client, Config.DEFAULT_MODEL

def _build_pdf_message_from_config(prompt: str, pdf_path: str, model_name: str) -> Dict[str, Any]:
    """基于现有配置构建包含PDF的消息"""
    
    if model_name == "gemini-2.0-flash":
        # Gemini使用base64编码
        with open(pdf_path, "rb") as f:
            pdf_data = base64.b64encode(f.read()).decode()
        
        return {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "file", "file": {
                    "filename": os.path.basename(pdf_path),
                    "file_data": pdf_data
                }}
            ]
        }
    
    elif model_name == "qwen-long":
        # 千问可能需要先上传文件，这里可以复用现有的agents.py中的文件上传逻辑
        # 暂时简化为文本处理
        return {"role": "user", "content": prompt}
    
    else:
        # 其他模型暂时只支持文本
        return {"role": "user", "content": prompt}

def extract_json_from_response(response_text: str) -> Dict[str, Any]:
    """从响应文本中提取JSON（复用现有逻辑）"""
    
    # 清理响应文本
    cleaned_text = _clean_response_text(response_text)
    
    # 尝试多种方法提取JSON
    extraction_methods = [
        _extract_direct_json,
        _extract_json_code_block,
        _extract_json_braces
    ]
    
    for method in extraction_methods:
        try:
            result = method(cleaned_text)
            if result:
                logging.info(f"成功使用 {method.__name__} 提取JSON")
                return result
        except Exception as e:
            logging.debug(f"{method.__name__} 提取失败: {str(e)}")
            continue
    
    # 所有方法都失败
    logging.error(f"无法从响应中提取JSON，响应内容: {response_text[:500]}...")
    raise ValueError("无法从响应中提取有效的JSON")

def _clean_response_text(text: str) -> str:
    """清理响应文本"""
    text = re.sub(r'\s+', ' ', text.strip())
    text = re.sub(r'```json\s*', '', text)
    text = re.sub(r'```\s*$', '', text)
    return text

def _extract_direct_json(text: str) -> Optional[Dict[str, Any]]:
    """尝试直接解析JSON"""
    return json.loads(text)

def _extract_json_code_block(text: str) -> Optional[Dict[str, Any]]:
    """从代码块中提取JSON"""
    patterns = [
        r'```json\s*(.*?)\s*```',
        r'```\s*(.*?)\s*```',
        r'`(.*?)`'
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text, re.DOTALL)
        if match:
            return json.loads(match.group(1).strip())
    
    return None

def _extract_json_braces(text: str) -> Optional[Dict[str, Any]]:
    """提取大括号内的JSON"""
    start = text.find('{')
    end = text.rfind('}')
    
    if start != -1 and end != -1 and end > start:
        json_str = text[start:end+1]
        return json.loads(json_str)
    
    return None

def save_uploaded_file_to_config_dir(uploaded_file, analysis_id: str, file_prefix: str = "") -> str:
    """将上传文件保存到配置指定的目录"""
    from werkzeug.utils import secure_filename
    
    # 使用现有配置的上传目录
    upload_dir = Config.UPLOAD_DIR
    
    # 确保目录存在
    os.makedirs(upload_dir, exist_ok=True)
    
    # 生成文件名
    filename = secure_filename(uploaded_file.filename)
    if file_prefix:
        filename = f"{analysis_id}_{file_prefix}_{filename}"
    else:
        filename = f"{analysis_id}_{filename}"
    
    file_path = os.path.join(upload_dir, filename)
    
    # 检查文件大小
    uploaded_file.seek(0, 2)  # 移动到文件末尾
    file_size = uploaded_file.tell()
    uploaded_file.seek(0)  # 重置到文件开头
    
    if file_size > Config.MAX_CONTENT_LENGTH:
        raise ValueError(f"文件大小 ({file_size / 1024 / 1024:.2f} MB) 超过限制 ({Config.MAX_CONTENT_LENGTH / 1024 / 1024:.2f} MB)")
    
    # 保存文件
    uploaded_file.save(file_path)
    logging.info(f"文件已保存到: {file_path}")
    
    return file_path

def cache_analysis_result(analysis_id: str, result_data: dict) -> str:
    """将分析结果缓存到配置指定的目录"""
    
    # 使用现有配置的缓存目录
    cache_dir = Config.CACHE_DIR
    
    # 确保目录存在
    os.makedirs(cache_dir, exist_ok=True)
    
    # 生成缓存文件路径
    cache_file = os.path.join(cache_dir, f"paper_analysis_{analysis_id}.json")
    
    # 保存结果
    with open(cache_file, 'w', encoding='utf-8') as f:
        json.dump(result_data, f, ensure_ascii=False, indent=2)
    
    logging.info(f"分析结果已缓存到: {cache_file}")
    return cache_file

def load_cached_analysis_result(analysis_id: str) -> Optional[dict]:
    """从缓存加载分析结果"""
    
    cache_file = os.path.join(Config.CACHE_DIR, f"paper_analysis_{analysis_id}.json")
    
    if os.path.exists(cache_file):
        try:
            with open(cache_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logging.error(f"加载缓存失败: {str(e)}")
    
    return None

def validate_analysis_result(result: Dict[str, Any], analysis_type: str) -> bool:
    """
    验证分析结果的完整性
    
    Args:
        result: 分析结果
        analysis_type: 分析类型 ('method_extraction', 'coverage_analysis', 'relation_coverage')
    
    Returns:
        bool: 是否有效
    """
    
    if analysis_type == 'method_extraction':
        return (
            'methods' in result and
            isinstance(result['methods'], list) and
            len(result['methods']) > 0
        )
    
    elif analysis_type == 'coverage_analysis':
        return (
            'method_coverage' in result and
            'algorithm_coverage' in result and
            'detailed_analysis' in result and
            isinstance(result['method_coverage'], dict) and
            isinstance(result['algorithm_coverage'], dict)
        )
    
    elif analysis_type == 'relation_coverage':
        return (
            'relation_coverage' in result and
            'detailed_analysis' in result and
            isinstance(result['relation_coverage'], dict)
        )
    
    return False

def estimate_token_count(text: str) -> int:
    """
    估算文本的token数量
    
    Args:
        text: 输入文本
    
    Returns:
        int: 估算的token数量
    """
    # 简单估算：英文约4字符=1token，中文约1.5字符=1token
    chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', text))
    english_chars = len(text) - chinese_chars
    
    estimated_tokens = chinese_chars / 1.5 + english_chars / 4
    return int(estimated_tokens)

def log_api_usage(model_name: str, prompt_tokens: int, completion_tokens: int, cost: float = 0.0):
    """
    记录API使用情况
    
    Args:
        model_name: 模型名称
        prompt_tokens: 输入token数
        completion_tokens: 输出token数
        cost: 费用（可选）
    """
    
    total_tokens = prompt_tokens + completion_tokens
    
    logging.info(f"API使用统计 - 模型: {model_name}")
    logging.info(f"输入tokens: {prompt_tokens}, 输出tokens: {completion_tokens}, 总计: {total_tokens}")
    
    if cost > 0:
        logging.info(f"预估费用: ${cost:.4f}")
    
    # 可以在这里添加到数据库记录API使用情况的逻辑
