import os
import logging
import json
import re
from qwen_agent.agents import Assistant
from qwen_agent.tools.base import BaseTool, register_tool
from pathlib import Path
from openai import OpenAI
from app.config import Config
import time
import datetime
import PyPDF2  # 添加PyPDF2导入
import hashlib
import traceback
import shutil
from app.modules.db_manager import db_manager  # 导入db_manager

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# 添加检查提取完成状态的公共函数
def check_extraction_complete(content):
    """
    检查提取响应中是否包含完成标志
    
    Args:
        content (str): API响应内容
    
    Returns:
        bool: 提取是否完成
    """
    is_complete = "EXTRACTION_COMPLETE: true" in content.lower()
    logging.info(f"提取完成状态: {'完成' if is_complete else '未完成'}")
    return is_complete

# 添加模拟实体数据，用于调试或禁用AI时的默认返回
MOCK_ENTITIES = [
    {
        "algorithm_entity": {
            "algorithm_id": "MockAlgo2023_TestAlgorithm",
            "entity_type": "Algorithm",
            "name": "模拟算法",
            "title": "测试算法论文",
            "year": 2023,
            "authors": ["测试作者1", "测试作者2"],
            "task": "测试任务",
            "dataset": ["测试数据集1", "测试数据集2"],
            "metrics": ["准确率", "精确率"],
            "architecture": {
                "components": ["测试组件1", "测试组件2"],
                "connections": ["测试连接"],
                "mechanisms": ["测试机制"]
            },
            "methodology": {
                "training_strategy": ["测试策略"],
                "parameter_tuning": ["测试参数"]
            },
            "feature_processing": ["测试处理方法"]
        }
    }
]

# 添加新的工具类用于提取PDF文本
@register_tool('pdf_text_extractor')
class PdfTextExtractor(BaseTool):
    description = 'PDF文本提取工具，输入PDF文件路径，返回提取的文本内容。'
    parameters = [{
        'name': 'pdf_path',
        'type': 'string',
        'description': 'PDF文件的路径',
        'required': True
    }]

    def call(self, params: str, **kwargs) -> str:
        import json
        import urllib.parse
        import os
        from pathlib import Path
        from app.config import Config
        
        # 解析参数
        pdf_path = json.loads(params)['pdf_path']
        
        # 生成缓存文件路径
        cache_dir = os.path.join(Config.CACHE_DIR, "pdf_text")
        os.makedirs(cache_dir, exist_ok=True)
        
        filename = Path(pdf_path).stem
        cache_file = os.path.join(cache_dir, f"{filename}_partial.json")
        
        # 返回文本内容，模型会处理提取逻辑
        return json.dumps([pdf_path, cache_file], ensure_ascii=False)

# 修改提取文本函数，添加缓存功能
def extract_text_from_pdf(pdf_path, task_id=None):
    """
    从PDF文件中提取文本内容，并缓存结果
    
    Args:
        pdf_path (str): PDF文件的路径
        task_id (str, optional): 任务ID，用于缓存标识
        
    Returns:
        str: 提取的文本内容
    """
    try:
        # 生成缓存文件路径
        from app.config import Config
        import os
        from pathlib import Path
        
        # 标准化路径分隔符，确保跨平台一致性
        pdf_path = os.path.normpath(pdf_path)
        
        cache_dir = os.path.join(Config.CACHE_DIR, "pdf_text")
        os.makedirs(cache_dir, exist_ok=True)
        
        # 使用原始文件名作为标识，确保缓存文件与上传文件名完全一致
        basename = os.path.basename(pdf_path)
        filename_without_ext = os.path.splitext(basename)[0]
        
        # 移除可能的任务ID前缀
        if task_id and filename_without_ext.startswith(f"task_{task_id}_"):
            filename_without_ext = filename_without_ext[len(f"task_{task_id}_"):]
        
        # 移除日期时间前缀格式 (如: 20250520_164839_)
        date_time_prefix_pattern = r"^\d{8}_\d{6}_"
        import re
        if re.match(date_time_prefix_pattern, filename_without_ext):
            logging.info(f"检测到日期时间前缀: {filename_without_ext}")
            filename_without_ext = re.sub(date_time_prefix_pattern, "", filename_without_ext)
            logging.info(f"移除前缀后的文件名: {filename_without_ext}")
        
        # 只使用JSON格式缓存文件
        cache_path = os.path.join(cache_dir, f"{filename_without_ext}_partial.json")
        
        logging.info(f"处理PDF文件文本: {pdf_path}, 缓存文件: {cache_path}")
        
        # 检查缓存文件是否存在
        if os.path.exists(cache_path) and os.path.getsize(cache_path) > 0:
            logging.info(f"从缓存加载: {cache_path}")
            try:
                with open(cache_path, 'r', encoding='utf-8') as f:
                    cache_data = json.load(f)
                    
                # 检查是否完成
                is_complete = cache_data.get('complete', False)
                text = cache_data.get('text', '')
                
                if is_complete and text.strip():
                    logging.info(f"成功从缓存加载已完成的文本，长度: {len(text)} 字符")
                    return text
                elif text.strip():
                    logging.info(f"从缓存加载部分提取的文本，长度: {len(text)} 字符，将继续提取")
                    extracted_text = text
                    # 将尝试次数重置为1
                    current_attempt = 1
                else:
                    logging.warning(f"缓存文件中没有有效文本内容，将重新提取")
                    extracted_text = ""
                    current_attempt = 0
            except Exception as e:
                logging.error(f"读取缓存文件出错: {str(e)}")
                extracted_text = ""
                current_attempt = 0
        else:
            # 如果缓存不存在，初始化变量
            extracted_text = ""
            current_attempt = 0
        
        # 尝试在缓存目录中查找可能匹配的文件
        if not extracted_text:
            matching_files = []
            normalized_name = re.sub(r'[^a-zA-Z0-9]', '', filename_without_ext.lower())
            
            for file in os.listdir(cache_dir):
                if file.endswith("_partial.json"):
                    file_base = file[:-13]  # 去掉 "_partial.json" 后缀
                    normalized_file = re.sub(r'[^a-zA-Z0-9]', '', file_base.lower())
                    
                    # 如果规范化后的名称匹配，或者包含关系，则认为可能匹配
                    if normalized_file == normalized_name or normalized_file in normalized_name or normalized_name in normalized_file:
                        matching_files.append(os.path.join(cache_dir, file))
            
            # 如果找到可能匹配的缓存文件，尝试使用第一个
            if matching_files:
                best_match = matching_files[0]
                logging.info(f"找到可能匹配的缓存文件: {best_match}")
                
                try:
                    with open(best_match, 'r', encoding='utf-8') as f:
                        match_data = json.load(f)
                        match_text = match_data.get('text', '')
                        match_complete = match_data.get('complete', False)
                        
                        if match_text.strip():
                            logging.info(f"成功从匹配的缓存文件加载文本，长度: {len(match_text)} 字符，完成状态: {match_complete}")
                            # 将内容复制到标准缓存路径以便下次直接使用
                            with open(cache_path, 'w', encoding='utf-8') as out_f:
                                json.dump({
                                    'text': match_text,
                                    'complete': match_complete,
                                    'attempt': 0,
                                    'timestamp': datetime.datetime.now().isoformat()
                                }, out_f, ensure_ascii=False)
                            
                            if match_complete:
                                return match_text
                            else:
                                extracted_text = match_text
                                current_attempt = 1
                except Exception as e:
                    logging.error(f"读取匹配的缓存文件时出错: {str(e)}")
        
        # 设置最大尝试次数
        max_attempts = 3
        is_extraction_complete = False
        
        # 如果未完成，则继续提取
        while current_attempt < max_attempts and not is_extraction_complete:
            current_attempt += 1
            try:
                if hasattr(Config, 'QWEN_API_KEY') and Config.QWEN_API_KEY:
                    logging.info(f"尝试使用千问API提取文本 (尝试 {current_attempt}/{max_attempts})")
                    # 使用千问API提取文本
                    from openai import OpenAI
                    
                    client = OpenAI(
                        api_key=Config.QWEN_API_KEY,
                        base_url=Config.QWEN_BASE_URL
                    )
                    
                    # 上传文件进行处理
                    file = client.files.create(file=Path(pdf_path), purpose="file-extract")
                    file_id = file.id
                    logging.info(f"文件上传成功，file_id: {file_id}")
                    
                    # 构建提示词
                    user_prompt = '请将PDF文件中的所有文本内容提取出来，原格式输出，不要添加任何注释或额外信息。要尽可能保留文本的完整性，包括所有页面内容、表格内容和参考文献。用原始段落格式输出，避免因过长而截断。'
                    
                    # 如果有之前的提取结果，添加到提示中
                    if extracted_text:
                        user_prompt = f'''我已经提取了部分文本内容如下:
                        
                        {extracted_text[:1000]}...
                        
                        请继续提取剩余的内容，确保不要重复已经提取的部分，只返回新内容。将新内容与之前内容无缝连接，不要有重复或遗漏。提取时要尽可能保留原始格式，包括所有页面内容、表格内容和参考文献，避免因过长而截断。请完成整个文档的提取。

                        在提取完成后，请加上一行单独的"EXTRACTION_COMPLETE: true"来表示你已完成提取整个文档。如果还有更多内容需要提取，则加上"EXTRACTION_COMPLETE: false"。'''
                    else:
                        user_prompt += '\n\n在提取完成后，请加上一行单独的"EXTRACTION_COMPLETE: true"来表示你已完成提取整个文档。如果还有更多内容需要提取，则加上"EXTRACTION_COMPLETE: false"。'
                    
                    # 构建消息
                    messages = [
                        {
                            'role': 'system',
                            'content': f'fileid://{file_id}'
                        },
                        {
                            'role': 'user',
                            'content': user_prompt
                        }
                    ]
                    
                    # 调用API，确保大模型可以返回最大的内容
                    completion = client.chat.completions.create(
                        model=Config.QWEN_MODEL or "qwen-long",
                        messages=messages,
                        temperature=0.0,
                        stream=True,
                        max_tokens=None  # 不限制token数量
                    )
                    
                    # 收集流式响应内容
                    new_text = ""
                    chunk_count = 0
                    
                    for chunk in completion:
                        chunk_count += 1
                        if chunk.choices and chunk.choices[0].delta.content:
                            new_text += chunk.choices[0].delta.content
                        
                        # 每100个块记录一次进度，避免日志过多
                        if chunk_count % 100 == 0:
                            logging.info(f"已收到 {chunk_count} 个响应块，当前文本长度: {len(new_text)} 字符")
                    
                    logging.info(f"API提取完成，总共收到 {chunk_count} 个响应块，新文本长度: {len(new_text)} 字符")
                    
                    # 检查是否包含完成标记
                    is_extraction_complete = "EXTRACTION_COMPLETE: true" in new_text.lower()
                    
                    # 从新文本中移除完成标记
                    new_text = new_text.replace("EXTRACTION_COMPLETE: true", "").replace("EXTRACTION_COMPLETE: false", "")
                    
                    # 合并文本内容
                    if not extracted_text:
                        extracted_text = new_text
                    else:
                        extracted_text += "\n" + new_text
                    
                    # 保存到缓存文件
                    with open(cache_path, 'w', encoding='utf-8') as f:
                        json.dump({
                            'text': extracted_text,
                            'complete': is_extraction_complete,
                            'attempt': current_attempt,
                            'timestamp': datetime.datetime.now().isoformat()
                        }, f, ensure_ascii=False)
                    
                    logging.info(f"已保存提取结果到缓存，当前文本长度: {len(extracted_text)} 字符，是否完成: {is_extraction_complete}")
                    
                    # 如果已完成，直接返回
                    if is_extraction_complete:
                        return extracted_text
            except Exception as e:
                logging.error(f"使用千问API提取文本时出错: {str(e)}")
                logging.error(traceback.format_exc())
                logging.info("将使用备用方法提取文本")
        
        # 如果提取成功，但未完成，仍然返回部分结果
        if extracted_text:
            logging.info(f"返回部分提取结果，长度: {len(extracted_text)} 字符")
            return extracted_text
        
        # 备用方法：使用PyPDF2提取
        logging.info(f"使用PyPDF2从PDF提取文本: {pdf_path}")
        text = ""
        try:
            with open(pdf_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                num_pages = len(reader.pages)
                
                logging.info(f"开始从PDF提取文本: {pdf_path}, 共 {num_pages} 页")
                
                # 提取每一页的文本
                for page_num in range(num_pages):
                    try:
                        page = reader.pages[page_num]
                        page_text = page.extract_text()
                        if page_text:
                            text += page_text + "\n\n"
                        # 每10页记录一次进度
                        if (page_num + 1) % 10 == 0 or page_num == num_pages - 1:
                            logging.info(f"已处理 {page_num+1}/{num_pages} 页")
                    except Exception as page_err:
                        logging.error(f"处理第 {page_num+1} 页时出错: {str(page_err)}")
                
                # 缓存提取的文本
                if text.strip():
                    with open(cache_path, 'w', encoding='utf-8') as f:
                        json.dump({
                            'text': text,
                            'complete': True,  # 假设PyPDF2能够完整提取
                            'attempt': max_attempts,
                            'timestamp': datetime.datetime.now().isoformat()
                        }, f, ensure_ascii=False)
                    logging.info(f"成功使用PyPDF2提取文本，已缓存到: {cache_path}")
                    return text
                else:
                    logging.warning("PyPDF2提取的文本为空")
        except Exception as pdf_err:
            logging.error(f"使用PyPDF2提取文本时出错: {str(pdf_err)}")
        
        # 如果所有方法都失败，尝试使用pdfminer作为最后的备用方法
        try:
            from pdfminer.high_level import extract_text as pdfminer_extract_text
            logging.info(f"使用pdfminer提取文本: {pdf_path}")
            text = pdfminer_extract_text(pdf_path)
            
            if text.strip():
                with open(cache_path, 'w', encoding='utf-8') as f:
                    json.dump({
                        'text': text,
                        'complete': True,  # 假设pdfminer能够完整提取
                        'attempt': max_attempts,
                        'timestamp': datetime.datetime.now().isoformat()
                    }, f, ensure_ascii=False)
                logging.info(f"成功使用pdfminer提取文本，已缓存到: {cache_path}")
                return text
            else:
                logging.warning("pdfminer提取的文本为空")
        except Exception as miner_err:
            logging.error(f"使用pdfminer提取文本时出错: {str(miner_err)}")
        
        logging.error(f"所有提取方法都失败，无法从PDF提取文本: {pdf_path}")
        return ""
    except Exception as e:
        logging.error(f"提取PDF文本过程中发生错误: {str(e)}")
        logging.error(traceback.format_exc())
        return ""

# 修改生成提取提示词的函数
def generate_entity_extraction_prompt(text, model_name, previous_entities=None, partial_extraction=False):
    """
    根据模型名称和上下文生成提取实体的提示词
    
    Args:
        text (str): 论文文本
        model_name (str): 模型名称
        previous_entities (list): 之前提取的实体列表，用于断点续传
        partial_extraction (bool): 是否为部分提取（断点续传）
        
    Returns:
        str: 生成的提示词
    """
    # 基础提示模板
    base_prompt = """
请从以下论文文本中提取算法、数据集和评价指标的实体信息，以JSON格式返回。

请识别以下类型的实体：
1. 算法：论文中描述的机器学习或深度学习算法，例如BERT、ResNet、LSTM等
2. 数据集：用于训练或评估算法的数据集，例如ImageNet、COCO、CIFAR-10等
3. 评价指标：用于评估算法性能的度量，例如准确率、精确率、召回率等

对于每个实体，请尽可能提取以下信息：
- 实体类型（Algorithm, Dataset, Metric）
- 实体ID（使用格式：实体名称_年份，例如BERT_2018）
- 实体名称
- 发表年份
- 作者
- 任务领域
- 用于评估的数据集（如果是算法）
- 用于评估的指标（如果是算法）
- 算法架构（如果是算法）
- 方法论（如果是算法）
- 特征处理方法（如果是算法）

请以JSON格式输出，确保包含以下结构：
```json
[
  {
    "algorithm_entity": {
      "algorithm_id": "算法名_年份",
      "entity_type": "Algorithm",
      "name": "算法名称",
      "title": "论文标题",
      "year": 发表年份,
      "authors": ["作者1", "作者2"],
      "task": "任务领域",
      "dataset": ["使用的数据集1", "使用的数据集2"],
      "metrics": ["使用的评价指标1", "使用的评价指标2"],
      "architecture": {
        "components": ["组件1", "组件2"],
        "connections": ["连接描述"],
        "mechanisms": ["机制描述"]
      },
      "methodology": {
        "training_strategy": ["训练策略"],
        "parameter_tuning": ["参数调整方法"]
      },
      "feature_processing": ["特征处理方法"]
    }
  },
  {
    "dataset_entity": {
      "dataset_id": "数据集名_年份",
      "entity_type": "Dataset",
      "name": "数据集名称",
      "year": 发表年份,
      "domain": "领域",
      "size": "数据集大小",
      "characteristics": ["特征1", "特征2"]
    }
  },
  {
    "metric_entity": {
      "metric_id": "指标名_年份",
      "entity_type": "Metric",
      "name": "指标名称",
      "description": "指标描述",
      "formula": "计算公式",
      "value_range": "取值范围",
      "interpretation": "解释"
    }
  }
]
```

只包含论文中明确提到的实体信息，如果某些字段信息不可用，可以省略。请确保JSON格式正确，避免语义错误。
"""

    # 如果有之前提取的实体，添加到提示中
    if previous_entities and len(previous_entities) > 0:
        # 格式化之前的实体为提示
        previous_entities_str = json.dumps(previous_entities[:5], ensure_ascii=False, indent=2)
        context_prompt = f"""
我之前已经从这篇论文中提取了部分实体，但提取不完整。以下是已知的部分实体：

{previous_entities_str}

请继续完善这些实体的信息，并提取论文中的其他实体。注意避免重复提取已有的实体，专注于补充已有实体的缺失信息并提取新的实体。
"""
        base_prompt = context_prompt + base_prompt
    
    # 添加完成状态请求
    completion_request = """
最后，请明确告知我提取是否已完成，还是需要继续提取更多实体。请根据你对文本的分析，判断是否已经提取了所有可能的实体。

在JSON返回后，请单独一行写明"EXTRACTION_COMPLETE: true"（如果你认为已经提取完所有实体）或"EXTRACTION_COMPLETE: false"（如果你认为还有更多实体需要提取）。
"""
    base_prompt += completion_request
    
    # 如果是部分提取，添加特殊说明
    if partial_extraction:
        continuation_note = """
注意：这是一个部分提取任务，你只需要关注未被提取的实体或已有实体的不完整信息。不要重复已经完全提取的实体。
"""
        base_prompt += continuation_note
    
    # 根据模型类型调整提示词
    if model_name.lower() == "qwen":
        # 千问模型提示
        prompt = base_prompt + f"\n\n论文文本：\n{text[:10000]}"  # 截取前10000个字符避免过长
    elif model_name.lower() == "openai":
        # OpenAI模型提示
        prompt = base_prompt + f"\n\n论文文本：\n{text[:8000]}"  # OpenAI模型限制更严格
    else:
        # 通用提示
        prompt = base_prompt + f"\n\n论文文本：\n{text[:8000]}"
    
    return prompt

# 更新 extract_entities_with_openai 函数
def extract_entities_with_openai(prompt, model_name="gpt-3.5-turbo", max_attempts=3, temp_cache_path=None):
    """
    使用OpenAI API提取实体
    
    Args:
        prompt (str): 提取实体的提示词
        model_name (str): 模型名称
        max_attempts (int): 最大尝试次数
        temp_cache_path (str, optional): 临时缓存文件路径
        
    Returns:
        tuple: (提取的实体列表, 是否处理完成)
    """
    if not hasattr(Config, 'OPENAI_API_KEY') or not Config.OPENAI_API_KEY:
        logging.error("未配置OpenAI API密钥")
        return [], False
    
    # 加载临时缓存（如果有）
    partial_results = {}
    if temp_cache_path and os.path.exists(temp_cache_path):
        try:
            with open(temp_cache_path, 'r', encoding='utf-8') as f:
                partial_results = json.load(f)
                logging.info(f"加载了临时缓存数据: {temp_cache_path}")
        except Exception as e:
            logging.error(f"读取临时缓存出错: {str(e)}")
    
    # 确保临时缓存目录存在
    if temp_cache_path:
        temp_dir = os.path.dirname(temp_cache_path)
        if temp_dir and not os.path.exists(temp_dir):
            try:
                os.makedirs(temp_dir, exist_ok=True)
                logging.info(f"创建临时缓存目录: {temp_dir}")
            except Exception as e:
                logging.error(f"创建临时缓存目录失败: {str(e)}")
                # 如果目录创建失败，不使用缓存继续执行
                temp_cache_path = None
    
    # 尝试多次提取
    attempt = 0
    while attempt < max_attempts:
        attempt += 1
        try:
            logging.info(f"尝试使用OpenAI提取实体 (尝试 {attempt}/{max_attempts})")
            # 创建OpenAI客户端
            client = OpenAI(
                api_key=Config.OPENAI_API_KEY
            )
            # 构建消息
            messages = [
                {"role": "system", "content": "你是一个专注于从学术论文中提取实体信息的AI助手。请从提供的文本中提取算法、数据集和评价指标等相关实体信息，并以JSON格式返回。"},
                {"role": "user", "content": prompt}
            ]
            # 调用API
            response = client.chat.completions.create(
                model=model_name,
                messages=messages,
                temperature=0.2,
                max_tokens=None,  # 不限制token数量
                stream=True
            )
            # 收集流式响应内容
            content = ""
            for chunk in response:
                if chunk.choices and chunk.choices[0].delta.content:
                    content += chunk.choices[0].delta.content
                # 保存部分结果到临时缓存
                if temp_cache_path:
                    try:
                        # 更新临时缓存
                        partial_results.update({
                            'response': content,
                            'attempt': attempt,
                            'timestamp': datetime.datetime.now().isoformat()
                        })
                        with open(temp_cache_path, 'w', encoding='utf-8') as f:
                            json.dump(partial_results, f, ensure_ascii=False, indent=2)
                        logging.info(f"已保存部分响应到临时缓存: {temp_cache_path}")
                    except Exception as e:
                        logging.error(f"保存临时缓存出错: {str(e)}")
                        # 继续执行，不因缓存错误中断主流程
                # 检查是否包含完成标志 - 使用公共函数
                is_complete = check_extraction_complete(content)
            # 尝试提取JSON部分
            json_text = extract_json_from_text(content)
            entities = []
            # 验证并解析JSON
            if json_text:
                try:
                    # 尝试直接解析
                    entities = json.loads(json_text)
                    if not isinstance(entities, list):
                        entities = [entities]  # 确保是列表格式
                except json.JSONDecodeError:
                    logging.warning(f"JSON解析错误，尝试清理后重新解析")
                    try:
                        # 清理JSON文本，处理常见错误
                        clean_json = json_text.replace('```', '').strip()
                        clean_json = re.sub(r',\s*]', ']', clean_json)  # 移除尾部逗号
                        clean_json = re.sub(r',\s*}', '}', clean_json)  # 移除尾部逗号
                        entities = json.loads(clean_json)
                        if not isinstance(entities, list):
                            entities = [entities]  # 确保是列表格式
                    except Exception as clean_err:
                        logging.error(f"清理后JSON仍解析失败: {str(clean_err)}")
            # 验证结果
            if entities and len(entities) > 0:
                logging.info(f"成功提取 {len(entities)} 个实体")
                return entities, is_complete
            else:
                logging.warning(f"提取结果格式不正确，未能提取有效实体")
                if attempt < max_attempts:
                    logging.info(f"将重试...")
                    time.sleep(2)  # 短暂延迟后重试
        except Exception as e:
            logging.error(f"使用OpenAI API提取实体时出错: {str(e)}")
            import traceback
            logging.error(traceback.format_exc())
            if attempt < max_attempts:
                logging.info(f"将重试...")
                time.sleep(2)
    logging.error(f"在 {max_attempts} 次尝试后仍未能提取实体")
    return [], False

# 更新 extract_entities_with_qwen_agent 函数
def extract_entities_with_qwen_agent(agent, prompt, max_attempts=3, temp_cache_path=None):
    """
    使用千问agent提取实体
    
    Args:
        agent: 千问agent实例
        prompt (str): 提取实体的提示词 
        max_attempts (int): 最大尝试次数
        temp_cache_path (str, optional): 临时缓存文件路径
        
    Returns:
        tuple: (提取的实体列表, 是否处理完成)
    """
    if not agent:
        logging.error("千问agent未初始化")
        return [], False
    
    # 加载临时缓存（如果有）
    partial_results = {}
    if temp_cache_path and os.path.exists(temp_cache_path):
        try:
            with open(temp_cache_path, 'r', encoding='utf-8') as f:
                partial_results = json.load(f)
                logging.info(f"加载了临时缓存数据: {temp_cache_path}")
        except Exception as e:
                logging.error(f"读取临时缓存出错: {str(e)}")
    
    # 确保临时缓存目录存在
    if temp_cache_path:
        temp_dir = os.path.dirname(temp_cache_path)
        if temp_dir and not os.path.exists(temp_dir):
            try:
                os.makedirs(temp_dir, exist_ok=True)
                logging.info(f"创建临时缓存目录: {temp_dir}")
            except Exception as e:
                logging.error(f"创建临时缓存目录失败: {str(e)}")
                # 如果目录创建失败，不使用缓存继续执行
                temp_cache_path = None
    
    # 尝试多次提取
    attempt = 0
    while attempt < max_attempts:
        attempt += 1
        try:
            logging.info(f"尝试提取实体 (尝试 {attempt}/{max_attempts})")
            
            # 调用agent进行提取
            response = agent.chat(prompt)
            logging.debug(f"千问API响应: {response}")
            
            # 保存部分结果到临时缓存
            if temp_cache_path:
                try:
                    # 更新临时缓存
                    partial_results.update({
                        'response': response,
                        'attempt': attempt,
                        'timestamp': datetime.datetime.now().isoformat()
                    })
                    with open(temp_cache_path, 'w', encoding='utf-8') as f:
                        json.dump(partial_results, f, ensure_ascii=False, indent=2)
                    logging.info(f"已保存部分响应到临时缓存: {temp_cache_path}")
                except Exception as e:
                    logging.error(f"保存临时缓存出错: {str(e)}")
                    # 继续执行，不因缓存错误中断主流程
            
            # 检查是否包含完成标志 - 使用公共函数
            is_complete = check_extraction_complete(response)
            
            # 尝试提取JSON部分
            json_text = extract_json_from_text(response)
            entities = []
            
            # 验证并解析JSON
            if json_text:
                try:
                    # 尝试直接解析
                    entities = json.loads(json_text)
                    if not isinstance(entities, list):
                        entities = [entities]  # 确保是列表格式
                except json.JSONDecodeError:
                    logging.warning(f"JSON解析错误，尝试清理后重新解析")
                    try:
                        # 清理JSON文本，处理常见错误
                        clean_json = json_text.replace('```', '').strip()
                        clean_json = re.sub(r',\s*]', ']', clean_json)  # 移除尾部逗号
                        clean_json = re.sub(r',\s*}', '}', clean_json)  # 移除尾部逗号
                        entities = json.loads(clean_json)
                        if not isinstance(entities, list):
                            entities = [entities]  # 确保是列表格式
                    except Exception as clean_err:
                        logging.error(f"清理后JSON仍解析失败: {str(clean_err)}")
            
            # 验证结果
            if entities and len(entities) > 0:
                logging.info(f"成功提取 {len(entities)} 个实体")
                return entities, is_complete
            else:
                logging.warning(f"提取结果格式不正确，未能提取有效实体")
                if attempt < max_attempts:
                    logging.info(f"将重试...")
                    time.sleep(2)  # 短暂延迟后重试
        except Exception as e:
            logging.error(f"使用千问API提取实体时出错: {str(e)}")
            logging.error(traceback.format_exc())
            if attempt < max_attempts:
                logging.info(f"将重试...")
                time.sleep(2)
    
    logging.error(f"在 {max_attempts} 次尝试后仍未能提取实体")
    return [], False

# 创建一个通用的实体提取函数，替代多个API特定的函数
def extract_entities_with_model(prompt, model_name="qwen", max_attempts=3, temp_cache_path=None, agent=None):
    """
    使用指定模型提取实体
    
    Args:
        prompt (str): 提取实体的提示词
        model_name (str): 模型名称 ("qwen", "openai")
        max_attempts (int): 最大尝试次数
        temp_cache_path (str, optional): 临时缓存文件路径
        agent: 可选的agent实例（如果使用agent调用）
        
    Returns:
        tuple: (提取的实体列表, 是否处理完成)
    """
    if model_name.lower() == "qwen" and agent:
        return extract_entities_with_qwen_agent(agent, prompt, max_attempts, temp_cache_path)
    elif model_name.lower() == "openai":
        return extract_entities_with_openai(prompt, Config.OPENAI_MODEL or "gpt-3.5-turbo", max_attempts, temp_cache_path)
    elif model_name.lower() == "qwen":
        # 使用千问API直接调用
        return extract_entities_with_qwen(prompt, max_attempts, temp_cache_path)
    else:
        logging.error(f"不支持的模型: {model_name}")
        return [], False

def setup_qwen_agent(pdf_path=None):
    """
    设置千问助手
    
    Args:
        pdf_path (str, optional): PDF文件路径
        
    Returns:
        Assistant: 千问助手实例或None（如果创建失败）
    """
    if Config.DISABLE_AI:
        logging.warning("AI功能已禁用，不创建Assistant对象")
        return None
        
    llm_cfg = {
        'model': 'qwen-long',
        'model_server': Config.QWEN_BASE_URL,
        'api_key': Config.QWEN_API_KEY,
        'generate_cfg': {
            'top_p': 0.7
        }
    }

    system_instruction = '''你是一名研究人员，负责提取文献中的算法实体及其要素。
    根据算法进化模式[组件，架构，特征，方法，任务，评估]提取本文中的算法实体及其要素。
    在收到用户的请求后，你应该：
    - 分析论文，提取出算法实体及其关键要素
    - 识别出算法之间的演化关系
    - 以JSON格式输出结果
    '''

    tools = ['algorithm_entity_extractor', 'pdf_text_extractor', 'code_interpreter']
    files = [pdf_path] if pdf_path else []

    try:
        # 尝试创建Assistant对象
        assistant = Assistant(llm=llm_cfg,
                     system_message=system_instruction,
                     function_list=tools,
                     files=files)
        # 测试Assistant对象能否正常工作
        logging.info("千问Assistant对象创建成功")
        return assistant
    except Exception as e:
        logging.error(f"创建Assistant对象时出错: {str(e)}")
        # 如果使用千问模型失败，返回None
        return None


def extract_evolution_relations(entities, pdf_path=None, task_id=None):
    """
    提取实体之间的演化关系，支持同时结合PDF文件和实体信息进行分析
    
    Args:
        entities (List[Dict]): 实体列表
        pdf_path (str, optional): PDF文件路径，如果提供则会同时分析文件内容
        task_id (str, optional): 任务ID，用于缓存标识
    
    Returns:
        List[Dict]: 演化关系列表
    """
    try:
        import os
        import json
        import time
        import tempfile
        import logging
        from pathlib import Path
        from app.config import Config
        
        logging.info(f"开始提取演化关系，实体数量: {len(entities)}")
        if pdf_path:
            logging.info(f"同时使用PDF文件进行分析: {pdf_path}")
        
        # 检查缓存
        cache_dir = Path(Config.CACHE_DIR) / "relations"
        cache_dir.mkdir(exist_ok=True, parents=True)
        
        # 使用PDF文件名、任务ID和实体哈希值生成缓存文件名
        if pdf_path:
            basename = os.path.basename(pdf_path)
            filename_without_ext = os.path.splitext(basename)[0]
            cache_key = f"task_{task_id}_{filename_without_ext}_relations" if task_id else f"{filename_without_ext}_relations"
        else:
            # 如果没有PDF文件，只使用实体的哈希值
            entities_hash = hashlib.md5(str(entities).encode()).hexdigest()
            cache_key = f"task_{task_id}_entities_{entities_hash}_relations" if task_id else f"entities_{entities_hash}_relations"
            
        cache_file = cache_dir / f"{cache_key}.json"
        
        # 如果缓存存在且有效，直接返回缓存的结果
        if cache_file.exists() and os.path.getsize(cache_file) > 0:
                try:
                    with open(cache_file, 'r', encoding='utf-8') as f:
                        cached_data = json.load(f)
                    logging.info(f"从缓存加载 {len(cached_data)} 条关系")
                    return cached_data
                except Exception as e:
                    logging.error(f"读取缓存文件时出错: {str(e)}")
        
        # 准备实体描述
        entity_descriptions = []
        for entity in entities:
            # 支持两种格式的实体数据
            if 'entity_id' in entity:
                # 直接格式
                entity_id = entity.get('entity_id', '')
                entity_type = entity.get('entity_type', 'Algorithm')
                name = entity.get('name', entity_id)
                description = entity.get('description', '')
                entity_info = f"实体ID: {entity_id}, 类型: {entity_type}, 名称: {name}, 描述: {description}"
                entity_descriptions.append(entity_info)
            elif 'algorithm_entity' in entity:
                # 嵌套格式
                algo = entity['algorithm_entity']
                entity_id = algo.get('algorithm_id', '')
                name = algo.get('name', entity_id)
                title = algo.get('title', '')
                year = algo.get('year', '')
                authors = ', '.join(algo.get('authors', []))
                task = algo.get('task', '')
                entity_type = algo.get('entity_type', 'Algorithm')
                
                entity_info = f"实体ID: {entity_id}, 类型: {entity_type}, 名称: {name}, 标题: {title}, 年份: {year}, 作者: {authors}, 任务: {task}"
                entity_descriptions.append(entity_info)
            elif 'dataset_entity' in entity:
                # 数据集格式
                dataset = entity['dataset_entity']
                entity_id = dataset.get('dataset_id', '')
                name = dataset.get('name', entity_id)
                description = dataset.get('description', '')
                entity_type = dataset.get('entity_type', 'Dataset')
                
                entity_info = f"实体ID: {entity_id}, 类型: {entity_type}, 名称: {name}, 描述: {description}"
                entity_descriptions.append(entity_info)
            elif 'metric_entity' in entity:
                # 指标格式
                metric = entity['metric_entity']
                entity_id = metric.get('metric_id', '')
                name = metric.get('name', entity_id)
                description = metric.get('description', '')
                entity_type = metric.get('entity_type', 'Metric')
                
                entity_info = f"实体ID: {entity_id}, 类型: {entity_type}, 名称: {name}, 描述: {description}"
                entity_descriptions.append(entity_info)
                
                # 构建提示词
        system_message = "你是一个算法演化关系分析专家，负责判断实体之间的演化关系，包括算法、数据集和评估指标。请尽可能详细分析关系并避免输出过程中的截断。"
        
        user_message = f"""分析这篇论文与以下{str(len(entity_descriptions))}个已知实体之间的演化关系。

我们将演化关系分为五种类型：

1. 改进（Improve）：指算法设计或结构上的创新，升级整体架构或机制，以实现新功能或提高适应性。
   表达方式：A improves B, A refines B, A enhances B, A boosts B, A advances B, A increases B, A strengthens B, A enriches B, A elevates B

2. 优化（Optimize）：通过参数调整、训练策略或配置细节来提高效率、精度或收敛速度，而不改变核心结构。优化强调资源效率、降低计算成本和提高性能。
   表达方式：A optimizes B, A fine-tunes B, A accelerates B, A minimizes resource usage in B, A reduces complexity in B, A streamlines B, A stabilizes B, A lowers computational cost of B

3. 扩展（Extend）：指在现有算法中添加新模块、功能或组件，扩展其范围或适应性，而不改变其核心逻辑。这通常用于处理更多样化的数据类型、支持复杂任务或拓宽应用场景。
   表达方式：A extends B, A expands B, A builds on B, A enables B, A incorporates B, A generalizes B, A adapts B, A broadens B, A introduces new capabilities to B

4. 替换（Replace）：涉及用更新的组件或方法替代某些组件或方法，通常是为了提高性能或解决局限性。替换在局部层面进行，但可能会对性能产生重大影响，通常使用经过验证的更优方法来增强原始算法。
   表达方式：A replaces B, A supersedes B, A substitutes B, A displaces B, A modifies B by replacing components, A changes core mechanisms in B, A swaps foundational techniques in B, A reconfigures the structure of B

5. 使用（Use）：表示一个实体使用另一个实体作为工具或资源。例如，算法使用特定的数据集进行训练或评估，或使用特定的指标进行性能评估，算法不能使用算法。
   表达方式：A uses B, A employs B, A utilizes B, A applies B, A leverages B, A adopts B, A implements B

重要说明：请注意实体类型可以是算法（Algorithm）、数据集（Dataset）或评估指标（Metric）。实体之间的演化关系可以是：
- 算法改进/优化/扩展/替换另一个算法
- 算法使用特定数据集
- 算法使用特定评估指标
- 数据集改进/扩展另一个数据集
- 评估指标改进/扩展另一个评估指标

请为每个实体判断是否与论文中的其他实体存在演化关系，并以JSON格式返回:
                ```json
                [
  {{
    "from_entities": [
      {{
        "entity_id": "源实体ID",
        "entity_type": "Algorithm/Dataset/Metric"
      }}
    ],
    "to_entities": [
      {{
        "entity_id": "目标实体ID",
        "entity_type": "Algorithm/Dataset/Metric"
      }}
    ],
    "relation_type": "Improve/Optimize/Extend/Replace/Use中的一种",
    "structure": "结构变化类别(Architecture.Component, Methodology.Training, Evaluation.Metric, Evaluation.Dataset等)",
    "detail": "详细说明关系内容",
    "evidence": "文中的具体证据",
                          "confidence": 0.95
  }},
  ...
]
```

最后，请明确告知我提取是否已完成，还是需要继续提取更多关系。在JSON返回后，请单独一行写明"EXTRACTION_COMPLETE: true/false"。

已知实体信息如下:
{chr(10).join(entity_descriptions)}"""
        
        file_id = None
        response = None
        
        try:
            # 统一使用千问API进行处理，同时支持PDF文件和实体列表
            from openai import OpenAI
            client = OpenAI(
                api_key=Config.QWEN_API_KEY,
                base_url=Config.QWEN_BASE_URL
            )
            # 构建消息列表
            messages = [
                {
                    'role': 'system',
                    'content': system_message
                }
            ]
            # 如果提供了PDF文件，上传并添加到消息中
            if pdf_path and os.path.exists(pdf_path):
                logging.info("上传PDF文件进行分析...")
                file = client.files.create(file=Path(pdf_path), purpose="file-extract")
                file_id = file.id
                logging.info(f"文件上传成功，file_id: {file_id}")
                # 添加文件引用
                messages.append({
                    'role': 'system',
                    'content': f'fileid://{file_id}'
                })
            # 添加用户消息，包含实体描述
            messages.append({
                'role': 'user',
                'content': user_message
            })
            # 调用千问API
            logging.info("调用千问API进行关系提取...")
            start_time = time.time()
            completion = client.chat.completions.create(
                model=Config.QWEN_MODEL or "qwen-long",
                messages=messages,
                temperature=0.2,
                stream=True,
                max_tokens=None  # 不限制token数量
            )
            # 收集流式响应
            full_response = ""
            for chunk in completion:
                if chunk.choices and chunk.choices[0].delta.content:
                    full_response += chunk.choices[0].delta.content
                # 每50个块记录一次
                if len(full_response) % 50 == 0:
                    logging.info(f"收到响应块 #{len(full_response)//50}，当前响应长度: {len(full_response)}")
            elapsed_time = time.time() - start_time
            logging.info(f"API调用完成，耗时: {elapsed_time:.2f}秒，响应长度: {len(full_response)}字符")
            # 检查是否包含完成标志
            is_complete = "EXTRACTION_COMPLETE: true" in full_response.lower()
            logging.info(f"提取完成状态: {'完成' if is_complete else '未完成'}")
            # 解析响应内容
            response = full_response
            # 解析响应
            if response:
                # 使用改进的JSON提取函数提取JSON部分
                json_str = extract_json_from_text(response)
                if json_str:
                    try:
                        relations = json.loads(json_str)
                        logging.info(f"成功提取 {len(relations)} 个演化关系")
                        # 验证关系格式
                        valid_relations = []
                        for relation in relations:
                            # 验证基本结构
                            if not isinstance(relation, dict):
                                continue
                            if 'from_entities' not in relation or 'to_entities' not in relation:
                                continue
                            if not isinstance(relation['from_entities'], list) or not isinstance(relation['to_entities'], list):
                                continue
                            if len(relation['from_entities']) == 0 or len(relation['to_entities']) == 0:
                                continue
                            # 验证实体ID
                            valid = True
                            for from_entity in relation['from_entities']:
                                if not isinstance(from_entity, dict) or 'entity_id' not in from_entity:
                                    valid = False
                                    break
                            for to_entity in relation['to_entities']:
                                if not isinstance(to_entity, dict) or 'entity_id' not in to_entity:
                                    valid = False
                                    break
                            if valid:
                                # 将每个from_entity和to_entity组合都创建一个单独的关系（扁平化）
                                for from_entity in relation['from_entities']:
                                    for to_entity in relation['to_entities']:
                                        # 创建数据库格式的关系对象
                                        db_relation = {
                                            "from_entity": from_entity["entity_id"],
                                            "to_entity": to_entity["entity_id"],
                                            "relation_type": relation.get("relation_type", ""),
                                            "structure": relation.get("structure", ""),
                                            "detail": relation.get("detail", ""),
                                            "evidence": relation.get("evidence", ""),
                                            "confidence": relation.get("confidence", 0.0),
                                            "from_entity_type": from_entity.get("entity_type", "Algorithm"),
                                            "to_entity_type": to_entity.get("entity_type", "Algorithm"),
                                            "extraction_complete": is_complete
                                        }
                                        valid_relations.append(db_relation)
                        # 缓存结果
                        with open(cache_file, 'w', encoding='utf-8') as f:
                            json.dump(valid_relations, f, indent=2, ensure_ascii=False)
                        return valid_relations
                    except json.JSONDecodeError as e:
                        logging.error(f"解析JSON时出错: {str(e)}")
                        # 尝试清理并重新解析
                        try:
                            # 手动处理逗号问题
                            cleaned_json = json_str.replace("},\n]", "}]").replace("},\n  ]", "}]")
                            relations = json.loads(cleaned_json)
                            logging.info(f"清理后成功解析JSON，共有 {len(relations)} 个关系")
                            # 验证和扁平化关系
                            valid_relations = []
                            for relation in relations:
                                if 'from_entities' in relation and 'to_entities' in relation:
                                    if isinstance(relation['from_entities'], list) and isinstance(relation['to_entities'], list):
                                        if len(relation['from_entities']) > 0 and len(relation['to_entities']) > 0:
                                            for from_entity in relation['from_entities']:
                                                for to_entity in relation['to_entities']:
                                                    if isinstance(from_entity, dict) and 'entity_id' in from_entity and isinstance(to_entity, dict) and 'entity_id' in to_entity:
                                                        db_relation = {
                                                            "from_entity": from_entity["entity_id"],
                                                            "to_entity": to_entity["entity_id"],
                                                            "relation_type": relation.get("relation_type", ""),
                                                            "structure": relation.get("structure", ""),
                                                            "detail": relation.get("detail", ""),
                                                            "evidence": relation.get("evidence", ""),
                                                            "confidence": relation.get("confidence", 0.0),
                                                            "from_entity_type": from_entity.get("entity_type", "Algorithm"),
                                                            "to_entity_type": to_entity.get("entity_type", "Algorithm"),
                                                            "extraction_complete": is_complete
                                                        }
                                                        valid_relations.append(db_relation)
                            # 缓存结果
                            with open(cache_file, 'w', encoding='utf-8') as f:
                                json.dump(valid_relations, f, indent=2, ensure_ascii=False)
                            return valid_relations
                        except Exception as e2:
                            logging.error(f"清理后仍无法解析JSON: {str(e2)}")
                else:
                    logging.error("未能提取到有效的JSON内容")
            else:
                logging.error("API返回内容为空")
        except Exception as e:
            logging.error(f"调用API时出错: {str(e)}")
            logging.error(traceback.format_exc())
        return []
            
    except Exception as e:
        logging.error(f"提取演化关系时出错: {str(e)}")
        logging.error(traceback.format_exc())
        return [] 

def extract_entities_with_qwen(prompt, max_attempts=3, temp_cache_path=None):
    """
    使用千问API提取实体
    
    Args:
        prompt (str): 提取实体的提示词
        max_attempts (int): 最大尝试次数
        temp_cache_path (str, optional): 临时缓存文件路径
        
    Returns:
        tuple: (提取的实体列表, 是否处理完成)
    """
    if not hasattr(Config, 'QWEN_API_KEY') or not Config.QWEN_API_KEY:
        logging.error("未配置千问API密钥")
        return [], False
    
    # 加载临时缓存（如果有）
    partial_results = {}
    if temp_cache_path and os.path.exists(temp_cache_path):
        try:
            with open(temp_cache_path, 'r', encoding='utf-8') as f:
                partial_results = json.load(f)
                logging.info(f"加载了临时缓存数据: {temp_cache_path}")
        except Exception as e:
            logging.error(f"读取临时缓存出错: {str(e)}")
    
    # 确保临时缓存目录存在
    if temp_cache_path:
        temp_dir = os.path.dirname(temp_cache_path)
        if temp_dir and not os.path.exists(temp_dir):
            try:
                os.makedirs(temp_dir, exist_ok=True)
                logging.info(f"创建临时缓存目录: {temp_dir}")
            except Exception as e:
                logging.error(f"创建临时缓存目录失败: {str(e)}")
                # 如果目录创建失败，不使用缓存继续执行
                temp_cache_path = None
    
    # 尝试多次提取
    attempt = 0
    while attempt < max_attempts:
        attempt += 1
        try:
            logging.info(f"尝试使用千问API提取实体 (尝试 {attempt}/{max_attempts})")
            
            # 创建OpenAI客户端（千问API兼容OpenAI格式）
            client = OpenAI(
                api_key=Config.QWEN_API_KEY,
                base_url=Config.QWEN_BASE_URL
            )
            
            # 构建消息
            messages = [
                {"role": "system", "content": "你是一个专注于从学术论文中提取实体信息的AI助手。请从提供的文本中提取算法、数据集和评价指标等相关实体信息，并以JSON格式返回。"},
                {"role": "user", "content": prompt}
            ]
            
            # 调用API
            response = client.chat.completions.create(
                model=Config.QWEN_MODEL or "qwen-long",
                messages=messages,
                temperature=0.2,
                max_tokens=None,  # 不限制token数量
                stream=True
            )
            
            # 收集流式响应内容
            content = ""
            for chunk in response:
                if chunk.choices and chunk.choices[0].delta.content:
                    content += chunk.choices[0].delta.content
            
            # 保存部分结果到临时缓存
            if temp_cache_path:
                try:
                    # 更新临时缓存
                    partial_results.update({
                        'response': content,
                        'attempt': attempt,
                        'timestamp': datetime.datetime.now().isoformat()
                    })
                    with open(temp_cache_path, 'w', encoding='utf-8') as f:
                        json.dump(partial_results, f, ensure_ascii=False, indent=2)
                    logging.info(f"已保存部分响应到临时缓存: {temp_cache_path}")
                except Exception as e:
                    logging.error(f"保存临时缓存出错: {str(e)}")
                    # 继续执行，不因缓存错误中断主流程
            
            # 检查是否包含完成标志 - 使用公共函数
            is_complete = check_extraction_complete(content)
            
            # 尝试提取JSON部分
            json_text = extract_json_from_text(content)
            entities = []
            
            # 验证并解析JSON
            if json_text:
                try:
                    # 尝试直接解析
                    entities = json.loads(json_text)
                    if not isinstance(entities, list):
                        entities = [entities]  # 确保是列表格式
                except json.JSONDecodeError:
                    logging.warning(f"JSON解析错误，尝试清理后重新解析")
                    try:
                        # 清理JSON文本，处理常见错误
                        clean_json = json_text.replace('```', '').strip()
                        clean_json = re.sub(r',\s*]', ']', clean_json)  # 移除尾部逗号
                        clean_json = re.sub(r',\s*}', '}', clean_json)  # 移除尾部逗号
                        entities = json.loads(clean_json)
                        if not isinstance(entities, list):
                            entities = [entities]  # 确保是列表格式
                    except Exception as clean_err:
                        logging.error(f"清理后JSON仍解析失败: {str(clean_err)}")
            
            # 验证结果
            if entities and len(entities) > 0:
                logging.info(f"成功提取 {len(entities)} 个实体")
                return entities, is_complete
            else:
                logging.warning(f"提取结果格式不正确，未能提取有效实体")
                if attempt < max_attempts:
                    logging.info(f"将重试...")
                    time.sleep(2)  # 短暂延迟后重试
        except Exception as e:
            logging.error(f"使用千问API提取实体时出错: {str(e)}")
            logging.error(traceback.format_exc())
            if attempt < max_attempts:
                logging.info(f"将重试...")
                time.sleep(2)
    
    logging.error(f"在 {max_attempts} 次尝试后仍未能提取实体")
    return [], False


def extract_paper_entities(pdf_paths, max_attempts=3, batch_size=20, force_reprocess=False, model_name="qwen", task_id=None):
    """
    从论文列表中提取所有实体
    
    Args:
        pdf_paths (List[str]): PDF文件路径列表
        max_attempts (int): 最大尝试次数
        batch_size (int): 批处理大小
        force_reprocess (bool): 是否强制重新处理
        model_name (str): 使用的模型名称
        task_id (str, optional): 任务ID
        
    Returns:
        List[Dict]: 提取的实体列表
    """
    if not pdf_paths:
        logging.warning("没有提供PDF文件路径")
        return []
    
    if isinstance(pdf_paths, str):
        pdf_paths = [pdf_paths]
    
    # 创建临时目录
    os.makedirs(Config.TEMP_FOLDER, exist_ok=True)
    
    # 验证有效的PDF路径
    valid_pdf_paths = []
    for path in pdf_paths:
        # 标准化路径分隔符
        path = os.path.normpath(path)
        if os.path.exists(path) and path.lower().endswith('.pdf'):
            valid_pdf_paths.append(path)
        else:
            logging.warning(f"无效的PDF文件路径: {path}")
    
    if not valid_pdf_paths:
        logging.warning("没有有效的PDF文件路径")
        return []
    
    # 分批处理PDF文件，避免处理太多文件
    total_files = len(valid_pdf_paths)
    processed_count = 0
    all_complete = True
    extracted_entities = []
    
    for i in range(0, len(valid_pdf_paths), batch_size):
        batch_paths = valid_pdf_paths[i:i+batch_size]
        logging.info(f"处理PDF批次 {i//batch_size + 1}/{(len(valid_pdf_paths)+batch_size-1)//batch_size}, 共 {len(batch_paths)} 个文件")
        
        for pdf_path in batch_paths:
            try:
                # 提取PDF文本内容
                logging.info(f"正在处理PDF: {pdf_path}")
                
                # 创建缓存目录
                cache_dir = os.path.join(Config.CACHE_DIR, "entities")
                os.makedirs(cache_dir, exist_ok=True)
                
                # 生成缓存文件名
                basename = os.path.basename(pdf_path)
                filename_without_ext = os.path.splitext(basename)[0]
                
                # 移除可能的任务ID前缀
                if task_id and filename_without_ext.startswith(f"task_{task_id}_"):
                    filename_without_ext = filename_without_ext[len(f"task_{task_id}_"):]
                
                # 移除日期时间前缀格式 (如: 20250520_164839_)
                date_time_prefix_pattern = r"^\d{8}_\d{6}_"
                import re
                if re.match(date_time_prefix_pattern, filename_without_ext):
                    logging.info(f"检测到日期时间前缀: {filename_without_ext}")
                    filename_without_ext = re.sub(date_time_prefix_pattern, "", filename_without_ext)
                    logging.info(f"移除前缀后的文件名: {filename_without_ext}")
                
                cache_key = f"{filename_without_ext}_entities.json"
                cache_path = os.path.join(cache_dir, cache_key)
                
                # 生成临时缓存文件路径
                temp_cache_dir = os.path.join(Config.TEMP_FOLDER, "entities")
                os.makedirs(temp_cache_dir, exist_ok=True)
                temp_cache_path = os.path.join(temp_cache_dir, f"{filename_without_ext}_partial.json")
                
                # 检查是否有缓存且不需要强制重新处理
                previous_entities = []
                if not force_reprocess and os.path.exists(cache_path):
                    try:
                        with open(cache_path, 'r', encoding='utf-8') as f:
                            previous_entities = json.load(f)
                            logging.info(f"从缓存加载实体数据: {cache_path}")
                        
                        if previous_entities:
                            extracted_entities.extend(previous_entities)
                            processed_count += 1
                            logging.info(f"已从缓存加载 {len(previous_entities)} 个实体")
                            
                            # 更新进度
                            progress = (1.0 * processed_count / total_files) * 100
                            if task_id:
                                db_manager.update_processing_status(
                                    task_id=task_id,
                                    current_stage=f'处理PDF文件 ({processed_count}/{total_files})',
                                    progress=progress,
                                    current_file=basename,
                                    message=f"从缓存加载 {basename} 数据"
                                )
                                
                            continue  # 跳过处理这个文件
                    except Exception as e:
                        logging.error(f"读取缓存文件时出错: {str(e)}")

                # 创建临时文件副本
                temp_file = os.path.join(Config.TEMP_FOLDER, basename)
                try:
                    # 复制文件以避免可能的权限问题
                    shutil.copy2(pdf_path, temp_file)
                    logging.info(f"已创建临时文件副本: {temp_file}")
                except Exception as e:
                    logging.error(f"创建临时文件副本时出错: {str(e)}")
                    # 直接使用原始文件
                    temp_file = pdf_path
                    
                # 提取论文中的实体
                try:
                    logging.info(f"从PDF提取实体: {basename}")
                    current_attempt = 0
                    
                    # 尝试多次提取，以处理可能的API故障
                    while current_attempt < max_attempts:
                        try:
                            current_attempt += 1
                            logging.info(f"第 {current_attempt} 次尝试提取实体...")
                            
                            # 提取PDF的文本内容
                            paper_text = extract_text_from_pdf(temp_file, task_id=task_id)
                            
                            if not paper_text or len(paper_text.strip()) < 100:
                                logging.error("无法从PDF提取足够的文本来分析关系")
                                return []
                            
                            # 生成提取实体的提示
                            prompt = generate_entity_extraction_prompt(paper_text, model_name, previous_entities, 
                                                                       partial_extraction=bool(previous_entities))
                            
                            # 调用统一的实体提取函数
                            entities = []
                            is_extraction_complete = False
                            
                            # 使用统一的实体提取模型
                            agent = None
                            if model_name.lower() == "qwen":
                                agent = setup_qwen_agent(pdf_path)
                            
                            entities, is_extraction_complete = extract_entities_with_model(
                                prompt, 
                                model_name, 
                                max_attempts,
                                temp_cache_path,
                                agent
                            )
                            
                            # 如果没有提取成功，设置完成标志为False
                            if not entities:
                                all_complete = False
                                continue
                            
                            # 合并或更新结果
                            final_entities = []
                            
                            # 如果有之前的提取结果，需要进行合并和更新
                            if previous_entities:
                                # 创建ID到索引的映射
                                id_to_index = {}
                                for idx, entity in enumerate(previous_entities):
                                    entity_id = None
                                    if 'algorithm_entity' in entity:
                                        entity_id = entity['algorithm_entity'].get('algorithm_id')
                                    elif 'dataset_entity' in entity:
                                        entity_id = entity['dataset_entity'].get('dataset_id')
                                    elif 'metric_entity' in entity:
                                        entity_id = entity['metric_entity'].get('metric_id')
                                    
                                    if entity_id:
                                        id_to_index[entity_id] = idx
                                
                                # 合并结果，更新已存在的实体，添加新实体
                                final_entities = previous_entities.copy()
                                
                                # 遍历新提取的实体
                                for new_entity in entities:
                                    # 为新实体添加完整性标记
                                    new_entity['is_complete'] = is_extraction_complete
                                    
                                    # 根据实体类型获取ID
                                    entity_id = None
                                    entity_type = None
                                    
                                    if 'algorithm_entity' in new_entity:
                                        entity_id = new_entity['algorithm_entity'].get('algorithm_id')
                                        entity_type = 'algorithm_entity'
                                    elif 'dataset_entity' in new_entity:
                                        entity_id = new_entity['dataset_entity'].get('dataset_id')
                                        entity_type = 'dataset_entity'
                                    elif 'metric_entity' in new_entity:
                                        entity_id = new_entity['metric_entity'].get('metric_id')
                                        entity_type = 'metric_entity'
                                    
                                    # 如果ID在现有结果中，更新
                                    if entity_id and entity_id in id_to_index:
                                        idx = id_to_index[entity_id]
                                        # 更新实体
                                        if entity_type:
                                            final_entities[idx][entity_type] = new_entity[entity_type]
                                            # 更新完整性标记
                                            final_entities[idx]['is_complete'] = is_extraction_complete
                                        else:
                                        # 添加新实体
                                            final_entities.append(new_entity)
                            else:
                                # 没有之前的提取结果，直接使用新提取的结果
                                for entity in entities:
                                    entity['is_complete'] = is_extraction_complete
                                final_entities = entities
                            
                            # 更新全局完成状态
                            if not is_extraction_complete:
                                all_complete = False
                            
                            # 缓存提取的实体
                            with open(cache_path, 'w', encoding='utf-8') as f:
                                json.dump(final_entities, f, ensure_ascii=False, indent=2)
                            
                            logging.info(f"提取结果: {len(final_entities)} 个实体, 完成状态: {'完成' if is_extraction_complete else '未完成'}")
                            extracted_entities.extend(final_entities)
                            break  # 成功提取，退出循环
                        
                        except Exception as e:
                            logging.error(f"处理 {pdf_path} 时出错: {str(e)}")
                            import traceback
                            logging.error(traceback.format_exc())
                            all_complete = False
                            break
                
                except Exception as e:
                    logging.error(f"从PDF提取实体时出错: {str(e)}")
                    import traceback
                    logging.error(traceback.format_exc())
                    all_complete = False
                    break
            
            except Exception as e:
                        logging.error(f"处理 {pdf_path} 时出错: {str(e)}")
                        import traceback
                        logging.error(traceback.format_exc())
                        all_complete = False
    
    return extracted_entities, all_complete
