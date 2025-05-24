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
import tempfile

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# 添加检查提取完成状态的公共函数
def check_extraction_complete(text):
    """
    检查API响应中是否包含完成标志
    
    Args:
        text (str): API响应文本
        
    Returns:
        bool: 是否已完成提取
    """
    # 寻找完成标志
    completion_patterns = [
        r'EXTRACTION_COMPLETE:\s*true',
        r'extraction_complete"?\s*:\s*true',
        r'{"extraction_complete"?\s*:\s*true}',
        r'提取完成'
    ]
    
    for pattern in completion_patterns:
        if re.search(pattern, text, re.IGNORECASE):
            return True
            
    # 检查是否明确未完成
    incomplete_patterns = [
        r'EXTRACTION_COMPLETE:\s*false',
        r'extraction_complete"?\s*:\s*false',
        r'{"extraction_complete"?\s*:\s*false}',
        r'需要继续提取',
        r'提取未完成'
    ]
    
    for pattern in incomplete_patterns:
        if re.search(pattern, text, re.IGNORECASE):
            return False
            
    # 默认为完成
    return True


# # 添加新的工具类用于提取PDF文本
# @register_tool('pdf_text_extractor')
# class PdfTextExtractor(BaseTool):
#     description = 'PDF文本提取工具，输入PDF文件路径，返回提取的文本内容。'
#     parameters = [{
#         'name': 'pdf_path',
#         'type': 'string',
#         'description': 'PDF文件的路径',
#         'required': True
#     }]

#     def call(self, params: str, **kwargs) -> str:
#         import json
#         import urllib.parse
#         import os
#         from pathlib import Path
#         from app.config import Config
        
#         # 解析参数
#         pdf_path = json.loads(params)['pdf_path']
        
#         # 生成缓存文件路径
#         cache_dir = os.path.join(Config.CACHE_DIR, "pdf_text")
#         os.makedirs(cache_dir, exist_ok=True)
        
#         filename = Path(pdf_path).stem
#         cache_file = os.path.join(cache_dir, f"{filename}_partial.json")
        
#         # 返回文本内容，模型会处理提取逻辑
#         return json.dumps([pdf_path, cache_file], ensure_ascii=False)

# 修改提取文本函数，添加缓存功能
# def extract_text_from_pdf(pdf_path, task_id=None):
#     """
#     从PDF文件中提取文本内容，并缓存结果
    
#     Args:
#         pdf_path (str): PDF文件的路径
#         task_id (str, optional): 任务ID，用于缓存标识
        
#     Returns:
#         str: 提取的文本内容
#     """
#     try:
#         # 生成缓存文件路径
#         from app.config import Config
#         import os
#         from pathlib import Path
        
#         # 标准化路径分隔符，确保跨平台一致性
#         pdf_path = os.path.normpath(pdf_path)
        
#         cache_dir = os.path.join(Config.CACHE_DIR, "pdf_text")
#         os.makedirs(cache_dir, exist_ok=True)
        
#         # 使用原始文件名作为标识，确保缓存文件与上传文件名完全一致
#         basename = os.path.basename(pdf_path)
#         filename_without_ext = os.path.splitext(basename)[0]
        
#         # 移除可能的任务ID前缀
#         if task_id and filename_without_ext.startswith(f"task_{task_id}_"):
#             filename_without_ext = filename_without_ext[len(f"task_{task_id}_"):]
        
#         # 移除日期时间前缀格式 (如: 20250520_164839_)
#         date_time_prefix_pattern = r"^\d{8}_\d{6}_"
#         import re
#         if re.match(date_time_prefix_pattern, filename_without_ext):
#             logging.info(f"检测到日期时间前缀: {filename_without_ext}")
#             filename_without_ext = re.sub(date_time_prefix_pattern, "", filename_without_ext)
#             logging.info(f"移除前缀后的文件名: {filename_without_ext}")
        
#         # 只使用JSON格式缓存文件
#         cache_path = os.path.join(cache_dir, f"{filename_without_ext}_partial.json")
        
#         logging.info(f"处理PDF文件文本: {pdf_path}, 缓存文件: {cache_path}")
        
#         # 检查缓存文件是否存在
#         extracted_text = ""
#         current_attempt = 0
        
#         if os.path.exists(cache_path) and os.path.getsize(cache_path) > 0:
#             logging.info(f"从缓存加载: {cache_path}")
#             try:
#                 with open(cache_path, 'r', encoding='utf-8') as f:
#                     cache_data = json.load(f)
#                 # 检查是否完成
#                 is_complete = cache_data.get('complete', False)
#                 text = cache_data.get('text', '')
#                 if is_complete and text.strip():
#                     logging.info(f"成功从缓存加载已完成的文本，长度: {len(text)} 字符")
#                     return text
#                 elif text.strip():
#                     logging.info(f"从缓存加载部分提取的文本，长度: {len(text)} 字符，将继续提取")
#                     extracted_text = text
#                     # 将尝试次数重置为1
#                     current_attempt = 1
#                 else:
#                     logging.warning(f"缓存文件中没有有效文本内容，将重新提取")
#             except Exception as e:
#                 logging.error(f"读取缓存文件出错: {str(e)}")
#                 extracted_text = ""
#                 current_attempt = 0
#         else:
#             # 如果缓存不存在，初始化变量
#             extracted_text = ""
#             current_attempt = 0
        
#         # 尝试在缓存目录中查找可能匹配的文件
#         if not extracted_text:
#             matching_files = []
#             normalized_name = re.sub(r'[^a-zA-Z0-9]', '', filename_without_ext.lower())
#             for file in os.listdir(cache_dir):
#                 if file.endswith("_partial.json"):
#                     file_base = file[:-13]  # 去掉 "_partial.json" 后缀
#                     normalized_file = re.sub(r'[^a-zA-Z0-9]', '', file_base.lower())
#                     # 如果规范化后的名称匹配，或者包含关系，则认为可能匹配
#                     if normalized_file == normalized_name or normalized_file in normalized_name or normalized_name in normalized_file:
#                         matching_files.append(os.path.join(cache_dir, file))
#             # 如果找到可能匹配的缓存文件，尝试使用第一个
#             if matching_files:
#                 best_match = matching_files[0]
#                 logging.info(f"找到可能匹配的缓存文件: {best_match}")
#                 try:
#                     with open(best_match, 'r', encoding='utf-8') as f:
#                         match_data = json.load(f)
#                         match_text = match_data.get('text', '')
#                         match_complete = match_data.get('complete', False)
#                         if match_text.strip():
#                             logging.info(f"成功从匹配的缓存文件加载文本，长度: {len(match_text)} 字符，完成状态: {match_complete}")
#                             # 将内容复制到标准缓存路径以便下次直接使用
#                             with open(cache_path, 'w', encoding='utf-8') as out_f:
#                                 json.dump({
#                                     'text': match_text,
#                                     'complete': match_complete,
#                                     'attempt': 0,
#                                     'timestamp': datetime.datetime.now().isoformat()
#                                 }, out_f, ensure_ascii=False)
#                             if match_complete:
#                                 return match_text
#                             else:
#                                 extracted_text = match_text
#                                 current_attempt = 1
#                 except Exception as e:
#                     logging.error(f"读取匹配的缓存文件时出错: {str(e)}")
#         # 设置最大尝试次数
#         max_attempts = 10
#         is_extraction_complete = False
        
#         # 如果未完成，则继续提取
#         while current_attempt < max_attempts and not is_extraction_complete:
#             current_attempt += 1
#             try:
#                 if hasattr(Config, 'QWEN_API_KEY') and Config.QWEN_API_KEY:
#                     logging.info(f"尝试使用千问API提取文本 (尝试 {current_attempt}/{max_attempts})")
#                     # 使用千问API提取文本
#                     from openai import OpenAI
                    
#                     client = OpenAI(
#                         api_key=Config.QWEN_API_KEY,
#                         base_url=Config.QWEN_BASE_URL
#                     )
                    
#                     # 上传文件进行处理
#                     file = client.files.create(file=Path(pdf_path), purpose="file-extract")
#                     file_id = file.id
#                     logging.info(f"文件上传成功，file_id: {file_id}")
                    
#                     # 构建提示词
#                     user_prompt = '请将PDF文件中的所有文本内容提取出来，原格式输出，不要添加任何注释或额外信息。要尽可能保留文本的完整性，包括所有页面内容、表格内容和参考文献。用原始段落格式输出，避免因过长而截断。'
#                     # 如果有之前的提取结果，添加到提示中
#                     if extracted_text:
#                         user_prompt = f'''我已经提取了部分文本内容如下:
                        
#                         {extracted_text}...
                        
#                         请继续提取剩余的内容，确保不要重复已经提取的部分，只返回新内容。将新内容与之前内容无缝连接，不要有重复或遗漏。提取时要尽可能保留原始格式，包括所有页面内容、表格内容和参考文献，避免因过长而截断。请完成整个文档的提取。

#                         在提取完成后，请加上一行单独的"EXTRACTION_COMPLETE: true"来表示你已完成提取整个文档。如果还有更多内容需要提取，则加上"EXTRACTION_COMPLETE: false"。'''
#                     else:
#                         user_prompt += '\n\n在提取完成后，请加上一行单独的"EXTRACTION_COMPLETE: true"来表示你已完成提取整个文档。如果还有更多内容需要提取，则加上"EXTRACTION_COMPLETE: false"。'
                
#                     # 构建消息
#                     messages = [
#                         {
#                             'role': 'system',
#                             'content': f'fileid://{file_id}'
#                         },
#                         {
#                             'role': 'user',
#                             'content': user_prompt
#                         }
#                     ]
                    
#                     # 调用API，确保大模型可以返回最大的内容
#                     completion = client.chat.completions.create(
#                         model=Config.QWEN_MODEL or "qwen-long",
#                         messages=messages,
#                         temperature=0.0,
#                         stream=True,
#                         max_tokens=None  # 不限制token数量
#                     )
                    
#                     # 收集流式响应内容
#                     new_text = ""
#                     chunk_count = 0
                    
#                     for chunk in completion:
#                         chunk_count += 1
#                         if chunk.choices and chunk.choices[0].delta.content:
#                             new_text += chunk.choices[0].delta.content
                        
#                         # 每100个块记录一次进度，避免日志过多
#                         if chunk_count % 100 == 0:
#                             logging.info(f"已收到 {chunk_count} 个响应块，当前文本长度: {len(new_text)} 字符")
                    
#                     logging.info(f"API提取完成，总共收到 {chunk_count} 个响应块，新文本长度: {len(new_text)} 字符")
                    
#                     # 检查是否包含完成标记
#                     is_extraction_complete = "EXTRACTION_COMPLETE: true" in new_text.lower()
                    
#                     # 从新文本中移除完成标记
#                     new_text = new_text.replace("EXTRACTION_COMPLETE: true", "").replace("EXTRACTION_COMPLETE: false", "")
                    
#                     # 合并文本内容
#                     if not extracted_text:
#                         extracted_text = new_text
#                     else:
#                         # 查找最后一个完整段落的边界，避免不完整内容合并
#                         last_paragraph_end = extracted_text.rfind("\n\n")
#                         if last_paragraph_end > 0:
#                             # 保留最后一段作为重叠检查
#                             last_paragraph = extracted_text[last_paragraph_end:]
#                             base_text = extracted_text[:last_paragraph_end]
#                             # 检查新文本是否包含上一段的内容，避免重复
#                             overlap_start = new_text.find(last_paragraph[:min(100, len(last_paragraph))])
#                             if overlap_start > 0:
#                                 # 如果找到重叠，只保留新内容
#                                 new_content = new_text[overlap_start + len(last_paragraph):]
#                                 extracted_text = base_text + last_paragraph + new_content
#                                 logging.info(f"检测到文本重叠，已去除重复内容。合并后长度: {len(extracted_text)}")
#                             else:
#                                 # 如果没找到明确重叠，直接追加
#                                 extracted_text += "\n\n" + new_text
#                                 logging.info(f"未检测到明确重叠，直接追加新内容。合并后长度: {len(extracted_text)}")
#                         else:
#                             # 如果没有明确的段落边界，简单追加
#                             extracted_text += "\n\n" + new_text
#                             logging.info(f"未找到段落边界，直接追加新内容。合并后长度: {len(extracted_text)}")
                    
#                     # 保存到缓存文件
#                     with open(cache_path, 'w', encoding='utf-8') as f:
#                         json.dump({
#                             'text': extracted_text,
#                             'complete': is_extraction_complete,
#                             'attempt': current_attempt,
#                             'timestamp': datetime.datetime.now().isoformat()
#                         }, f, ensure_ascii=False)
                    
#                     logging.info(f"已保存提取结果到缓存，当前文本长度: {len(extracted_text)} 字符，是否完成: {is_extraction_complete}")
                    
#                     # 如果已完成，直接返回
#                     if is_extraction_complete:
#                         return extracted_text
#             except Exception as e:
#                 logging.error(f"使用千问API提取文本时出错: {str(e)}")
#                 logging.error(traceback.format_exc())
#                 logging.info("将使用备用方法提取文本")
        
#         # 如果提取成功，但未完成，仍然返回部分结果
#         if extracted_text:
#             logging.info(f"返回部分提取结果，长度: {len(extracted_text)} 字符")
#             return extracted_text
        
#         # 备用方法：使用PyPDF2提取
#         logging.info(f"使用PyPDF2从PDF提取文本: {pdf_path}")
#         text = ""
#         try:
#             with open(pdf_path, 'rb') as file:
#                 reader = PyPDF2.PdfReader(file)
#                 num_pages = len(reader.pages)
#                 logging.info(f"开始从PDF提取文本: {pdf_path}, 共 {num_pages} 页")
#                 # 提取每一页的文本
#                 for page_num in range(num_pages):
#                     try:
#                         page = reader.pages[page_num]
#                         page_text = page.extract_text()
#                         if page_text:
#                             text += page_text + "\n\n"
#                         # 每10页记录一次进度
#                         if (page_num + 1) % 10 == 0 or page_num == num_pages - 1:
#                             logging.info(f"已处理 {page_num+1}/{num_pages} 页")
#                     except Exception as page_err:
#                         logging.error(f"处理第 {page_num+1} 页时出错: {str(page_err)}")
                
#                 # 缓存提取的文本
#                 if text.strip():
#                     with open(cache_path, 'w', encoding='utf-8') as f:
#                         json.dump({
#                             'text': text,
#                             'complete': True,  # 假设PyPDF2能够完整提取
#                             'attempt': max_attempts,
#                             'timestamp': datetime.datetime.now().isoformat()
#                         }, f, ensure_ascii=False)
#                     logging.info(f"成功使用PyPDF2提取文本，已缓存到: {cache_path}")
#                     return text
#                 else:
#                     logging.warning("PyPDF2提取的文本为空")
#         except Exception as pdf_err:
#             logging.error(f"使用PyPDF2提取文本时出错: {str(pdf_err)}")
        
#         # 如果所有方法都失败，尝试使用pdfminer作为最后的备用方法
#         try:
#             from pdfminer.high_level import extract_text as pdfminer_extract_text
#             logging.info(f"使用pdfminer提取文本: {pdf_path}")
#             text = pdfminer_extract_text(pdf_path)
#             if text.strip():
#                 with open(cache_path, 'w', encoding='utf-8') as f:
#                     json.dump({
#                         'text': text,
#                         'complete': True,  # 假设pdfminer能够完整提取
#                         'attempt': max_attempts,
#                         'timestamp': datetime.datetime.now().isoformat()
#                     }, f, ensure_ascii=False)
#                 logging.info(f"成功使用pdfminer提取文本，已缓存到: {cache_path}")
#                 return text
#             else:
#                 logging.warning("pdfminer提取的文本为空")
#         except Exception as miner_err:
#             logging.error(f"使用pdfminer提取文本时出错: {str(miner_err)}")
        
#         logging.error(f"所有提取方法都失败，无法从PDF提取文本: {pdf_path}")
#         return ""
#     except Exception as e:
#         logging.error(f"提取PDF文本过程中发生错误: {str(e)}")
#         logging.error(traceback.format_exc())
#         return ""

# 修改生成提取提示词的函数
def generate_entity_extraction_prompt(model_name="qwen", previous_entities=None, partial_extraction=False):
    """
    生成用于实体提取的提示词，不再需要包含论文文本内容
    
    Args:
        model_name (str): 使用的模型名称
        previous_entities (list): 之前提取的实体列表，用于断点续传
        partial_extraction (bool): 是否为部分提取（有先前的实体）
    
    Returns:
        str: 实体提取的提示词
    """
    # 基础提示
    base_prompt = """
你是一位专业的学术论文分析专家。请仔细阅读PDF文件内容，提取出所有提到的算法、数据集和评价指标实体，以 JSON 格式返回。

请识别以下类型的实体：
1. 算法：论文中描述的机器学习或深度学习算法，例如BERT、ResNet、LSTM等
2. 数据集：用于训练或评估算法的数据集，例如ImageNet、COCO、CIFAR-10等
3. 评价指标：用于评估算法性能的度量，例如准确率、精确率、召回率等

对于每个实体，请尽可能提取以下信息：
- 实体类型（Algorithm, Dataset, Metric）不能省略
- 算法实体ID（使用格式: 作者年份_实体名称，例如Zhang2016_TemplateSolver）不能省略
- 数据集实体ID（使用格式: 实体名称_年份，例如MNIST_2010）不能省略
- 评价实体ID（使用格式: 实体名称_类别，例如Accuracy_Classification）不能省略
- 实体名称，不能省略
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
                          "algorithm_id": "Zhang2016_TemplateSolver",
                          "entity_type": "Algorithm",
                          "name": "TemplateSolver",
                          "title": "论文标题",
                          "year": 2016,
                          "authors": ["Zhang, Y.", "Li, W.", ...],
                          "task": "任务类型",
                          "dataset": ["数据集1", "数据集2", ...],
                          "metrics": ["评价指标1", "评价指标2", ...],
                          "architecture": {
                            "components": ["组件1", "组件2", ...],
                            "connections": ["连接1", "连接2", ...],
                            "mechanisms": ["机制1", "机制2", ...]
                          },
                          "methodology": {
                            "training_strategy": ["策略1", "策略2", ...],
                            "parameter_tuning": ["参数1", "参数2", ...]
                          },
                          "feature_processing": ["处理方法1", "处理方法2", ...],
                          "evolution_relations": [
                            {
                              "from_entity": "Wang2015_PriorAlgorithm",
                              "to_entity": "Zhang2016_TemplateSolver",
                              "relation_type": "Improve",
                              "structure": "Architecture.Mechanism",
                              "detail": "具体改进内容",
                              "evidence": "证据文本",
                              "confidence": 0.95
                            }
                          ]
                        }
                      },
                      {
                        "dataset_entity": {
                          "dataset_id": "MNIST_2010",
                          "entity_type": "Dataset",
                          "name": "MNIST",
                          "description": "手写数字识别数据集",
                          "domain": "计算机视觉",
                          "size": 70000,
                          "year": 2010,
                          "creators": ["LeCun, Y.", "Cortes, C.", ...],
                          "evolution_relations": [
                            {
                              "from_entity": "OldDataset_2005",
                              "to_entity": "MNIST_2010",
                              "relation_type": "Extend",
                              "detail": "扩展了样本数量",
                              "evidence": "证据文本",
                              "confidence": 0.9
                            }
                          ]
                        }
                      },
                      {
                        "metric_entity": {
                          "metric_id": "Accuracy_Classification",
                          "entity_type": "Metric",
                          "name": "Accuracy",
                          "description": "分类准确率",
                          "category": "分类评估",
                          "formula": "正确分类样本数/总样本数",
                          "evolution_relations": [
                            {
                              "from_entity": "OldMetric_2000",
                              "to_entity": "Accuracy_Classification",
                              "relation_type": "Improve",
                              "detail": "改进了计算方式",
                              "evidence": "证据文本",
                              "confidence": 0.85
                            }
                          ]
                        }
                      },...//其他实体
                    ]
                    ```

尽量提取论文中所有可能的实体信息，如果某些字段信息不可用，可以省略（如果论文中出现实体名字也可以抽取）。
注意:
1.尤其注意带有大写或代表性名称，表格内或加黑的实体[例子:RNN,CNN,LSTM等],尤其注意请生成全部实体
2.例如如果有130个引文，尽量生成120个实体
3.你可以分段提取
请确保JSON格式正确，避免语义错误。
"""

    # 添加完成状态请求
    completion_request = """
最后，请明确告知我提取是否已完成，还是需要继续提取更多实体。请根据你对文本的分析，判断是否已经提取了所有可能的实体。

在JSON返回后，请单独一行写明"EXTRACTION_COMPLETE: true"（找不到任何实体了）或"EXTRACTION_COMPLETE: false"（如果有任何没有被抽取的实体存在或分段提取,请继续提取）。
"""
    base_prompt += completion_request

    # 如果有之前提取的实体，添加到提示中
    if previous_entities and len(previous_entities) > 0:
        entity_examples = []
        for i, entity in enumerate(previous_entities):  # 只显示最多5个示例
            if "algorithm_entity" in entity:
                entity_type = "算法"
                entity_name = entity["algorithm_entity"].get("name", "未知算法")
                entity_examples.append(f"{entity_type}: {entity_name}")
            elif "dataset_entity" in entity:
                entity_type = "数据集"
                entity_name = entity["dataset_entity"].get("name", "未知数据集")
                entity_examples.append(f"{entity_type}: {entity_name}")
            elif "metric_entity" in entity:
                entity_type = "评价指标"
                entity_name = entity["metric_entity"].get("name", "未知指标")
                entity_examples.append(f"{entity_type}: {entity_name}")
        
        if partial_extraction:
            # 添加部分提取的上下文
            previous_entities_hint = "\n\n以下实体已经被提取过，请不要重复提取，并继续识别其他实体：\n- "
            previous_entities_hint += "\n- ".join(entity_examples)
            previous_entities_hint += "\n\n请确保你提取的是新实体，不要包含上述已提取的实体。"
            base_prompt += previous_entities_hint
    
    return base_prompt

# 更新 extract_entities_with_openai 函数
def extract_entities_with_openai(prompt, model_name="gpt-3.5-turbo", max_attempts=3, temp_cache_path=None):
    """
    使用OpenAI API提取实体
    
    Args:
        prompt (str): 提取实体的提示词
        model_name (str): 模型名称
        max_attempts (int): 最大尝试次数
        temp_cache_path (str, optional): 临时缓存文件路径，已弃用，保留参数是为了兼容性
        
    Returns:
        tuple: (提取的实体列表, 是否处理完成)
    """
    if not hasattr(Config, 'OPENAI_API_KEY') or not Config.OPENAI_API_KEY:
        logging.error("未配置OpenAI API密钥")
        return [], False
    
    # 不再使用临时缓存文件，而是在内存中保存结果
    response_content = ""
    
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
            # 保存响应到内存变量
            response_content = content
            logging.info(f"已收到OpenAI API响应，长度: {len(content)} 字符")
            # 检查是否包含完成标志 - 使用公共函数
            is_complete = check_extraction_complete(content)
            # 尝试提取JSON部分
            json_text = extract_json_from_text(content)
            entities = []
            # 验证并解析JSON
            if json_text:
                try:
                    entities = json.loads(json_text)
                    if not isinstance(entities, list):
                        entities = [entities]  # 确保是列表格式
                except json.JSONDecodeError:
                    logging.warning(f"JSON解析错误，尝试清理后重新解析")
                    try:
                        clean_json = json_text.replace('```', '').strip()
                        clean_json = re.sub(r',\s*]', ']', clean_json)  # 移除尾部逗号
                        clean_json = re.sub(r',\s*}', '}', clean_json)  # 移除尾部逗号
                        entities = json.loads(clean_json)
                        if not isinstance(entities, list):
                            entities = [entities]  # 确保是列表格式
                    except Exception as clean_err:
                        logging.error(f"清理后JSON仍解析失败: {str(clean_err)}")
            if entities and len(entities) > 0:
                logging.info(f"成功从OpenAI响应提取 {len(entities)} 个实体")
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

# # 更新 extract_entities_with_qwen_agent 函数
# def extract_entities_with_qwen_agent(agent, prompt, max_attempts=3, temp_cache_path=None):
#     """
#     使用千问agent提取实体
    
#     Args:
#         agent: 千问agent实例
#         prompt (str): 提取实体的提示词 
#         max_attempts (int): 最大尝试次数
#         temp_cache_path (str, optional): 临时缓存文件路径，已弃用，保留参数是为了兼容性
        
#     Returns:
#         tuple: (提取的实体列表, 是否处理完成)
#     """
#     if not agent:
#         logging.error("千问agent未初始化")
#         return [], False
    
#     # 在内存中处理结果
#     response_content = ""
    
#     # 尝试多次提取
#     attempt = 0
#     while attempt < max_attempts:
#         attempt += 1
#         try:
#             logging.info(f"尝试使用千问agent提取实体 (尝试 {attempt}/{max_attempts})")
            
#             # 调用agent进行提取
#             response = agent.chat(prompt)
#             response_content = response
#             logging.debug(f"千问agent响应: {response}")
#             logging.info(f"已收到千问agent响应，长度: {len(response)} 字符")
            
#             # 检查是否包含完成标志 - 使用公共函数
#             is_complete = check_extraction_complete(response)
        
#         # 尝试提取JSON部分
#             json_text = extract_json_from_text(response)
#             entities = []
            
#             # 验证并解析JSON
#             if json_text:
#                 try:
#                     # 尝试直接解析
#                     entities = json.loads(json_text)
#                     if not isinstance(entities, list):
#                         entities = [entities]  # 确保是列表格式
#                 except json.JSONDecodeError:
#                     logging.warning(f"JSON解析错误，尝试清理后重新解析")
#                     try:
#                         # 清理JSON文本，处理常见错误
#                         clean_json = json_text.replace('```', '').strip()
#                         clean_json = re.sub(r',\s*]', ']', clean_json)  # 移除尾部逗号
#                         clean_json = re.sub(r',\s*}', '}', clean_json)  # 移除尾部逗号
#                         entities = json.loads(clean_json)
#                         if not isinstance(entities, list):
#                             entities = [entities]  # 确保是列表格式
#                     except Exception as clean_err:
#                         logging.error(f"清理后JSON仍解析失败: {str(clean_err)}")
            
#             # 验证结果
#             if entities and len(entities) > 0:
#                 logging.info(f"成功从千问agent响应提取 {len(entities)} 个实体")
#                 return entities, is_complete
#             else:
#                 logging.warning(f"提取结果格式不正确，未能提取有效实体")
#                 if attempt < max_attempts:
#                     logging.info(f"将重试...")
#                     time.sleep(2)  # 短暂延迟后重试
#         except Exception as e:
#             logging.error(f"使用千问API提取实体时出错: {str(e)}")
#             logging.error(traceback.format_exc())
#             if attempt < max_attempts:
#                 logging.info(f"将重试...")
#                 time.sleep(2)
    
#     logging.error(f"在 {max_attempts} 次尝试后仍未能从千问agent提取实体")
#     return [], False

# 创建一个通用的实体提取函数，替代多个API特定的函数
def extract_entities_with_model(pdf_paths, model_name="qwen", max_attempts=3, previous_entities=None, agent=None):
    """
    使用指定模型从PDF文件中提取实体，支持多个文件和file-id方式
    
    Args:
        pdf_paths (str/list): PDF文件路径或路径列表
        model_name (str): 模型名称 ("qwen", "openai")
        max_attempts (int): 最大尝试次数
        previous_entities (list, optional): 之前提取的实体，用于断点续传
        agent: 可选的agent实例（如果使用agent调用）
        
    Returns:
        tuple: (提取的实体列表, 是否处理完成)
    """
    # 转换单个路径为列表
    if isinstance(pdf_paths, str):
        pdf_paths = [pdf_paths]
    
    logging.info(f"使用模型 {model_name} 从 {len(pdf_paths)} 个PDF文件提取实体")
    
    # 准备file-id列表
    file_ids = []
    for pdf_path in pdf_paths:
        if os.path.exists(pdf_path):
            file_id = upload_and_cache_file(pdf_path)
            if file_id:
                file_ids.append(file_id)
    
    if not file_ids:
        logging.error("没有有效的file-id，无法提取实体")
        return [], False
        
    # 生成提取实体的提示
    prompt = generate_entity_extraction_prompt(
        model_name, 
        previous_entities=previous_entities, 
        partial_extraction=previous_entities and len(previous_entities) > 0
    )
    
    all_entities = []  # 所有已提取的实体
    current_attempt = 0
    is_extraction_complete = False
    
    while current_attempt < max_attempts and not is_extraction_complete:
        current_attempt += 1
        logging.info(f"提取尝试 {current_attempt}/{max_attempts}")
        
        # 如果已有提取的实体，将其加入到提示词中
        current_prompt = prompt
        if all_entities:
            # 构建已提取的实体列表摘要 
            entity_summary = []
            for idx, entity in enumerate(all_entities):
                entity_type = "未知"
                entity_name = "未命名"
                
                if 'algorithm_entity' in entity:
                    entity_type = "算法"
                    entity_name = entity['algorithm_entity'].get('name', '未命名算法')
                elif 'dataset_entity' in entity:
                    entity_type = "数据集"
                    entity_name = entity['dataset_entity'].get('name', '未命名数据集')
                elif 'metric_entity' in entity:
                    entity_type = "评价指标"
                    entity_name = entity['metric_entity'].get('name', '未命名指标')
                
                entity_summary.append(f"{entity_type}: {entity_name}")
            
            if len(all_entities) > 5:
                entity_summary.append(f"... 等 {len(all_entities)} 个实体")
            
            # 添加已提取实体的提示
            entities_hint = "\n\n已经提取的实体有：\n" + "\n".join(entity_summary)
            entities_hint += "\n\n请继续提取文本中其他尚未提取的实体，不要重复已提取的实体。"
            current_prompt = current_prompt + entities_hint
            logging.info(f"添加了 {len(all_entities)} 个已提取实体的提示信息")
        
        # 使用千问API提取实体
        entities = []
        try:
            from openai import OpenAI
            client = OpenAI(
                api_key=Config.QWEN_API_KEY,
                base_url=Config.QWEN_BASE_URL
            )
            
            # 构建消息
            messages = [
                {"role": "system", "content": "你是一个专注于从学术论文中提取实体信息的AI助手，负责提取算法、数据集和评价指标等实体信息。"}
            ]
            
            # 添加file-id引用
            if file_ids:
                file_content = ",".join([f"fileid://{fid}" for fid in file_ids])
                messages.append({"role": "system", "content": file_content})
            
            # 添加用户提示
            messages.append({"role": "user", "content": current_prompt})
            
            # 调用API
            logging.info(f"调用千问API提取实体，文件数: {len(file_ids)}")
            response = client.chat.completions.create(
                model=Config.QWEN_MODEL or "qwen-long",
                messages=messages,
                temperature=0.2,
                stream=True,
                max_tokens=None  # 不限制token数量
            )
            
            # 收集流式响应内容
            content = ""
            chunk_count = 0
            for chunk in response:
                chunk_count += 1
                if chunk.choices and chunk.choices[0].delta.content:
                    content += chunk.choices[0].delta.content
                # 每100个块记录一次
                if chunk_count % 100 == 0:
                    logging.info(f"收到响应块 #{chunk_count}，当前响应长度: {len(content)} 字符")
            
            logging.info(f"响应接收完成，共 {chunk_count} 个响应块，总长度: {len(content)} 字符")
            
            # 检查是否包含完成标志
            is_complete = check_extraction_complete(content)
            is_extraction_complete=is_complete
            # 提取JSON部分
            json_text = extract_json_from_text(content)
            if json_text:
                try:
                    entities = json.loads(json_text)
                    if not isinstance(entities, list):
                        entities = [entities]  # 确保是列表格式
                    logging.info(f"成功从API响应提取 {len(entities)} 个实体")
                except json.JSONDecodeError as e:
                    logging.warning(f"JSON解析错误: {str(e)}，尝试清理后重新解析")
                    try:
                        clean_json = json_text.replace('```', '').strip()
                        clean_json = re.sub(r',\s*]', ']', clean_json)
                        clean_json = re.sub(r',\s*}', '}', clean_json)
                        entities = json.loads(clean_json)
                        if not isinstance(entities, list):
                            entities = [entities]  # 确保是列表格式
                        logging.info(f"清理后成功解析 {len(entities)} 个实体")
                    except Exception as clean_err:
                        logging.error(f"清理后JSON仍解析失败: {str(clean_err)}")
                        logging.error(f"原始JSON错误: {str(e)}")
                        logging.debug(f"问题JSON片段: {json_text[:100]}...{json_text[-100:] if len(json_text) > 100 else ''}")
                        entities = []
            
            # 合并新提取的实体（去重）
            if all_entities:
                # 创建ID到实体的映射
                entity_id_map = {}
                for entity in all_entities:
                    entity_id = _get_entity_id(entity)
                    if entity_id:
                        entity_id_map[entity_id] = entity
                
                # 添加或更新实体
                for new_entity in entities:
                    entity_id = _get_entity_id(new_entity)
                    if entity_id and entity_id in entity_id_map:
                        # 更新已存在的实体
                        continue
                    else:
                        # 添加新实体
                        all_entities.append(new_entity)
            else:
                # 如果是首次提取，直接使用结果
                all_entities = entities
            
            # 如果提取已完成，或者连续两次提取结果相同（无新增实体），则认为提取完成
            if is_complete or (entities and len(all_entities) == len(entities) and current_attempt > 1):
                is_extraction_complete = True
                break
                
        except Exception as e:
            logging.error(f"提取实体时出错: {str(e)}")
            logging.error(traceback.format_exc())
            # 继续下一次尝试
    
    logging.info(f"完成实体提取，共 {len(all_entities)} 个实体，完成状态: {is_extraction_complete}")
    return all_entities, is_extraction_complete

# def extract_entities_with_qwen(prompt, max_attempts=3, temp_cache_path=None):
#     """
#     使用千问API提取实体
    
#     Args:
#         prompt (str): 提取实体的提示词
#         max_attempts (int): 最大尝试次数
#         temp_cache_path (str, optional): 临时缓存文件路径，已弃用，保留参数是为了兼容性
        
#     Returns:
#         tuple: (提取的实体列表, 是否处理完成)
#     """
#     if not hasattr(Config, 'QWEN_API_KEY') or not Config.QWEN_API_KEY:
#         logging.error("未配置千问API密钥")
#         return [], False
    
#     # 在内存中处理结果
#     response_content = ""
    
#     # 尝试多次提取
#     attempt = 0
#     while attempt < max_attempts:
#         attempt += 1
#         try:
#             logging.info(f"尝试使用千问API提取实体 (尝试 {attempt}/{max_attempts})")
            
#             # 创建OpenAI客户端（千问API兼容OpenAI格式）
#             client = OpenAI(
#                 api_key=Config.QWEN_API_KEY,
#                 base_url=Config.QWEN_BASE_URL
#             )
            
#             # 构建消息
#             messages = [
#                 {"role": "system", "content": "你是一个专注于从学术论文中提取实体信息的AI助手。请从提供的文本中提取算法、数据集和评价指标等相关实体信息，并以JSON格式返回。"},
#                 {"role": "user", "content": prompt}
#             ]
            
#             # 调用API
#             response = client.chat.completions.create(
#                 model=Config.QWEN_MODEL or "qwen-long",
#                 messages=messages,
#                 temperature=0.2,
#                 max_tokens=None,  # 不限制token数量
#                 stream=True
#             )
            
#             # 收集流式响应内容
#             content = ""
#             for chunk in response:
#                 if chunk.choices and chunk.choices[0].delta.content:
#                     content += chunk.choices[0].delta.content
            
#             response_content = content
#             logging.info(f"已收到千问API响应，长度: {len(content)} 字符")
            
#             # 检查是否包含完成标志 - 使用公共函数
#             is_complete = check_extraction_complete(content)
            
#             # 尝试提取JSON部分
#             json_text = extract_json_from_text(content)
#             entities = []
            
#             # 验证并解析JSON
#             if json_text:
#                 try:
#                     # 尝试直接解析
#                     entities = json.loads(json_text)
#                     if not isinstance(entities, list):
#                         entities = [entities]  # 确保是列表格式
#                 except json.JSONDecodeError:
#                     logging.warning(f"JSON解析错误，尝试清理后重新解析")
#                     try:
#                         # 清理JSON文本，处理常见错误
#                         clean_json = json_text.replace('```', '').strip()
#                         clean_json = re.sub(r',\s*]', ']', clean_json)  # 移除尾部逗号
#                         clean_json = re.sub(r',\s*}', '}', clean_json)  # 移除尾部逗号
#                         entities = json.loads(clean_json)
#                         if not isinstance(entities, list):
#                             entities = [entities]  # 确保是列表格式
#                     except Exception as clean_err:
#                         logging.error(f"清理后JSON仍解析失败: {str(clean_err)}")
            
#             # 验证结果
#             if entities and len(entities) > 0:
#                 logging.info(f"成功从千问API响应提取 {len(entities)} 个实体")
#                 return entities, is_complete
#             else:
#                 logging.warning(f"提取结果格式不正确，未能提取有效实体")
#                 if attempt < max_attempts:
#                     logging.info(f"将重试...")
#                     time.sleep(2)  # 短暂延迟后重试
#         except Exception as e:
#             logging.error(f"使用千问API提取实体时出错: {str(e)}")
#             logging.error(traceback.format_exc())
#             if attempt < max_attempts:
#                 logging.info(f"将重试...")
#                 time.sleep(2)
    
#     logging.error(f"在 {max_attempts} 次尝试后仍未能从千问API提取实体")
#     return [], False

def extract_json_from_text(text):
    """
    从文本中提取JSON格式的内容
    
    Args:
        text (str): 包含JSON的文本
        
    Returns:
        str: 提取的JSON文本，如果未找到则返回None
    """
    if not text:
        return None
        
    logging.debug(f"开始从文本中提取JSON，文本长度：{len(text)} 字符")
    
    # 首先尝试从代码块中提取JSON
    json_block_pattern = r'```(?:json)?\s*([\s\S]*?)(?:\s*```|\s*EXTRACTION_COMPLETE\s*:)'
    matches = re.findall(json_block_pattern, text)
    if matches:
        json_candidate = matches[0].strip()
        # 确保JSON以合适的结尾符号结束
        if json_candidate.endswith(','):
            json_candidate = json_candidate[:-1]
        if json_candidate.endswith(']') or json_candidate.endswith('}'):
            logging.debug(f"从代码块提取到可能的JSON，长度: {len(json_candidate)}")
            # 验证JSON是否有效
            try:
                json.loads(json_candidate)
                return json_candidate
            except json.JSONDecodeError as e:
                logging.warning(f"从代码块提取的JSON无效: {str(e)}，尝试其他方法")
                logging.debug(f"出错文本：{text}")
    
    # 尝试提取最外层的JSON数组 [...] 或对象 {...}
    # 首先尝试数组
    array_pattern = r'\[\s*\{[\s\S]*?\}\s*\]'
    matches = re.findall(array_pattern, text)
    if matches:
        for match in sorted(matches, key=len, reverse=True):
            try:
                json.loads(match)
                logging.debug(f"提取到有效的JSON数组，长度: {len(match)}")
                return match
            except json.JSONDecodeError:
                continue
    
    # 然后尝试对象
    object_pattern = r'\{[\s\S]*?\}'
    matches = re.findall(object_pattern, text)
    if matches:
        for match in sorted(matches, key=len, reverse=True):
            try:
                json.loads(match)
                logging.debug(f"提取到有效的JSON对象，长度: {len(match)}")
                return match
            except json.JSONDecodeError:
                continue
    
    # 如果上述方法都失败，尝试手动查找JSON边界
    if '[' in text and ']' in text:
        try:
            # 找到第一个 [ 和最后一个 ]
            start_idx = text.find('[')
            last_idx = text.rfind(']')
            if start_idx != -1 and last_idx != -1 and start_idx < last_idx:
                potential_json = text[start_idx:last_idx+1]
                try:
                    json.loads(potential_json)
                    logging.debug(f"使用索引方法提取到有效的JSON，长度: {len(potential_json)}")
                    return potential_json
                except json.JSONDecodeError:
                    # 尝试清理JSON
                    clean_json = re.sub(r',\s*]', ']', potential_json)
                    clean_json = re.sub(r',\s*}', '}', clean_json)
                    try:
                        json.loads(clean_json)
                        logging.debug(f"清理后提取到有效的JSON，长度: {len(clean_json)}")
                        return clean_json
                    except json.JSONDecodeError:
                        pass
        except Exception as e:
            logging.warning(f"手动查找JSON边界时出错: {str(e)}")
    
    # 最后的尝试：去除任何EXTRACTION_COMPLETE标记，然后再解析
    extraction_complete_pattern = r'([\s\S]*?)(?:\s*EXTRACTION_COMPLETE\s*:)'
    matches = re.findall(extraction_complete_pattern, text)
    if matches:
        cleaned_text = matches[0].strip()
        # 如果文本以代码块结束符号结束，去掉它
        if cleaned_text.endswith('```'):
            cleaned_text = cleaned_text[:-3].strip()
        
        # 检查开头和结尾，确保是完整的JSON
        if (cleaned_text.startswith('[') and cleaned_text.endswith(']')) or \
           (cleaned_text.startswith('{') and cleaned_text.endswith('}')):
            try:
                json.loads(cleaned_text)
                logging.debug(f"从去除标记后的文本中提取到有效JSON，长度: {len(cleaned_text)}")
                return cleaned_text
            except json.JSONDecodeError as e:
                # 尝试修复最常见的JSON错误
                if str(e).startswith('Extra data'):
                    error_pos = int(re.search(r'char (\d+)', str(e)).group(1))
                    cleaned_text = cleaned_text[:error_pos]
                    if (cleaned_text.startswith('[') and cleaned_text.endswith(']')) or \
                       (cleaned_text.startswith('{') and cleaned_text.endswith('}')):
                        try:
                            json.loads(cleaned_text)
                            logging.debug(f"从修复后的文本中提取到有效JSON，长度: {len(cleaned_text)}")
                            return cleaned_text
                        except json.JSONDecodeError:
                            pass
    
    logging.warning("未能从文本中提取有效的JSON")
    return None

def extract_paper_entities(pdf_paths, max_attempts=5, batch_size=1, model_name="qwen", task_id=None):
    """
    从PDF文件中提取实体，支持批量处理和文件ID模式
    
    Args:
        pdf_paths (str/list): 单个PDF路径或PDF路径列表
        max_attempts (int): 最大尝试次数
        batch_size (int): 批处理大小
        model_name (str): 使用的模型名称
        task_id (str): 任务ID，用于缓存
        
    Returns:
        tuple: (提取的实体列表, 处理是否完成标志)
    """
    # 转换单个路径为列表
    if isinstance(pdf_paths, str):
        pdf_paths = [pdf_paths]
        
    # 验证并过滤PDF路径
    valid_paths = []
    for path in pdf_paths:
        if os.path.exists(path) and path.lower().endswith('.pdf'):
            valid_paths.append(path)
        else:
            logging.warning(f"无效的PDF文件路径: {path}")
    
    if not valid_paths:
        logging.error("没有有效的PDF文件路径")
        return [], False
    
    logging.info(f"从 {len(valid_paths)} 个PDF文件中提取实体")
    
    # 生成缓存目录和文件路径
    cache_dir = os.path.join(Config.CACHE_DIR, 'entities')
    os.makedirs(cache_dir, exist_ok=True)
    
    # 生成缓存键
    cache_key_parts = []
    for path in valid_paths:
        filename = os.path.basename(path)
        file_stat = os.stat(path)
        file_size = file_stat.st_size
        file_mtime = int(file_stat.st_mtime)
        # 使用文件名、大小和修改时间生成唯一键
        cache_key_parts.append(f"{filename}_{file_size}_{file_mtime}")
    
    # 排序以确保相同文件集合的一致键值
    cache_key_parts.sort()
    
    # 添加可选的任务ID
    if task_id:
        cache_key = f"{task_id}_{'_'.join(cache_key_parts)}"
    else:
        cache_key = '_'.join(cache_key_parts)
    
    # 生成MD5哈希，避免文件名过长
    cache_key = hashlib.md5(cache_key.encode()).hexdigest()
    cache_file = os.path.join(cache_dir, f"{cache_key}.json")
    
    # 检查缓存
    if os.path.exists(cache_file):
        try:
            with open(cache_file, 'r', encoding='utf-8') as f:
                cached_data = json.load(f)
            is_complete = cached_data.get("is_complete", False)
            entities = cached_data.get("entities", [])
            logging.info(f"从缓存加载了 {len(entities)} 个实体，完成状态: {is_complete}")
            return entities, is_complete
        except Exception as e:
            logging.error(f"读取缓存文件出错: {str(e)}")
    
    # 如果没有缓存或读取缓存失败，执行提取
    if batch_size > 1 and len(valid_paths) > batch_size:
        # 批量处理
        all_entities = []
        is_complete = True
        
        # 将文件分批处理
        for i in range(0, len(valid_paths), batch_size):
            batch_paths = valid_paths[i:i+batch_size]
            logging.info(f"处理批次 {i//batch_size + 1}/{(len(valid_paths) + batch_size - 1)//batch_size}，包含 {len(batch_paths)} 个PDF文件")
            
            # 提取当前批次的实体
            batch_entities, batch_complete = extract_entities_with_model(
                batch_paths, 
                model_name=model_name,
                max_attempts=max_attempts,
                previous_entities=all_entities if all_entities else None
            )
            
            # 合并实体（去重）
            if all_entities:
                entity_ids = {_get_entity_id(e): e for e in all_entities if _get_entity_id(e)}
                for entity in batch_entities:
                    entity_id = _get_entity_id(entity)
                    if entity_id and entity_id not in entity_ids:
                        all_entities.append(entity)
                        entity_ids[entity_id] = entity
            else:
                all_entities = batch_entities
            
            # 如果任何批次未完成，则标记整体为未完成
            if not batch_complete:
                is_complete = False
                logging.warning(f"批次 {i//batch_size + 1} 实体提取未完成")
            
            logging.info(f"当前已提取 {len(all_entities)} 个唯一实体")
    else:
        # 直接处理所有文件
        all_entities, is_complete = extract_entities_with_model(
            valid_paths, 
            model_name=model_name,
            max_attempts=max_attempts
        )
    
    # 缓存结果
    try:
        with open(cache_file, 'w', encoding='utf-8') as f:
            cache_data = {
                "entities": all_entities,
                "is_complete": is_complete,
                "extraction_time": time.time()
            }
            json.dump(cache_data, f, indent=2, ensure_ascii=False)
        logging.info(f"成功缓存 {len(all_entities)} 个实体到 {cache_file}")
    except Exception as e:
        logging.error(f"缓存实体到文件时出错: {str(e)}")
    
    unique_entities = len({_get_entity_id(e) for e in all_entities if _get_entity_id(e)})
    logging.info(f"完成从 {len(valid_paths)} 个PDF文件中提取 {len(all_entities)} 个实体（{unique_entities} 个唯一实体），完成状态: {is_complete}")
    
    return all_entities, is_complete

def test_json_extraction_and_completion_status(text):
    """
    测试JSON提取和完成状态检测功能
    
    Args:
        text (str): 要测试的文本
        
    Returns:
        tuple: (提取的JSON字符串, 完成状态)
    """
    logging.info("开始测试JSON提取和完成状态检测")
    
    # 检查完成状态
    is_complete = check_extraction_complete(text)
    logging.info(f"完成状态检测结果: {is_complete}")
    
    # 提取JSON
    json_str = extract_json_from_text(text)
    if json_str:
        logging.info(f"成功提取JSON，长度: {len(json_str)}")
        json_preview = json_str[:200] + "..." if len(json_str) > 200 else json_str
        logging.info(f"JSON预览: {json_preview}")
    else:
        logging.error("未能提取到有效的JSON")
        
    # 如果提取到了JSON，尝试解析
    entities = []
    if json_str:
        try:
            entities = json.loads(json_str)
            if not isinstance(entities, list):
                entities = [entities]
            logging.info(f"解析到 {len(entities)} 个实体")
        except Exception as e:
            logging.error(f"JSON解析错误: {str(e)}")
    
    return json_str, is_complete, entities

# 添加一个函数用于上传文件并缓存file-id
def upload_and_cache_file(file_path, purpose="file-extract"):
    """
    上传文件到OpenAI并缓存file-id，避免重复上传
    
    Args:
        file_path (str): 文件路径
        purpose (str): 文件用途，默认为"file-extract"
        
    Returns:
        str: 文件ID，如果上传失败则返回None
    """
    if not os.path.exists(file_path):
        logging.error(f"文件不存在: {file_path}")
        return None
        
    # 创建缓存目录
    cache_dir = os.path.join(Config.CACHE_DIR, "file_ids")
    os.makedirs(cache_dir, exist_ok=True)
    
    # 获取文件信息作为缓存键
    file_name = os.path.basename(file_path)
    file_size = os.path.getsize(file_path)
    # 生成缓存键
    cache_key = f"{file_name}_{file_size}"
    cache_key = hashlib.md5(cache_key.encode()).hexdigest()
    cache_file = os.path.join(cache_dir, f"{cache_key}.json")
    
    # 检查缓存
    if os.path.exists(cache_file):
        try:
            with open(cache_file, 'r', encoding='utf-8') as f:
                cached_data = json.load(f)
                file_id = cached_data.get("file_id")
                if file_id:
                    logging.info(f"从缓存获取file-id: {file_id} (文件: {file_name})")
                    return file_id
        except Exception as e:
            logging.error(f"读取缓存文件出错: {str(e)}")
    
    # 如果没有缓存或缓存无效，上传文件
    try:
        from openai import OpenAI
        client = OpenAI(
            api_key=Config.OPENAI_API_KEY or Config.QWEN_API_KEY,
            base_url=Config.QWEN_BASE_URL
        )
        
        logging.info(f"上传文件到千问API: {file_name}")
        with open(file_path, "rb") as file:
            response = client.files.create(
                file=file,
                purpose=purpose
            )
            
        if hasattr(response, "id") and response.id:
            file_id = response.id
            logging.info(f"文件上传成功，file-id: {file_id}")
            
            # 缓存file-id
            with open(cache_file, 'w', encoding='utf-8') as f:
                cache_data = {
                    "file_id": file_id,
                    "file_name": file_name,
                    "upload_time": time.time()
                }
                json.dump(cache_data, f, indent=2, ensure_ascii=False)
            logging.info(f"file-id已缓存到 {cache_file}")
            return file_id
        else:
            logging.error("上传文件失败，API没有返回有效的file-id")
            return None
            
    except Exception as e:
        logging.error(f"上传文件时出错: {str(e)}")
        logging.error(traceback.format_exc())
        return None

def _get_entity_id(entity):
    """
    从实体字典中获取实体ID
    
    Args:
        entity (dict): 实体字典
        
    Returns:
        str: 实体ID，如果没有则返回None
    """
    if not entity or not isinstance(entity, dict):
        return None
        
    if "algorithm_entity" in entity and isinstance(entity["algorithm_entity"], dict):
        return entity["algorithm_entity"].get("algorithm_id")
    elif "dataset_entity" in entity and isinstance(entity["dataset_entity"], dict):
        return entity["dataset_entity"].get("dataset_id")
    elif "metric_entity" in entity and isinstance(entity["metric_entity"], dict):
        return entity["metric_entity"].get("metric_id")
    return None

def extract_evolution_relations(entities, pdf_paths=None, task_id=None, previous_relations=None):
    """
    从实体列表和PDF文件中提取演化关系，支持多个PDF文件和file-id方式
    
    Args:
        entities (list): 实体列表
        pdf_paths (str/list, optional): PDF文件路径或路径列表
        task_id (str, optional): 任务ID，用于缓存和日志
        previous_relations (list, optional): 之前提取的关系，用于断点续传
        
    Returns:
        list: 提取的演化关系列表
    """
    try:
        # 转换单个路径为列表
        if isinstance(pdf_paths, str):
            pdf_paths = [pdf_paths]
            
        if pdf_paths:
            logging.info(f"从 {len(pdf_paths)} 个PDF文件中提取演化关系")
        else:
            logging.info("没有提供PDF文件，仅基于实体列表分析演化关系")
            
        # 验证实体列表
        if not entities or not isinstance(entities, list) or len(entities) == 0:
            logging.error("没有提供实体或实体列表为空，无法提取关系")
            return []
            
        logging.info(f"基于 {len(entities)} 个实体提取演化关系" + 
                    (f"，已有 {len(previous_relations)} 个之前的关系" if previous_relations else ""))
        
        # 生成缓存目录和文件路径
        cache_dir = os.path.join(Config.CACHE_DIR, 'relations')
        os.makedirs(cache_dir, exist_ok=True)
        
        # 生成缓存键
        cache_key_parts = []
        
        # 添加PDF文件信息到缓存键
        if pdf_paths:
            for path in pdf_paths:
                if os.path.exists(path):
                    filename = os.path.basename(path)
                    file_stat = os.stat(path)
                    file_size = file_stat.st_size
                    file_mtime = int(file_stat.st_mtime)
                    cache_key_parts.append(f"{filename}_{file_size}_{file_mtime}")
        
        # 添加实体ID到缓存键
        entity_ids = []
        for entity in entities:
            entity_id = _get_entity_id(entity)
            if entity_id:
                entity_ids.append(entity_id)
        
        # 排序以确保相同文件集合的一致键值
        entity_ids.sort()
        cache_key_parts.extend(entity_ids)
        
        # 添加可选的任务ID
        if task_id:
            cache_key = f"{task_id}_{'_'.join(cache_key_parts)}"
        else:
            cache_key = '_'.join(cache_key_parts)
        
        # 生成MD5哈希，避免文件名过长
        cache_key = hashlib.md5(cache_key.encode()).hexdigest()
        cache_file = os.path.join(cache_dir, f"{cache_key}.json")
        
        # 检查缓存
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'r', encoding='utf-8') as f:
                    cached_relations = json.load(f)
                logging.info(f"从缓存加载了 {len(cached_relations)} 个关系")
                return cached_relations
            except Exception as e:
                logging.error(f"读取缓存文件出错: {str(e)}")
        
        # 准备file-id列表
        file_ids = []
        if pdf_paths:
            for pdf_path in pdf_paths:
                if os.path.exists(pdf_path):
                    file_id = upload_and_cache_file(pdf_path)
                    if file_id:
                        file_ids.append(file_id)
            
            if not file_ids:
                logging.warning("没有有效的file-id，将仅基于实体列表分析关系")
        
        # 生成关系提取提示
        system_message = """你是一位专业的算法演化关系分析专家。请分析提供的论文和实体列表，识别算法之间的演化关系。
        
演化关系是指一个算法对另一个算法的改进、扩展、优化或替代关系。请仔细分析这些关系，并以JSON格式返回。
        
1.实体ID（使用格式: 作者年份_实体名称，例如Zhang2016_TemplateSolver）
2.已知实体提取的entity_id需要保持一致
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
请根据论文内容，识别实体列表中算法之间的演化关系，并提供详细信息。如果找不到明确的关系，请勿创建虚构的关系。"""
        
        # 构建用户提示信息
        user_message = """请分析以下实体列表和论文内容，识别算法之间的演化关系。

实体列表（包含算法、数据集和评价指标）：
"""
        
        # 添加实体到提示中
        entity_json = json.dumps(entities, ensure_ascii=False, indent=2)
        user_message += entity_json
        
        user_message += """
        

重要说明：请注意实体类型可以是算法（Algorithm）、数据集（Dataset）或评估指标（Metric）。实体之间的演化关系可以是：
- 算法改进/优化/扩展/替换另一个算法
- 算法使用特定数据集
- 算法使用特定评估指标
- 数据集改进/扩展另一个数据集
- 评估指标改进/扩展另一个评估指标

请以以下格式返回关系列表：
[
  {
    "from_entities": [
      {"entity_id": "算法A的ID", "entity_type": "Algorithm"}
    ],
    "to_entities": [
      {"entity_id": "算法B的ID", "entity_type": "Algorithm"}
    ],
    "relation_type": "关系类型：Improve/Optimize/Extend/Replace/Use",
    "structure": "关系的结构描述",
    "detail": "关系的详细说明",
    "evidence": "支持这种关系判断的证据（引用论文中的文字）",
    "confidence": 置信度（0.0-1.0）
  },
  ...
]

请确保每个关系都有足够的证据支持，避免虚构不存在的关系。"""
        
        # 调用API进行关系提取
        try:
            from openai import OpenAI
            client = OpenAI(
                api_key=Config.QWEN_API_KEY,
                base_url=Config.QWEN_BASE_URL
            )
            
            # 构建消息
            messages = [
                {"role": "system", "content": system_message}
            ]
            
            # 添加file-id引用
            if file_ids:
                file_content = ",".join([f"fileid://{fid}" for fid in file_ids])
                messages.append({"role": "system", "content": file_content})
            
            # 添加用户提示
            messages.append({"role": "user", "content": user_message})
            
            # 调用API
            logging.info(f"调用千问API提取关系，文件数: {len(file_ids) if file_ids else 0}")
            response = client.chat.completions.create(
                model=Config.QWEN_MODEL or "qwen-long",
                messages=messages,
                temperature=0.2,
                stream=True,
                max_tokens=None  # 不限制token数量
            )
            
            # 收集流式响应内容
            content = ""
            chunk_count = 0
            for chunk in response:
                chunk_count += 1
                if chunk.choices and chunk.choices[0].delta.content:
                    content += chunk.choices[0].delta.content
                # 每100个块记录一次
                if chunk_count % 100 == 0:
                    logging.info(f"收到响应块 #{chunk_count}，当前响应长度: {len(content)} 字符")
            
            logging.info(f"响应接收完成，共 {chunk_count} 个响应块，总长度: {len(content)} 字符")
            
            # 检查是否包含完成标志
            is_complete = check_extraction_complete(content)
            
            # 提取JSON部分
            json_text = extract_json_from_text(content)
            if json_text:
                try:
                    relations = json.loads(json_text)
                    if not isinstance(relations, list):
                        relations = [relations]  # 确保是列表格式
                    logging.info(f"成功从API响应提取 {len(relations)} 个关系")
                except json.JSONDecodeError as e:
                    logging.warning(f"JSON解析错误: {str(e)}，尝试清理后重新解析")
                    try:
                        clean_json = json_text.replace('```', '').strip()
                        clean_json = re.sub(r',\s*]', ']', clean_json)
                        clean_json = re.sub(r',\s*}', '}', clean_json)
                        relations = json.loads(clean_json)
                        if not isinstance(relations, list):
                            relations = [relations]  # 确保是列表格式
                        logging.info(f"清理后成功解析 {len(relations)} 个关系")
                    except Exception as clean_err:
                        logging.error(f"清理后JSON仍解析失败: {str(clean_err)}")
                        logging.error(f"原始JSON错误: {str(e)}")
                        logging.debug(f"问题JSON片段: {json_text[:100]}...{json_text[-100:] if len(json_text) > 100 else ''}")
                        relations = []
                
                # 验证关系格式并转换为标准格式
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
                
                # 合并之前的关系和新提取的关系
                final_relations = []
                if previous_relations:
                    # 使用已有关系的集合来检查重复
                    existing_relation_keys = set()
                    for rel in previous_relations:
                        key = (rel.get("from_entity", ""), rel.get("to_entity", ""), rel.get("relation_type", ""))
                        existing_relation_keys.add(key)
                        final_relations.append(rel)
                    # 添加不重复的新关系
                    for rel in valid_relations:
                        key = (rel.get("from_entity", ""), rel.get("to_entity", ""), rel.get("relation_type", ""))
                        if key not in existing_relation_keys:
                            final_relations.append(rel)
                            existing_relation_keys.add(key)
                    logging.info(f"合并后有 {len(final_relations)} 条关系，其中 {len(previous_relations)} 条来自之前的结果，{len(final_relations) - len(previous_relations)} 条是新提取的")
                else:
                    final_relations = valid_relations
                
                # 缓存结果
                with open(cache_file, 'w', encoding='utf-8') as f:
                    json.dump(final_relations, f, indent=2, ensure_ascii=False)
                logging.info(f"成功缓存 {len(final_relations)} 条关系到 {cache_file}")
                return final_relations
            else:
                logging.error("未能从API响应中提取有效的关系")
                return []
        except Exception as e:
            logging.error(f"调用API时出错: {str(e)}")
            logging.error(traceback.format_exc())
            return []
    except Exception as e:
        logging.error(f"提取演化关系时出错: {str(e)}")
        logging.error(traceback.format_exc())
        return []
