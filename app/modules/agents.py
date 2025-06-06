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
            
    # 默认为未完成
    return False




# 修改生成提取提示词的函数
def generate_entity_extraction_prompt(model_name="qwen", previous_entities=None):
    """
    生成用于实体提取的提示词，不再需要包含论文文本内容
    
    Args:
        model_name (str): 使用的模型名称
        previous_entities (list): 之前提取的实体列表，用于断点续传
    
    Returns:
        str: 实体提取的提示词
    """
    # 获取基础提示词
    base_prompt = generate_entity_extraction_prompt_base()
    
    # 添加特征信息
    features_prompt = generate_entity_extraction_prompt_with_features()
    
    # 组合提示词
    prompt = base_prompt + "\n\n" + features_prompt
    
    # 添加完成状态请求
    completion_request = """
最后，请明确告知我提取是否已完成，还是需要继续提取更多实体。请根据你对文本的分析，判断是否已经提取了所有可能的实体。

在JSON返回后，请单独一行写明"EXTRACTION_COMPLETE: true"（找不到任何实体请输出）或"EXTRACTION_COMPLETE: false"（如果有任何没有被抽取的实体存在或需要分段提取,请输出）。
"""
    prompt += completion_request

    # 如果有之前提取的实体，添加到提示中
    if previous_entities and len(previous_entities) > 0:
        entity_examples = []
        for i, entity in enumerate(previous_entities): 
            if "algorithm_entity" in entity:
                entity_type = "算法"
                entity_name = entity["algorithm_entity"].get("algorithm_id")
                entity_examples.append(f"{entity_name}")
            elif "dataset_entity" in entity:
                entity_type = "数据集"
                entity_name = entity["dataset_entity"].get("dataset_id")
                entity_examples.append(f"{entity_name}")
            elif "metric_entity" in entity:
                entity_type = "评价指标"
                entity_name = entity["metric_entity"].get("metric_id")
                entity_examples.append(f"{entity_name}")
        
        # 添加部分提取的上下文
        logging.info(f"已提取的实体: {entity_examples}")
        previous_entities_hint = "\n\n以下实体已经被提取过，请不要重复提取，并继续识别其他实体：\n- "
        previous_entities_hint += "\n- ".join(entity_examples)
        previous_entities_hint += "\n\n请确保你提取的是新实体，不要包含上述已提取的实体。"
        prompt += previous_entities_hint
    
    return prompt

def generate_entity_extraction_prompt_base():
    """
    生成实体提取的基础提示词，不包含特征信息
    
    Returns:
        str: 实体提取的基础提示词
    """
    return """
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

请以JSON格式输出，确保包含以下结构，不要包含任何其他内容：
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
                          "feature_processing": ["处理方法1", "处理方法2", ...]
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
                          "creators": ["LeCun, Y.", "Cortes, C.", ...]
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
                          "formula": "正确分类样本数/总样本数"
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

def generate_entity_extraction_prompt_with_features():
    """
    生成带有特征信息的实体提取提示词
    
    Returns:
        str: 带有特征信息的实体提取提示词
    """
    return """
识别特征
"prompt_features": {
    "entities": {
      "Algorithm": {
        "surface_forms": ["Proper Nouns", "Acronyms", "CamelCase", "ALL_CAPS", "Author's names with method names"],
        "cue_words": ["algorithm", "method", "approach", "model", "framework", "system", "architecture"],
        "sentence_patterns": ["We propose X", "X improves upon Y", "X is based on Y", "X extends Y", "X replaces Y"],
        "verbs": ["propose", "introduce", "extend", "improve", "outperform", "use", "apply", "develop"],
        "typical_location": ["Title", "Abstract", "Introduction", "Methodology", "Related Work", "Algorithm pseudocode", "Figures/Tables captions"],
        "formatting": ["Bold", "Italics", "Small caps"]
      },
      "Dataset": {
        "surface_forms": ["Proper nouns", "Acronyms", "Capitalized compound nouns", "Names with numeric suffixes"],
        "cue_words": ["dataset", "benchmark", "corpus", "data", "set"],
        "sentence_patterns": ["evaluated on X", "tested using X", "experiments on X", "trained on X"],
        "verbs": ["use", "evaluate", "test", "train", "benchmark"],
        "typical_location": ["Abstract", "Experimental Setup", "Evaluation", "Tables", "Figures captions", "Results section"],
        "formatting": ["Italics", "Quotes"]
      },
      "Metric": {
        "surface_forms": ["Common nouns", "Acronyms", "Numerical values", "% symbols"],
        "cue_words": ["metric", "score", "rate", "performance", "accuracy", "precision", "recall", "error", "F1"],
        "sentence_patterns": ["measured by X", "achieved X", "improved X by", "in terms of X"],
        "verbs": ["achieve", "improve", "optimize", "measure", "report"],
        "typical_location": ["Abstract", "Results", "Evaluation section", "Table headers", "Figures axes labels"],
        "formatting": ["Lowercase", "Uppercase acronyms"]
      },
      "Task": {
        "surface_forms": ["Gerund phrases", "Noun phrases", "Adjective + noun combinations"],
        "cue_words": ["task", "problem", "challenge", "objective", "goal"],
        "sentence_patterns": ["X addresses the problem of Y", "for the task of X", "tackling X", "solving X"],
        "verbs": ["address", "tackle", "solve", "focus", "target"],
        "typical_location": ["Title", "Abstract", "Introduction", "Experimental Setup", "Evaluation"],
        "formatting": ["Lowercase", "Sometimes capitalized if proper noun-based"]
      },
    "semantic_features": ["Comparative statements", "Improvement terms", "Performance descriptors", "Explicit relation verbs"]
    }
    }
}

实体结构示例:
{
  "algorithm_entity": {
    "algorithm_id": "Huang2017_NeuralSolver",
    "entity_type": "Algorithm",
    "name": "NeuralSolver",
    "title": "A Neural Solver for Math Word Problems",
    "year": 2017,
    "authors": ["Huang", "Zhang"],
    "task": ["Math Word Problem Solving"],
    "datasets": ["Math23K"],
    "metrics": ["Accuracy"],
    "architecture": {
      "components": ["Encoder", "Decoder"],
      "connections": ["Attention"],
      "mechanisms": ["Gated Recurrent Unit"]
    },
    "methodology": {
      "training_strategy": ["CrossEntropyLoss", "Normalization"],
      "parameter_tuning": ["Adam", "Dropout"]
    },
    "feature_processing": ["Tokenization", "Stopword Removal"]
  }
}
"""

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



# 创建一个通用的实体提取函数，替代多个API特定的函数
def extract_entities_with_model(pdf_paths, model_name="qwen", max_attempts=5, previous_entities=None, agent=None):
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
        
    
    all_entities = []  # 所有已提取的实体
    current_attempt = 0
    is_extraction_complete = False
    
    while current_attempt < max_attempts and not is_extraction_complete:
        current_attempt += 1
        logging.info(f"提取尝试 {current_attempt}/{max_attempts},是否完成: {is_extraction_complete}")
        
        # 如果已有提取的实体，将其加入到提示词中
        current_prompt = generate_entity_extraction_prompt(
        model_name, 
        previous_entities=all_entities, 
    )
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
            logging.info(f"提取到的文本长度: {len(content)}")
            json_text = extract_json_from_text(content)
            logging.info(f"提取到的JSON文本长度: {len(json_text)}")
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
                        logging.info(f"已存在的实体: {entity_id}")
                        # 更新已存在的实体
                        continue
                    else:
                        # 添加新实体
                        all_entities.append(new_entity)
            else:
                # 如果是首次提取，直接使用结果
                all_entities = entities
            previous_entities = all_entities
            # 如果提取已完成
            if is_complete:
                is_extraction_complete = True
                break
                
        except Exception as e:
            logging.error(f"提取实体时出错: {str(e)}")
            logging.error(traceback.format_exc())
            # 继续下一次尝试
    
    logging.info(f"完成实体提取，共 {len(all_entities)} 个实体，完成状态: {is_extraction_complete}")
    return all_entities, is_extraction_complete

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
        
    logging.info(f"开始从文本中提取JSON，文本长度：{len(text)} 字符")
    
    # 首先尝试从代码块中提取JSON
    json_block_pattern = r'```(?:json)?\s*([\s\S]*?)(?:\s*```|\s*EXTRACTION_COMPLETE\s*:)'
    matches = re.findall(json_block_pattern, text)
    if matches:
        json_candidate = matches[0].strip()
        # 确保JSON以合适的结尾符号结束
        if json_candidate.endswith(','):
            json_candidate = json_candidate[:-1]
        if json_candidate.endswith(']') or json_candidate.endswith('}'):
            logging.info(f"从代码块提取到可能的JSON，长度: {len(json_candidate)}")
            # 验证JSON是否有效
            try:
                json.loads(json_candidate)
                # 检查提取后内容长度是否明显小于原文本
                if len(json_candidate) < len(text) - 500:
                    # 保存原文本和提取后的内容到缓存目录
                    os.makedirs("data/cache/test", exist_ok=True)
                    timestamp = int(time.time())
                    with open(f"data/cache/test/original_text_{timestamp}.txt", "w", encoding="utf-8") as f:
                        f.write(text)
                    with open(f"data/cache/test/extracted_json_{timestamp}.json", "w", encoding="utf-8") as f:
                        f.write(json_candidate)
                    logging.info(f"已保存原文本和提取JSON到data/cache/test目录，时间戳: {timestamp}")
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
                logging.info(f"提取到有效的JSON数组，长度: {len(match)}")
                # 检查提取后内容长度是否明显小于原文本
                if len(match) < len(text) - 500:
                    # 保存原文本和提取后的内容到缓存目录
                    os.makedirs("data/cache/test", exist_ok=True)
                    timestamp = int(time.time())
                    with open(f"data/cache/test/original_text_{timestamp}.txt", "w", encoding="utf-8") as f:
                        f.write(text)
                    with open(f"data/cache/test/extracted_json_{timestamp}.json", "w", encoding="utf-8") as f:
                        f.write(match)
                    logging.info(f"已保存原文本和提取JSON到data/cache/test目录，时间戳: {timestamp}")
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
                if len(match) < len(text) - 500:
                    # 保存原文本和提取后的内容到缓存目录
                    os.makedirs("data/cache/test", exist_ok=True)
                    timestamp = int(time.time())
                    with open(f"data/cache/test/original_text_{timestamp}.txt", "w", encoding="utf-8") as f:
                        f.write(text)
                    with open(f"data/cache/test/extracted_json_{timestamp}.json", "w", encoding="utf-8") as f:
                        f.write(match)
                    logging.info(f"已保存原文本和提取JSON到data/cache/test目录，时间戳: {timestamp}")
                return match
                logging.info(f"提取到有效的JSON对象，长度: {len(match)}")
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
                    if len(potential_json) < len(text) - 500:
                        # 保存原文本和提取后的内容到缓存目录
                        os.makedirs("data/cache/test", exist_ok=True)
                        timestamp = int(time.time())
                        with open(f"data/cache/test/original_text_{timestamp}.txt", "w", encoding="utf-8") as f:
                            f.write(text)
                        with open(f"data/cache/test/extracted_json_{timestamp}.json", "w", encoding="utf-8") as f:
                            f.write(potential_json)
                        logging.info(f"已保存原文本和提取JSON到data/cache/test目录，时间戳: {timestamp}")
                    logging.info(f"使用索引方法提取到有效的JSON，长度: {len(potential_json)}")
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
                if len(cleaned_text) < len(text) - 500:
                    # 保存原文本和提取后的内容到缓存目录
                    os.makedirs("data/cache/test", exist_ok=True)
                    timestamp = int(time.time())
                    with open(f"data/cache/test/original_text_{timestamp}.txt", "w", encoding="utf-8") as f:
                        f.write(text)
                    with open(f"data/cache/test/extracted_json_{timestamp}.json", "w", encoding="utf-8") as f:
                        f.write(cleaned_text)
                    logging.info(f"已保存原文本和提取JSON到data/cache/test目录，时间戳: {timestamp}")
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
        # 使用文件名、大小生成唯一键
        cache_key_parts.append(f"{filename}_{file_size}")
    
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
    all_entities = []
    is_complete = True
    
    # 统一分批处理，无论batch_size为多少
    for i in range(0, len(valid_paths), batch_size):
        batch_paths = valid_paths[i:i+batch_size]
        logging.info(f"处理批次 {i//batch_size + 1}/{(len(valid_paths) + batch_size - 1)//batch_size}，包含 {len(batch_paths)} 个PDF文件")
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

def extract_evolution_relations(entities, pdf_paths=None, task_id=None, previous_relations=None, max_attempts=5, batch_size=100):
    """
    从实体列表和PDF文件中提取演化关系，支持多个PDF文件和file-id方式，支持批量处理
    
    Args:
        entities (list): 实体列表
        pdf_paths (str/list, optional): PDF文件路径或路径列表
        task_id (str, optional): 任务ID，用于缓存和日志
        previous_relations (list, optional): 之前提取的关系，用于断点续传
        max_attempts (int): 最大尝试次数
        batch_size (int): 批处理大小，每次处理的PDF文件数量
        
    Returns:
        list: 提取的演化关系列表
    """
    try:
        # 转换单个路径为列表
        if isinstance(pdf_paths, str):
            pdf_paths = [pdf_paths]
            
        if pdf_paths:
            logging.info(f"从 {len(pdf_paths)} 个PDF文件中提取演化关系，批处理大小: {batch_size}")
        else:
            logging.info("没有提供PDF文件，仅基于实体列表分析演化关系")
            
        # 验证实体列表
        if not entities or not isinstance(entities, list) or len(entities) == 0:
            logging.error("没有提供实体或实体列表为空，无法提取关系")
            return []
            
        logging.info(f"基于 {len(entities)} 个实体提取演化关系" + 
                    (f"，已有 {len(previous_relations)} 个之前的关系" if previous_relations else ""))
        
        # 初始化变量
        all_relations = [] if not previous_relations else previous_relations.copy()
        is_extraction_complete = False
        
        # 如果有PDF文件，则分批处理
        if pdf_paths and len(pdf_paths) > 0:
            # 统一分批处理，无论batch_size为多少
            for i in range(0, len(pdf_paths), batch_size):
                batch_paths = pdf_paths[i:i+batch_size]
                logging.info(f"处理批次 {i//batch_size + 1}/{(len(pdf_paths) + batch_size - 1)//batch_size}，包含 {len(batch_paths)} 个PDF文件")
                
                # 为每个批次单独处理
                batch_relations, batch_complete = _process_relations_batch(
                    entities=entities,
                    pdf_paths=batch_paths,
                    previous_relations=all_relations,
                    max_attempts=max_attempts
                )
                
                # 合并关系并避免重复
                if batch_relations:
                    # 更新所有关系列表，去重
                    existing_ids = {f"{rel.get('from_entity')}_{rel.get('to_entity')}_{rel.get('relation_type')}" 
                                   for rel in all_relations if rel}
                    
                    for rel in batch_relations:
                        if rel:
                            rel_id = f"{rel.get('from_entity')}_{rel.get('to_entity')}_{rel.get('relation_type')}"
                            if rel_id not in existing_ids:
                                all_relations.append(rel)
                                existing_ids.add(rel_id)
                
                is_extraction_complete = batch_complete
                
                logging.info(f"批次 {i//batch_size + 1} 完成，提取了 {len(batch_relations)} 个关系，累计 {len(all_relations)} 个关系，完成状态: {batch_complete}")
                
                # 如果提取尚未完成且有更多批次，记录状态
                if not is_extraction_complete and i + batch_size < len(pdf_paths):
                    logging.warning(f"批次 {i//batch_size + 1} 提取未完成，将继续下一批次")
        else:
            # 如果没有PDF文件，直接处理实体列表
            all_relations, is_extraction_complete = _process_relations_batch(
                entities=entities,
                pdf_paths=None,
                previous_relations=all_relations,
                max_attempts=max_attempts
            )
        
        logging.info(f"关系提取完成，共 {len(all_relations)} 个关系，完成状态: {is_extraction_complete}")
        return all_relations
        
    except Exception as e:
        logging.error(f"提取演化关系时出错: {str(e)}")
        import traceback
        logging.error(traceback.format_exc())
        return all_relations if 'all_relations' in locals() else []

def _process_relations_batch(entities, pdf_paths=None, previous_relations=None, max_attempts=5):
    """
    处理一批PDF文件中的关系
    
    Args:
        entities (list): 实体列表
        pdf_paths (list, optional): PDF文件路径列表
        previous_relations (list, optional): 之前提取的关系，用于断点续传
        max_attempts (int): 最大尝试次数
        
    Returns:
        tuple: (提取的关系列表, 完成状态)
    """
    # 初始化变量
    all_relations = [] if not previous_relations else previous_relations.copy()
    current_attempt = 0
    is_extraction_complete = False
    initial_relation_count = len(all_relations)
    consecutive_no_new_relation = 0  # 跟踪连续未找到新关系的次数

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
    
    # 创建临时文件保存所有实体
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8') as temp_file:
        temp_filename = temp_file.name
        # 将所有实体以JSON格式写入TXT文件
        json_content = json.dumps(entities, ensure_ascii=False, indent=2)
        temp_file.write(json_content)
        logging.info(f"已将 {len(entities)} 个实体以JSON格式写入TXT文件: {temp_filename}")
    
    # 上传实体文件并获取file-id
    entities_file_id = upload_and_cache_file(temp_filename, purpose="file-extract")
    
    # 删除临时文件
    try:
        os.unlink(temp_filename)
    except Exception as e:
        logging.warning(f"删除临时文件时出错: {str(e)}")
    
    if not entities_file_id:
        logging.error("上传实体文件失败，无法获取file-id")
        return all_relations, False
    
    # 添加file-id引用（包括PDF文件和实体文件）
    all_file_ids = []
    if file_ids:
        all_file_ids.extend(file_ids)
    all_file_ids.append(entities_file_id)
    
    # 添加重试循环结构
    while current_attempt < max_attempts and not is_extraction_complete:
        current_attempt += 1
        logging.info(f"关系提取尝试 {current_attempt}/{max_attempts}")
        
        # 获取当前关系数量
        current_relation_count = len(all_relations)
        
        # 生成关系提取提示
        system_message, user_message = generate_evolution_relation_prompt(all_relations)
        
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
            file_content = ",".join([f"fileid://{fid}" for fid in all_file_ids])
            messages.append({"role": "system", "content": file_content})
            
            # 添加用户提示
            messages.append({"role": "user", "content": user_message})
            
            # 调用API
            logging.info(f"调用千问API提取关系，包含 {len(file_ids) if file_ids else 0} 个PDF文件和 1 个JSON内容的TXT文件(ID: {entities_file_id})，共 {len(all_file_ids)} 个文件")
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
                    logging.info(f"成功从API响应提取 {len(relations)} 个关系,是否完成: {is_complete}")
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
                    
                    # 清理从_entities和to_entities中的relation_type
                    if 'from_entities' in relation and isinstance(relation['from_entities'], list):
                        for entity in relation['from_entities']:
                            if 'relation_type' in entity:
                                del entity['relation_type']
                    
                    if 'to_entities' in relation and isinstance(relation['to_entities'], list):
                        for entity in relation['to_entities']:
                            if 'relation_type' in entity:
                                del entity['relation_type']
                    
                    # 验证必要字段
                    if 'from_entity' not in relation and 'from_entities' not in relation:
                        continue
                    if 'to_entity' not in relation and 'to_entities' not in relation:
                        continue
                    if 'relation_type' not in relation:
                        continue
                    
                    # 清理后加入有效关系
                    valid_relations.append(relation)
                
                # 添加到全部关系中，并追踪新增的关系
                new_relations_added = 0
                if valid_relations:
                    # 创建已有关系ID集合
                    existing_ids = {f"{rel.get('from_entity')}_{rel.get('to_entity')}_{rel.get('relation_type')}" 
                                   for rel in all_relations if rel}
                    
                    for rel in valid_relations:
                        rel_id = f"{rel.get('from_entity')}_{rel.get('to_entity')}_{rel.get('relation_type')}"
                        if rel_id not in existing_ids:
                            all_relations.append(rel)
                            existing_ids.add(rel_id)
                            new_relations_added += 1
                    
                    logging.info(f"已将 {new_relations_added} 个新关系添加到结果中，当前总关系数: {len(all_relations)}")
                
                # 判断是否需要继续提取
                if new_relations_added == 0:
                    consecutive_no_new_relation += 1
                    logging.warning(f"连续 {consecutive_no_new_relation} 次未提取到新关系")
                else:
                    consecutive_no_new_relation = 0
                
                # 如果连续两次没有提取到新关系，或者显式标记为完成，则结束提取
                if consecutive_no_new_relation >= 2 or is_complete:
                    is_extraction_complete = True
                    logging.info(f"{'模型明确标记提取完成' if is_complete else '连续多次未提取到新关系'}，结束提取")
                    break
            else:
                logging.warning("未能从API响应中提取JSON内容")
                # 如果是最后一次尝试，记录为提取未完成
                if current_attempt >= max_attempts:
                    logging.warning("达到最大尝试次数，但未能提取到有效关系")
                    is_extraction_complete = False
        
        except Exception as e:
            logging.error(f"调用API提取关系时出错: {str(e)}")
            import traceback
            logging.error(traceback.format_exc())
            # 如果是最后一次尝试，则将is_complete标记为False
            if current_attempt >= max_attempts:
                is_extraction_complete = False
    
    # 最终检查是否有新增关系
    if len(all_relations) > initial_relation_count:
        logging.info(f"总共提取了 {len(all_relations) - initial_relation_count} 个新关系，总计 {len(all_relations)} 个关系")
    else:
        logging.warning("未能提取到任何新关系")
        
    return all_relations, is_extraction_complete

def generate_evolution_relation_prompt(previous_relations=None, entities=None, is_complete=None):
    """
    生成提取演化关系的提示词
    
    Args:
        previous_relations (list, optional): 之前已提取的演化关系列表，默认为None。
        entities (list, optional): 实体列表，默认为None。
        is_complete (bool, optional): 是否完成所有关系提取，默认为None。
        
    Returns:
        tuple: (系统消息, 用户消息)
    """
    # 系统消息
    system_message = "你是一个专注于从学术论文和实体列表中提取演化关系的AI助手，能够理解算法之间的改进、扩展、替换等关系，以及算法与数据集、评价指标之间的关系。"
    
    # 用户消息
    user_message = generate_relation_extraction_prompt_base()
    
    # 添加特征信息
    feature_message = generate_relation_extraction_prompt_with_features()
    user_message += "\n\n" + feature_message
    
    # 添加已有关系信息
    if previous_relations and len(previous_relations) > 0:
        # 创建简化版的关系列表（只包含实体ID和关系类型）
        simplified_relations = []
        for rel in previous_relations:
            simplified_relations.append({
                "from_entity": rel.get("from_entity"),
                "to_entity": rel.get("to_entity"),
                "relation_type": rel.get("relation_type")
            })
        
        # 转换为JSON格式
        relations_json = json.dumps(simplified_relations, ensure_ascii=False)
        
        user_message += f"\n\n我们已经提取了 {len(previous_relations)} 个关系。以下是所有已知关系的简化列表（仅包含实体ID和关系类型）:\n{relations_json}"
        user_message += f"\n\n重要：请勿重复提取上述任何关系组合，而是发掘新的、尚未提取的关系。"
        user_message += f"\n特别是考虑那些由不同来源引用的实体之间可能存在的关系，或者不同任务领域之间的跨领域关系。"
        
        # 创建已处理实体的集合，便于统计
        processed_entities = set()
        for rel in previous_relations:
            if rel.get('from_entity'):
                processed_entities.add(rel.get('from_entity'))
            if rel.get('to_entity'):
                processed_entities.add(rel.get('to_entity'))
        
        user_message += f"\n\n目前已涉及 {len(processed_entities)} 个实体，请尝试发现更多实体之间的关系，尤其是尚未处理过的实体。"
    
    # 提取完成度请求
    user_message += f"\n\n如果你认为所有有意义的关系都已经提取完毕，请在JSON响应后单独一行写明：EXTRACTION_COMPLETE: true，否则写EXTRACTION_COMPLETE: false"
    
    return system_message, user_message

def generate_relation_extraction_prompt_base():
    """
    生成基础关系提取提示词
    
    Returns:
        str: 基础关系提取提示词
    """
    return """请分析已上传的实体文件和PDF论文内容，识别算法之间的演化关系。

重要提示：
1. 所有实体信息都在已上传的文本文件中，内容是JSON格式，包含算法、数据集和评价指标实体
2. 请不要创建文件中不存在的实体，必须严格使用文件中提供的实体ID
3. 对于每个关系，请提供它解决的问题（problem_addressed字段）

实体之间的演化关系可以是：
- 算法改进/优化/扩展/替换另一个算法
- 算法使用特定数据集
- 算法使用特定评估指标
- 数据集改进/扩展另一个数据集
- 评估指标改进/扩展另一个评估指标

请以以下格式返回关系列表，不要包含任何其他内容：
[
  {
    "from_entity": "实体A的ID",
    "to_entity": "实体B的ID",
    "relation_type": "主要关系类型：Improve/Optimize/Extend/Replace/Use",
    "structure": "关系的结构描述",
    "detail": "关系的详细说明",
    "problem_addressed": "该关系解决的问题",
    "evidence": "支持这种关系判断的证据（引用论文中的文字）",
    "confidence": 置信度（0.0-1.0）
  }
]

关系结构示例:
{
  "from_entity": "Zhang2016_TemplateSolver",
  "from_entity_type": "Algorithm",
  "to_entity": "Huang2017_NeuralSolver",
  "to_entity_type": "Algorithm",
  "relation_type": "Replace",
  "structure": "Architecture.Mechanism",
  "detail": "Template-based parser replaced by neural encoder-decoder",
  "problem_addressed": "Low adaptability of rule-based template parser",
  "evidence": "Instead of using a rule-based parser (Zhang et al. 2016), we adopt a neural sequence-to-sequence architecture.",
  "confidence": 0.91
}"""

def generate_relation_extraction_prompt_with_features():
    """
    生成带有特征信息的关系提取提示词
    
    Returns:
        str: 带有特征信息的关系提取提示词
    """
    return """请分析已上传的实体文件和PDF论文内容，识别算法之间的演化关系。

重要提示：
1. 所有实体信息都在已上传的文本文件中，内容是JSON格式，包含算法、数据集和评价指标实体
2. 请不要创建文件中不存在的实体，必须严格使用文件中提供的实体ID
3. 如果实体文件中有数百个实体，请尽可能全面分析它们之间的关系
4. 对于每个关系，请提供它解决的问题（problem_addressed字段）

实体之间的演化关系可以是：
- 算法改进/优化/扩展/替换另一个算法
- 算法使用特定数据集
- 算法使用特定评估指标
- 数据集改进/扩展另一个数据集
- 评估指标改进/扩展另一个评估指标

识别特征：
"relation_patterns": {
      "Improvement": ["improves upon", "outperforms", "better than", "surpasses", "exceeds"],
      "Usage": ["use", "apply", "employ", "adopt"],
      "Extension": ["extends", "generalizes", "based on", "builds on"],
      "Evaluation": ["evaluated on", "tested using", "in terms of", "measured by", "achieved"],
      "Comparison": ["compared to", "unlike", "versus", "whereas"]
    },
    "structural_features": ["Sentence-level context", "Paragraph context", "Citation contexts", "Section headings", "Figure/Table captions", "Pseudocode blocks"]

请以以下格式返回关系列表，不要包含任何其他内容：
[
  {
    "from_entity": "实体A的ID",
    "to_entity": "实体B的ID",
    "relation_type": "主要关系类型：Improve/Optimize/Extend/Replace/Use",
    "structure": "关系的结构描述",
    "detail": "关系的详细说明",
    "problem_addressed": "该关系解决的问题",
    "evidence": "支持这种关系判断的证据（引用论文中的文字）",
    "confidence": 置信度（0.0-1.0）
  }
]

关系结构示例:
{
  "from_entity": "Zhang2016_TemplateSolver",
  "from_entity_type": "Algorithm",
  "to_entity": "Huang2017_NeuralSolver",
  "to_entity_type": "Algorithm",
  "relation_type": "Replace",
  "structure": "Architecture.Mechanism",
  "detail": "Template-based parser replaced by neural encoder-decoder",
  "problem_addressed": "Low adaptability of rule-based template parser",
  "evidence": "Instead of using a rule-based parser (Zhang et al. 2016), we adopt a neural sequence-to-sequence architecture.",
  "confidence": 0.91
}"""

def validate_and_add_relation(db_manager, relation_data, task_id):
    """
    验证并添加关系
    
    Args:
        db_manager (DBManager): 数据库管理器
        relation_data (dict): 关系数据
        task_id (str): 任务ID
        
    Returns:
        tuple: (成功状态, 消息)
    """
    required_fields = ['from_entity', 'to_entity', 'relation_type', 'structure', 'detail', 'evidence', 'confidence']
    for field in required_fields:
        if field not in relation_data:
            return False, f"关系缺少必要字段：{field}"
    
    if not isinstance(relation_data.get('confidence'), (int, float)):
        try:
            relation_data['confidence'] = float(relation_data.get('confidence', 0))
        except:
            relation_data['confidence'] = 0.5
    
    # 检查实体是否存在
    from_entity = db_manager.get_entity_by_id(relation_data['from_entity'])
    to_entity = db_manager.get_entity_by_id(relation_data['to_entity'])
    
    if not from_entity:
        return False, f"源实体不存在：{relation_data['from_entity']}"
    if not to_entity:
        return False, f"目标实体不存在：{relation_data['to_entity']}"
    
    # 添加问题解决字段
    if 'problem_addressed' not in relation_data:
        relation_data['problem_addressed'] = ""
    
    # 添加关系
    success = db_manager.store_algorithm_relation(
        from_entity=relation_data['from_entity'],
        to_entity=relation_data['to_entity'],
        relation_type=relation_data['relation_type'],
        structure=relation_data['structure'],
        detail=relation_data['detail'],
        evidence=relation_data['evidence'],
        confidence=relation_data['confidence'],
        task_id=task_id,
        problem_addressed=relation_data.get('problem_addressed', "")
    )
    
    if success:
        return True, "关系添加成功"
    else:
        return False, "关系添加失败"
