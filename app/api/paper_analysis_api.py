# 论文分析API接口
# 文件：app/api/paper_analysis_api.py

from flask import Blueprint, request, jsonify
import uuid
import os
import json
import logging
import threading
import base64
import re
from datetime import datetime
from typing import Optional, Dict, Any
from werkzeug.utils import secure_filename
from openai import OpenAI
from app.config import Config
from app.modules.db_manager import db_manager
from app.modules.debug_logger import DebugLogger

# 支持的模型列表（基于现有配置）
SUPPORTED_MODELS = [
    'qwen-long',                    # 默认模型
    'gemini-2.0-flash',            # 支持PDF分析
    'claude-3-7-sonnet-20250219',  # 推理能力强
    'gpt-3.5-turbo',               # 通用模型
    'gpt-4.1-mini',                # 多模态支持
    'deepseek-v3'                  # 代码和推理
]

# 创建蓝图
paper_analysis_api = Blueprint('paper_analysis_api', __name__)

def save_extracted_results(analysis_id: str, extracted_data: dict, analysis_type: str):
    """保存提取结果到文件，供后续复用"""
    try:
        # 创建保存目录
        save_dir = os.path.join(Config.UPLOAD_DIR, "extracted_results")
        os.makedirs(save_dir, exist_ok=True)

        # 生成文件名
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{analysis_id}_{analysis_type}_{timestamp}.json"
        file_path = os.path.join(save_dir, filename)

        # 添加元数据
        result_data = {
            "analysis_id": analysis_id,
            "analysis_type": analysis_type,
            "timestamp": datetime.now().isoformat(),
            "extracted_data": extracted_data
        }

        # 保存到文件
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(result_data, f, ensure_ascii=False, indent=2)

        logging.info(f"提取结果已保存到: {file_path}")
        return file_path

    except Exception as e:
        logging.error(f"保存提取结果失败: {str(e)}")
        return None

# ==================== 工具函数 ====================

def call_llm_with_prompt(prompt: str, model_name: str = None, pdf_path: Optional[str] = None) -> str:
    """通用大模型调用函数，基于现有Config配置"""
    if model_name is None:
        model_name = Config.DEFAULT_MODEL

    try:
        client, actual_model = _get_model_client_from_config(model_name)

        messages = [
            {"role": "system", "content": "You are a professional academic analysis assistant."}
        ]

        # 根据不同模型使用不同的PDF处理方式
        if pdf_path and os.path.exists(pdf_path):
            if model_name == "qwen-long":
                # qwen-long使用file-id方式
                from app.modules.agents import upload_and_cache_file
                file_id = upload_and_cache_file(pdf_path, model_name=model_name)
                if file_id:
                    # 使用file-id方式
                    messages = [
                        {"role": "system", "content": "请分析文件内容并回答问题。"},
                        {"role": "system", "content": f"fileid://{file_id}"},
                        {"role": "user", "content": prompt}
                    ]
                else:
                    # 文件上传失败，使用纯文本
                    messages.append({"role": "user", "content": prompt})
            else:
                # 其他模型使用文本方式
                from app.modules.agents import extract_text_from_pdf
                pdf_text = extract_text_from_pdf(pdf_path)
                if pdf_text:
                    # 根据模型限制文本长度
                    if model_name == "deepseek-v3" or model_name.startswith("claude") or model_name == "gemini-2.0-flash":
                        # DeepSeek 和 Claude 截取前6万个字符
                        pdf_text = pdf_text[:50000]

                    combined_prompt = f"{prompt}\n\nPDF文件内容:\n\n{pdf_text}"
                    messages.append({"role": "user", "content": combined_prompt})
                else:
                    # 文本提取失败，跳过PDF处理
                    logging.warning(f"PDF文本提取失败: {pdf_path}")
                    messages.append({"role": "user", "content": prompt})
        else:
            messages.append({"role": "user", "content": prompt})

        logging.info(f"调用{model_name}模型进行分析")

        for attempt in range(Config.MAX_RETRY_ATTEMPTS):
            try:
                response = client.chat.completions.create(
                    model=actual_model,
                    messages=messages,
                    temperature=0.1,
                    stream=True
                )

                content = ""
                for chunk in response:
                    if chunk.choices and chunk.choices[0].delta.content:
                        content += chunk.choices[0].delta.content

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
        logging.warning(f"未知模型 {model_name}，使用默认模型 {Config.DEFAULT_MODEL}")
        client = OpenAI(
            api_key=Config.QWEN_API_KEY,
            base_url=Config.QWEN_BASE_URL
        )
        return client, Config.DEFAULT_MODEL

def extract_json_from_response(response_text: str) -> Dict[str, Any]:
    """从响应文本中提取JSON，使用json_repair工具"""

    # 记录原始响应用于调试
    logging.info(f"原始响应长度: {len(response_text)} 字符")
    logging.debug(f"原始响应前200字符: {response_text[:200]}")

    try:
        # 使用现有的extract_json_from_text函数（来自agents.py）
        from app.modules.agents import extract_json_from_text

        # 提取并修复JSON
        repaired_json_str = extract_json_from_text(response_text)

        if repaired_json_str:
            # 解析修复后的JSON
            result = json.loads(repaired_json_str)

            if result and isinstance(result, dict):
                logging.info("成功使用json_repair工具提取JSON")
                return result
            else:
                logging.warning("json_repair返回的不是有效的字典对象")
        else:
            logging.warning("json_repair未能提取到JSON内容")

    except Exception as e:
        logging.error(f"使用json_repair提取JSON失败: {str(e)}")

    # 如果json_repair失败，尝试备用方法
    logging.info("尝试备用JSON提取方法...")

    try:
        # 备用方法1: 直接解析
        result = json.loads(response_text.strip())
        if result and isinstance(result, dict):
            logging.info("成功使用直接解析方法")
            return result
    except:
        pass

    try:
        # 备用方法2: 提取大括号内容
        start = response_text.find('{')
        end = response_text.rfind('}')

        if start != -1 and end != -1 and end > start:
            json_str = response_text[start:end+1]
            result = json.loads(json_str)
            if result and isinstance(result, dict):
                logging.info("成功使用大括号提取方法")
                return result
    except:
        pass

    try:
        # 备用方法3: 提取代码块
        import re
        match = re.search(r'```(?:json)?\s*(.*?)\s*```', response_text, re.DOTALL)
        if match:
            json_str = match.group(1).strip()
            result = json.loads(json_str)
            if result and isinstance(result, dict):
                logging.info("成功使用代码块提取方法")
                return result
    except:
        pass

    # 所有方法都失败，记录详细错误信息
    logging.error(f"所有JSON提取方法都失败，响应内容: {response_text[:500]}...")
    raise ValueError(f"无法从响应中提取有效的JSON。响应开头: {response_text[:100]}...")

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

def cache_analysis_result(analysis_id: str, result_data: dict) -> str:
    """将分析结果缓存到配置指定的目录"""
    os.makedirs(Config.CACHE_DIR, exist_ok=True)
    cache_file = os.path.join(Config.CACHE_DIR, f"paper_analysis_{analysis_id}.json")

    with open(cache_file, 'w', encoding='utf-8') as f:
        json.dump(result_data, f, ensure_ascii=False, indent=2)

    logging.info(f"分析结果已缓存到: {cache_file}")
    return cache_file

# ==================== 提示词模板 ====================

def get_method_extraction_prompt():
    """获取方法提取提示词（专业实体提取标准）"""
    return """
你是一位专业的学术论文分析专家。请仔细阅读论文内容，按照学术标准提取研究方法和算法实体。

【提取标准】：
1. **方法（Methods）**：抽象的研究方法论，如Deep Learning Methods, Statistical Learning Methods, Supervised Learning Methods等
2. **算法（Algorithms）**：具体的、明确的算法实体，如BERT、ResNet、LSTM、CNN、SVM、Random Forest等

【算法提取要求】：
- 只提取具体的、有明确名称的算法
- 包括：神经网络架构（BERT、ResNet、LSTM）、机器学习算法（SVM、Random Forest）、优化算法（Adam、SGD）
- 排除：抽象概念（如"梯度下降"、"反向传播"）、通用术语（如"深度学习"、"机器学习"）
- 特别注意：带有大写或代表性名称的算法，表格内或加粗的算法名称

【重点关注】：
1. 论文标题、摘要、方法章节中的具体算法名称
2. 实验部分对比的基线算法
3. 表格中列出的算法名称
4. 引用文献中提到的经典算法

【输出要求】：
- 严格按照JSON格式输出
- 所有内容必须为英文
- 方法名称要体现研究方法论的抽象性
- 算法名称必须是具体的、标准的算法术语

JSON格式：
{
  "methods": [
    {
      "name": "研究方法名称（英文，抽象层面，如：Deep Learning Methods, Convolutional Neural Network Methods等）",
      "section": "对应章节号",
      "description": "方法的简要描述（英文）",
      "algorithms": [
        {
          "name": "具体算法名称（英文，如：BERT, ResNet-50, LSTM, SVM, Adam等）",
          "description": "算法的简要描述和应用场景（英文）",
          "context": "在论文中的具体应用上下文（英文）"
        }
      ]
    }
  ]
}

请开始分析论文内容：
"""

def get_coverage_analysis_prompt_safe(reference_data_str: str, comparison_data_str: str) -> str:
    """获取智能覆盖率分析提示词（基于语义理解的智能匹配）"""

    # 限制数据长度以避免token超限（增加长度确保数据完整性）
    max_ref_length = 1115000  # 增加到15000字符
    max_comp_length = 1112000  # 增加到12000字符

    if len(reference_data_str) > max_ref_length:
        reference_data_str = reference_data_str[:max_ref_length] + "...[数据截断]"
        logging.warning(f"引文数据过长，已截断到 {max_ref_length} 字符")

    if len(comparison_data_str) > max_comp_length:
        comparison_data_str = comparison_data_str[:max_comp_length] + "...[数据截断]"
        logging.warning(f"论文数据过长，已截断到 {max_comp_length} 字符")

    return f"""
你是一个专业的学术内容智能分析助手。请基于论文中提取的研究方法，智能地从引文数据中识别和匹配对应的方法，计算覆盖率。

【核心任务】：
不是简单的名称匹配，而是基于论文方法的语义内容，智能识别引文数据中的对应技术和方法。

【严格匹配策略】：
1. **方法匹配（严格标准）**：
   - 只匹配技术名称高度相似的方法
   - 避免过度泛化，如"Supervised Learning"不应匹配"Neural Networks"
   - 要求技术本质和应用场景都相近
   - 示例：Deep Learning Methods ↔ Deep Learning Methods（名称相近）

2. **算法匹配（精确标准）**：
   - 只匹配具体的、明确的算法名称
   - 严格的名称匹配，避免功能性匹配
   - 直接匹配：完全相同的算法名称（如LSTM↔LSTM、Dropout↔Dropout）
   - 变体匹配：同一算法的明确变体（如CNN↔Convolutional Neural Network）
   - 严格排除：不将抽象概念匹配具体算法

【严格原则】：
- 名称相似性优先：优先考虑名称的直接相似性
- 避免功能性匹配：不因为功能相关就匹配
- 提高匹配门槛：宁可漏掉也不要错误匹配
- 真实反映覆盖：覆盖率应该反映真实的重叠程度

【分析数据】：
引文数据（技术资源库）：
{reference_data_str}

论文方法（待分析内容）：
{comparison_data_str}

【严格分析要求】：
1. 对于论文中的每个方法，严格地在引文数据中寻找名称相似的技术
2. 优先进行名称匹配，避免过度的语义解释
3. 只考虑直接的技术对应关系，不进行远距离的演进推理
4. 避免跨领域的强行关联，保持匹配的准确性
5. 提供保守但准确的覆盖率计算，宁可低估也不高估

【重要计算要求】：
请仔细统计数据中的实际数量，确保以下数值准确：
- total_reference_methods = 引文数据中的方法总数（仔细计数）
- total_paper_methods = 论文数据中的方法总数（仔细计数）
- total_reference_algorithms = 引文数据中的算法总数（仔细计数）
- total_paper_algorithms = 论文数据中的算法总数（仔细计数）

【覆盖率计算公式】：
- method_coverage_ratio = intelligently_matched_methods / total_paper_methods
- algorithm_coverage_ratio = intelligently_matched_algorithms / total_paper_algorithms

注意：分母必须使用论文数据的总数，分子是智能匹配的数量

【严格匹配类型定义】：
1. **direct**: 完全相同的名称
   - "Dropout" ↔ "Dropout"
   - "LSTM" ↔ "LSTM"

2. **variant**: 同一技术的不同表达方式
   - "CNN" ↔ "Convolutional Neural Network"
   - "RNN" ↔ "Recurrent Neural Network"

3. **similar_name**: 名称高度相似的技术
   - "Deep Learning" ↔ "Deep Learning Methods"
   - "Neural Networks" ↔ "Neural Network Methods"

❌ 严格禁止的匹配：
- 功能相关但名称不同的技术
- 抽象概念与具体算法的匹配
- 跨领域的强行关联

【输出要求】：
- 严格按照JSON格式输出
- 所有内容必须为英文
- 提供保守但准确的匹配说明
- 计算基于严格标准的覆盖率
- 确保所有数量统计准确无误
- 宁可低估覆盖率也不要高估

【输出格式】：
{{
  "method_coverage": {{
    "total_reference_methods": 数字,
    "total_paper_methods": 数字,
    "intelligently_matched_methods": 数字,
    "coverage_ratio": 小数,
    "intelligent_matches": [
      {{
        "paper_method": "论文中的方法名（英文）",
        "matched_reference_methods": ["引文数据中的对应方法（英文）"],
        "match_type": "direct|variant|similar_name",
        "match_reasoning": "匹配推理说明（英文）",
        "confidence_score": 小数
      }}
    ],
    "unmatched_paper_methods": ["无法在引文数据中找到对应的论文方法（英文）"]
  }},
  "algorithm_coverage": {{
    "total_reference_algorithms": 数字,
    "total_paper_algorithms": 数字,
    "intelligently_matched_algorithms": 数字,
    "coverage_ratio": 小数,
    "intelligent_matches": [
      {{
        "paper_algorithm": "论文算法名（英文）",
        "matched_reference_algorithms": ["引文数据中的对应算法（英文）"],
        "match_type": "direct|variant|similar_name",
        "match_reasoning": "匹配推理说明（英文）",
        "confidence_score": 小数
      }}
    ],
    "unmatched_paper_algorithms": ["无法在引文数据中找到对应的论文算法（英文）"]
  }},
  "detailed_analysis": "基于智能语义匹配的详细覆盖率分析说明（必须为英文）"
}}

请确保：
1. 所有数值计算准确无误
2. 匹配推理合理且有说服力
3. 覆盖率反映真实的技术重叠程度
4. JSON格式完全正确
"""

def get_coverage_analysis_prompt():
    """获取覆盖率分析提示词（中文提示词，英文输出）- 已弃用，使用get_coverage_analysis_prompt_safe"""
    return """
你是一个专业的学术内容比较分析助手。请比较两组研究方法和算法数据，计算覆盖率。

比较规则：
1. 方法层面比较：比较方法名称的语义相似性，相似度>0.7认为匹配
2. 算法层面比较：在匹配的方法下，比较算法的覆盖情况
3. 考虑同义词和不同表述方式（如CNN vs 卷积神经网络）
4. 提供详细的匹配分析和未匹配项说明

参考数据（基准）：
{reference_data}

待比较数据（论文）：
{comparison_data}

输出要求：
- 严格按照JSON格式输出
- 所有内容必须为英文
- 计算平均覆盖率（average_coverage_ratio）
- 提供详细的英文分析说明

输出格式：
{
  "method_coverage": {
    "total_reference_methods": 数字,
    "total_paper_methods": 数字,
    "matched_methods": 数字,
    "coverage_ratio": 小数,
    "matches": [
      {
        "paper_method": "论文中的方法名（英文）",
        "reference_method": "参考数据中的方法名（英文）",
        "similarity_score": 相似度分数,
        "is_matched": true/false
      }
    ]
  },
  "algorithm_coverage": {
    "total_reference_algorithms": 数字,
    "total_paper_algorithms": 数字,
    "matched_algorithms": 数字,
    "average_coverage_ratio": 小数,
    "by_method": [
      {
        "method_name": "方法名称（英文）",
        "reference_algorithms": 数字,
        "paper_algorithms": 数字,
        "matched_algorithms": 数字,
        "coverage_ratio": 小数
      }
    ]
  },
  "detailed_analysis": "详细分析说明（必须为英文）"
}

请开始分析：
"""

def _summarize_relations_for_analysis(relations):
    """为分析摘要关系数据，避免数据过大"""
    if not relations:
        return []

    # 限制关系数量，避免提示词过长
    max_relations = 30  # 最多处理30个关系

    # 如果关系数量超过限制，进行采样
    if len(relations) > max_relations:
        # 按关系类型分组采样
        relation_types = {}
        for rel in relations:
            rel_type = rel.get('relation_type', 'unknown')
            if rel_type not in relation_types:
                relation_types[rel_type] = []
            relation_types[rel_type].append(rel)

        # 从每个类型中采样
        sampled_relations = []
        relations_per_type = max_relations // max(len(relation_types), 1)

        for rel_type, type_relations in relation_types.items():
            sample_count = min(relations_per_type, len(type_relations))
            sampled_relations.extend(type_relations[:sample_count])

        # 如果还有剩余配额，随机补充
        if len(sampled_relations) < max_relations:
            remaining = max_relations - len(sampled_relations)
            all_remaining = [r for r in relations if r not in sampled_relations]
            sampled_relations.extend(all_remaining[:remaining])

        relations = sampled_relations

    # 简化关系数据结构，只保留关键信息
    simplified_relations = []
    for rel in relations:
        simplified = {
            'relation_type': rel.get('relation_type', 'unknown'),
            'source_entity': rel.get('source_entity', '')[:50],  # 限制长度
            'target_entity': rel.get('target_entity', '')[:50],  # 限制长度
            'description': rel.get('description', '')[:100] if rel.get('description') else ''  # 限制长度
        }
        simplified_relations.append(simplified)

    return simplified_relations

def _summarize_enhanced_relations_for_analysis(enhanced_relations):
    """为分析摘要增强后的关系数据，保留实体名称信息"""
    if not enhanced_relations:
        return []

    # 由于使用1M上下文模型，大幅提高关系数量限制
    max_relations = 500  # 最多处理500个关系，避免重要数据丢失

    # 如果关系数量超过限制，进行采样
    if len(enhanced_relations) > max_relations:
        # 按关系类型分组采样
        relation_types = {}
        for rel in enhanced_relations:
            rel_type = rel.get('relation_type', 'unknown')
            if rel_type not in relation_types:
                relation_types[rel_type] = []
            relation_types[rel_type].append(rel)

        # 从每个类型中采样
        sampled_relations = []
        relations_per_type = max_relations // max(len(relation_types), 1)

        for rel_type, type_relations in relation_types.items():
            sample_count = min(relations_per_type, len(type_relations))
            sampled_relations.extend(type_relations[:sample_count])

        # 如果还有剩余配额，随机补充
        if len(sampled_relations) < max_relations:
            remaining = max_relations - len(sampled_relations)
            all_remaining = [r for r in enhanced_relations if r not in sampled_relations]
            sampled_relations.extend(all_remaining[:remaining])

        enhanced_relations = sampled_relations

    # 简化关系数据结构，保留增强的实体名称信息
    simplified_relations = []
    for rel in enhanced_relations:
        simplified = {
            'relation_type': rel.get('relation_type', 'unknown'),
            'source_entity': rel.get('source_entity', '')[:50],  # 原始ID
            'target_entity': rel.get('target_entity', '')[:50],  # 原始ID
            'source_entity_name': rel.get('source_entity_name', '')[:100],  # 增强的实体名称
            'target_entity_name': rel.get('target_entity_name', '')[:100],  # 增强的实体名称
            'source_entity_type': rel.get('source_entity_type', ''),
            'target_entity_type': rel.get('target_entity_type', ''),
            'description': rel.get('description', '')[:150] if rel.get('description') else '',
            'data_completeness': {
                'has_source_name': bool(rel.get('source_entity_name')),
                'has_target_name': bool(rel.get('target_entity_name')),
                'has_description': bool(rel.get('description'))
            }
        }
        simplified_relations.append(simplified)

    return simplified_relations

def get_task_relation_coverage_prompt():
    """获取增强版任务关系覆盖率分析提示词（处理数据不完整情况）"""
    return """
你是一个专业的学术关系分析助手。请分析综述关系和引文关系之间的覆盖率，特别注意处理数据不完整的情况。

【核心任务】：
计算有多少综述关系被引文关系覆盖，即使在数据不完整的情况下也要尽可能识别重叠关系。
覆盖率计算公式：重合关系数 / 综述关系总数

【数据输入】：
综述关系数据: {review_relations}
引文关系数据: {citation_relations}

【匹配策略】：
   - 相同的关系类型 + 语义相似的实体名称匹配
   - 考虑同义词和变体（如CNN vs Convolutional Neural Network）


【数据不完整处理策略】：
- 记录数据质量对分析结果的影响

【输出要求】：
请严格按照以下JSON格式输出，detailed_analysis部分使用中文：
{{
  "relation_coverage": {{
    "total_review_relations": 数字,
    "total_citation_relations": 数字,
    "overlapping_relations": 数字,
    "overall_coverage_ratio": 小数,
    "coverage_by_type": {{
      "improve": 小数,
      "optimize": 小数,
      "extend": 小数,
      "replace": 小数,
      "use": 小数
    }},
    "match_quality_breakdown": {{
      "high_confidence_matches": 数字,
      "medium_confidence_matches": 数字,
      "low_confidence_matches": 数字
    }}
  }},
  "data_quality_assessment": {{
    "review_data_completeness": 小数,
    "citation_data_completeness": 小数,
    "missing_fields_impact": "数据缺失对分析的影响说明（中文）",
    "confidence_level": 小数
  }},
  "detailed_analysis": "详细的中文分析说明，包括：1）匹配策略使用情况，2）数据质量评估，3）覆盖率解释，4）未匹配原因分析，5）数据改进建议",
  "improvement_suggestions": [
    "具体的数据质量改进建议（中文）"
  ]
}}

【重要提醒】：
- 即使数据不完整，也要尽力识别可能的关系重叠
- 不要因为实体名称缺失就完全放弃匹配
- 提供透明的数据质量评估和分析限制说明
- 给出具体可行的数据改进建议

注意：只返回JSON格式，不要其他文字。detailed_analysis字段必须使用中文。
"""

def get_entity_classification_prompt():
    """获取实体分类提示词（中文提示词，英文输出）"""
    return """
你是一个专业的算法分类专家。请将以下算法按照研究方法进行分类。

分析要求：
1. 根据算法的特点和应用领域进行分类
2. 将相似的算法归类到同一个方法类别中
3. 使用标准的学术方法分类
4. 所有输出内容必须为英文

算法数据：
{entities_data}

常见方法类别参考：
- Deep Learning Methods（深度学习方法）
- Traditional Machine Learning Methods（传统机器学习方法）
- Computer Vision Methods（计算机视觉方法）
- Natural Language Processing Methods（自然语言处理方法）
- Optimization Methods（优化方法）
- Statistical Methods（统计方法）

输出格式：
{
  "methods": [
    {
      "name": "方法类别名称（英文）",
      "description": "方法类别的简要描述（英文）",
      "algorithms": [
        {
          "name": "算法名称（英文）",
          "description": "算法描述（英文）",
          "context": "应用上下文（英文）"
        }
      ]
    }
  ]
}

请开始分类：
"""

# ==================== 后台任务处理函数 ====================

def enhance_relations_with_entity_names(relations: list, debug_logger=None) -> list:
    """增强关系数据：补充实体名称信息"""
    enhanced_relations = []
    entity_cache = {}  # 缓存实体信息避免重复查询

    missing_source_names = 0
    missing_target_names = 0
    missing_descriptions = 0

    for relation in relations:
        enhanced_relation = relation.copy()

        # 获取source_entity名称
        source_entity_id = relation.get('source_entity') or relation.get('from_entity')
        if source_entity_id:
            if source_entity_id not in entity_cache:
                try:
                    entity_data = db_manager.get_entity_by_id(source_entity_id)
                    if entity_data:
                        if 'algorithm_entity' in entity_data:
                            entity_cache[source_entity_id] = {
                                'name': entity_data['algorithm_entity'].get('name', source_entity_id),
                                'type': 'Algorithm'
                            }
                        elif 'dataset_entity' in entity_data:
                            entity_cache[source_entity_id] = {
                                'name': entity_data['dataset_entity'].get('name', source_entity_id),
                                'type': 'Dataset'
                            }
                        elif 'metric_entity' in entity_data:
                            entity_cache[source_entity_id] = {
                                'name': entity_data['metric_entity'].get('name', source_entity_id),
                                'type': 'Metric'
                            }
                        else:
                            entity_cache[source_entity_id] = {'name': source_entity_id, 'type': 'Unknown'}
                    else:
                        entity_cache[source_entity_id] = {'name': source_entity_id, 'type': 'Unknown'}
                except Exception as e:
                    logging.warning(f"获取实体 {source_entity_id} 信息失败: {str(e)}")
                    entity_cache[source_entity_id] = {'name': source_entity_id, 'type': 'Unknown'}

            enhanced_relation['source_entity_name'] = entity_cache[source_entity_id]['name']
            enhanced_relation['source_entity_type'] = entity_cache[source_entity_id]['type']
        else:
            enhanced_relation['source_entity_name'] = ''
            enhanced_relation['source_entity_type'] = ''
            missing_source_names += 1

        # 获取target_entity名称
        target_entity_id = relation.get('target_entity') or relation.get('to_entity')
        if target_entity_id:
            if target_entity_id not in entity_cache:
                try:
                    entity_data = db_manager.get_entity_by_id(target_entity_id)
                    if entity_data:
                        if 'algorithm_entity' in entity_data:
                            entity_cache[target_entity_id] = {
                                'name': entity_data['algorithm_entity'].get('name', target_entity_id),
                                'type': 'Algorithm'
                            }
                        elif 'dataset_entity' in entity_data:
                            entity_cache[target_entity_id] = {
                                'name': entity_data['dataset_entity'].get('name', target_entity_id),
                                'type': 'Dataset'
                            }
                        elif 'metric_entity' in entity_data:
                            entity_cache[target_entity_id] = {
                                'name': entity_data['metric_entity'].get('name', target_entity_id),
                                'type': 'Metric'
                            }
                        else:
                            entity_cache[target_entity_id] = {'name': target_entity_id, 'type': 'Unknown'}
                    else:
                        entity_cache[target_entity_id] = {'name': target_entity_id, 'type': 'Unknown'}
                except Exception as e:
                    logging.warning(f"获取实体 {target_entity_id} 信息失败: {str(e)}")
                    entity_cache[target_entity_id] = {'name': target_entity_id, 'type': 'Unknown'}

            enhanced_relation['target_entity_name'] = entity_cache[target_entity_id]['name']
            enhanced_relation['target_entity_type'] = entity_cache[target_entity_id]['type']
        else:
            enhanced_relation['target_entity_name'] = ''
            enhanced_relation['target_entity_type'] = ''
            missing_target_names += 1

        # 检查描述字段
        if not enhanced_relation.get('description'):
            missing_descriptions += 1

        enhanced_relations.append(enhanced_relation)

    # 记录数据增强统计
    logging.info(f"关系数据增强完成: {len(enhanced_relations)}个关系, "
                f"缺失source名称: {missing_source_names}, "
                f"缺失target名称: {missing_target_names}, "
                f"缺失描述: {missing_descriptions}")

    return enhanced_relations

def assess_data_quality(entities: list, relations: list) -> dict:
    """评估数据质量"""
    quality_report = {
        "entities_analysis": {
            "total_entities": len(entities),
            "entities_with_names": 0,
            "entities_with_types": 0,
            "entities_with_descriptions": 0
        },
        "relations_analysis": {
            "total_relations": len(relations),
            "relations_with_source_names": 0,
            "relations_with_target_names": 0,
            "relations_with_descriptions": 0,
            "relations_with_complete_info": 0
        },
        "data_completeness_score": 0.0,
        "quality_issues": [],
        "improvement_suggestions": []
    }

    # 分析实体质量
    for entity in entities:
        if entity.get('name') or entity.get('algorithm_entity', {}).get('name') or \
           entity.get('dataset_entity', {}).get('name') or entity.get('metric_entity', {}).get('name'):
            quality_report["entities_analysis"]["entities_with_names"] += 1

        if entity.get('type') or entity.get('entity_type'):
            quality_report["entities_analysis"]["entities_with_types"] += 1

        if entity.get('description'):
            quality_report["entities_analysis"]["entities_with_descriptions"] += 1

    # 分析关系质量
    for relation in relations:
        if relation.get('source_entity_name'):
            quality_report["relations_analysis"]["relations_with_source_names"] += 1

        if relation.get('target_entity_name'):
            quality_report["relations_analysis"]["relations_with_target_names"] += 1

        if relation.get('description'):
            quality_report["relations_analysis"]["relations_with_descriptions"] += 1

        if (relation.get('source_entity_name') and relation.get('target_entity_name') and
            relation.get('relation_type')):
            quality_report["relations_analysis"]["relations_with_complete_info"] += 1

    # 计算数据完整性评分
    if len(relations) > 0:
        completeness_score = quality_report["relations_analysis"]["relations_with_complete_info"] / len(relations)
        quality_report["data_completeness_score"] = completeness_score

    # 识别质量问题和改进建议
    if quality_report["relations_analysis"]["relations_with_source_names"] < len(relations) * 0.8:
        quality_report["quality_issues"].append("大量关系缺少源实体名称")
        quality_report["improvement_suggestions"].append("补充实体名称映射，确保关系数据完整性")

    if quality_report["relations_analysis"]["relations_with_descriptions"] < len(relations) * 0.5:
        quality_report["quality_issues"].append("关系描述信息不足")
        quality_report["improvement_suggestions"].append("添加关系描述信息以提高匹配准确性")

    if quality_report["data_completeness_score"] < 0.3:
        quality_report["quality_issues"].append("数据完整性严重不足，可能影响分析准确性")
        quality_report["improvement_suggestions"].append("优先补充实体名称和关系类型信息")

    return quality_report

def get_task_citation_data_with_config(task_id: str, debug_logger=None) -> dict:
    """获取任务的引文数据（优化版：补充实体名称和数据质量检查）"""
    all_entities = db_manager.get_entities_by_task(task_id)
    all_relations = db_manager.get_relations_by_task(task_id)

    citation_entities = [e for e in all_entities if e.get('source') == '引文']
    citation_relations = [r for r in all_relations if r.get('source') == '引文']

    logging.info(f"获取到 {len(citation_entities)} 个引文实体，{len(citation_relations)} 个引文关系")

    # 数据质量检查和增强
    enhanced_relations = enhance_relations_with_entity_names(citation_relations, debug_logger)
    data_quality_report = assess_data_quality(citation_entities, enhanced_relations)

    if debug_logger:
        debug_logger.save_json_result(
            step_name="data_quality_report",
            data=data_quality_report,
            description="数据质量评估报告"
        )

    return {
        "entities": citation_entities,
        "relations": enhanced_relations,
        "data_quality": data_quality_report
    }

def convert_relations_to_methods_with_config(relations: list, model_name: str, debug_logger=None) -> dict:
    """从引文关系中提取方法-算法结构（基于实体连通图分析）"""

    if not relations:
        logging.warning("没有找到引文关系数据")
        return {"methods": []}

    # 导入数据库管理器
    from app.modules.db_manager import DatabaseManager
    db_manager = DatabaseManager()

    # 第一步：从关系中提取所有实体ID并获取实体名称
    entity_names = {}  # entity_id -> entity_name
    entity_types = {}  # entity_id -> entity_type

    logging.info(f"开始从 {len(relations)} 个关系中提取实体信息")

    for relation in relations:
        from_entity_id = relation.get('from_entity')
        to_entity_id = relation.get('to_entity')

        # 获取from_entity的名称
        if from_entity_id and from_entity_id not in entity_names:
            try:
                entity_data = db_manager.get_entity_by_id(from_entity_id)
                if entity_data:
                    if 'algorithm_entity' in entity_data:
                        entity_names[from_entity_id] = entity_data['algorithm_entity'].get('name', from_entity_id)
                        entity_types[from_entity_id] = 'Algorithm'
                    elif 'dataset_entity' in entity_data:
                        entity_names[from_entity_id] = entity_data['dataset_entity'].get('name', from_entity_id)
                        entity_types[from_entity_id] = 'Dataset'
                    elif 'metric_entity' in entity_data:
                        entity_names[from_entity_id] = entity_data['metric_entity'].get('name', from_entity_id)
                        entity_types[from_entity_id] = 'Metric'
                else:
                    entity_names[from_entity_id] = from_entity_id
                    entity_types[from_entity_id] = relation.get('from_entity_type', 'Algorithm')
            except Exception as e:
                logging.warning(f"获取实体 {from_entity_id} 信息失败: {str(e)}")
                entity_names[from_entity_id] = from_entity_id
                entity_types[from_entity_id] = relation.get('from_entity_type', 'Algorithm')

        # 获取to_entity的名称
        if to_entity_id and to_entity_id not in entity_names:
            try:
                entity_data = db_manager.get_entity_by_id(to_entity_id)
                if entity_data:
                    if 'algorithm_entity' in entity_data:
                        entity_names[to_entity_id] = entity_data['algorithm_entity'].get('name', to_entity_id)
                        entity_types[to_entity_id] = 'Algorithm'
                    elif 'dataset_entity' in entity_data:
                        entity_names[to_entity_id] = entity_data['dataset_entity'].get('name', to_entity_id)
                        entity_types[to_entity_id] = 'Dataset'
                    elif 'metric_entity' in entity_data:
                        entity_names[to_entity_id] = entity_data['metric_entity'].get('name', to_entity_id)
                        entity_types[to_entity_id] = 'Metric'
                else:
                    entity_names[to_entity_id] = to_entity_id
                    entity_types[to_entity_id] = relation.get('to_entity_type', 'Algorithm')
            except Exception as e:
                logging.warning(f"获取实体 {to_entity_id} 信息失败: {str(e)}")
                entity_names[to_entity_id] = to_entity_id
                entity_types[to_entity_id] = relation.get('to_entity_type', 'Algorithm')

    logging.info(f"成功获取 {len(entity_names)} 个实体的名称信息")

    # 保存实体信息到调试记录
    if debug_logger:
        debug_logger.save_json_result(
            step_name="extracted_entities",
            data={
                "entity_names": entity_names,
                "entity_types": entity_types,
                "total_entities": len(entity_names)
            },
            description="从关系中提取的实体信息"
        )

    # 第二步：构建连通图，基于实体名称进行聚类
    import networkx as nx
    G = nx.Graph()

    # 添加所有实体作为节点
    for entity_id, entity_name in entity_names.items():
        G.add_node(entity_id, name=entity_name, type=entity_types[entity_id])

    # 添加关系作为边
    edges_added = 0
    for relation in relations:
        from_entity_id = relation.get('from_entity')
        to_entity_id = relation.get('to_entity')
        relation_type = relation.get('relation_type', 'Unknown')

        if from_entity_id and to_entity_id and from_entity_id in entity_names and to_entity_id in entity_names:
            G.add_edge(from_entity_id, to_entity_id, relation_type=relation_type)
            edges_added += 1

    # 第三步：找到连通分量，每个连通分量代表一个研究方法
    connected_components = list(nx.connected_components(G))
    logging.info(f"发现 {len(connected_components)} 个连通分量（研究方法）")

    # 保存连通图信息到调试记录
    if debug_logger:
        graph_data = {
            "nodes": [{"id": node, "name": entity_names.get(node, node), "type": entity_types.get(node, "Unknown")}
                     for node in G.nodes()],
            "edges": [{"from": edge[0], "to": edge[1], "relation_type": G.edges[edge].get('relation_type', 'Unknown')}
                     for edge in G.edges()],
            "connected_components": [
                {
                    "component_id": i,
                    "nodes": list(component),
                    "size": len(component),
                    "node_names": [entity_names.get(node, node) for node in component]
                }
                for i, component in enumerate(connected_components)
            ],
            "graph_stats": {
                "total_nodes": G.number_of_nodes(),
                "total_edges": G.number_of_edges(),
                "edges_added": edges_added,
                "components_count": len(connected_components)
            }
        }

        debug_logger.save_graph_data(
            step_name="citation_graph_analysis",
            graph_data=graph_data,
            description="引文关系连通图分析结果"
        )

    # 第四步：为每个连通分量生成方法信息
    methods = []

    for i, component in enumerate(connected_components):
        # 保留所有连通分量，包括单节点分量（可能代表重要的独立方法）
        # if len(component) < 2:  # 不再跳过单节点分量
        #     continue

        # 获取分量中的实体名称
        component_entities = []
        algorithm_entities = []
        dataset_entities = []
        metric_entities = []

        for entity_id in component:
            entity_name = entity_names[entity_id]
            entity_type = entity_types[entity_id]

            component_entities.append({
                'id': entity_id,
                'name': entity_name,
                'type': entity_type
            })

            if entity_type == 'Algorithm':
                algorithm_entities.append(entity_name)
            elif entity_type == 'Dataset':
                dataset_entities.append(entity_name)
            elif entity_type == 'Metric':
                metric_entities.append(entity_name)

        # 生成方法名称（改进单节点分量的处理）
        if len(component) == 1:
            # 单节点分量：直接使用实体名称和类型
            single_entity = component_entities[0]
            entity_name = single_entity['name']
            entity_type = single_entity['type']

            if entity_type == 'Algorithm':
                method_name = f"{entity_name} Algorithm Methods"
            elif entity_type == 'Dataset':
                method_name = f"{entity_name} Dataset Methods"
            elif entity_type == 'Metric':
                method_name = f"{entity_name} Evaluation Methods"
            else:
                method_name = f"{entity_name} Related Methods"
        elif algorithm_entities:
            # 多节点分量：找到最核心的算法（连接度最高的）
            core_algorithm = None
            max_degree = 0
            for entity_id in component:
                if entity_types[entity_id] == 'Algorithm':
                    degree = G.degree(entity_id)
                    if degree > max_degree:
                        max_degree = degree
                        core_algorithm = entity_names[entity_id]

            if core_algorithm:
                method_name = f"{core_algorithm} Related Methods"
            else:
                method_name = f"{algorithm_entities[0]} Related Methods"
        else:
            # 没有算法实体的分量
            if component_entities:
                primary_entity = component_entities[0]['name']
                method_name = f"{primary_entity} Related Methods"
            else:
                method_name = f"Method {i+1}"

        # 构建方法描述（改进单节点分量的描述）
        if len(component) == 1:
            # 单节点分量的描述
            single_entity = component_entities[0]
            entity_name = single_entity['name']
            entity_type = single_entity['type']
            description = f"Independent {entity_type.lower()} entity: {entity_name}. This represents a standalone research method or technique that may be important for coverage analysis."
        else:
            # 多节点分量的描述
            description_parts = []
            if algorithm_entities:
                description_parts.append(f"Algorithms: {', '.join(algorithm_entities[:3])}")
            if dataset_entities:
                description_parts.append(f"Datasets: {', '.join(dataset_entities[:2])}")
            if metric_entities:
                description_parts.append(f"Metrics: {', '.join(metric_entities[:2])}")

            if description_parts:
                description = f"Connected research method with {len(component)} entities. " + "; ".join(description_parts)
            else:
                description = f"Research method derived from network analysis with {len(component)} connected entities."

        # 构建算法列表（改进单节点分量的处理）
        algorithms = []

        if len(component) == 1:
            # 单节点分量：直接使用该实体作为算法
            single_entity = component_entities[0]
            algorithms.append({
                "name": single_entity['name'],
                "description": f"Standalone {single_entity['type'].lower()} entity that represents an independent research method or technique.",
                "context": f"This {single_entity['type'].lower()} appears as an isolated entity in the citation network, potentially representing a specialized or emerging technique."
            })
        else:
            # 多节点分量：处理算法实体
            for algo_name in algorithm_entities:
                connection_count = len(component) - 1
                algorithms.append({
                    "name": algo_name,
                    "description": f"Algorithm used in this research method, connected to {connection_count} other entities in the citation network.",
                    "context": f"Part of a connected component with {len(component)} entities, indicating its integration with other research elements."
                })

            # 如果没有算法实体，使用其他类型的实体作为"算法"
            if not algorithms and component_entities:
                for entity in component_entities[:3]:  # 最多取3个
                    algorithms.append({
                        "name": entity['name'],
                        "description": f"{entity['type']} entity that plays an important role in this research method.",
                        "context": f"Non-algorithm entity that contributes to the research methodology within a {len(component)}-entity connected component."
                    })

        methods.append({
            "name": method_name,
            "section": "从关系网络推断",
            "description": description,
            "algorithms": algorithms,
            "entity_count": len(component),
            "relation_types": list(set([G[u][v]['relation_type'] for u, v in G.edges() if u in component and v in component]))
        })

    logging.info(f"从关系网络中成功构建 {len(methods)} 个研究方法")
    return {"methods": methods}

def convert_entities_to_methods_with_config(entities: list, model_name: str) -> dict:
    """将实体数据转换为方法-算法结构（使用现有配置）"""

    if not entities:
        logging.warning("没有找到引文实体数据")
        return {"methods": []}

    # 构建实体信息文本
    entities_text = ""
    for entity in entities:
        if 'algorithm_entity' in entity:
            algo = entity['algorithm_entity']
            entities_text += f"Algorithm: {algo.get('name', '')}\n"
            entities_text += f"Description: {algo.get('description', '')}\n"
            entities_text += f"Application: {algo.get('application_domain', '')}\n\n"

    if not entities_text.strip():
        logging.warning("没有找到有效的算法实体数据")
        return {"methods": []}

    # 使用大模型进行方法分类
    classification_prompt = get_entity_classification_prompt().format(
        entities_data=entities_text
    )

    try:
        response = call_llm_with_prompt(classification_prompt, model_name)
        methods_data = extract_json_from_response(response)

        logging.info(f"成功分类了 {len(methods_data.get('methods', []))} 个方法类别")
        return methods_data
    except Exception as e:
        logging.error(f"实体分类失败: {str(e)}")
        return {"methods": []}

def start_paper_task_analysis_with_config(analysis_id: str, paper_path: str, task_id: str, model_name: str, extracted_data: dict = None):
    """启动论文与任务比较分析的后台任务（基于现有配置）"""

    def analysis_worker():
        # 创建调试记录器
        debug_logger = DebugLogger(analysis_id)
        debug_logger.log_step(
            step_name="analysis_start",
            step_type="processing",
            description="开始论文与任务比较分析",
            paper_path=paper_path,
            task_id=task_id,
            model_name=model_name
        )

        try:
            # 第0步：先获取引文数据作为参考（优化提取效果）
            db_manager.update_processing_status(
                task_id=analysis_id,
                current_stage='获取引文参考数据',
                progress=10,
                message='正在获取任务引文数据作为提取参考'
            )

            citation_data = get_task_citation_data_with_config(task_id, debug_logger)

            # 保存引文数据
            debug_logger.save_json_result(
                step_name="citation_data",
                data=citation_data,
                description="获取的任务引文数据"
            )

            # 第1步：获取论文方法数据（优先使用已提取的数据）
            if extracted_data:
                db_manager.update_processing_status(
                    task_id=analysis_id,
                    current_stage='使用已提取数据',
                    progress=25,
                    message='正在使用已提取的论文方法数据'
                )

                # 检查提取数据的结构，支持两种格式
                if "extracted_data" in extracted_data:
                    # 格式1：包含元数据的完整结构
                    paper_methods = extracted_data["extracted_data"]
                    logging.info(f"使用已提取的论文方法数据（完整格式），包含 {len(paper_methods.get('methods', []))} 个方法")
                else:
                    # 格式2：直接的方法数据
                    paper_methods = extracted_data
                    logging.info(f"使用已提取的论文方法数据（简单格式），包含 {len(paper_methods.get('methods', []))} 个方法")

                # 保存使用的提取数据
                debug_logger.save_json_result(
                    step_name="paper_methods_from_extracted",
                    data=paper_methods,
                    description="使用的已提取论文方法数据"
                )
            else:
                # 使用增强版提示词从论文提取方法（第1次大模型调用）
                db_manager.update_processing_status(
                    task_id=analysis_id,
                    current_stage='提取论文方法',
                    progress=25,
                    message='正在从论文中全面提取研究方法和算法（参考引文数据）'
                )

                # 准备引文数据的简化版本作为参考
                try:
                    if citation_data["entities"] or citation_data["relations"]:
                        # 创建引文数据的简化描述
                        reference_summary = f"该领域包含约{len(citation_data['entities'])}个研究实体和{len(citation_data['relations'])}个关系，"
                        reference_summary += "涉及深度学习、机器学习、计算机视觉、自然语言处理等多个技术领域。"

                        # 使用增强版提示词
                        from app.modules.paper_analysis_prompts import get_enhanced_method_extraction_prompt_with_reference
                        paper_prompt = get_enhanced_method_extraction_prompt_with_reference(reference_summary)
                    else:
                        # 如果没有引文数据，使用标准增强版提示词
                        paper_prompt = get_method_extraction_prompt()
                except Exception as e:
                    logging.warning(f"准备增强版提示词失败，使用标准提示词: {str(e)}")
                    paper_prompt = get_method_extraction_prompt()

                # 保存论文方法提取提示词
                debug_logger.save_prompt(
                    step_name="paper_method_extraction",
                    prompt=paper_prompt,
                    model_name=model_name,
                    pdf_path=paper_path,
                    prompt_type="enhanced" if "enhanced" in str(type(paper_prompt)) else "standard"
                )

                paper_response = call_llm_with_prompt(
                    prompt=paper_prompt,
                    model_name=model_name,
                    pdf_path=paper_path
                )

                # 保存论文方法提取响应
                debug_logger.save_llm_response(
                    step_name="paper_method_extraction",
                    response=paper_response,
                    model_name=model_name,
                    pdf_path=paper_path
                )

                try:
                    paper_methods = extract_json_from_response(paper_response)
                    logging.info("论文方法提取JSON解析成功")

                    # 保存解析成功的论文方法
                    debug_logger.save_json_result(
                        step_name="paper_methods_parsed",
                        data=paper_methods,
                        description="成功解析的论文方法数据"
                    )

                    # 保存首次提取的结果到文件
                    save_extracted_results(analysis_id, paper_methods, "paper_task")

                except Exception as e:
                    logging.error(f"论文方法JSON解析失败: {str(e)}")

                    # 保存解析错误
                    debug_logger.save_error(
                        step_name="paper_methods_parsing",
                        error=e,
                        context="论文方法提取JSON解析失败"
                    )

                    # 创建默认结构
                    paper_methods = {
                        "methods": [
                            {
                                "name": "JSON解析失败",
                                "section": "N/A",
                                "description": f"无法解析模型响应: {str(e)}",
                                "algorithms": []
                            }
                        ]
                    }
                    logging.warning("使用默认论文方法结构继续处理")

                # 保存默认结构
                debug_logger.save_json_result(
                    step_name="paper_methods_fallback",
                    data=paper_methods,
                    description="JSON解析失败后的默认论文方法结构"
                )

            # 第2步：从任务引文数据提取方法（第2次大模型调用）
            db_manager.update_processing_status(
                task_id=analysis_id,
                current_stage='提取引文方法',
                progress=55,
                message='正在从任务引文数据中提取研究方法和算法'
            )

            # 如果没有引文实体，尝试从引文关系中提取方法信息
            if not citation_data["entities"] and citation_data["relations"]:
                logging.info(f"没有引文实体，尝试从 {len(citation_data['relations'])} 个引文关系中提取方法信息")

                debug_logger.log_step(
                    step_name="citation_method_extraction_from_relations",
                    step_type="processing",
                    description="从引文关系中提取方法信息",
                    relations_count=len(citation_data["relations"]),
                    model_name=model_name
                )

                citation_methods = convert_relations_to_methods_with_config(citation_data["relations"], model_name, debug_logger)
            else:
                debug_logger.log_step(
                    step_name="citation_method_extraction_from_entities",
                    step_type="processing",
                    description="从引文实体中提取方法信息",
                    entities_count=len(citation_data["entities"]),
                    model_name=model_name
                )

                citation_methods = convert_entities_to_methods_with_config(citation_data["entities"], model_name)

            # 保存引文方法提取结果
            debug_logger.save_json_result(
                step_name="citation_methods",
                data=citation_methods,
                description="从引文数据中提取的方法信息"
            )

            # 第3步：比较计算覆盖率（第3次大模型调用）
            db_manager.update_processing_status(
                task_id=analysis_id,
                current_stage='计算覆盖率',
                progress=80,
                message='正在比较分析并计算覆盖率'
            )

            # 安全地构建覆盖率分析提示词（避免格式化问题）
            try:
                reference_data_str = json.dumps(citation_methods, ensure_ascii=False, indent=2)
                comparison_data_str = json.dumps(paper_methods, ensure_ascii=False, indent=2)

                # 使用字符串拼接而不是.format()来避免大括号冲突
                coverage_prompt = get_coverage_analysis_prompt_safe(reference_data_str, comparison_data_str)

                debug_logger.log_step(
                    step_name="coverage_prompt_construction",
                    step_type="processing",
                    description="成功构建覆盖率分析提示词",
                    reference_data_length=len(reference_data_str),
                    comparison_data_length=len(comparison_data_str)
                )

            except Exception as format_error:
                logging.error(f"构建覆盖率分析提示词失败: {str(format_error)}")

                debug_logger.save_error(
                    step_name="coverage_prompt_construction",
                    error=format_error,
                    context="构建覆盖率分析提示词时发生格式化错误"
                )

                # 使用简化的数据进行分析
                reference_data_str = str(citation_methods)
                comparison_data_str = str(paper_methods)
                coverage_prompt = get_coverage_analysis_prompt_safe(reference_data_str, comparison_data_str)

                debug_logger.log_step(
                    step_name="coverage_prompt_fallback",
                    step_type="processing",
                    description="使用简化数据构建覆盖率分析提示词",
                    reference_data_length=len(reference_data_str),
                    comparison_data_length=len(comparison_data_str)
                )

            # 计算论文1的方法数和算法数
            paper_methods_list = paper_methods.get('methods', [])
            paper_methods_count = len(paper_methods_list)

            # 计算论文1的算法数
            paper_algorithms_count = 0
            for method in paper_methods_list:
                if 'algorithms' in method:
                    paper_algorithms_count += len(method['algorithms'])

            # 准备额外参数传递给大模型
            reference_methods_count = len(citation_methods.get('methods', []))

            # 在提示词中包含额外参数
            coverage_prompt_with_params = coverage_prompt + f"""

【额外参数信息】：
以下是准确统计的数量，请在计算时参考这些数值：
- reference_methods_count: {reference_methods_count} (引文数据中的方法总数)
- paper_methods_count: {paper_methods_count} (论文数据中的方法总数)
- paper_algorithms_count: {paper_algorithms_count} (论文数据中的算法总数)
- data_source: {"extracted_data" if extracted_data else "pdf_extraction"} (数据来源)

请确保您的计算结果中：
- total_reference_methods = {reference_methods_count}
- total_paper_methods = {paper_methods_count}
- total_paper_algorithms = {paper_algorithms_count}
- method_coverage_ratio = intelligently_matched_methods / {paper_methods_count}
- algorithm_coverage_ratio = intelligently_matched_algorithms / {paper_algorithms_count}
"""

            # 保存覆盖率分析提示词
            debug_logger.save_prompt(
                step_name="coverage_analysis",
                prompt=coverage_prompt_with_params,
                model_name=model_name,
                reference_methods_count=reference_methods_count,
                paper_methods_count=paper_methods_count,
                paper_algorithms_count=paper_algorithms_count,
                data_source="extracted_data" if extracted_data else "pdf_extraction"
            )

            coverage_response = call_llm_with_prompt(
                prompt=coverage_prompt_with_params,
                model_name=model_name
            )

            # 保存覆盖率分析响应
            debug_logger.save_llm_response(
                step_name="coverage_analysis",
                response=coverage_response,
                model_name=model_name
            )

            try:
                coverage_result = extract_json_from_response(coverage_response)
                logging.info("覆盖率分析JSON解析成功")

                # 简单的数量验证（只添加验证信息，不修改大模型结果）
                if 'method_coverage' in coverage_result:
                    method_coverage = coverage_result['method_coverage']
                    paper_methods_list = paper_methods.get('methods', [])
                    total_paper_methods = len(paper_methods_list)

                    # 计算论文总算法数
                    total_paper_algorithms = 0
                    for method in paper_methods_list:
                        if 'algorithms' in method:
                            total_paper_algorithms += len(method['algorithms'])

                    # 验证大模型计算的数量是否正确
                    expected_total_paper_methods = total_paper_methods
                    expected_total_paper_algorithms = total_paper_algorithms
                    expected_reference_methods = len(citation_methods.get('methods', []))

                    actual_total_paper_methods = method_coverage.get('total_paper_methods', 0)
                    actual_total_paper_algorithms = coverage_result.get('algorithm_coverage', {}).get('total_paper_algorithms', 0)
                    actual_reference_methods = method_coverage.get('total_reference_methods', 0)

                    # 记录数量验证结果
                    if actual_total_paper_methods != expected_total_paper_methods:
                        logging.warning(f"论文方法数不匹配: 大模型={actual_total_paper_methods}, 实际={expected_total_paper_methods}")
                    if actual_total_paper_algorithms != expected_total_paper_algorithms:
                        logging.warning(f"论文算法数不匹配: 大模型={actual_total_paper_algorithms}, 实际={expected_total_paper_algorithms}")
                    if actual_reference_methods != expected_reference_methods:
                        logging.warning(f"引用方法数不匹配: 大模型={actual_reference_methods}, 实际={expected_reference_methods}")

                    # 添加计算验证信息（包含期望值和实际值对比）
                    method_coverage['calculation_verification'] = {
                        "expected_total_paper_methods": expected_total_paper_methods,
                        "expected_total_paper_algorithms": expected_total_paper_algorithms,
                        "expected_reference_methods": expected_reference_methods,
                        "actual_total_paper_methods": actual_total_paper_methods,
                        "actual_total_paper_algorithms": actual_total_paper_algorithms,
                        "actual_reference_methods": actual_reference_methods,
                        "reference_methods_count": expected_reference_methods,
                        "paper_methods_count": expected_total_paper_methods,
                        "paper_algorithms_count": expected_total_paper_algorithms,
                        "data_source": "extracted_data" if extracted_data else "pdf_extraction",
                        "consistency_note": "使用已提取结果确保一致性" if extracted_data else "首次提取，已保存结果供后续使用",
                        "calculation_timestamp": datetime.now().isoformat(),
                        "quantity_verification": {
                            "paper_methods_match": actual_total_paper_methods == expected_total_paper_methods,
                            "paper_algorithms_match": actual_total_paper_algorithms == expected_total_paper_algorithms,
                            "reference_methods_match": actual_reference_methods == expected_reference_methods
                        }
                    }

                    # 记录大模型返回的结果（用于调试）
                    logging.info(f"大模型返回的覆盖率结果 - 方法覆盖率: {method_coverage.get('coverage_ratio', 'N/A')}, 算法覆盖率: {coverage_result.get('algorithm_coverage', {}).get('coverage_ratio', 'N/A')}")
                    logging.info(f"数量验证 - 论文方法: {actual_total_paper_methods}/{expected_total_paper_methods}, 论文算法: {actual_total_paper_algorithms}/{expected_total_paper_algorithms}, 引用方法: {actual_reference_methods}/{expected_reference_methods}")

                # 保存解析成功的覆盖率结果
                debug_logger.save_json_result(
                    step_name="coverage_result_parsed",
                    data=coverage_result,
                    description="成功解析的覆盖率分析结果（已验证计算公式）"
                )

            except Exception as json_error:
                logging.error(f"覆盖率分析JSON解析失败: {str(json_error)}")
                logging.error(f"原始响应前500字符: {coverage_response[:500]}")

                # 保存解析错误
                debug_logger.save_error(
                    step_name="coverage_result_parsing",
                    error=json_error,
                    context=f"覆盖率分析JSON解析失败，原始响应长度：{len(coverage_response)}字符"
                )

                # 创建默认的覆盖率结果
                coverage_result = {
                    "method_coverage": {
                        "method_coverage_ratio": 0.0,
                        "algorithm_coverage_ratio": 0.0,
                        "covered_methods": [],
                        "uncovered_methods": [],
                        "covered_algorithms": [],
                        "uncovered_algorithms": []
                    },
                    "detailed_analysis": f"论文与任务覆盖率分析失败：JSON解析错误。错误原因：{str(json_error)}。原始响应长度：{len(coverage_response)}字符。建议尝试使用其他模型（如Claude或DeepSeek）重新分析。"
                }
                logging.warning("使用默认覆盖率结果继续处理")

                # 保存默认覆盖率结果
                debug_logger.save_json_result(
                    step_name="coverage_result_fallback",
                    data=coverage_result,
                    description="JSON解析失败后的默认覆盖率结果"
                )

            # 保存最终结果
            final_results = {
                "paper_methods": paper_methods,
                "citation_methods": citation_methods,
                "method_coverage": coverage_result,
                "model_used": model_name
            }

            # 缓存结果到配置目录
            cache_analysis_result(analysis_id, final_results)

            # 完成调试记录
            debug_logger.finalize(final_results)

            db_manager.update_processing_status(
                task_id=analysis_id,
                current_stage='分析完成',
                progress=100,
                message='论文与任务比较分析完成'
            )

        except Exception as e:
            error_msg = str(e)
            # 安全地记录错误，避免格式化问题
            logging.error("论文与任务分析失败: %s", error_msg)

            # 保存错误到调试记录
            debug_logger.save_error(
                step_name="analysis_fatal_error",
                error=e,
                context="论文与任务比较分析过程中发生致命错误"
            )

            # 完成调试记录（即使失败也要保存调试信息）
            debug_logger.finalize()

            # 清理错误消息中的特殊字符，避免格式化问题
            safe_error_msg = error_msg.replace('"', "'").replace('\n', ' ').replace('\r', '')[:200]

            db_manager.update_processing_status(
                task_id=analysis_id,
                current_stage='分析失败',
                progress=-1,
                message=f'分析失败: {safe_error_msg}'
            )

    # 启动后台线程
    thread = threading.Thread(target=analysis_worker)
    thread.daemon = True
    thread.start()

def start_paper_paper_analysis_with_config(analysis_id: str, paper1_path: str, paper2_path: str, model_name: str, extracted_data: dict = None):
    """启动论文与论文比较分析的后台任务（基于现有配置）"""

    def analysis_worker():
        # 创建调试记录器
        debug_logger = DebugLogger(analysis_id)
        debug_logger.log_step(
            step_name="paper_paper_analysis_start",
            step_type="processing",
            description="开始论文与论文比较分析",
            paper1_path=paper1_path,
            paper2_path=paper2_path,
            model_name=model_name,
            has_extracted_data=extracted_data is not None
        )

        try:
            # 第1步：获取论文1方法数据（优先使用已提取的数据）
            if extracted_data:
                db_manager.update_processing_status(
                    task_id=analysis_id,
                    current_stage='使用已提取数据',
                    progress=25,
                    message='正在使用已提取的论文1方法数据'
                )

                # 检查提取数据的结构，支持两种格式
                if "extracted_data" in extracted_data:
                    # 格式1：包含元数据的完整结构
                    paper1_methods = extracted_data["extracted_data"]
                    logging.info(f"使用已提取的论文1方法数据（完整格式），包含 {len(paper1_methods.get('methods', []))} 个方法")
                else:
                    # 格式2：直接的方法数据
                    paper1_methods = extracted_data
                    logging.info(f"使用已提取的论文1方法数据（简单格式），包含 {len(paper1_methods.get('methods', []))} 个方法")
            else:
                db_manager.update_processing_status(
                    task_id=analysis_id,
                    current_stage='提取论文1方法',
                    progress=25,
                    message='正在从第一篇论文中提取研究方法和算法'
                )

                paper1_response = call_llm_with_prompt(
                    prompt=get_method_extraction_prompt(),
                    model_name=model_name,
                    pdf_path=paper1_path
                )
                paper1_methods = extract_json_from_response(paper1_response)

                # 保存首次提取的结果到文件
                save_extracted_results(analysis_id, paper1_methods, "paper_paper")

            # 第2步：从论文2提取方法
            db_manager.update_processing_status(
                task_id=analysis_id,
                current_stage='提取论文2方法',
                progress=50,
                message='正在从第二篇论文中提取研究方法和算法'
            )

            paper2_response = call_llm_with_prompt(
                prompt=get_method_extraction_prompt(),
                model_name=model_name,
                pdf_path=paper2_path
            )
            paper2_methods = extract_json_from_response(paper2_response)

            # 第3步：比较计算覆盖率
            db_manager.update_processing_status(
                task_id=analysis_id,
                current_stage='计算覆盖率',
                progress=80,
                message='正在比较两篇论文并计算覆盖率'
            )

            # 复用现有代码：论文1当论文，论文2当引文数据
            try:
                # 论文2作为引文数据（参考基准）
                reference_data_str = json.dumps(paper2_methods, ensure_ascii=False, indent=2)
                # 论文1作为待分析论文
                comparison_data_str = json.dumps(paper1_methods, ensure_ascii=False, indent=2)
                coverage_prompt = get_coverage_analysis_prompt_safe(reference_data_str, comparison_data_str)
            except Exception as format_error:
                logging.error(f"构建论文比较提示词失败: {str(format_error)}")
                # 使用简化的数据进行分析
                reference_data_str = str(paper2_methods)[:1000] + "..."
                comparison_data_str = str(paper1_methods)[:1000] + "..."
                coverage_prompt = get_coverage_analysis_prompt_safe(reference_data_str, comparison_data_str)

            # 计算论文1和论文2的方法数和算法数（用于调试日志）
            paper1_methods_list = paper1_methods.get('methods', [])
            paper2_methods_list = paper2_methods.get('methods', [])
            paper1_methods_count = len(paper1_methods_list)
            paper2_methods_count = len(paper2_methods_list)

            # 计算算法数
            paper1_algorithms_count = 0
            for method in paper1_methods_list:
                if 'algorithms' in method:
                    paper1_algorithms_count += len(method['algorithms'])

            paper2_algorithms_count = 0
            for method in paper2_methods_list:
                if 'algorithms' in method:
                    paper2_algorithms_count += len(method['algorithms'])

            # 复用现有逻辑：论文1当论文，论文2当引文数据
            # 所以参数需要对调
            reference_methods_count = paper2_methods_count  # 论文2作为引文数据
            paper_methods_count = paper1_methods_count      # 论文1作为待分析论文
            paper_algorithms_count = paper1_algorithms_count # 论文1作为待分析论文

            coverage_prompt_with_params = coverage_prompt + f"""

【额外参数信息】：
以下是准确统计的数量，请在计算时参考这些数值：
- reference_methods_count: {reference_methods_count} (引文数据中的方法总数)
- paper_methods_count: {paper_methods_count} (论文数据中的方法总数)
- paper_algorithms_count: {paper_algorithms_count} (论文数据中的算法总数)
- data_source: {"extracted_data" if extracted_data else "pdf_extraction"} (数据来源)

请确保您的计算结果中：
- total_reference_methods = {reference_methods_count}
- total_paper_methods = {paper_methods_count}
- total_paper_algorithms = {paper_algorithms_count}
- method_coverage_ratio = intelligently_matched_methods / {paper_methods_count}
- algorithm_coverage_ratio = intelligently_matched_algorithms / {paper_algorithms_count}
"""

            # 保存覆盖率分析提示词（复用逻辑：论文1当论文，论文2当引文数据）
            debug_logger.save_prompt(
                step_name="paper_paper_coverage_analysis",
                prompt=coverage_prompt_with_params,
                model_name=model_name,
                reference_methods_count=paper2_methods_count,  # 论文2作为引文数据
                paper_methods_count=paper1_methods_count,      # 论文1作为待分析论文
                paper_algorithms_count=paper1_algorithms_count,
                data_source="extracted_data" if extracted_data else "pdf_extraction"
            )

            coverage_response = call_llm_with_prompt(
                prompt=coverage_prompt_with_params,
                model_name=model_name
            )

            try:
                coverage_result = extract_json_from_response(coverage_response)
                logging.info("论文比较覆盖率JSON解析成功")

                # 简单的数量验证（复用现有逻辑：论文1当论文，论文2当引文数据）
                if 'method_coverage' in coverage_result:
                    method_coverage = coverage_result['method_coverage']
                    paper1_methods_list = paper1_methods.get('methods', [])  # 待分析论文
                    paper2_methods_list = paper2_methods.get('methods', [])  # 引文数据
                    total_paper1_methods = len(paper1_methods_list)
                    total_paper2_methods = len(paper2_methods_list)

                    # 计算论文1总算法数（待分析论文）
                    total_paper1_algorithms = 0
                    for method in paper1_methods_list:
                        if 'algorithms' in method:
                            total_paper1_algorithms += len(method['algorithms'])

                    # 计算论文2总算法数（引文数据）
                    total_paper2_algorithms = 0
                    for method in paper2_methods_list:
                        if 'algorithms' in method:
                            total_paper2_algorithms += len(method['algorithms'])

                    # 验证大模型返回的数量（复用逻辑：论文1当论文，论文2当引文数据）
                    actual_reference_methods = method_coverage.get('total_reference_methods', 0)  # 应该是paper2（引文数据）
                    actual_total_paper_methods = method_coverage.get('total_paper_methods', 0)  # 应该是paper1（待分析论文）
                    actual_reference_algorithms = coverage_result.get('algorithm_coverage', {}).get('total_reference_algorithms', 0)  # 应该是paper2（引文数据）
                    actual_total_paper_algorithms = coverage_result.get('algorithm_coverage', {}).get('total_paper_algorithms', 0)  # 应该是paper1（待分析论文）

                    # 记录数量验证结果
                    if actual_reference_methods != total_paper2_methods:
                        logging.warning(f"引文数据方法数不匹配: 大模型={actual_reference_methods}, 实际={total_paper2_methods}")
                    if actual_total_paper_methods != total_paper1_methods:
                        logging.warning(f"论文方法数不匹配: 大模型={actual_total_paper_methods}, 实际={total_paper1_methods}")
                    if actual_reference_algorithms != total_paper2_algorithms:
                        logging.warning(f"引文数据算法数不匹配: 大模型={actual_reference_algorithms}, 实际={total_paper2_algorithms}")
                    if actual_total_paper_algorithms != total_paper1_algorithms:
                        logging.warning(f"论文算法数不匹配: 大模型={actual_total_paper_algorithms}, 实际={total_paper1_algorithms}")

                    # 添加计算验证信息（复用逻辑：论文1当论文，论文2当引文数据）
                    method_coverage['calculation_verification'] = {
                        "total_paper1_methods": total_paper1_methods,      # 论文1（待分析论文）
                        "total_paper1_algorithms": total_paper1_algorithms,
                        "total_paper2_methods": total_paper2_methods,      # 论文2（引文数据）
                        "total_paper2_algorithms": total_paper2_algorithms,
                        "expected_reference_methods": total_paper2_methods,    # 期望的引文数据方法数
                        "expected_paper_methods": total_paper1_methods,        # 期望的论文方法数
                        "expected_reference_algorithms": total_paper2_algorithms,  # 期望的引文数据算法数
                        "expected_paper_algorithms": total_paper1_algorithms,      # 期望的论文算法数
                        "actual_reference_methods": actual_reference_methods,
                        "actual_paper_methods": actual_total_paper_methods,
                        "actual_reference_algorithms": actual_reference_algorithms,
                        "actual_paper_algorithms": actual_total_paper_algorithms,
                        "reference_methods_count": total_paper2_methods,    # 论文2作为引文数据
                        "paper_methods_count": total_paper1_methods,        # 论文1作为待分析论文
                        "paper_algorithms_count": total_paper1_algorithms,  # 论文1作为待分析论文
                        "comparison_methods_count": total_paper2_methods,   # 论文2的方法数
                        "comparison_algorithms_count": total_paper2_algorithms, # 论文2的算法数
                        "data_source": "extracted_data" if extracted_data else "pdf_extraction",
                        "consistency_note": "论文对论文分析：复用现有逻辑，论文1当待分析论文，论文2当引文数据",
                        "calculation_timestamp": datetime.now().isoformat(),
                        "quantity_verification": {
                            "reference_methods_match": actual_reference_methods == total_paper2_methods,
                            "paper_methods_match": actual_total_paper_methods == total_paper1_methods,
                            "reference_algorithms_match": actual_reference_algorithms == total_paper2_algorithms,
                            "paper_algorithms_match": actual_total_paper_algorithms == total_paper1_algorithms
                        }
                    }

                    # 记录大模型返回的结果（用于调试）
                    logging.info(f"大模型返回的论文对论文覆盖率结果 - 方法覆盖率: {method_coverage.get('coverage_ratio', 'N/A')}, 算法覆盖率: {coverage_result.get('algorithm_coverage', {}).get('coverage_ratio', 'N/A')}")
                    logging.info(f"论文对论文数量验证（复用逻辑） - 引文数据方法: {actual_reference_methods}/{total_paper2_methods}, 论文方法: {actual_total_paper_methods}/{total_paper1_methods}, 引文数据算法: {actual_reference_algorithms}/{total_paper2_algorithms}, 论文算法: {actual_total_paper_algorithms}/{total_paper1_algorithms}")

            except Exception as json_error:
                logging.error(f"论文比较覆盖率JSON解析失败: {str(json_error)}")
                logging.error(f"原始响应前500字符: {coverage_response[:500]}")

                # 创建默认的覆盖率结果
                coverage_result = {
                    "method_coverage": {
                        "method_coverage_ratio": 0.0,
                        "algorithm_coverage_ratio": 0.0,
                        "covered_methods": [],
                        "uncovered_methods": [],
                        "covered_algorithms": [],
                        "uncovered_algorithms": []
                    },
                    "detailed_analysis": f"论文与论文比较分析失败：JSON解析错误。错误原因：{str(json_error)}。原始响应长度：{len(coverage_response)}字符。建议尝试使用其他模型（如Claude或DeepSeek）重新分析。"
                }
                logging.warning("使用默认论文比较结果继续处理")

            # 保存最终结果
            final_results = {
                "paper1_methods": paper1_methods,
                "paper2_methods": paper2_methods,
                "method_coverage": coverage_result,
                "model_used": model_name
            }

            # 缓存结果
            cache_analysis_result(analysis_id, final_results)

            db_manager.update_processing_status(
                task_id=analysis_id,
                current_stage='分析完成',
                progress=100,
                message='论文与论文比较分析完成'
            )

        except Exception as e:
            error_msg = str(e)
            # 安全地记录错误，避免格式化问题
            logging.error("论文对比分析失败: %s", error_msg)

            # 清理错误消息中的特殊字符，避免格式化问题
            safe_error_msg = error_msg.replace('"', "'").replace('\n', ' ').replace('\r', '')[:200]

            db_manager.update_processing_status(
                task_id=analysis_id,
                current_stage='分析失败',
                progress=-1,
                message=f'分析失败: {safe_error_msg}'
            )

    # 启动后台线程
    thread = threading.Thread(target=analysis_worker)
    thread.daemon = True
    thread.start()

def start_relation_coverage_analysis_with_config(analysis_id: str, task_id: str, model_name: str):
    """启动任务关系覆盖率分析的后台任务（优化版：集成调试系统和数据质量检查）"""

    def analysis_worker():
        # 创建调试记录器
        debug_logger = DebugLogger(analysis_id)
        debug_logger.log_step(
            step_name="relation_coverage_analysis_start",
            step_type="processing",
            description="开始任务关系覆盖率分析",
            task_id=task_id,
            model_name=model_name
        )

        try:
            # 获取关系数据（复用现有功能）
            db_manager.update_processing_status(
                task_id=analysis_id,
                current_stage='获取关系数据',
                progress=20,
                message='正在获取任务的综述和引文关系数据'
            )

            all_relations = db_manager.get_relations_by_task(task_id)
            if not all_relations:
                all_relations = []

            review_relations = [r for r in all_relations if r.get('source') == '综述']
            citation_relations = [r for r in all_relations if r.get('source') == '引文']

            logging.info(f"获取到 {len(review_relations)} 个综述关系，{len(citation_relations)} 个引文关系")

            # 保存原始关系数据
            debug_logger.save_json_result(
                step_name="raw_relations_data",
                data={
                    "review_relations": review_relations,
                    "citation_relations": citation_relations,
                    "review_count": len(review_relations),
                    "citation_count": len(citation_relations)
                },
                description="原始关系数据"
            )

            # 数据增强和质量检查
            db_manager.update_processing_status(
                task_id=analysis_id,
                current_stage='数据质量检查',
                progress=35,
                message='正在进行数据增强和质量检查'
            )

            enhanced_review_relations = enhance_relations_with_entity_names(review_relations, debug_logger)
            enhanced_citation_relations = enhance_relations_with_entity_names(citation_relations, debug_logger)

            # 评估数据质量
            review_quality = assess_data_quality([], enhanced_review_relations)
            citation_quality = assess_data_quality([], enhanced_citation_relations)

            combined_quality_report = {
                "review_data_quality": review_quality,
                "citation_data_quality": citation_quality,
                "overall_assessment": {
                    "review_completeness": review_quality["data_completeness_score"],
                    "citation_completeness": citation_quality["data_completeness_score"],
                    "analysis_feasibility": "high" if min(review_quality["data_completeness_score"],
                                                        citation_quality["data_completeness_score"]) > 0.3 else "low"
                }
            }

            debug_logger.save_json_result(
                step_name="enhanced_relations_data",
                data={
                    "enhanced_review_relations": enhanced_review_relations,
                    "enhanced_citation_relations": enhanced_citation_relations,
                    "data_quality_report": combined_quality_report
                },
                description="增强后的关系数据和质量报告"
            )

            # 如果没有关系数据，直接返回空分析结果
            if len(review_relations) == 0 and len(citation_relations) == 0:
                logging.warning("任务没有关系数据，返回空分析结果")

                empty_result = {
                    "relation_coverage": {
                        "total_review_relations": 0,
                        "total_citation_relations": 0,
                        "overlapping_relations": 0,
                        "overall_coverage_ratio": 0.0,
                        "coverage_by_type": {
                            "improve": 0.0,
                            "optimize": 0.0,
                            "extend": 0.0,
                            "replace": 0.0,
                            "use": 0.0
                        }
                    },
                    "detailed_analysis": "该任务没有关系数据，无法进行覆盖率分析。请确保任务包含综述关系和引文关系数据。"
                }

                final_results = {
                    "review_relations_count": 0,
                    "citation_relations_count": 0,
                    "relation_coverage": empty_result,
                    "model_used": model_name
                }

                # 缓存结果
                cache_analysis_result(analysis_id, final_results)

                db_manager.update_processing_status(
                    task_id=analysis_id,
                    current_stage='分析完成',
                    progress=100,
                    message='任务关系覆盖率分析完成（无关系数据）'
                )
                return

            # 分析关系覆盖率（1次大模型调用）
            db_manager.update_processing_status(
                task_id=analysis_id,
                current_stage='分析关系覆盖率',
                progress=60,
                message='正在分析综述和引文关系的覆盖率'
            )

            # 使用增强后的数据进行分析，由于1M上下文，优先使用完整数据
            if len(enhanced_review_relations) <= 500 and len(enhanced_citation_relations) <= 500:
                # 数据量在1M上下文范围内，使用完整的增强数据
                review_data_for_analysis = enhanced_review_relations
                citation_data_for_analysis = enhanced_citation_relations
                data_processing_note = "使用完整的增强数据进行分析"
                is_sampled = False
            else:
                # 数据量过大，使用摘要但保留增强的实体名称信息
                review_data_for_analysis = _summarize_enhanced_relations_for_analysis(enhanced_review_relations)
                citation_data_for_analysis = _summarize_enhanced_relations_for_analysis(enhanced_citation_relations)
                data_processing_note = f"使用摘要数据进行分析（原始: 综述{len(enhanced_review_relations)}条/引文{len(enhanced_citation_relations)}条 → 采样: 综述{len(review_data_for_analysis)}条/引文{len(citation_data_for_analysis)}条）"
                is_sampled = True

            logging.info(f"关系数据处理 - {data_processing_note}")

            # 构建增强版提示词，包含原始数据量信息用于正确计算覆盖率
            coverage_calculation_note = f"""
重要提醒：覆盖率计算必须基于原始数据量
- 原始综述关系总数：{len(enhanced_review_relations)}条
- 原始引文关系总数：{len(enhanced_citation_relations)}条
- 分析用数据：综述{len(review_data_for_analysis)}条，引文{len(citation_data_for_analysis)}条
- 覆盖率计算公式：重合关系数 / 原始综述关系总数({len(enhanced_review_relations)})
"""

            relation_prompt = get_task_relation_coverage_prompt().format(
                review_relations=json.dumps(review_data_for_analysis, ensure_ascii=False, indent=2),
                citation_relations=json.dumps(citation_data_for_analysis, ensure_ascii=False, indent=2)
            ) + coverage_calculation_note

            # 保存提示词
            debug_logger.save_prompt(
                step_name="relation_coverage_analysis",
                prompt=relation_prompt,
                model_name=model_name,
                original_review_relations_count=len(enhanced_review_relations),
                original_citation_relations_count=len(enhanced_citation_relations),
                analysis_review_relations_count=len(review_data_for_analysis),
                analysis_citation_relations_count=len(citation_data_for_analysis),
                data_processing_note=data_processing_note,
                coverage_calculation_base=len(enhanced_review_relations)
            )

            relation_response = call_llm_with_prompt(
                prompt=relation_prompt,
                model_name=model_name
            )

            # 保存LLM响应
            debug_logger.save_llm_response(
                step_name="relation_coverage_analysis",
                response=relation_response,
                model_name=model_name
            )

            # 记录响应用于调试
            logging.info(f"关系分析响应长度: {len(relation_response)} 字符")
            logging.info(f"关系分析响应前200字符: {repr(relation_response[:200])}")

            # 检查响应是否过短或被截断
            if len(relation_response.strip()) < 100:
                logging.warning(f"大模型响应过短，可能被截断。完整响应: {repr(relation_response)}")

                # 如果响应过短，直接使用默认结果
                relation_result = {
                    "relation_coverage": {
                        "total_review_relations": len(enhanced_review_relations),
                        "total_citation_relations": len(enhanced_citation_relations),
                        "overlapping_relations": 0,
                        "overall_coverage_ratio": 0.0,
                        "coverage_by_type": {
                            "improve": 0.0,
                            "optimize": 0.0,
                            "extend": 0.0,
                            "replace": 0.0,
                            "use": 0.0
                        },
                        "match_quality_breakdown": {
                            "high_confidence_matches": 0,
                            "medium_confidence_matches": 0,
                            "low_confidence_matches": 0
                        }
                    },
                    "data_quality_assessment": {
                        "review_data_completeness": combined_quality_report["review_data_quality"]["data_completeness_score"],
                        "citation_data_completeness": combined_quality_report["citation_data_quality"]["data_completeness_score"],
                        "missing_fields_impact": "响应过短，无法进行数据质量评估",
                        "confidence_level": 0.0
                    },
                    "detailed_analysis": f"关系覆盖率分析失败：大模型响应过短或被截断（响应长度：{len(relation_response)}字符）。这可能是由于网络问题或模型服务异常导致的。建议尝试使用其他模型（如Claude或DeepSeek）重新分析，或检查网络连接状态。",
                    "improvement_suggestions": [
                        "检查网络连接状态",
                        "尝试使用其他大模型",
                        "减少输入数据量重新分析"
                    ]
                }
                logging.warning("使用默认结果结构（响应过短）")

                # 保存错误信息
                debug_logger.save_error(
                    step_name="relation_coverage_response_too_short",
                    error=ValueError(f"响应过短: {len(relation_response)}字符"),
                    context="大模型响应过短或被截断"
                )
            else:
                # 响应长度正常，尝试解析JSON
                try:
                    relation_result = extract_json_from_response(relation_response)
                    logging.info("关系覆盖率JSON解析成功")

                    # 验证和修正覆盖率计算基数
                    if 'relation_coverage' in relation_result:
                        coverage_data = relation_result['relation_coverage']

                        # 确保使用原始数据量作为分母
                        if 'total_review_relations' in coverage_data:
                            if coverage_data['total_review_relations'] != len(enhanced_review_relations):
                                logging.warning(f"修正覆盖率计算基数：{coverage_data['total_review_relations']} → {len(enhanced_review_relations)}")
                                coverage_data['total_review_relations'] = len(enhanced_review_relations)

                                # 重新计算覆盖率
                                overlapping = coverage_data.get('overlapping_relations', 0)
                                coverage_data['overall_coverage_ratio'] = overlapping / len(enhanced_review_relations) if len(enhanced_review_relations) > 0 else 0.0

                        if 'total_citation_relations' in coverage_data:
                            if coverage_data['total_citation_relations'] != len(enhanced_citation_relations):
                                logging.warning(f"修正引文关系总数：{coverage_data['total_citation_relations']} → {len(enhanced_citation_relations)}")
                                coverage_data['total_citation_relations'] = len(enhanced_citation_relations)

                        # 添加计算验证信息
                        coverage_data['calculation_verification'] = {
                            "original_review_count": len(enhanced_review_relations),
                            "original_citation_count": len(enhanced_citation_relations),
                            "analysis_review_count": len(review_data_for_analysis),
                            "analysis_citation_count": len(citation_data_for_analysis),
                            "coverage_formula": f"overlapping_relations / original_review_count({len(enhanced_review_relations)})",
                            "data_sampling_applied": len(review_data_for_analysis) < len(enhanced_review_relations)
                        }

                    # 保存解析成功的结果
                    debug_logger.save_json_result(
                        step_name="relation_coverage_result_parsed",
                        data=relation_result,
                        description="成功解析的关系覆盖率分析结果（已验证计算基数）"
                    )

                    # 验证结果结构是否正确
                    if not isinstance(relation_result, dict) or 'relation_coverage' not in relation_result:
                        logging.warning("JSON解析成功但结构不正确，尝试修复")
                        if isinstance(relation_result, str):
                            raise ValueError(f"返回的是字符串而不是JSON对象: {relation_result}")
                        else:
                            relation_result = {
                                "relation_coverage": relation_result,
                                "detailed_analysis": "JSON结构已自动修复",
                                "data_quality_assessment": {
                                    "review_data_completeness": combined_quality_report["review_data_quality"]["data_completeness_score"],
                                    "citation_data_completeness": combined_quality_report["citation_data_quality"]["data_completeness_score"],
                                    "missing_fields_impact": "结构修复后的分析结果",
                                    "confidence_level": 0.5
                                }
                            }

                except Exception as json_error:
                    logging.error(f"关系覆盖率JSON解析失败: {str(json_error)}")
                    logging.error(f"原始响应: {repr(relation_response)}")

                    # 保存解析错误
                    debug_logger.save_error(
                        step_name="relation_coverage_result_parsing",
                        error=json_error,
                        context=f"关系覆盖率分析JSON解析失败，原始响应长度：{len(relation_response)}字符"
                    )

                    # 创建默认的结果结构
                    relation_result = {
                        "relation_coverage": {
                            "total_review_relations": len(enhanced_review_relations),
                            "total_citation_relations": len(enhanced_citation_relations),
                            "overlapping_relations": 0,
                            "overall_coverage_ratio": 0.0,
                            "coverage_by_type": {
                                "improve": 0.0,
                                "optimize": 0.0,
                                "extend": 0.0,
                                "replace": 0.0,
                                "use": 0.0
                            },
                            "match_quality_breakdown": {
                                "high_confidence_matches": 0,
                                "medium_confidence_matches": 0,
                                "low_confidence_matches": 0
                            }
                        },
                        "data_quality_assessment": {
                            "review_data_completeness": combined_quality_report["review_data_quality"]["data_completeness_score"],
                            "citation_data_completeness": combined_quality_report["citation_data_quality"]["data_completeness_score"],
                            "missing_fields_impact": f"JSON解析失败，无法完成分析。错误：{str(json_error)}",
                            "confidence_level": 0.0
                        },
                        "detailed_analysis": f"关系覆盖率分析失败：JSON解析错误。错误原因：{str(json_error)}。原始响应长度：{len(relation_response)}字符。建议尝试使用其他模型（如Claude或DeepSeek）重新分析。",
                        "improvement_suggestions": [
                            "尝试使用其他大模型重新分析",
                            "检查提示词格式是否正确",
                            "减少输入数据量后重试"
                        ]
                    }
                    logging.warning("使用默认结果结构（JSON解析失败）")

                    # 保存默认结果
                    debug_logger.save_json_result(
                        step_name="relation_coverage_result_fallback",
                        data=relation_result,
                        description="JSON解析失败后的默认关系覆盖率结果"
                    )



            # 保存最终结果
            final_results = {
                "review_relations_count": len(enhanced_review_relations),
                "citation_relations_count": len(enhanced_citation_relations),
                "relation_coverage": relation_result,
                "model_used": model_name,
                "data_quality_report": combined_quality_report,
                "analysis_metadata": {
                    "data_enhancement_applied": True,
                    "data_processing_note": data_processing_note,
                    "analysis_timestamp": datetime.now().isoformat()
                }
            }

            # 缓存结果
            cache_analysis_result(analysis_id, final_results)

            # 完成调试记录
            debug_logger.finalize(final_results)

            db_manager.update_processing_status(
                task_id=analysis_id,
                current_stage='分析完成',
                progress=100,
                message='任务关系覆盖率分析完成'
            )

        except Exception as e:
            error_msg = str(e)
            # 安全地记录错误，避免格式化问题
            logging.error("关系覆盖率分析失败: %s", error_msg)

            # 保存错误到调试记录
            debug_logger.save_error(
                step_name="relation_coverage_analysis_fatal_error",
                error=e,
                context="关系覆盖率分析过程中发生致命错误"
            )

            # 完成调试记录（即使失败也要保存调试信息）
            debug_logger.finalize()

            # 清理错误消息中的特殊字符，避免格式化问题
            safe_error_msg = error_msg.replace('"', "'").replace('\n', ' ').replace('\r', '')[:200]

            db_manager.update_processing_status(
                task_id=analysis_id,
                current_stage='分析失败',
                progress=-1,
                message=f'分析失败: {safe_error_msg}'
            )

    # 启动后台线程
    thread = threading.Thread(target=analysis_worker)
    thread.daemon = True
    thread.start()

@paper_analysis_api.route('/analysis/paper-task', methods=['POST'])
def analyze_paper_task():
    """论文与任务比较分析（基于现有配置）"""
    try:
        # 验证请求参数
        if 'task_id' not in request.form:
            return jsonify({"success": False, "message": "缺少任务ID"}), 400

        task_id = request.form['task_id']
        model_name = request.form.get('model', Config.DEFAULT_MODEL)

        # 检查是否有提取结果文件（优先级更高）
        extracted_file = request.files.get('extracted_file')
        paper_file = request.files.get('paper_file')

        if not extracted_file and not paper_file:
            return jsonify({"success": False, "message": "请上传论文PDF文件或提取结果JSON文件"}), 400
        
        # 验证模型支持
        if model_name not in SUPPORTED_MODELS:
            return jsonify({"success": False, "message": f"不支持的模型: {model_name}，支持的模型: {SUPPORTED_MODELS}"}), 400

        # 生成分析ID
        analysis_id = str(uuid.uuid4())

        # 确保目录存在
        os.makedirs(Config.UPLOAD_DIR, exist_ok=True)

        paper_path = None
        extracted_data = None

        # 处理提取结果文件（优先级更高）
        if extracted_file:
            if not extracted_file.filename.lower().endswith('.json'):
                return jsonify({"success": False, "message": "提取结果文件必须是JSON格式"}), 400

            try:
                extracted_content = extracted_file.read().decode('utf-8')
                extracted_data = json.loads(extracted_content)
                logging.info(f"使用提取结果文件: {extracted_file.filename}")
            except Exception as e:
                return jsonify({"success": False, "message": f"提取结果文件格式错误: {str(e)}"}), 400

        # 处理PDF文件
        if paper_file:
            if not paper_file.filename.lower().endswith('.pdf'):
                return jsonify({"success": False, "message": "论文文件必须是PDF格式"}), 400

            # 检查文件大小
            paper_file.seek(0, 2)
            file_size = paper_file.tell()
            paper_file.seek(0)

            if file_size > Config.MAX_CONTENT_LENGTH:
                return jsonify({
                    "success": False,
                    "message": f"文件大小 ({file_size / 1024 / 1024:.2f} MB) 超过限制 ({Config.MAX_CONTENT_LENGTH / 1024 / 1024:.2f} MB)"
                }), 400

            # 保存文件
            filename = secure_filename(paper_file.filename)
            paper_path = os.path.join(Config.UPLOAD_DIR, f"{analysis_id}_paper_{filename}")
            paper_file.save(paper_path)
            logging.info(f"论文文件已保存到: {paper_path}")
        
        # 创建分析任务记录（复用现有的状态管理）
        db_manager.update_processing_status(
            task_id=analysis_id,
            current_stage='初始化',
            progress=0,
            message='论文与任务比较分析任务已创建'
        )
        
        # 启动后台分析任务
        start_paper_task_analysis_with_config(analysis_id, paper_path, task_id, model_name, extracted_data)
        
        return jsonify({
            "success": True,
            "analysis_id": analysis_id,
            "message": "论文与任务比较分析任务已启动",
            "model_used": model_name
        })
        
    except Exception as e:
        logging.error(f"启动论文分析失败: {str(e)}")
        return jsonify({"success": False, "message": str(e)}), 500

@paper_analysis_api.route('/analysis/paper-paper', methods=['POST'])
def analyze_paper_paper():
    """论文与论文比较分析（基于现有配置）"""
    try:
        # 验证请求参数
        if 'paper2_file' not in request.files:
            return jsonify({"success": False, "message": "缺少论文2文件"}), 400

        paper2_file = request.files['paper2_file']
        model_name = request.form.get('model', Config.DEFAULT_MODEL)

        # 检查是否有提取结果文件（优先级更高）
        extracted_file = request.files.get('extracted_file')
        paper1_file = request.files.get('paper1_file')

        if not extracted_file and not paper1_file:
            return jsonify({"success": False, "message": "请上传论文1文件或提取结果JSON文件"}), 400
        
        # 验证模型支持
        if model_name not in SUPPORTED_MODELS:
            return jsonify({"success": False, "message": f"不支持的模型: {model_name}，支持的模型: {SUPPORTED_MODELS}"}), 400

        # 生成分析ID
        analysis_id = str(uuid.uuid4())

        # 确保目录存在
        os.makedirs(Config.UPLOAD_DIR, exist_ok=True)

        paper1_path = None
        extracted_data = None

        # 处理提取结果文件（优先级更高）
        if extracted_file:
            if not extracted_file.filename.lower().endswith('.json'):
                return jsonify({"success": False, "message": "提取结果文件必须是JSON格式"}), 400

            try:
                extracted_content = extracted_file.read().decode('utf-8')
                extracted_data = json.loads(extracted_content)
                logging.info(f"使用论文1提取结果文件: {extracted_file.filename}")
            except Exception as e:
                return jsonify({"success": False, "message": f"提取结果文件格式错误: {str(e)}"}), 400

        # 处理论文1 PDF文件
        if paper1_file:
            if not paper1_file.filename.lower().endswith('.pdf'):
                return jsonify({"success": False, "message": "论文1文件必须是PDF格式"}), 400

            # 检查文件大小
            paper1_file.seek(0, 2)
            file_size = paper1_file.tell()
            paper1_file.seek(0)

            if file_size > Config.MAX_CONTENT_LENGTH:
                return jsonify({
                    "success": False,
                    "message": f"论文1文件大小 ({file_size / 1024 / 1024:.2f} MB) 超过限制"
                }), 400

            # 保存文件
            filename1 = secure_filename(paper1_file.filename)
            paper1_path = os.path.join(Config.UPLOAD_DIR, f"{analysis_id}_paper1_{filename1}")
            paper1_file.save(paper1_path)
            logging.info(f"论文1文件已保存到: {paper1_path}")

        # 处理论文2 PDF文件
        if not paper2_file.filename.lower().endswith('.pdf'):
            return jsonify({"success": False, "message": "论文2文件必须是PDF格式"}), 400

        # 检查论文2文件大小
        paper2_file.seek(0, 2)
        file_size = paper2_file.tell()
        paper2_file.seek(0)

        if file_size > Config.MAX_CONTENT_LENGTH:
            return jsonify({
                "success": False,
                "message": f"论文2文件大小 ({file_size / 1024 / 1024:.2f} MB) 超过限制"
            }), 400

        # 保存论文2文件
        filename2 = secure_filename(paper2_file.filename)
        paper2_path = os.path.join(Config.UPLOAD_DIR, f"{analysis_id}_paper2_{filename2}")
        paper2_file.save(paper2_path)
        logging.info(f"论文2文件已保存到: {paper2_path}")

        # 创建分析任务记录
        db_manager.update_processing_status(
            task_id=analysis_id,
            current_stage='初始化',
            progress=0,
            message='论文与论文比较分析任务已创建'
        )

        # 启动后台分析任务
        start_paper_paper_analysis_with_config(analysis_id, paper1_path, paper2_path, model_name, extracted_data)
        
        return jsonify({
            "success": True,
            "analysis_id": analysis_id,
            "message": "论文与论文比较分析任务已启动",
            "model_used": model_name
        })
        
    except Exception as e:
        logging.error(f"启动论文对比分析失败: {str(e)}")
        return jsonify({"success": False, "message": str(e)}), 500

@paper_analysis_api.route('/analysis/task-relation-coverage', methods=['POST'])
def analyze_task_relation_coverage():
    """任务关系覆盖率分析（基于现有配置）"""
    try:
        # 获取请求参数
        data = request.get_json()
        if not data or 'task_id' not in data:
            return jsonify({"success": False, "message": "缺少任务ID"}), 400
        
        task_id = data['task_id']
        model_name = data.get('model', Config.DEFAULT_MODEL)
        
        # 验证模型支持
        if model_name not in SUPPORTED_MODELS:
            return jsonify({"success": False, "message": f"不支持的模型: {model_name}，支持的模型: {SUPPORTED_MODELS}"}), 400
        
        # 验证任务是否存在（宽松验证，允许没有关系数据的任务）
        try:
            # 尝试获取任务数据以验证任务存在
            task_relations = db_manager.get_relations_by_task(task_id)
            task_entities = db_manager.get_entities_by_task(task_id)

            # 只要任务存在（有实体或关系数据）就允许分析
            if not task_relations and not task_entities:
                return jsonify({"success": False, "message": f"任务 {task_id} 不存在或没有任何数据"}), 404

            # 如果没有关系数据，给出警告但仍然允许分析
            if not task_relations:
                logging.warning(f"任务 {task_id} 没有关系数据，将进行空数据分析")

        except Exception as e:
            return jsonify({"success": False, "message": f"无法访问任务数据: {str(e)}"}), 404
        
        # 生成分析ID
        analysis_id = str(uuid.uuid4())
        
        # 创建分析任务记录
        db_manager.update_processing_status(
            task_id=analysis_id,
            current_stage='初始化',
            progress=0,
            message='任务关系覆盖率分析任务已创建'
        )
        
        # 启动后台分析任务
        start_relation_coverage_analysis_with_config(analysis_id, task_id, model_name)
        
        return jsonify({
            "success": True,
            "analysis_id": analysis_id,
            "message": "任务关系覆盖率分析任务已启动",
            "model_used": model_name
        })
        
    except Exception as e:
        logging.error(f"启动关系覆盖率分析失败: {str(e)}")
        return jsonify({"success": False, "message": str(e)}), 500

@paper_analysis_api.route('/analysis/history', methods=['GET'])
def get_analysis_history():
    """获取论文分析历史记录"""
    try:
        # 获取所有论文分析相关的缓存文件
        cache_dir = "data/cache"
        history_records = []

        if os.path.exists(cache_dir):
            for filename in os.listdir(cache_dir):
                if filename.startswith("paper_analysis_") and filename.endswith(".json"):
                    file_path = os.path.join(cache_dir, filename)
                    try:
                        # 获取文件修改时间
                        file_stat = os.stat(file_path)
                        created_time = datetime.fromtimestamp(file_stat.st_mtime)

                        # 从文件名提取分析ID
                        analysis_id = filename.replace("paper_analysis_", "").replace(".json", "")

                        # 尝试读取文件内容获取更多信息
                        with open(file_path, 'r', encoding='utf-8') as f:
                            analysis_data = json.load(f)

                        # 确定分析类型
                        analysis_type = "未知类型"
                        if "relation_coverage" in analysis_data:
                            analysis_type = "关系覆盖率分析"
                        elif "coverage_analysis" in analysis_data:
                            analysis_type = "论文与任务比较"
                        elif "paper_comparison" in analysis_data:
                            analysis_type = "论文与论文比较"

                        # 获取任务ID（如果有）
                        task_id = analysis_data.get("task_id", "未知")

                        history_records.append({
                            "analysis_id": analysis_id,
                            "analysis_type": analysis_type,
                            "task_id": task_id,
                            "created_time": created_time.strftime('%Y-%m-%d %H:%M:%S'),
                            "status": "已完成",
                            "file_size": file_stat.st_size
                        })

                    except Exception as e:
                        logging.warning(f"读取分析文件 {filename} 失败: {str(e)}")
                        continue

        # 按创建时间倒序排列
        history_records.sort(key=lambda x: x['created_time'], reverse=True)

        return jsonify({
            "success": True,
            "analyses": history_records,
            "total_count": len(history_records),
            "message": f"找到 {len(history_records)} 条分析记录"
        })

    except Exception as e:
        logging.error(f"获取论文分析历史记录失败: {str(e)}")
        return jsonify({
            "success": False,
            "message": f"获取历史记录失败: {str(e)}"
        }), 500

@paper_analysis_api.route('/analysis/<analysis_id>/results', methods=['GET'])
def get_analysis_results(analysis_id):
    """获取分析结果（基于现有配置）"""
    try:
        # 先尝试从缓存加载
        cached_result = load_cached_analysis_result(analysis_id)
        if cached_result:
            return jsonify({
                "success": True,
                "analysis_id": analysis_id,
                "status": "completed",
                "progress": 100,
                "current_stage": "分析完成",
                "message": "分析已完成（从缓存加载）",
                "results": cached_result
            })

        # 从数据库获取分析结果（复用现有的状态管理）
        task_data = db_manager.get_processing_status(analysis_id)

        if not task_data:
            return jsonify({"success": False, "message": "分析任务不存在"}), 404
        
        return jsonify({
            "success": True,
            "analysis_id": analysis_id,
            "status": task_data.get('status', 'unknown'),
            "progress": task_data.get('progress', 0),
            "current_stage": task_data.get('current_stage', ''),
            "message": task_data.get('message', ''),
            "results": task_data.get('results')
        })
        
    except Exception as e:
        logging.error(f"获取分析结果失败: {str(e)}")
        return jsonify({"success": False, "message": str(e)}), 500

@paper_analysis_api.route('/analysis/models', methods=['GET'])
def get_supported_models():
    """获取支持的模型列表"""
    try:
        # 模型能力映射（内置）
        MODEL_CAPABILITIES = {
            'qwen-long': {
                'supports_pdf': True,
                'max_tokens': 32000,
                'good_for': ['text_analysis', 'chinese_content']
            },
            'gemini-2.0-flash': {
                'supports_pdf': True,
                'max_tokens': 128000,
                'good_for': ['multimodal', 'pdf_analysis']
            },
            'claude-3-7-sonnet-20250219': {
                'supports_pdf': False,
                'max_tokens': 200000,
                'good_for': ['reasoning', 'analysis']
            },
            'gpt-3.5-turbo': {
                'supports_pdf': False,
                'max_tokens': 16000,
                'good_for': ['general_purpose']
            },
            'gpt-4.1-mini': {
                'supports_pdf': True,
                'max_tokens': 128000,
                'good_for': ['complex_reasoning', 'multimodal']
            },
            'deepseek-v3': {
                'supports_pdf': False,
                'max_tokens': 64000,
                'good_for': ['coding', 'reasoning']
            }
        }

        models_info = []
        for model in SUPPORTED_MODELS:
            model_info = {
                "name": model,
                "is_default": model == Config.DEFAULT_MODEL,
                "capabilities": MODEL_CAPABILITIES.get(model, {})
            }
            models_info.append(model_info)

        return jsonify({
            "success": True,
            "models": models_info,
            "default_model": Config.DEFAULT_MODEL
        })

    except Exception as e:
        logging.error(f"获取模型列表失败: {str(e)}")
        return jsonify({"success": False, "message": str(e)}), 500

@paper_analysis_api.route('/analysis/tasks', methods=['GET'])
def get_available_tasks():
    """获取可用的任务列表，复用现有的比较分析历史记录"""
    try:
        # 使用现有的比较分析历史记录作为任务列表
        tasks = db_manager.get_comparison_history(limit=200)

        # 为论文分析页面格式化任务数据
        formatted_tasks = []
        for task in tasks:
            # 确保每个任务都有必要的字段
            formatted_task = {
                "task_id": task.get('task_id', ''),
                "id": task.get('task_id', ''),  # 兼容不同的ID字段名
                "name": task.get('task_name', task.get('name', f"任务 {task.get('task_id', 'Unknown')}")),
                "description": task.get('description', f"创建于 {task.get('created_at', 'Unknown')}"),
                "status": task.get('status', 'completed'),
                "created_at": task.get('created_at', ''),
                "entity_count": task.get('entity_count', 0),
                "relation_count": task.get('relation_count', 0)
            }

            # 如果任务名为空或只是任务ID，尝试生成更友好的名称
            if not formatted_task["name"] or formatted_task["name"] == formatted_task["task_id"]:
                if formatted_task["entity_count"] > 0:
                    formatted_task["name"] = f"任务 {formatted_task['task_id'][:8]}... ({formatted_task['entity_count']}个实体)"
                else:
                    formatted_task["name"] = f"任务 {formatted_task['task_id'][:8]}..."

            # 包含所有任务，但会检查是否有实际数据
            # 先添加所有任务，后面会验证数据
            formatted_tasks.append(formatted_task)

        # 暂时跳过数据验证以提高性能，直接使用所有任务
        # 用户可以选择任务后，在实际分析时再验证数据
        # 这样可以避免加载页面时的长时间等待

        # 为所有任务设置默认状态
        for task in formatted_tasks:
            if not task.get("name") or task["name"] == task["task_id"]:
                task["name"] = f"任务 {task['task_id'][:8]}..."
            task["status"] = "available"  # 标记为可用，实际验证在分析时进行

        # 按创建时间倒序排列，最新的在前面
        formatted_tasks.sort(key=lambda x: x.get('created_at', ''), reverse=True)

        # 如果没有找到任务，返回示例数据
        if not formatted_tasks:
            formatted_tasks = [
                {
                    "task_id": "example_task_1",
                    "id": "example_task_1",
                    "name": "示例任务1 (请先上传论文生成任务)",
                    "description": "这是一个示例任务，请先通过上传页面处理论文生成实际任务",
                    "status": "example",
                    "entity_count": 0,
                    "relation_count": 0
                }
            ]

        return jsonify({
            "success": True,
            "tasks": formatted_tasks,
            "total_count": len(formatted_tasks),
            "message": f"找到 {len(formatted_tasks)} 个可用任务" if formatted_tasks and formatted_tasks[0]["status"] != "example" else "暂无实际任务，请先上传论文生成任务数据"
        })

    except Exception as e:
        logging.error(f"获取任务列表失败: {str(e)}")
        # 返回示例数据而不是完全失败
        return jsonify({
            "success": True,
            "tasks": [
                {
                    "task_id": "example_task_1",
                    "id": "example_task_1",
                    "name": "示例任务1 (数据库连接失败)",
                    "description": "无法连接到数据库，这是示例数据",
                    "status": "error",
                    "entity_count": 0,
                    "relation_count": 0
                }
            ],
            "message": f"获取任务列表时出错: {str(e)}，显示示例数据"
        }), 200  # 返回200而不是500，避免前端报错

@paper_analysis_api.route('/analysis/<analysis_id>', methods=['DELETE'])
def delete_analysis_task(analysis_id):
    """删除分析任务和相关文件"""
    try:
        # 删除缓存文件
        cache_file = os.path.join(Config.CACHE_DIR, f"paper_analysis_{analysis_id}.json")
        if os.path.exists(cache_file):
            os.remove(cache_file)
            logging.info(f"已删除缓存文件: {cache_file}")
        
        # 删除上传的PDF文件
        upload_dir = Config.UPLOAD_DIR
        for filename in os.listdir(upload_dir):
            if filename.startswith(analysis_id):
                file_path = os.path.join(upload_dir, filename)
                os.remove(file_path)
                logging.info(f"已删除上传文件: {file_path}")
        
        # 删除数据库记录
        db_manager.delete_processing_status(analysis_id)
        
        return jsonify({
            "success": True,
            "message": "分析任务已删除"
        })
        
    except Exception as e:
        logging.error(f"删除分析任务失败: {str(e)}")
        return jsonify({"success": False, "message": str(e)}), 500

# API蓝图已完成，可以直接使用
