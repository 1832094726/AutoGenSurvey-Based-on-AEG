# 论文分析提示词模板
# 文件：app/modules/paper_analysis_prompts.py

def get_method_extraction_prompt():
    """获取增强版方法提取提示词（中文提示词，英文输出）"""
    return """
你是一个专业的学术论文分析助手。请仔细阅读提供的论文，全面深入地提取其中的研究方法和相关算法。

【重要】提取丰富度要求：
- 目标提取10-20个不同的研究方法
- 每个方法下提取3-8个具体算法/技术
- 总算法数量应达到30-60个
- 确保覆盖论文中提到的所有技术细节

【全面分析要求】：
1. **主要研究方法**：识别论文的核心方法论章节和技术路线
2. **具体技术方法**：提取每个算法、模型、技术的具体名称
3. **实验方法**：包含数据处理、特征提取、模型训练、评估方法
4. **对比方法**：提取论文中用于对比的基线方法和算法
5. **创新方法**：识别论文提出的新方法、改进方法、组合方法
6. **数据处理方法**：包含预处理、后处理、数据增强等技术
7. **评估方法**：提取评估指标、验证方法、测试策略
8. **优化方法**：包含训练策略、超参数优化、正则化技术

【提取范围扩展】：
- 不仅限于方法论章节，还要扫描引言、相关工作、实验、结论等所有章节
- 提取论文中提到的所有具体算法名称（如CNN、LSTM、Transformer、ResNet等）
- 包含所有模型架构、网络结构、算法变体
- 提取数据集处理方法、特征工程技术
- 包含所有评估指标和验证方法
- 提取优化算法、损失函数、激活函数等技术细节

【算法提取细化】：
- 提取具体的模型名称（如ResNet-50、BERT-base、GPT-3等）
- 包含算法的变体和改进版本
- 提取组合方法和集成技术
- 包含预训练模型、微调方法
- 提取注意力机制、正则化技术等组件

输出要求：
- 严格按照JSON格式输出
- 所有内容必须为英文
- 方法名称要准确反映论文内容，可以是具体的技术领域
- 算法名称使用标准术语，包含具体模型名称
- 提供简要但准确的描述

JSON格式：
{
  "methods": [
    {
      "name": "方法名称（英文，如：Deep Learning Methods, Computer Vision Methods, Natural Language Processing Methods, Data Preprocessing Methods, Evaluation Methods等）",
      "section": "对应章节号或ALL（如果跨章节）",
      "description": "方法的简要描述（英文）",
      "algorithms": [
        {
          "name": "算法名称（英文，如：CNN, ResNet-50, BERT, Transformer, Adam Optimizer等）",
          "description": "算法的简要描述和应用场景（英文）",
          "context": "在论文中的具体应用上下文（英文）"
        }
      ]
    }
  ]
}

【提取示例参考】：
如果论文涉及深度学习图像分类，应该提取类似以下丰富内容：
- Deep Learning Methods: CNN, ResNet, DenseNet, EfficientNet, Vision Transformer
- Data Processing Methods: Data Augmentation, Normalization, Resize, Crop
- Training Methods: Transfer Learning, Fine-tuning, Multi-task Learning
- Optimization Methods: Adam, SGD, Learning Rate Scheduling, Dropout
- Evaluation Methods: Cross-validation, Accuracy, Precision, Recall, F1-score
- Computer Vision Methods: Object Detection, Feature Extraction, Image Classification

请开始全面深入地分析论文内容，确保提取的方法和算法数量丰富且全面：
"""

def get_enhanced_method_extraction_prompt_with_reference(reference_data_str):
    """获取参考引文数据的增强版方法提取提示词"""
    return f"""
你是一个专业的学术论文分析助手。请仔细阅读提供的论文，全面深入地提取其中的研究方法和相关算法。

【参考引文关系数据】：
以下是相关任务的引文关系数据，包含了该领域的主要研究方法和算法，请参考这些内容来指导你的提取工作：

{reference_data_str}

【重要】基于参考数据的提取要求：
- 参考数据显示该领域有丰富的研究方法和算法
- 你的提取结果应该达到相似的丰富度和覆盖面
- 目标提取15-25个不同的研究方法
- 每个方法下提取4-10个具体算法/技术
- 总算法数量应达到50-80个，与参考数据相当

【全面分析要求】：
1. **核心技术方法**：深度挖掘论文中的所有技术细节
2. **算法模型**：提取所有提到的具体算法、模型、架构名称
3. **实验技术**：包含数据处理、特征工程、模型训练的所有技术
4. **评估方法**：提取所有评估指标、验证策略、测试方法
5. **优化技术**：包含训练策略、超参数调优、正则化等
6. **数据方法**：数据预处理、增强、采样、标注等技术
7. **对比基线**：论文中用于对比的所有基线方法和算法
8. **创新技术**：论文提出的新方法、改进方案、组合策略

【提取策略】：
- 逐段扫描论文，不遗漏任何技术细节
- 特别关注方法论、实验设计、相关工作等章节
- 提取所有具体的算法名称、模型架构、技术组件
- 包含论文中引用的经典方法和最新技术
- 提取不同粒度的方法：从高层框架到具体实现细节

【算法提取细化】：
- 具体模型：ResNet-50, BERT-large, GPT-3, Transformer等
- 算法变体：CNN variants, RNN variants, Attention mechanisms
- 优化器：Adam, SGD, RMSprop, AdamW等
- 损失函数：Cross-entropy, MSE, Focal Loss等
- 正则化：Dropout, Batch Normalization, Layer Normalization
- 激活函数：ReLU, GELU, Swish, Sigmoid等

输出要求：
- 严格按照JSON格式输出
- 所有内容必须为英文
- 方法名称要具体且有区分度
- 算法名称使用标准术语，包含版本信息
- 确保提取的丰富度与参考数据相当

JSON格式：
{{
  "methods": [
    {{
      "name": "具体方法名称（如：Convolutional Neural Network Methods, Transformer-based Methods, Data Augmentation Methods等）",
      "section": "对应章节号或ALL",
      "description": "方法的详细描述（英文）",
      "algorithms": [
        {{
          "name": "具体算法名称（如：ResNet-50, BERT-base, Adam Optimizer, Cross-Entropy Loss等）",
          "description": "算法的详细描述和应用场景（英文）",
          "context": "在论文中的具体应用上下文和作用（英文）"
        }}
      ]
    }}
  ]
}}

请基于参考数据的丰富度，全面深入地分析论文内容，确保提取结果的数量和质量与引文数据相当：
"""

def get_coverage_analysis_prompt():
    """获取智能覆盖率分析提示词（基于语义理解的智能匹配）"""
    return """
你是一个专业的学术内容智能分析助手。请基于论文中提取的研究方法，智能地从引文数据中识别和匹配对应的方法，计算覆盖率。

【核心任务】：
不是简单的名称匹配，而是基于论文方法的语义内容，智能识别引文数据中的对应技术和方法。

【方法的智能匹配策略】：
1. **语义理解匹配**：理解论文方法的技术本质，在引文数据中寻找相同或相关的技术
2. **技术领域映射**：将论文方法映射到引文数据的技术领域和算法集合
3. **多层次匹配**：
   - 直接匹配：相同的算法名称（如CNN、BERT、ResNet）
   - 类别匹配：相同技术类别（如深度学习方法、自然语言处理方法）
   - 功能匹配：相同功能目的（如图像分类、文本分析、特征提取）
算法匹配依靠语义匹配
【分析数据】：
引文数据（技术资源库）：
{reference_data}

论文方法（待分析内容）：
{comparison_data}

【分析要求】：
1. 对于论文中的每个方法，智能地在引文数据中寻找对应的技术
2. 不要局限于严格的名称匹配，要理解技术的本质和关联性
3. 考虑技术演进关系（如CNN → ResNet → EfficientNet）
4. 考虑应用领域关联（如计算机视觉方法可能对应多种图像处理算法）
5. 提供合理的匹配解释和覆盖率计算

【输出要求】：
- 严格按照JSON格式输出
- 所有内容必须为英文
- 提供智能匹配的详细说明
- 计算基于语义理解的覆盖率

【输出格式】：
{
  "method_coverage": {
    "total_reference_methods": 数字,
    "total_paper_methods": 数字,
    "intelligently_matched_methods": 数字,
    "coverage_ratio": 小数,
    "intelligent_matches": [
      {
        "paper_method": "论文中的方法名（英文）",
        "matched_reference_methods": ["引文数据中的对应方法（英文）"],
        "match_type": "direct|category|functional|evolutionary",
        "match_reasoning": "匹配推理说明（英文）",
        "confidence_score": 小数
      }
    ],
    "unmatched_paper_methods": ["无法在引文数据中找到对应的论文方法（英文）"]
  },
  "algorithm_coverage": {
    "total_reference_algorithms": 数字,
    "total_paper_algorithms": 数字,
    "intelligently_matched_algorithms": 数字,
    "coverage_ratio": 小数,
    "intelligent_matches": [
      {
        "paper_algorithm": "论文算法名（英文）",
        "matched_reference_algorithms": ["引文数据中的对应算法（英文）"],
        "match_type": "direct|variant|category|functional",
        "match_reasoning": "匹配推理说明（英文）",
        "confidence_score": 小数
      }
    ],
    "unmatched_paper_algorithms": ["无法在引文数据中找到对应的论文算法（英文）"]
  },
  "detailed_analysis": "基于智能语义匹配的详细覆盖率分析说明（必须为英文）"
}

请基于语义理解和技术关联性进行智能分析：
"""

def get_task_relation_coverage_prompt():
    """获取任务关系覆盖率分析提示词（中文提示词，英文输出）"""
    return """
你是一个专业的学术关系分析助手。请分析任务数据中综述关系和引文关系之间的覆盖率。

分析要求：
1. 比较综述关系（来自综述论文）与引文关系（来自引用论文）
2. 计算重合比例：重合关系数 / 综述关系总数
3. 考虑关系匹配的语义相似性
4. 提供详细的覆盖模式分析

任务数据：
综述关系：{review_relations}
引文关系：{citation_relations}

匹配规则：
- 关系被认为匹配的条件：
  1. 相同的关系类型（improve, optimize, extend, replace, use）
  2. 相似的实体对（考虑同义词和变体）
  3. 语义相似度 > 0.7

输出要求：
- 严格按照JSON格式输出
- 所有内容必须为英文
- 提供详细的英文分析说明

输出格式：
{
  "relation_coverage": {
    "total_review_relations": 数字,
    "total_citation_relations": 数字,
    "overlapping_relations": 数字,
    "overall_coverage_ratio": 小数,
    "coverage_by_type": {
      "improve": 小数,
      "optimize": 小数,
      "extend": 小数,
      "replace": 小数,
      "use": 小数
    },
    "relation_type_breakdown": {
      "improve": {
        "review_count": 数字,
        "citation_count": 数字,
        "overlap_count": 数字,
        "coverage_ratio": 小数
      },
      "optimize": {
        "review_count": 数字,
        "citation_count": 数字,
        "overlap_count": 数字,
        "coverage_ratio": 小数
      },
      "extend": {
        "review_count": 数字,
        "citation_count": 数字,
        "overlap_count": 数字,
        "coverage_ratio": 小数
      },
      "replace": {
        "review_count": 数字,
        "citation_count": 数字,
        "overlap_count": 数字,
        "coverage_ratio": 小数
      },
      "use": {
        "review_count": 数字,
        "citation_count": 数字,
        "overlap_count": 数字,
        "coverage_ratio": 小数
      }
    }
  },
  "detailed_analysis": "关系覆盖模式的详细分析（必须为英文），包括哪些类型的关系覆盖良好，哪些在引文中相比综述缺失的见解"
}

请开始分析：
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

输出要求：
- 严格按照JSON格式输出
- 所有内容必须为英文
- 方法名称使用标准学术术语

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

# 支持的模型列表（基于现有配置）
SUPPORTED_MODELS = [
    'qwen-long',           # 默认模型
    'gemini-2.0-flash',
    'claude-3-7-sonnet-20250219',
    'gpt-3.5-turbo',
    'gpt-4.1-mini',
    'deepseek-v3'
]

# 模型能力映射
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

def get_recommended_model_for_task(task_type: str) -> str:
    """根据任务类型推荐最适合的模型"""
    from app.config import Config
    
    
    # 默认返回配置中的默认模型
    return Config.DEFAULT_MODEL
