# AutoSurvey API 集成规范

## 概述

本文档详细说明了与AutoSurvey系统集成的API规范、调用方式、认证机制和限制条件。

## AutoSurvey 系统架构

### 核心组件
- **数据库模块**: 存储arXiv论文数据和嵌入向量
- **检索模块**: 基于FAISS的向量检索
- **生成模块**: 基于大语言模型的综述生成
- **评估模块**: 综述质量评估

### 输入输出格式

#### 标准输入格式
```json
{
  "topic": "综述主题",
  "entities": [
    {
      "id": "实体ID",
      "type": "实体类型(algorithm/dataset/metric)",
      "name": "实体名称",
      "title": "实体标题",
      "year": 年份,
      "authors": ["作者列表"],
      "description": "描述",
      "metadata": {
        "task": "任务类型",
        "source": "数据来源"
      }
    }
  ],
  "relations": [
    {
      "source": "源实体ID",
      "target": "目标实体ID",
      "type": "关系类型",
      "source_type": "源实体类型",
      "target_type": "目标实体类型",
      "description": "关系描述",
      "evidence": "证据",
      "confidence": 置信度
    }
  ],
  "generation_params": {
    "section_num": 7,
    "subsection_len": 700,
    "rag_num": 60,
    "outline_reference_num": 1500,
    "model": "gpt-4o-2024-05-13"
  }
}
```

#### 标准输出格式
```json
{
  "survey_id": "综述ID",
  "topic": "综述主题",
  "content": "综述内容(Markdown格式)",
  "outline": {
    "sections": [
      {
        "title": "章节标题",
        "subsections": ["子章节列表"]
      }
    ]
  },
  "references": [
    {
      "title": "论文标题",
      "authors": ["作者列表"],
      "year": 年份,
      "venue": "发表场所",
      "url": "论文链接"
    }
  ],
  "metadata": {
    "generation_time": "生成时间",
    "model": "使用的模型",
    "word_count": 字数,
    "section_count": 章节数,
    "reference_count": 参考文献数
  },
  "quality_metrics": {
    "completeness_score": 完整性评分,
    "coherence_score": 连贯性评分,
    "novelty_score": 新颖性评分,
    "overall_score": 总体评分
  }
}
```

## API 调用方式

### 方式一：命令行调用（推荐用于集成）

```bash
python main.py \
  --topic "深度学习在自然语言处理中的应用" \
  --input_file "input_data.json" \
  --output_dir "./output/" \
  --model "gpt-4o-2024-05-13" \
  --section_num 7 \
  --subsection_len 700 \
  --rag_num 60 \
  --outline_reference_num 1500 \
  --api_key "your-api-key" \
  --api_url "https://api.openai.com/v1/chat/completions"
```

### 方式二：Python API调用

```python
from autosurvey import AutoSurveyGenerator

# 初始化生成器
generator = AutoSurveyGenerator(
    api_key="your-api-key",
    api_url="https://api.openai.com/v1/chat/completions",
    model="gpt-4o-2024-05-13",
    db_path="./database"
)

# 生成综述
result = generator.generate_survey(
    topic="深度学习在自然语言处理中的应用",
    input_data=input_data,
    section_num=7,
    subsection_len=700,
    rag_num=60,
    outline_reference_num=1500
)
```

### 方式三：HTTP API调用（需要包装）

由于AutoSurvey原生不提供HTTP API，我们需要创建包装服务：

```python
# 包装服务示例
from flask import Flask, request, jsonify
import subprocess
import json
import os

app = Flask(__name__)

@app.route('/api/generate_survey', methods=['POST'])
def generate_survey():
    data = request.json
    
    # 保存输入数据到临时文件
    input_file = f"temp_input_{uuid.uuid4()}.json"
    with open(input_file, 'w') as f:
        json.dump(data['input_data'], f)
    
    # 调用AutoSurvey
    cmd = [
        'python', 'main.py',
        '--topic', data['topic'],
        '--input_file', input_file,
        '--output_dir', './temp_output/',
        '--model', data.get('model', 'gpt-4o-2024-05-13'),
        '--api_key', data['api_key']
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    # 清理临时文件
    os.remove(input_file)
    
    if result.returncode == 0:
        # 读取输出结果
        output_file = './temp_output/survey_result.json'
        with open(output_file, 'r') as f:
            survey_result = json.load(f)
        return jsonify({"success": True, "result": survey_result})
    else:
        return jsonify({"success": False, "error": result.stderr})
```

## 认证机制

### OpenAI API认证
AutoSurvey使用OpenAI API进行文本生成，需要提供有效的API密钥：

```python
# 环境变量方式
export OPENAI_API_KEY="sk-your-api-key"

# 参数传递方式
--api_key "sk-your-api-key"

# 配置文件方式
{
  "api_key": "sk-your-api-key",
  "api_url": "https://api.openai.com/v1/chat/completions"
}
```

### 数据库访问认证
如果使用自定义数据库，需要配置数据库连接：

```python
database_config = {
    "type": "mysql",  # 或 "sqlite", "postgresql"
    "host": "localhost",
    "port": 3306,
    "username": "user",
    "password": "password",
    "database": "autosurvey_db"
}
```

## 限制条件和约束

### 1. API调用限制
- **OpenAI API限制**: 
  - 每分钟请求数限制（根据API密钥等级）
  - 每月token使用量限制
  - 单次请求最大token数限制

### 2. 数据库限制
- **FAISS索引大小**: 默认数据库包含53万篇论文
- **检索性能**: 大规模检索可能影响响应时间
- **存储空间**: 完整数据库需要约10GB存储空间

### 3. 生成参数限制
```python
PARAMETER_LIMITS = {
    "section_num": {"min": 3, "max": 15, "default": 7},
    "subsection_len": {"min": 300, "max": 1500, "default": 700},
    "rag_num": {"min": 20, "max": 200, "default": 60},
    "outline_reference_num": {"min": 500, "max": 3000, "default": 1500}
}
```

### 4. 输入数据限制
- **实体数量**: 建议不超过1000个实体
- **关系数量**: 建议不超过5000个关系
- **主题长度**: 建议不超过200字符
- **描述长度**: 单个实体描述不超过1000字符

### 5. 输出限制
- **综述长度**: 通常在5000-20000字之间
- **参考文献数**: 通常在50-500篇之间
- **生成时间**: 根据复杂度，5-30分钟不等

## 错误处理

### 常见错误类型
1. **API认证错误**: 无效的API密钥
2. **参数错误**: 超出限制范围的参数
3. **数据格式错误**: 输入数据格式不正确
4. **资源不足错误**: 内存或存储空间不足
5. **网络错误**: API调用超时或失败

### 错误响应格式
```json
{
  "success": false,
  "error_code": "INVALID_API_KEY",
  "error_message": "提供的API密钥无效",
  "details": {
    "timestamp": "2024-01-01T12:00:00Z",
    "request_id": "req_123456"
  }
}
```

### 重试机制
```python
RETRY_CONFIG = {
    "max_retries": 3,
    "backoff_factor": 2,
    "retry_on_errors": [
        "RATE_LIMIT_EXCEEDED",
        "NETWORK_TIMEOUT",
        "TEMPORARY_UNAVAILABLE"
    ]
}
```

## 性能优化建议

### 1. 数据预处理
- 提前清洗和验证输入数据
- 去除重复实体和关系
- 优化实体描述长度

### 2. 参数调优
- 根据需求调整`rag_num`和`outline_reference_num`
- 平衡生成质量和速度
- 使用合适的模型（GPT-4 vs GPT-3.5）

### 3. 缓存策略
- 缓存相似主题的检索结果
- 复用已生成的章节内容
- 缓存实体嵌入向量

### 4. 并行处理
- 并行处理多个章节生成
- 异步执行检索和生成任务
- 使用队列管理长时间运行的任务

## 集成最佳实践

### 1. 数据准备
```python
def prepare_autosurvey_input(task_data):
    """准备AutoSurvey输入数据"""
    # 数据清洗
    entities = clean_entities(task_data.entities)
    relations = clean_relations(task_data.relations)
    
    # 格式转换
    autosurvey_entities = convert_entities_format(entities)
    autosurvey_relations = convert_relations_format(relations)
    
    # 验证数据
    validate_input_data(autosurvey_entities, autosurvey_relations)
    
    return {
        "entities": autosurvey_entities,
        "relations": autosurvey_relations
    }
```

### 2. 异步处理
```python
import asyncio
from concurrent.futures import ThreadPoolExecutor

async def generate_survey_async(input_data, topic, params):
    """异步生成综述"""
    loop = asyncio.get_event_loop()
    
    with ThreadPoolExecutor() as executor:
        result = await loop.run_in_executor(
            executor, 
            call_autosurvey, 
            input_data, topic, params
        )
    
    return result
```

### 3. 监控和日志
```python
import logging
from datetime import datetime

def setup_autosurvey_logging():
    """设置AutoSurvey集成日志"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('autosurvey_integration.log'),
            logging.StreamHandler()
        ]
    )

def log_generation_metrics(start_time, end_time, result):
    """记录生成指标"""
    duration = (end_time - start_time).total_seconds()
    
    logging.info(f"综述生成完成: "
                f"耗时={duration:.2f}秒, "
                f"字数={result.get('word_count', 0)}, "
                f"章节数={result.get('section_count', 0)}")
```

## 版本兼容性

### AutoSurvey版本支持
- **v1.0**: 基础功能支持
- **v1.1**: 增强的实体关系处理
- **v1.2**: 改进的质量评估（当前推荐版本）

### 依赖版本要求
```
python>=3.10
torch>=1.9.0
transformers>=4.20.0
faiss-cpu>=1.7.0
openai>=1.0.0
```

## 故障排除

### 常见问题及解决方案

1. **内存不足**
   - 减少`rag_num`参数
   - 分批处理大量实体
   - 使用更小的嵌入模型

2. **生成质量不佳**
   - 增加`outline_reference_num`
   - 使用更高级的模型
   - 改进输入数据质量

3. **生成时间过长**
   - 减少章节数量
   - 降低子章节长度
   - 使用更快的模型

4. **API调用失败**
   - 检查网络连接
   - 验证API密钥
   - 实施重试机制
