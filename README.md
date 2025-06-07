# 算法要素关系图生成系统

## 项目介绍

本系统是一个算法要素关系图生成系统，通过分析综述论文及其引用文献，提取算法实体及其要素，构建算法进化关系图，并自动生成算法评论。系统支持以下主要功能：

- 通过上传PDF文档自动提取算法实体及其关系
- 支持算法实体的查看、编辑和管理
- 支持算法演化关系的查看、添加和修改
- 提供交互式的关系图可视化展示
- 支持从Excel/CSV表格导入数据
- 基于图数据生成算法评论

## 系统架构

系统采用前后端分离架构，主要包括以下模块：

- 数据获取与解析模块：解析PDF文档，提取算法实体和关系
- 数据处理与存储模块：处理和规范化数据，存储到数据库
- 知识图谱构建模块：基于实体和关系构建知识图谱
- API服务模块：提供RESTful API接口
- 前端展示与交互模块：提供用户界面和交互功能

## 环境要求

- Python 3.8+
- Flask 2.0+
- Neo4j 4.0+ 

## 安装与部署

1. 克隆代码库

```bash
git clone <repository-url>
cd <repository-directory>
```

2. 创建并激活虚拟环境

```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
```

3. 安装依赖

```bash
pip install -r requirements.txt
```

4. 配置环境变量（可选）

创建`.env`文件，添加以下内容：

```
FLASK_APP=run.py
FLASK_ENV=development
SECRET_KEY=your-secret-key
```

5. 初始化数据库

```bash
python -c "from app.modules.db_manager import DatabaseManager; DatabaseManager().init_db()"
```

6. 启动应用

```bash
python run.py
```

或者

```bash
flask run
```

7. 访问应用

打开浏览器，访问 http://localhost:5000

## 使用说明

1. 首页：通过上传PDF文档或导入表格数据创建算法实体和关系
2. 算法实体表格：查看、添加、编辑和删除算法实体
3. 关系图展示：交互式查看算法演化关系图，可以添加新关系

## 最近更新

- **2023-12-20**: 修复了从嵌套数据结构中正确获取`source`字段的问题。现在系统能够正确处理嵌套在`algorithm_entity`、`dataset_entity`和`metric_entity`对象中的数据。
- **2023-12-20**: 增强了数据库操作，现在能够识别并处理嵌套结构中的实体数据，保障数据的一致性和完整性。
- **2023-12-20**: 更新了文档，明确说明了系统中使用的嵌套数据结构及其正确的访问方式。

## 数据格式

本系统使用嵌套结构存储实体数据，实体可能包含在嵌套对象中。

### 实体嵌套结构

系统中的实体数据可能以以下两种形式存在：

1. 直接实体:
```json
{
  "entity_id": "AlexNet_2012",
  "name": "AlexNet",
  "entity_type": "Algorithm",
  "year": 2012,
  "source": "综述",
  // 其他字段...
}
```

2. 嵌套实体（推荐格式）:
```json
{
  "algorithm_entity": {
    "entity_id": "AlexNet_2012",
    "name": "AlexNet",
    "entity_type": "Algorithm", 
    "year": 2012,
    "source": "综述",
    // 其他字段...
  }
}
```

**重要**: 当处理实体时，必须检查是否存在嵌套结构，并从正确的位置获取字段值。例如，获取`source`字段的正确方式：

```python
def get_source(entity):
    if 'algorithm_entity' in entity:
        return entity['algorithm_entity'].get('source', '未知')
    elif 'dataset_entity' in entity:
        return entity['dataset_entity'].get('source', '未知')
    elif 'metric_entity' in entity:
        return entity['metric_entity'].get('source', '未知')
    else:
        return entity.get('source', '未知')
```

### 算法实体

```json
{
  "id": 1,
  "name": "AlexNet",
  "entity_type": "Algorithm",
  "year": 2012,
  "authors": ["Alex Krizhevsky", "Ilya Sutskever", "Geoffrey Hinton"],
  "task": "图像分类",
  "dataset": ["ImageNet"],
  "metrics": ["Top-1 Accuracy", "Top-5 Accuracy"],
  "architecture": {
    "components": ["Conv", "ReLU", "MaxPool", "FC"],
    "connections": ["Sequential"],
    "mechanisms": ["Dropout"]
  },
  "methodology": {
    "training_strategy": ["SGD with momentum", "Learning rate decay"],
    "parameter_tuning": ["Grid search"]
  },
  "feature_processing": ["Image augmentation"],
  "source": "综述"  // 实体来源: "综述", "引文" 或 "未知"
}
```

### 演化关系

```json
{
  "id": 1,
  "from_entity_id": 1,
  "to_entity_id": 2,
  "relation_type": "Improve",
  "structure": "Architecture.Components",
  "detail": "增加了更深的网络层",
  "evidence": "论文第3节",
  "confidence": 0.95,
  "source": "综述"  // 关系来源: "综述", "引文" 或 "未知"
}
```

## 贡献者

- [Your Name]

## 许可证

[Your License] 