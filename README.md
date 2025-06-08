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

- **2023-12-27**: 改进数据库结构，将所有表的主键从单一ID改为联合主键(ID, task_id, source)，提升多任务数据管理能力和避免跨任务ID冲突。
- **2023-12-20**: 修复了从嵌套数据结构中正确获取`source`字段的问题。现在系统能够正确处理嵌套在`algorithm_entity`、`dataset_entity`和`metric_entity`对象中的数据。
- **2023-12-20**: 增强了数据库操作，现在能够识别并处理嵌套结构中的实体数据，保障数据的一致性和完整性。
- **2023-12-20**: 更新了文档，明确说明了系统中使用的嵌套数据结构及其正确的访问方式。

## 数据库结构

### 联合主键设计

本系统的数据库表使用联合主键(ID, task_id, source)设计，具有以下优势：

1. **多任务数据隔离**：不同任务的数据可以使用相同ID但不会冲突
2. **数据来源区分**：可以区分相同实体在不同来源(综述/引文)中的信息
3. **数据完整性**：确保每条记录的唯一性由多个关键字段共同决定

主要数据库表的联合主键设计：

- **Algorithms表**: (algorithm_id, task_id, source)
- **Datasets表**: (dataset_id, task_id, source) 
- **Metrics表**: (metric_id, task_id, source)
- **EvolutionRelations表**: (relation_id, task_id, source)

### 表间关系

- **实体表与关系表**: EvolutionRelations表通过from_entity和to_entity字段关联到算法/数据集/指标实体
- **任务关联**: 所有表都包含task_id字段，用于关联到特定任务
- **来源标记**: 所有表都包含source字段，标明数据来源("综述"/"引文"/"未知")

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

## 数据库连接池使用说明

本系统使用了MySQL数据库连接池来优化数据库连接的管理，提高系统性能和稳定性。

### 连接池架构

系统采用两层数据库访问架构：

1. **连接池层（db_pool.py）**：
   - `MySQLConnectionPool`类：管理数据库连接的创建、回收和重用
   - `DBUtils`类：提供基本的数据库操作接口，如执行查询、插入等
   - 全局实例`db_utils`：应用程序中使用的数据库操作工具实例

2. **数据管理层（db_manager.py）**：
   - `DatabaseManager`类：提供业务逻辑相关的数据库操作，如存储算法实体、关系等
   - 全局实例`db_manager`：应用程序中使用的数据库管理器实例

### 如何使用数据库连接池

在代码中使用数据库时，应遵循以下原则：

1. **基本查询操作**：
   - 直接使用`db_utils`实例进行基本的CRUD操作
   ```python
   # 查询示例
   results = db_utils.select_all('SELECT * FROM Algorithms WHERE task_id = %s', (task_id,))
   
   # 插入示例
   db_utils.insert_one('INSERT INTO Algorithms (algorithm_id, name) VALUES (%s, %s)', (id, name))
   
   # 更新示例
   db_utils.update_one('UPDATE Algorithms SET name = %s WHERE algorithm_id = %s', (new_name, id))
   ```

2. **业务逻辑操作**：
   - 使用`db_manager`实例进行业务相关的数据操作
   ```python
   # 获取实体
   entities = db_manager.get_entities_by_task(task_id)
   
   # 存储实体
   db_manager.store_algorithm_entity(entity_data, task_id)
   ```

3. **最佳实践**：
   - 不要手动获取或管理数据库连接
   - 不要使用`_reconnect_if_needed()`等旧方法
   - 使用`db_utils`的封装方法而不是直接执行SQL
   - 在执行大量操作时，合理控制事务范围

### 连接池配置

连接池配置在`app/config.py`中定义：

- `MYSQL_HOST`: 数据库服务器地址
- `MYSQL_PORT`: 数据库端口
- `MYSQL_USER`: 数据库用户名
- `MYSQL_PASSWORD`: 数据库密码
- `MYSQL_DB`: 数据库名称

连接池参数：
- 初始连接数: 3
- 最大连接数: 10

### 故障排除

如果遇到数据库连接问题，可以通过以下API检查连接池状态：
```
GET /api/system/db-status
```

此API将返回连接池的状态信息和连接测试结果。

## 其他功能

[此处可添加系统其他功能的说明...] 