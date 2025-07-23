# 算法要素关系图生成系统

## 项目介绍

本系统是一个智能算法要素关系图生成与综述自动化系统，通过分析综述论文及其引用文献，提取算法实体及其要素，构建算法进化关系图，并集成AutoSurvey功能自动生成高质量学术综述。系统支持以下主要功能：

### 🔍 **数据提取与管理**
- 通过上传PDF文档自动提取算法实体及其关系
- 支持算法实体的查看、编辑和管理
- 支持算法演化关系的查看、添加和修改
- 支持从Excel/CSV表格导入数据

### 📊 **可视化与分析**
- 提供交互式的关系图可视化展示
- 算法演进图谱构建和分析
- 关键算法节点识别和影响力评估
- 技术发展路径和趋势分析

### 🤖 **AutoSurvey集成功能** ⭐ **新功能**
- **智能任务选择**: 从多个任务中选择和组合实体关系数据
- **算法脉络分析**: 自动构建算法演进图谱，识别关键技术节点
- **综述自动生成**: 集成AutoSurvey系统，基于算法脉络生成学术综述
- **多格式输出**: 支持Markdown、HTML、PDF、Word等多种格式
- **版本管理**: 综述结果的存储、版本控制和历史记录
- **质量评估**: 自动评估综述质量和完整性
- **异步处理**: 支持长时间运行的综述生成任务

### 💡 **智能评论生成**
- 基于图数据生成算法评论
- 算法发展脉络的文字描述生成
- 技术关联分析和影响力评估

## 系统架构

系统采用前后端分离架构，主要包括以下模块：

### 🏗️ **核心模块**
- **数据获取与解析模块**：解析PDF文档，提取算法实体和关系
- **数据处理与存储模块**：处理和规范化数据，存储到数据库
- **知识图谱构建模块**：基于实体和关系构建知识图谱
- **API服务模块**：提供RESTful API接口
- **前端展示与交互模块**：提供用户界面和交互功能

### 🤖 **AutoSurvey集成模块** ⭐ **新增**
- **任务选择器**：智能任务选择和数据提取
- **实体关系提取器**：从任务数据中提取和格式化实体关系
- **数据格式转换器**：转换为AutoSurvey标准输入格式
- **算法脉络分析引擎**：构建算法演进图谱，识别关键节点和路径
- **AutoSurvey连接器**：与AutoSurvey系统的API集成和异步处理
- **综述生成引擎**：基于模板和脉络分析生成增强综述内容
- **存储管理系统**：综述结果的版本管理、多格式存储和检索
- **格式化模块**：支持Markdown、HTML、PDF、Word等多种输出格式

## 环境要求

### 🔧 **基础环境**
- Python 3.8+
- Flask 2.0+
- Neo4j 4.0+

### 🤖 **AutoSurvey集成功能额外要求**
- **必需依赖**：
  - `asyncio` - 异步处理支持
  - `sqlite3` - 综述存储数据库
  - `python-docx` - Word文档生成（可选）
  - `psutil` - 系统资源监控（可选）
- **外部工具**：
  - `pdflatex` - PDF生成支持（可选，需要LaTeX环境）
  - `pandoc` - 文档格式转换（可选）
- **API配置**：
  - OpenAI API密钥（用于调用AutoSurvey）
  - 网络连接（用于API调用）

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

6. **配置AutoSurvey集成** ⭐ **新功能配置**

在`app/config.py`中添加AutoSurvey相关配置，或通过环境变量设置：

```bash
# 设置OpenAI API密钥
export OPENAI_API_KEY="your-openai-api-key"
export AUTOSURVEY_API_URL="https://api.openai.com/v1/chat/completions"
```

7. **验证环境配置**

```bash
# 运行环境验证脚本
python scripts/validate_autosurvey_setup.py

# 运行集成测试（可选）
python scripts/test_autosurvey_integration.py
```

8. **启动应用**

```bash
python run.py
```

或者

```bash
flask run
```

9. **访问应用**

- **主系统**：http://localhost:5000
- **AutoSurvey集成功能**：http://localhost:5000/autosurvey ⭐ **新功能入口**

## 使用说明

### 📖 **基础功能**
1. **首页**：通过上传PDF文档或导入表格数据创建算法实体和关系
2. **算法实体表格**：查看、添加、编辑和删除算法实体
3. **关系图展示**：交互式查看算法演化关系图，可以添加新关系

### 🤖 **AutoSurvey集成功能使用指南** ⭐ **新功能**

#### 1. 访问AutoSurvey功能
- 在浏览器中访问：`http://localhost:5000/autosurvey`
- 或从主页导航菜单进入AutoSurvey集成页面

#### 2. 任务选择和配置
- **选择任务**：从任务列表中选择一个或多个相关任务
- **设置主题**：输入综述的主题和描述
- **配置参数**：
  - 选择生成模型（推荐：gpt-4o-2024-05-13）
  - 设置章节数量（默认：7章）
  - 配置子章节长度（默认：700字）
  - 选择输出格式（Markdown、HTML、PDF、Word）

#### 3. 生成综述
- 点击"开始生成"按钮启动综述生成流程
- 系统将自动执行以下步骤：
  1. 提取选定任务的实体关系数据
  2. 构建算法演进图谱和脉络分析
  3. 调用AutoSurvey API生成基础综述
  4. 基于算法脉络增强综述内容
  5. 生成多种格式的输出文件

#### 4. 监控进度
- 实时查看生成进度和当前处理阶段
- 查看详细的处理日志和状态信息
- 如需要可以取消正在进行的任务

#### 5. 查看和下载结果
- 预览生成的综述内容
- 查看算法脉络分析结果
- 下载多种格式的综述文件
- 查看质量评估报告

#### 6. 结果管理
- 浏览历史生成的综述列表
- 按主题、状态、时间等条件搜索
- 管理综述版本和标签
- 导出和分享综述结果

## 最近更新

### 🚀 **2024-01-23**: AutoSurvey集成功能正式发布 ⭐ **重大更新**
- **新增AutoSurvey集成模块**：完整实现与AutoSurvey系统的深度集成
- **算法脉络分析引擎**：自动构建算法演进图谱，识别关键技术节点和发展路径
- **智能综述生成**：基于任务实体关系数据自动生成高质量学术综述
- **多格式输出支持**：支持Markdown、HTML、PDF、Word等多种格式导出
- **异步处理机制**：支持长时间运行的综述生成任务，实时进度监控
- **版本管理系统**：综述结果的存储、版本控制和历史记录管理
- **质量评估功能**：自动评估综述质量和完整性，提供改进建议
- **用户界面优化**：全新的任务选择界面和参数配置功能

### 📈 **历史更新**
- **2023-12-27**: 改进数据库结构，将所有表的主键从单一ID改为联合主键(ID, task_id, source)，提升多任务数据管理能力和避免跨任务ID冲突。
- **2023-12-20**: 修复了从嵌套数据结构中正确获取`source`字段的问题。现在系统能够正确处理嵌套在`algorithm_entity`、`dataset_entity`和`metric_entity`对象中的数据。
- **2023-12-20**: 增强了数据库操作，现在能够识别并处理嵌套结构中的实体数据，保障数据的一致性和完整性。
- **2023-12-20**: 更新了文档，明确说明了系统中使用的嵌套数据结构及其正确的访问方式。

## 数据库结构

### 联合主键设计

本系统的主要实体表使用联合主键(ID, task_id, source)设计，具有以下优势：

1. **多任务数据隔离**：不同任务的数据可以使用相同ID但不会冲突
2. **数据来源区分**：可以区分相同实体在不同来源(综述/引文)中的信息
3. **数据完整性**：确保每条记录的唯一性由多个关键字段共同决定

主要数据库表的主键设计：

- **Algorithms表**: (algorithm_id, task_id, source) - 联合主键
- **Datasets表**: (dataset_id, task_id, source) - 联合主键
- **Metrics表**: (metric_id, task_id, source) - 联合主键
- **EvolutionRelations表**: relation_id - 单一主键

使用单一主键的EvolutionRelations表能够支持跨任务和跨来源的关系引用，适合存储复杂网络结构的演化关系。这种设计使得不同来源和不同任务的关系能够被统一管理，便于执行全局关系分析。

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

## AutoSurvey集成功能详细说明 ⭐ **新功能**

### 🎯 **功能概述**

AutoSurvey集成功能是本系统的重要扩展，它将现有的算法实体关系数据与AutoSurvey系统深度集成，实现了从数据提取到综述生成的全自动化流程。

### 🔧 **核心组件**

#### 1. **任务选择器 (TaskSelector)**
- 智能任务筛选和选择
- 支持多任务数据合并
- 数据质量评估和验证

#### 2. **实体关系提取器 (EntityRelationExtractor)**
- 自动提取算法、数据集、指标实体
- 解析实体间的演化关系
- 数据标准化和清洗

#### 3. **算法脉络分析引擎 (AlgorithmLineageAnalyzer)**
- 构建算法演进图谱
- 识别关键技术节点
- 分析发展路径和趋势

#### 4. **AutoSurvey连接器 (AutoSurveyConnector)**
- 与AutoSurvey API的安全连接
- 异步任务处理和状态监控
- 错误处理和重试机制

#### 5. **综述生成引擎 (SurveyContentGenerator)**
- 基于模板的内容生成
- 算法脉络描述集成
- 质量评估和优化

#### 6. **存储管理系统 (SurveyStorageManager)**
- 多版本综述存储
- 多格式文件管理
- 检索和导出功能

### 📊 **数据流程**

```
任务选择 → 数据提取 → 格式转换 → 脉络分析 → AutoSurvey调用 → 内容增强 → 格式化输出 → 结果存储
```

### 🚀 **使用场景**

1. **学术研究**：快速生成特定领域的技术综述
2. **技术调研**：了解算法技术的发展脉络和趋势
3. **论文写作**：为学术论文提供相关工作部分的参考
4. **教学辅助**：生成教学用的技术发展历程材料

### 📈 **性能特点**

- **高效处理**：异步处理机制，支持大规模数据
- **智能分析**：基于图算法的脉络分析
- **质量保证**：多层次的质量评估和验证
- **用户友好**：直观的界面和实时进度反馈

### 🔒 **安全考虑**

- API密钥安全存储和传输
- 数据访问权限控制
- 错误信息脱敏处理
- 审计日志记录

### 📚 **相关文档**

- [AutoSurvey API规范](docs/autosurvey_api_specification.md)
- [集成功能详细文档](README_AutoSurvey_Integration.md)
- [测试和验证指南](tests/test_autosurvey_integration.py)

---

## 其他功能

[此处可添加系统其他功能的说明...]

---

**项目名称**: AutoGenSurvey-Based-on-AEG
**最后更新**: 2024年1月23日
**版本**: v2.0.0 (包含AutoSurvey集成功能)
