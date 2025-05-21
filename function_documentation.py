"""
该文件整理了代码库中主要函数的作用和输入输出格式，便于查阅和维护。
"""

# ===================== app/modules/agents.py 函数 =====================

def extract_text_from_pdf(pdf_path):
    """
    从PDF文件中提取文本内容，并缓存结果
    
    输入:
        pdf_path (str): PDF文件的路径
        
    输出:
        str: 提取的文本内容
    
    功能:
        1. 首先检查缓存，如果存在且有效则直接返回
        2. 尝试使用千问API提取文本
        3. 如API调用失败，则使用PyPDF2作为备用方法
        4. 将提取的文本缓存到本地文件
    """

def extract_entities_with_openai(prompt, model_name="gpt-3.5-turbo"):
    """
    使用OpenAI API提取实体信息
    
    输入:
        prompt (str): 提示词
        model_name (str): 要使用的模型名称
        
    输出:
        str: API返回的JSON字符串
    
    功能:
        1. 创建OpenAI客户端并发送请求
        2. 记录API调用时间和返回结果
        3. 尝试从返回内容中提取JSON部分
    """

def extract_entities_with_zhipu(prompt, model_name="Qwen-long"):
    """
    使用千问API提取实体信息
    
    输入:
        prompt (str): 提示词
        model_name (str): 要使用的模型名称
        
    输出:
        str: API返回的JSON字符串
    
    功能:
        1. 创建千问API客户端并发送请求
        2. 支持流式响应，累积返回内容
        3. 尝试从返回内容中提取JSON部分
    """

def setup_qwen_agent(pdf_path=None):
    """
    设置千问助手
    
    输入:
        pdf_path (str, optional): PDF文件路径
        
    输出:
        Assistant: 千问助手实例
    
    功能:
        1. 配置千问助手的参数，如模型、API密钥等
        2. 设置系统指令和可用工具
        3. 返回配置好的助手实例
    """

def extract_paper_entities(pdf_paths, max_attempts=3, batch_size=20, force_reprocess=False, model_name="qwen", task_id=None):
    """
    从PDF论文中提取实体信息，支持多种模型
    
    输入:
        pdf_paths (str or list): PDF文件路径或路径列表
        max_attempts (int): 最大尝试次数，当模型返回需要继续提取时会继续
        batch_size (int): 一次处理的PDF文件数量
        force_reprocess (bool): 是否强制重新处理，忽略缓存
        model_name (str): 要使用的模型名称
        task_id (str): 任务ID，用于缓存标识
    
    输出:
        tuple: (提取的实体列表, 是否完成提取)
    
    功能:
        1. 分批处理多个PDF文件
        2. 优先使用缓存的结果
        3. 支持选择不同的模型进行提取
        4. 缓存提取结果以便后续使用
    """

def extract_evolution_relations(entities, pdf_path=None):
    """
    提取实体之间的演化关系，支持同时结合PDF文件和实体信息进行分析
    
    输入:
        entities (List[Dict]): 实体列表
        pdf_path (str, optional): PDF文件路径
    
    输出:
        List[Dict]: 演化关系列表
    
    功能:
        1. 构建实体描述列表
        2. 使用千问API分析实体间的演化关系
        3. 如果提供了PDF文件，会一并分析
        4. 验证和格式化关系数据
        5. 缓存结果以避免重复分析
    """

def extract_json_from_text(text):
    """
    从文本中提取JSON内容
    
    输入:
        text (str): 包含JSON的文本
    
    输出:
        str or None: 提取的JSON字符串，如果未找到则返回None
    
    功能:
        1. 尝试多种模式匹配方法找出JSON内容
        2. 支持code block、数组和对象格式的JSON
    """

# ===================== app/modules/data_extraction.py 函数 =====================

def parse_review_paper(file_path, task_id=None):
    """
    解析综述文章，提取引用文献列表。
    
    输入:
        file_path (str): 综述文章的文件路径
        task_id (str, optional): 处理任务ID，用于更新处理状态
        
    输出:
        List[str]: 引用文献的DOI或唯一标识列表
    
    功能:
        1. 使用千问API解析PDF文件中的引用文献
        2. 缓存解析结果，支持断点续传
        3. 上传文件到API进行处理
        4. 处理API返回的流式响应
        5. 从响应中提取引用文献信息
    """

def retrieve_referenced_papers(citation_list, download_dir=None):
    """
    根据引用列表检索并下载相关文献的PDF文件。
    
    输入:
        citation_list (List[str]): 引用文献列表
        download_dir (str): 下载目录路径
        
    输出:
        List[str]: 下载的PDF文件路径列表
    
    功能:
        1. 使用arxiv客户端检索文献
        2. 下载找到的文献到指定目录
        3. 使用临时文件确保下载安全
    """

def get_entity_from_cache(paper_id):
    """
    从缓存获取已处理的实体信息
    
    输入:
        paper_id (str): 论文ID
    
    输出:
        dict or None: 缓存的实体数据，如果不存在则返回None
    
    功能:
        1. 查找并读取缓存文件
        2. 处理可能的异常情况
    """

def save_entity_to_cache(paper_id, entity_data):
    """
    保存实体信息到缓存
    
    输入:
        paper_id (str): 论文ID
        entity_data (dict): 要缓存的实体数据
    
    输出:
        None
    
    功能:
        1. 确保缓存目录存在
        2. 将实体数据序列化为JSON并保存
    """

def extract_entities_from_paper(pdf_path, task_id=None, sub_progress=None):
    """
    从单篇文献中提取算法实体及其要素。
    
    输入:
        pdf_path (str): 单篇文献的PDF文件路径
        task_id (str, optional): 处理任务ID，用于更新处理状态
        sub_progress (tuple, optional): 批处理进度，格式为(当前索引, 总数)
        
    输出:
        List[Dict]: 结构化的算法实体JSON数据列表
    
    功能:
        1. 检查缓存，避免重复处理
        2. 创建临时文件副本以避免权限问题
        3. 调用千问API处理PDF
        4. 规范化实体格式，确保ID和类型字段的一致性
        5. 保存结果到缓存
    """

def _clean_json_string(json_str):
    """
    清理和修复常见的JSON格式错误
    
    输入:
        json_str (str): JSON字符串
    
    输出:
        str: 清理后的JSON字符串
    
    功能:
        1. 修复括号不匹配问题
        2. 修复缺少逗号的问题
        3. 修复转义字符问题
        4. 修复未闭合的引号
        5. 处理非标准JSON(如注释或尾随逗号)
        6. 处理缺少键名的情况
    """

def process_papers_and_extract_data(review_pdf_path, task_id=None, citation_paths=None):
    """
    处理综述文章和引用文献，提取实体和关系数据。
    
    输入:
        review_pdf_path (str): 综述文章的PDF文件路径
        task_id (str, optional): 处理任务ID，用于更新处理状态
        citation_paths (list, optional): 引用文献的PDF文件路径列表
        
    输出:
        Tuple[List[Dict], List[Dict], Dict]: 提取的实体列表、关系列表和指标
    
    功能:
        1. 确保所有必要的目录存在
        2. 处理主综述文章，提取实体
        3. 收集并处理所有引用文献
        4. 使用提取的实体生成演化关系
        5. 保存数据到数据库和本地文件
        6. 计算比较指标
    """

def calculate_comparison_metrics(review_entities, citation_entities, relations):
    """
    计算各种比较指标
    
    输入:
        review_entities (list): 从综述中提取的实体列表
        citation_entities (list): 从引用文献中提取的实体列表
        relations (list): 提取的演化关系列表
    
    输出:
        dict: 包含各种指标的字典
    
    功能:
        1. 分类计数实体(算法、数据集、评价指标)
        2. 计算实体精确率和召回率
        3. 统计关系类型分布
        4. 计算聚类指标
    """

def build_algorithm_clusters(relations):
    """
    根据算法之间的演化关系构建聚类
    
    输入:
        relations (list): 演化关系列表
        
    输出:
        list: 聚类列表，每个聚类是一组相关的算法ID
    
    功能:
        1. 使用图算法找出连通分量
        2. 仅考虑算法之间的关系
        3. 将连通分量转换为聚类列表
    """

def save_progress(task_id, entities, processed_files, progress):
    """
    保存处理进度到文件
    
    输入:
        task_id (str): 处理任务ID
        entities (list): 已提取的实体列表
        processed_files (set): 已处理的文件集合
        progress (float): 当前进度(0-1)
    
    输出:
        None
    
    功能:
        1. 将处理状态序列化为JSON
        2. 保存到进度文件中，便于断点续传
    """

# ===================== app/modules/db_manager.py 函数 =====================

def _connect_mysql(self):
    """
    连接MySQL数据库
    
    输入:
        None
    
    输出:
        None
    
    功能:
        1. 使用配置信息创建数据库连接
        2. 确保数据库存在，并选择使用它
    """

def _check_and_init_tables(self):
    """
    检查表是否存在，并在需要时初始化表
    
    输入:
        None
    
    输出:
        None
    
    功能:
        1. 检查所需的表是否存在
        2. 不存在时创建相应的表结构
    """

def store_entities(self, entities):
    """
    将算法实体存储到数据库中
    
    输入:
        entities (List[Dict]): 算法实体列表
    
    输出:
        None
    
    功能:
        1. 遍历实体列表
        2. 调用相应的存储方法
    """

def store_relations(self, relations):
    """
    将一组算法实体之间的演化关系存储到数据库
    
    输入:
        relations (List[Dict]): 演化关系数据列表
            
    输出:
        bool: 是否成功存储
    
    功能:
        1. 验证关系数据
        2. 循环处理每条关系
        3. 记录存储结果
    """

def store_algorithm_relation(self, relation):
    """
    将单条实体演化关系存储到数据库
    
    输入:
        relation (dict): 演化关系数据，支持两种格式：
            1. 数据库格式：{from_entity, to_entity, relation_type, ...}
            2. API格式：{from_entities, to_entities, relation_type, ...}
        
    输出:
        bool: 是否成功存储
    
    功能:
        1. 验证关系数据格式
        2. 处理不同格式的关系数据
        3. 创建和存储关系记录
    """

def _store_relation_mysql(self, relation_data):
    """
    将实体演化关系存储到MySQL数据库
    
    输入:
        relation_data (dict): 演化关系数据
        
    输出:
        bool: 是否成功存储
    
    功能:
        1. 验证关系数据
        2. 检查实体是否存在
        3. 检查是否有重复关系
        4. 根据情况执行插入或更新操作
    """

def _check_entity_exists(self, entity_id, entity_type="Algorithm"):
    """
    检查实体是否存在于数据库中
    
    输入:
        entity_id (str): 实体ID
        entity_type (str): 实体类型
    
    输出:
        bool: 实体是否存在
    
    功能:
        1. 根据实体类型选择表
        2. 构建查询语句检查实体是否存在
    """

def get_all_entities(self):
    """
    获取所有类型的实体信息，包括算法、数据集和评价指标
    
    输入:
        None
    
    输出:
        List[Dict]: 所有实体列表
    
    功能:
        1. 分别获取不同类型的实体
        2. 将不同类型实体转换为统一格式
        3. 合并所有实体到一个列表
    """

def get_all_relations(self):
    """
    获取所有演化关系的信息
    
    输入:
        None
    
    输出:
        List[Dict]: 演化关系列表
    
    功能:
        1. 查询所有演化关系
        2. 格式化关系数据
        3. 处理可能的数据类型转换
    """

def update_entity(self, entity_id, updated_data):
    """
    更新指定算法实体的信息
    
    输入:
        entity_id (str): 算法实体ID
        updated_data (Dict): 更新后的数据
        
    输出:
        bool: 操作是否成功
    
    功能:
        1. 检查实体是否存在
        2. 构建更新语句和参数
        3. 执行更新操作
    """

def add_relation(self, relation):
    """
    添加新的演化关系
    
    输入:
        relation (Dict): 新关系的数据
        
    输出:
        bool: 操作是否成功
    
    功能:
        1. 验证关系数据
        2. 构建插入语句
        3. 执行插入操作
    """

def modify_relation(self, relation_id, updated_relation):
    """
    修改现有的演化关系
    
    输入:
        relation_id (int/str): 关系ID
        updated_relation (Dict): 更新后的关系数据
        
    输出:
        bool: 操作是否成功
    
    功能:
        1. 检查关系是否存在
        2. 构建更新语句和参数
        3. 执行更新操作
    """

def delete_relation(self, relation_id):
    """
    删除指定的演化关系
    
    输入:
        relation_id (int/str): 关系ID
        
    输出:
        bool: 操作是否成功
    
    功能:
        1. 构建删除语句
        2. 执行删除操作
    """

def create_processing_task(self, task_id, task_name):
    """
    创建新的处理任务记录
    
    输入:
        task_id (str): 任务ID
        task_name (str): 任务名称或描述
        
    输出:
        bool: 操作是否成功
    
    功能:
        1. 记录任务开始时间和初始状态
        2. 保存任务信息到数据库
    """

def update_processing_status(self, task_id, **kwargs):
    """
    更新处理任务的状态
    
    输入:
        task_id (str): 任务ID
        **kwargs: 可包含以下参数:
            status (str): 状态 ('waiting', 'processing', 'completed', 'failed')
            current_stage (str): 当前处理阶段
            progress (float): 进度 (0.0 到 1.0)
            current_file (str): 当前处理的文件
            message (str): 状态消息
            completed (bool): 是否已完成
            
    输出:
        bool: 操作是否成功
    
    功能:
        1. 检查任务是否存在
        2. 构建更新语句和参数
        3. 执行更新操作
    """

def get_processing_status(self, task_id=None):
    """
    获取处理任务的状态
    
    输入:
        task_id (str, optional): 任务ID，如果为None则返回所有任务
        
    输出:
        dict or list: 任务状态信息
    
    功能:
        1. 查询任务状态
        2. 格式化状态数据
        3. 返回单个任务或任务列表
    """

def get_entity_by_id(self, entity_id):
    """
    通过ID获取任意类型的实体
    
    输入:
        entity_id (str): 实体ID
        
    输出:
        dict: 实体信息字典
    
    功能:
        1. 查询不同类型的实体表
        2. 解析JSON字段
        3. 构建标准格式的实体对象
        4. 获取相关的演化关系
    """

def init_db(self):
    """
    初始化数据库
    
    输入:
        None
    
    输出:
        bool: 是否成功初始化
    
    功能:
        1. 连接数据库
        2. 完成初始化操作
    """

def _get_entity_relations(self, entity_id):
    """
    获取指定实体的演化关系
    
    输入:
        entity_id (str): 实体ID
        
    输出:
        List[Dict]: 演化关系列表
    
    功能:
        1. 查询指向该实体的关系
        2. 格式化关系数据
    """

def delete_entity(self, entity_id):
    """
    删除指定的算法实体及其相关关系
    
    输入:
        entity_id (str): 实体ID
        
    输出:
        bool: 操作是否成功
    
    功能:
        1. 删除与此实体相关的所有关系
        2. 删除实体本身
    """

def store_algorithm_entity(self, entity):
    """
    存储算法实体到MySQL数据库
    
    输入:
        entity (dict): 要存储的算法实体数据
        
    输出:
        bool: 是否成功存储
    
    功能:
        1. 确保ID字段存在且一致
        2. 根据实体类型选择存储方法
        3. 执行存储操作
    """

def _store_algorithm_mysql(self, entity):
    """
    将算法实体存储到MySQL数据库
    
    输入:
        entity (dict): 算法实体数据
    
    输出:
        bool: 是否成功存储
    
    功能:
        1. 提取实体字段
        2. 处理列表和字典字段转为JSON
        3. 检查实体是否已存在
        4. 执行插入或更新操作
    """

def _store_dataset_mysql(self, entity):
    """
    将数据集实体存储到MySQL数据库
    
    输入:
        entity (dict): 数据集实体数据
    
    输出:
        bool: 是否成功存储
    
    功能:
        1. 提取数据集字段
        2. 处理特殊字段如size
        3. 检查实体是否已存在
        4. 执行插入或更新操作
    """

def _store_metric_mysql(self, entity):
    """
    将评价指标实体存储到MySQL数据库
    
    输入:
        entity (dict): 评价指标实体数据
    
    输出:
        bool: 是否成功存储
    
    功能:
        1. 提取评价指标字段
        2. 检查实体是否已存在
        3. 执行插入或更新操作
    """

def get_all_datasets(self):
    """
    获取所有数据集实体
    
    输入:
        None
    
    输出:
        List[Dict]: 数据集列表
    
    功能:
        1. 查询所有数据集
        2. 将查询结果转换为字典列表
    """

def get_all_metrics(self):
    """
    获取所有评估指标实体
    
    输入:
        None
    
    输出:
        List[Dict]: 评估指标列表
    
    功能:
        1. 查询所有评估指标
        2. 将查询结果转换为字典列表
    """

def get_dataset_by_id(self, dataset_id):
    """
    获取指定ID的数据集详细信息
    
    输入:
        dataset_id (str): 数据集ID
    
    输出:
        dict or None: 数据集信息，不存在则返回None
    
    功能:
        1. 查询指定ID的数据集
        2. 将查询结果转换为字典
    """

def get_metric_by_id(self, metric_id):
    """
    获取指定ID的评估指标详细信息
    
    输入:
        metric_id (str): 评估指标ID
    
    输出:
        dict or None: 评估指标信息，不存在则返回None
    
    功能:
        1. 查询指定ID的评估指标
        2. 将查询结果转换为字典
    """

def get_relation_types(self):
    """
    获取系统中所有的演化关系类型
    
    输入:
        None
    
    输出:
        List[str]: 关系类型列表
    
    功能:
        1. 查询不同的关系类型
        2. 如果没有数据，返回默认关系类型
    """

def get_incoming_relations(self, entity_id):
    """
    获取指向指定实体的关系（传入关系）
    
    输入:
        entity_id (str): 实体ID
    
    输出:
        List[Dict]: 关系列表
    
    功能:
        1. 查询以该实体为目标的所有关系
        2. 格式化关系数据
    """

def get_outgoing_relations(self, entity_id):
    """
    获取从指定实体出发的所有关系
    
    输入:
        entity_id (str): 实体ID
    
    输出:
        List[Dict]: 关系列表
    
    功能:
        1. 查询以该实体为源的所有关系
        2. 格式化关系数据
    """

def clear_all_data(self):
    """
    清除所有数据库中的数据
    
    输入:
        None
    
    输出:
        bool: 是否成功清除数据
    
    功能:
        1. 清空所有相关表中的数据
        2. 记录清除操作日志
    """ 