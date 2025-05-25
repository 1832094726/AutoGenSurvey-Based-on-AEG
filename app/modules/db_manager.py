import os
import logging
import json
import datetime
import time  # 添加time模块用于睡眠
import mysql.connector  # 导入MySQL连接器包
import traceback
from mysql.connector import Error as MySQLError  # 导入MySQL错误类型便于捕获
from app.config import Config

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class DatabaseManager:
    """数据库管理器，处理算法实体和关系的存储与读取"""
    
    def __init__(self):
        """初始化数据库连接"""
        self.db_type = 'mysql'  # 固定为MySQL
        # 初始化连接与游标为None
        self.conn = None
        self.cursor = None
        self._connect_mysql()
        
        # 检查并初始化表
        self._check_and_init_tables()
    
    def _connect_mysql(self):
        """
        连接到MySQL数据库
        """
        max_attempts = 5
        attempt = 0
        
        while attempt < max_attempts:
            attempt += 1
            try:
                # 关闭现有连接
                if hasattr(self, 'conn') and self.conn:
                    try:
                        self.conn.close()
                    except Exception as e:
                        logging.warning(f"关闭旧连接时出错: {str(e)}")
                
                # 建立新连接，增加连接超时和使用纯Python实现
                self.conn = mysql.connector.connect(
                    host=Config.MYSQL_HOST,
                    port=Config.MYSQL_PORT,
                    user=Config.MYSQL_USER,
                    password=Config.MYSQL_PASSWORD,
                    database=Config.MYSQL_DB,
                    charset='utf8mb4',
                    use_pure=True,  # 使用纯Python实现，增加稳定性
                    connection_timeout=30,  # 增加连接超时时间
                    autocommit=False,  # 显式控制事务
                    pool_size=5,  # 使用连接池
                    pool_name="algorithm_pool",
                    pool_reset_session=True,
                    get_warnings=True,
                    raise_on_warnings=False  # 不对警告抛出异常
                )
                self.cursor = self.conn.cursor(buffered=True)  # 使用缓冲游标
                logging.info("MySQL数据库连接成功")
                return True
            except (mysql.connector.Error, MySQLError) as e:
                logging.error(f"连接MySQL数据库时出错 (尝试 {attempt}/{max_attempts}): {str(e)}")
                if attempt < max_attempts:
                    time.sleep(2)  # 等待2秒后重试
                else:
                    logging.error("已达到最大重试次数，无法连接到MySQL数据库")
                    raise
            except Exception as e:
                logging.error(f"连接MySQL数据库时发生未知错误: {str(e)}")
                import traceback
                logging.error(traceback.format_exc())
                raise
        return False
    
    def _reconnect_if_needed(self):
        """
        检查MySQL连接状态并在需要时重连
        """
        max_retries = 3
        retry_count = 0
        
        while retry_count < max_retries:
            try:
                # 尝试执行简单的查询来检查连接
                if self.conn is None or not self.conn.is_connected():
                    logging.warning("MySQL连接已断开，尝试重连")
                    self._connect_mysql()
                    return True
                # 测试连接是否有效
                try:
                    self.cursor.execute("SELECT 1")
                    self.cursor.fetchone()
                    return True  # 连接正常
                except Exception as e:
                    logging.warning(f"连接测试失败: {str(e)}，尝试重连")
                    self._connect_mysql()
                    return True
            except Exception as e:
                retry_count += 1
                logging.error(f"检查MySQL连接时出错 (尝试 {retry_count}/{max_retries}): {str(e)}")
                if retry_count < max_retries:
                    logging.error("无法建立MySQL连接，等待5秒后重试")
                    time.sleep(5)
                else:
                    logging.error("达到最大重试次数，无法重新连接MySQL")
                    return False
        return False
    
    def _check_and_init_tables(self):
        """检查表是否存在，并在需要时初始化表"""
        self._check_and_init_tables_mysql()
    
    def _check_and_init_tables_mysql(self):
        """检查并初始化MySQL表"""
        try:
            # 检查算法表是否存在
            self.cursor.execute("SHOW TABLES LIKE 'Algorithms'")
            if not self.cursor.fetchone():
                # 创建算法表
                self.cursor.execute('''
                CREATE TABLE IF NOT EXISTS Algorithms (
                    algorithm_id VARCHAR(255) PRIMARY KEY,
                    name VARCHAR(255),
                    title TEXT,
                    year VARCHAR(50),
                    authors TEXT,
                    task VARCHAR(255),
                    dataset TEXT,
                    metrics TEXT,
                    architecture_components TEXT,
                    architecture_connections TEXT,
                    architecture_mechanisms TEXT,
                    methodology_training_strategy TEXT,
                    methodology_parameter_tuning TEXT,
                    feature_processing TEXT,
                    entity_type VARCHAR(50) DEFAULT 'Algorithm'
                )
                ''')
                logging.info("创建MySQL Algorithms表")
            # 检查数据集表是否存在
            self.cursor.execute("SHOW TABLES LIKE 'Datasets'")
            if not self.cursor.fetchone():
                # 创建数据集表
                self.cursor.execute('''
                CREATE TABLE IF NOT EXISTS Datasets (
                    dataset_id VARCHAR(255) PRIMARY KEY,
                    name VARCHAR(255),
                    description TEXT,
                    domain VARCHAR(255),
                    size INT,
                    year VARCHAR(50),
                    creators TEXT,
                    entity_type VARCHAR(50) DEFAULT 'Dataset'
                )
                ''')
                logging.info("创建MySQL Datasets表")
            # 检查指标表是否存在
            self.cursor.execute("SHOW TABLES LIKE 'Metrics'")
            if not self.cursor.fetchone():
                # 创建指标表
                self.cursor.execute('''
                CREATE TABLE IF NOT EXISTS Metrics (
                    metric_id VARCHAR(255) PRIMARY KEY,
                    name VARCHAR(255),
                    description TEXT,
                    category VARCHAR(255),
                    formula TEXT,
                    entity_type VARCHAR(50) DEFAULT 'Metric'
                )
                ''')
                logging.info("创建MySQL Metrics表")
            # 检查关系表是否存在
            self.cursor.execute("SHOW TABLES LIKE 'EvolutionRelations'")
            if not self.cursor.fetchone():
                # 创建关系表，确保relation_type非空
                self.cursor.execute('''
                CREATE TABLE IF NOT EXISTS EvolutionRelations (
                    relation_id INT AUTO_INCREMENT PRIMARY KEY,
                    from_entity VARCHAR(255) NOT NULL,
                    to_entity VARCHAR(255) NOT NULL,
                    relation_type VARCHAR(100) NOT NULL,
                    structure VARCHAR(255),
                    detail TEXT,
                    evidence TEXT,
                    confidence FLOAT,
                    from_entity_type VARCHAR(50),
                    to_entity_type VARCHAR(50),
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
                )
                ''')
                logging.info("创建MySQL EvolutionRelations表")
            # 检查处理状态表是否存在
            self.cursor.execute("SHOW TABLES LIKE 'ProcessingStatus'")
            if not self.cursor.fetchone():
                # 创建处理状态表
                self.cursor.execute('''
                CREATE TABLE IF NOT EXISTS ProcessingStatus (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    task_id VARCHAR(255) UNIQUE,
                    task_name VARCHAR(255),
                    status VARCHAR(50),  -- 'waiting', 'processing', 'completed', 'failed'
                    current_stage VARCHAR(100),
                    progress FLOAT,  -- 0.0到1.0之间的进度
                    current_file TEXT,
                    message TEXT,
                    start_time DATETIME,
                    update_time DATETIME,
                    end_time DATETIME,
                    completed TINYINT(1) DEFAULT 0
                )
                ''')
                logging.info("创建MySQL ProcessingStatus表")
            self.conn.commit()
        except (mysql.connector.Error, MySQLError) as e:
            logging.error(f"初始化数据库表时出错: {str(e)}")
            self.conn.rollback()
        except Exception as e:
            logging.error(f"初始化数据库表时出错: {str(e)}")
            import traceback
            logging.error(traceback.format_exc())
            self.conn.rollback()
    
    def __del__(self):
        """析构函数，关闭数据库连接"""
        if hasattr(self, 'conn'):
            self.conn.close()
    
    def store_entities(self, entities):
        """
        将算法实体存储到数据库中。
        
        Args:
            entities (List[Dict]): 算法实体列表
        """
        self._store_entities_mysql(entities)
    
    def _store_entities_mysql(self, entities):
        """将实体存储到MySQL数据库"""
        max_retries = 3
        retry_count = 0
        
        while retry_count < max_retries:
            try:
                # 检查连接状态
                if not self._reconnect_if_needed():
                    logging.error("无法建立MySQL连接，等待5秒后重试")
                    retry_count += 1
                    time.sleep(5)
                    continue
                for entity in entities:
                    if 'algorithm_entity' in entity:
                        algo = entity['algorithm_entity']
                        # 转换列表/字典为JSON字符串
                        authors = json.dumps(algo.get('authors', []), ensure_ascii=False)
                        dataset = json.dumps(algo.get('dataset', []), ensure_ascii=False)
                        metrics = json.dumps(algo.get('metrics', []), ensure_ascii=False)
                        architecture = algo.get('architecture', {})
                        arch_components = json.dumps(architecture.get('components', []), ensure_ascii=False)
                        arch_connections = json.dumps(architecture.get('connections', []), ensure_ascii=False)
                        arch_mechanisms = json.dumps(architecture.get('mechanisms', []), ensure_ascii=False)
                        methodology = algo.get('methodology', {})
                        meth_training = json.dumps(methodology.get('training_strategy', []), ensure_ascii=False)
                        meth_params = json.dumps(methodology.get('parameter_tuning', []), ensure_ascii=False)
                        feature_processing = json.dumps(algo.get('feature_processing', []), ensure_ascii=False)
                        # MySQL的INSERT ON DUPLICATE KEY UPDATE语法
                        sql = '''
                        INSERT INTO Algorithms 
                        (algorithm_id, name, title, year, authors, task, dataset, metrics,
                        architecture_components, architecture_connections, architecture_mechanisms,
                        methodology_training_strategy, methodology_parameter_tuning, feature_processing)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                        ON DUPLICATE KEY UPDATE 
                        name = VALUES(name), title = VALUES(title), year = VALUES(year), 
                        authors = VALUES(authors), task = VALUES(task), dataset = VALUES(dataset),
                        metrics = VALUES(metrics), architecture_components = VALUES(architecture_components),
                        architecture_connections = VALUES(architecture_connections), 
                        architecture_mechanisms = VALUES(architecture_mechanisms),
                        methodology_training_strategy = VALUES(methodology_training_strategy),
                        methodology_parameter_tuning = VALUES(methodology_parameter_tuning),
                        feature_processing = VALUES(feature_processing)
                        '''
                        self.cursor.execute(sql, (
                            algo.get('algorithm_id', ''),
                            algo.get('name', ''),
                            algo.get('title', ''),
                            algo.get('year', ''),
                            authors,
                            algo.get('task', ''),
                            dataset,
                            metrics,
                            arch_components,
                            arch_connections,
                            arch_mechanisms,
                            meth_training,
                            meth_params,
                            feature_processing
                        ))
                self.conn.commit()
                logging.info(f"已将 {len(entities)} 个实体存储到MySQL数据库")
                return
            except (mysql.connector.Error, MySQLError) as e:
                retry_count += 1
                logging.error(f"存储实体到MySQL时出错 (尝试 {retry_count}/{max_retries}): {str(e)}")
                # 如果是连接问题，等待后重试
                time.sleep(5)
                # 尝试重新连接
                self._connect_mysql()
                if retry_count >= max_retries:
                    logging.error("重试次数已达上限，无法存储实体到数据库")
                    break
            except Exception as e:
                logging.error(f"存储实体到MySQL时出错: {str(e)}")
                import traceback
                logging.error(traceback.format_exc())
                break
        logging.error(f"无法将 {len(entities)} 个实体存储到MySQL数据库")
    
    def store_relations(self, relations):
        """
        将一组算法实体之间的演化关系存储到数据库
        
        Args:
            relations (List[Dict]): 演化关系数据列表，支持两种格式：
                1. 数据库格式：{from_entity, to_entity, relation_type, ...}
                2. API格式：{from_entities, to_entities, relation_type, ...}
            
        Returns:
            bool: 是否成功存储
        """
        if not relations:
            logging.warning("没有提供关系数据")
            return False
            
        stored_count = 0
        for relation in relations:
            if self.store_algorithm_relation(relation):
                stored_count += 1
                
        logging.info(f"成功存储 {stored_count}/{len(relations)} 条关系数据")
        return stored_count > 0
        
    def store_algorithm_relation(self, relation):
        """
        将单条实体演化关系存储到数据库
        
        Args:
            relation (dict): 演化关系数据，支持两种格式：
                1. 数据库格式：{from_entity, to_entity, relation_type, ...}
                2. API格式：{from_entities, to_entities, relation_type, ...}
            
        Returns:
            bool: 是否成功存储
        """
        try:
            # 验证关系数据
            if not isinstance(relation, dict):
                logging.error(f"关系数据格式错误: {relation}")
                return False
            # 检查是否是数据库格式（from_entity/to_entity）
            if "from_entity" in relation and "to_entity" in relation:
                # 直接使用数据库格式
                return self._store_relation_mysql(relation)
            # 检查是否是API格式（from_entities/to_entities数组）
            # 验证必要字段
            required_fields = ["from_entities", "to_entities", "relation_type"]
            for field in required_fields:
                if field not in relation:
                    logging.error(f"关系数据缺少必要字段 '{field}': {relation}")
                    return False
            # 验证from_entities和to_entities
            if not isinstance(relation["from_entities"], list) or not isinstance(relation["to_entities"], list):
                logging.error(f"from_entities或to_entities必须是列表: {relation}")
                return False
            if len(relation["from_entities"]) == 0 or len(relation["to_entities"]) == 0:
                logging.error(f"from_entities或to_entities不能为空: {relation}")
                return False
            # 对每一组from_entity和to_entity创建关系
            success_count = 0
            relation_count = 0
            for from_entity in relation["from_entities"]:
                for to_entity in relation["to_entities"]:
                    relation_count += 1
                    # 验证实体数据
                    if not isinstance(from_entity, dict) or not isinstance(to_entity, dict):
                        logging.warning(f"实体数据格式错误: {from_entity} -> {to_entity}")
                        continue
                    if "entity_id" not in from_entity or "entity_id" not in to_entity:
                        logging.warning(f"实体数据缺少entity_id字段: {from_entity} -> {to_entity}")
                        continue
                    from_entity_id = from_entity["entity_id"]
                    to_entity_id = to_entity["entity_id"]
                    from_entity_type = from_entity.get("entity_type", "Algorithm")
                    to_entity_type = to_entity.get("entity_type", "Algorithm")
                    # 创建关系记录
                    relation_data = {
                        "from_entity": from_entity_id,
                        "to_entity": to_entity_id,
                        "relation_type": relation.get("relation_type", "Improve"),
                        "structure": relation.get("structure", ""),
                        "detail": relation.get("detail", ""),
                        "evidence": relation.get("evidence", ""),
                        "confidence": relation.get("confidence", 0.0),
                        "from_entity_type": from_entity_type,
                        "to_entity_type": to_entity_type
                    }
                    # 调用数据库存储方法
                    if self._store_relation_mysql(relation_data):
                        success_count += 1
            logging.info(f"成功存储 {success_count}/{relation_count} 条演化关系")
            return success_count > 0
        except Exception as e:
            logging.error(f"存储演化关系时出错: {str(e)}")
            import traceback
            logging.error(traceback.format_exc())
            return False
    
    def _store_relation_mysql(self, relation_data):
        """
        将实体演化关系存储到MySQL数据库
        
        Args:
            relation_data (dict): 演化关系数据
            
        Returns:
            bool: 是否成功存储
        """
        try:
            # 准备数据
            from_entity = relation_data["from_entity"]
            to_entity = relation_data["to_entity"]
            relation_type = relation_data["relation_type"]
            structure = relation_data["structure"]
            detail = relation_data["detail"]
            evidence = relation_data["evidence"]
            confidence = relation_data["confidence"]
            from_entity_type = relation_data["from_entity_type"]
            to_entity_type = relation_data["to_entity_type"]
            # 检查实体是否存在
            from_exists = self._check_entity_exists(from_entity, from_entity_type)
            to_exists = self._check_entity_exists(to_entity, to_entity_type)
            if not from_exists:
                logging.warning(f"来源实体不存在: {from_entity} (类型: {from_entity_type})")
            if not to_exists:
                logging.warning(f"目标实体不存在: {to_entity} (类型: {to_entity_type})")
            # 即使实体不存在也尝试创建关系，因为实体可能在后续步骤中创建
            # 构建关系ID字符串
            relation_id_str = f"{from_entity}_{to_entity}_{relation_type}"
            # 首先检查是否存在相同的关系
            check_sql = """
            SELECT relation_id FROM EvolutionRelations 
            WHERE from_entity = %s AND to_entity = %s AND relation_type = %s
            """
            self.cursor.execute(check_sql, (from_entity, to_entity, relation_type))
            result = self.cursor.fetchone()
            if result:
                # 关系已存在，执行更新
                update_sql = """
                UPDATE EvolutionRelations 
                SET structure = %s, detail = %s, evidence = %s, confidence = %s,
                    from_entity_type = %s, to_entity_type = %s
                WHERE from_entity = %s AND to_entity = %s AND relation_type = %s
                """
                self.cursor.execute(
                    update_sql, 
                    (structure, detail, evidence, confidence, 
                     from_entity_type, to_entity_type,
                     from_entity, to_entity, relation_type)
                )
                logging.info(f"更新演化关系: {from_entity} -> {to_entity} ({relation_type})")
            else:
                # 关系不存在，执行插入
                insert_sql = """
                INSERT INTO EvolutionRelations (
                    from_entity, to_entity, relation_type, 
                    structure, detail, evidence, confidence, 
                    from_entity_type, to_entity_type
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                """
                self.cursor.execute(
                    insert_sql, 
                    (from_entity, to_entity, relation_type, 
                     structure, detail, evidence, confidence,
                     from_entity_type, to_entity_type)
                )
                logging.info(f"创建新演化关系: {from_entity} -> {to_entity} ({relation_type})")
            self.conn.commit()
            return True
        except Exception as e:
            self.conn.rollback()
            logging.error(f"存储演化关系到MySQL时出错: {str(e)}")
            import traceback
            logging.error(traceback.format_exc())
            return False
    
    def _check_entity_exists(self, entity_id, entity_type="Algorithm"):
        """检查实体是否存在于数据库中"""
        try:
            # 根据实体类型选择表
            if entity_type == "Algorithm":
                table_name = "Algorithms"
                id_column = "algorithm_id"
            elif entity_type == "Dataset":
                table_name = "Datasets"
                id_column = "dataset_id"
            elif entity_type == "Metric":
                table_name = "Metrics"
                id_column = "metric_id"
            else:
                logging.warning(f"未知的实体类型: {entity_type}")
                return False
            # 构建SQL查询
            check_sql = f"SELECT COUNT(*) FROM {table_name} WHERE {id_column} = %s"
            self.cursor.execute(check_sql, (entity_id,))
            result = self.cursor.fetchone()
            return result and result[0] > 0
        except Exception as e:
            logging.error(f"检查实体 {entity_id} 是否存在时出错: {str(e)}")
            return False
    
    def get_all_entities(self):
        """
        获取所有类型的实体信息，包括算法、数据集和评价指标。
        
        Returns:
            List[Dict]: 所有实体列表
        """
        logging.warning("开始获取所有实体...")
        entities = []
        try:
            # 获取算法实体
            algorithm_entities = self._get_all_entities_mysql()
            if algorithm_entities:
                entities.extend(algorithm_entities)
                logging.warning(f"获取到 {len(algorithm_entities)} 个算法实体")
            # 获取数据集实体
            dataset_entities = self.get_all_datasets()
            if dataset_entities:
                # 转换为统一格式
                for dataset in dataset_entities:
                    dataset['entity_type'] = 'Dataset'
                    dataset['entity_id'] = dataset['dataset_id']
                    entities.append({'dataset_entity': dataset})
                logging.warning(f"获取到 {len(dataset_entities)} 个数据集实体")
            # 获取评价指标实体
            metric_entities = self.get_all_metrics()
            if metric_entities:
                # 转换为统一格式
                for metric in metric_entities:
                    metric['entity_type'] = 'Metric'
                    metric['entity_id'] = metric['metric_id']
                    entities.append({'metric_entity': metric})
                logging.warning(f"获取到 {len(metric_entities)} 个评价指标实体")
        except Exception as e:
            logging.error(f"获取实体列表出错: {str(e)}")
            import traceback
            logging.error(traceback.format_exc())
            
        logging.warning(f"总共获取到 {len(entities)} 个实体")
        return entities
            
    def _get_all_entities_mysql(self):
        """从MySQL数据库获取所有算法实体"""
        max_retries = 3
        retry_count = 0
        
        while retry_count < max_retries:
            try:
                # 检查连接是否已关闭并重新连接
                if not self._reconnect_if_needed():
                    logging.error("无法建立MySQL连接，等待5秒后重试")
                    retry_count += 1
                    time.sleep(5)
                    continue
                
                self.cursor.execute("SELECT * FROM Algorithms")
                rows = self.cursor.fetchall()
                entities = []
                column_names = [description[0] for description in self.cursor.description]
                logging.warning("从MySQL获取到 %d 行数据", len(rows))
                for row in rows:
                    # 将行数据转换为字典
                    entity_dict = dict(zip(column_names, row))
                    # 处理JSON字段
                    for field in ['authors', 'dataset', 'metrics', 'architecture_components', 
                                  'architecture_connections', 'architecture_mechanisms',
                                  'methodology_training_strategy', 'methodology_parameter_tuning', 
                                  'feature_processing']:
                        if entity_dict.get(field) and isinstance(entity_dict[field], str):
                            try:
                                entity_dict[field] = json.loads(entity_dict[field])
                            except json.JSONDecodeError:
                                # 如果不是有效的JSON，尝试按逗号分隔
                                if ',' in entity_dict[field]:
                                    entity_dict[field] = [item.strip() for item in entity_dict[field].split(',')]
                                else:
                                    entity_dict[field] = [entity_dict[field]]
                    # 构建规范化的实体对象
                    algorithm_entity = {
                        'algorithm_id': entity_dict['algorithm_id'],
                        'entity_id': entity_dict['algorithm_id'],  # 增加统一的entity_id字段
                        'name': entity_dict['name'],
                        'title': entity_dict.get('title', ''),
                        'year': entity_dict.get('year', ''),
                        'authors': entity_dict.get('authors', []),
                        'task': entity_dict.get('task', ''),
                        'dataset': entity_dict.get('dataset', []),
                        'metrics': entity_dict.get('metrics', []),
                        'architecture': {
                            'components': entity_dict.get('architecture_components', []),
                            'connections': entity_dict.get('architecture_connections', []),
                            'mechanisms': entity_dict.get('architecture_mechanisms', [])
                        },
                        'methodology': {
                            'training_strategy': entity_dict.get('methodology_training_strategy', []),
                            'parameter_tuning': entity_dict.get('methodology_parameter_tuning', [])
                        },
                        'feature_processing': entity_dict.get('feature_processing', []),
                        'entity_type': 'Algorithm'
                    }
                    # 添加到实体列表
                    entities.append({'algorithm_entity': algorithm_entity})
                if not entities:
                    logging.warning("没有找到任何算法实体，返回空列表")
                else:
                    logging.warning("返回 %d 个算法实体", len(entities))
                return entities
            except (mysql.connector.Error, MySQLError) as e:
                retry_count += 1
                logging.error(f"从MySQL获取实体时出错 (尝试 {retry_count}/{max_retries}): {str(e)}")
                # 如果是连接问题，等待后重试
                time.sleep(5)
                # 尝试重新连接
                self._connect_mysql()
                if retry_count >= max_retries:
                    logging.error("重试次数已达上限，无法从数据库获取实体")
                    break
            except Exception as e:
                logging.error("从MySQL获取实体时出错: %s", str(e))
                import traceback
                logging.error(traceback.format_exc())
                break
        return []
            
    def get_all_relations(self):
        """
        获取所有演化关系的信息。
        
        Returns:
            List[Dict]: 演化关系列表
        """
        logging.warning("开始获取所有关系...")
        return self._get_all_relations_mysql()
            
    def _get_all_relations_mysql(self):
        """从MySQL获取所有演化关系"""
        max_retries = 3
        retry_count = 0
        while retry_count < max_retries:
            try:
                # 检查连接是否已关闭
                if not self._reconnect_if_needed():
                    logging.error("无法建立MySQL连接，等待5秒后重试")
                    retry_count += 1
                    time.sleep(5)
                    continue
                self.cursor.execute("""
                SELECT relation_id, from_entity, to_entity, relation_type, structure, 
                       detail, evidence, confidence, from_entity_type, to_entity_type
                FROM EvolutionRelations
                """)
                rows = self.cursor.fetchall()
                relations = []
                column_names = [description[0] for description in self.cursor.description]
                logging.warning("从MySQL获取到 %d 行关系数据", len(rows))
                for row in rows:
                    # 将行数据转换为字典
                    relation_dict = dict(zip(column_names, row))
                    # 确保confidence是浮点数
                    if 'confidence' in relation_dict and relation_dict['confidence'] is not None:
                        relation_dict['confidence'] = float(relation_dict['confidence'])
                    relations.append(relation_dict)
                if not relations:
                    logging.warning("没有找到任何关系，返回空列表")
                else:
                    logging.warning("返回 %d 个关系，示例: %s", 
                                 len(relations), 
                                 json.dumps(relations[0], ensure_ascii=False) if relations else "无")
                return relations
            except (mysql.connector.Error, MySQLError) as e:
                retry_count += 1
                logging.error(f"从MySQL获取关系时出错 (尝试 {retry_count}/{max_retries}): {str(e)}")
                # 如果是连接问题，等待后重试
                time.sleep(5)
                # 尝试重新连接
                self._connect_mysql()
                if retry_count >= max_retries:
                    logging.error("重试次数已达上限，无法从数据库获取关系")
                    break
            except Exception as e:
                logging.error("从MySQL获取关系时出错: %s", str(e))
                import traceback
                logging.error(traceback.format_exc())
                break
        return []
    
    def update_entity(self, entity_id, updated_data):
        """
        更新指定算法实体的信息。
        
        Args:
            entity_id (str): 算法实体ID
            updated_data (Dict): 更新后的数据
            
        Returns:
            bool: 操作是否成功
        """
        try:
            return self._update_entity_mysql(entity_id, updated_data)
        except Exception as e:
            logging.error(f"更新实体 {entity_id} 时出错: {str(e)}")
            return False
    
    def _update_entity_mysql(self, entity_id, updated_data):
        """在MySQL中更新实体"""
        # 检查实体是否存在
        self.cursor.execute('SELECT * FROM Algorithms WHERE algorithm_id = %s', (entity_id,))
        if not self.cursor.fetchone():
            logging.error(f"实体 {entity_id} 不存在")
            return False
        
        # 构建SET子句和参数
        set_clauses = []
        params = []
        
        if 'name' in updated_data:
            set_clauses.append('name = %s')
            params.append(updated_data['name'])
        if 'title' in updated_data:
            set_clauses.append('title = %s')
            params.append(updated_data['title'])
        if 'year' in updated_data:
            set_clauses.append('year = %s')
            params.append(updated_data['year'])
        if 'authors' in updated_data:
            set_clauses.append('authors = %s')
            params.append(json.dumps(updated_data['authors'], ensure_ascii=False))
        if 'task' in updated_data:
            set_clauses.append('task = %s')
            params.append(updated_data['task'])
        if 'dataset' in updated_data:
            set_clauses.append('dataset = %s')
            params.append(json.dumps(updated_data['dataset'], ensure_ascii=False))
        if 'metrics' in updated_data:
            set_clauses.append('metrics = %s')
            params.append(json.dumps(updated_data['metrics'], ensure_ascii=False))
        
        # 处理架构信息
        if 'architecture' in updated_data:
            arch = updated_data['architecture']
            if 'components' in arch:
                set_clauses.append('architecture_components = %s')
                params.append(json.dumps(arch['components'], ensure_ascii=False))
            if 'connections' in arch:
                set_clauses.append('architecture_connections = %s')
                params.append(json.dumps(arch['connections'], ensure_ascii=False))
            if 'mechanisms' in arch:
                set_clauses.append('architecture_mechanisms = %s')
                params.append(json.dumps(arch['mechanisms'], ensure_ascii=False))
        
        # 处理方法学信息
        if 'methodology' in updated_data:
            meth = updated_data['methodology']
            if 'training_strategy' in meth:
                set_clauses.append('methodology_training_strategy = %s')
                params.append(json.dumps(meth['training_strategy'], ensure_ascii=False))
            if 'parameter_tuning' in meth:
                set_clauses.append('methodology_parameter_tuning = %s')
                params.append(json.dumps(meth['parameter_tuning'], ensure_ascii=False))
        
        # 处理特征处理信息
        if 'feature_processing' in updated_data:
            set_clauses.append('feature_processing = %s')
            params.append(json.dumps(updated_data['feature_processing'], ensure_ascii=False))
        
        # 如果没有更新字段，直接返回成功
        if not set_clauses:
            return True
        
        # 完成SQL语句
        update_sql = 'UPDATE Algorithms SET ' + ', '.join(set_clauses) + ' WHERE algorithm_id = %s'
        params.append(entity_id)
        
        # 执行更新
        self.cursor.execute(update_sql, params)
        self.conn.commit()
        
        return True
    
    def add_relation(self, relation):
        """
        添加新的演化关系。
        
        Args:
            relation (Dict): 新关系的数据
            
        Returns:
            bool: 操作是否成功
        """
        try:
            return self._add_relation_mysql(relation)
        except Exception as e:
            logging.error(f"添加关系时出错: {str(e)}")
            return False
    
    def _add_relation_mysql(self, relation):
        """在MySQL中添加关系"""
        # 检查目标实体是否存在 - 移除外键约束检查
        to_entity = relation.get('to_entity', '')
        if not to_entity:
            logging.error("缺少目标实体ID")
            return False
        
        # 获取实体类型
        from_entity_type = relation.get('from_entity_type', 'Algorithm')
        to_entity_type = relation.get('to_entity_type', 'Algorithm')
        
        # 确保关系类型存在且不为空
        relation_type = relation.get('relation_type', '')

        
        # 插入关系
        sql = '''
        INSERT INTO EvolutionRelations
        (from_entity, to_entity, relation_type, structure, detail, evidence, confidence, from_entity_type, to_entity_type)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
        '''
        
        self.cursor.execute(sql, (
            relation.get('from_entity', ''),
            to_entity,
            relation_type,
            relation.get('structure', ''),
            relation.get('detail', ''),
            relation.get('evidence', ''),
            relation.get('confidence', 0.0),
            from_entity_type,
            to_entity_type
        ))
        
        self.conn.commit()
        return True
    
    def modify_relation(self, relation_id, updated_relation):
        """
        修改现有的演化关系。
        
        Args:
            relation_id (int/str): 关系ID
            updated_relation (Dict): 更新后的关系数据
            
        Returns:
            bool: 操作是否成功
        """
        try:
            return self._modify_relation_mysql(relation_id, updated_relation)
        except Exception as e:
            logging.error(f"修改关系 {relation_id} 时出错: {str(e)}")
            return False
    
    def _modify_relation_mysql(self, relation_id, updated_relation):
        """在MySQL中修改关系"""
        # 检查关系是否存在
        self.cursor.execute('SELECT * FROM EvolutionRelations WHERE relation_id = %s', (relation_id,))
        if not self.cursor.fetchone():
            logging.error(f"关系 {relation_id} 不存在")
            return False
        
        # 构建SET子句和参数
        set_clauses = []
        params = []
        
        if 'from_entity' in updated_relation:
            set_clauses.append('from_entity = %s')
            params.append(updated_relation['from_entity'])
        if 'to_entity' in updated_relation:
            set_clauses.append('to_entity = %s')
            params.append(updated_relation['to_entity'])
        if 'relation_type' in updated_relation:
            set_clauses.append('relation_type = %s')
            params.append(updated_relation['relation_type'])
        if 'structure' in updated_relation:
            set_clauses.append('structure = %s')
            params.append(updated_relation['structure'])
        if 'detail' in updated_relation:
            set_clauses.append('detail = %s')
            params.append(updated_relation['detail'])
        if 'evidence' in updated_relation:
            set_clauses.append('evidence = %s')
            params.append(updated_relation['evidence'])
        if 'confidence' in updated_relation:
            set_clauses.append('confidence = %s')
            params.append(updated_relation['confidence'])
        
        # 如果没有更新字段，直接返回成功
        if not set_clauses:
            return True
        
        # 完成SQL语句
        update_sql = 'UPDATE EvolutionRelations SET ' + ', '.join(set_clauses) + ' WHERE relation_id = %s'
        params.append(relation_id)
        
        # 执行更新
        self.cursor.execute(update_sql, params)
        self.conn.commit()
        
        return True
    
    def delete_relation(self, relation_id):
        """
        删除指定的演化关系。
        
        Args:
            relation_id (int/str): 关系ID
            
        Returns:
            bool: 操作是否成功
        """
        try:
            return self._delete_relation_mysql(relation_id)
        except Exception as e:
            logging.error(f"删除关系 {relation_id} 时出错: {str(e)}")
            return False
    
    def _delete_relation_mysql(self, relation_id):
        """在MySQL中删除关系"""
        self.cursor.execute('DELETE FROM EvolutionRelations WHERE relation_id = %s', (relation_id,))
        self.conn.commit()
        
        return True

    def create_processing_task(self, task_id, task_name):
        """
        创建新的处理任务记录
        
        Args:
            task_id (str): 任务ID
            task_name (str): 任务名称或描述
            
        Returns:
            bool: 操作是否成功
        """
        try:
            now = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            
            self.cursor.execute('''
            INSERT INTO ProcessingStatus 
            (task_id, status, current_stage, progress, message, start_time, update_time)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            ON DUPLICATE KEY UPDATE
            status = VALUES(status),
            current_stage = VALUES(current_stage),
            progress = VALUES(progress),
            message = VALUES(message),
            update_time = VALUES(update_time)
            ''', (
                task_id, 'waiting', '初始化', 0.0, f'创建任务: {task_name}', now, now
            ))
            
            self.conn.commit()
            return True
        
        except Exception as e:
            logging.error(f"创建处理任务记录时出错: {str(e)}")
            return False
    
    def update_processing_status(self, task_id, status=None, current_stage=None, progress=None, 
                                 current_file=None, message=None, completed=None):
        """
        更新任务处理状态
        
        参数:
            task_id: 任务ID
            status: 任务状态 (如'pending', 'processing', 'completed', 'failed')
            current_stage: 当前处理阶段
            progress: 进度 (0-100)
            current_file: 当前处理的文件
            message: 状态消息
            completed: 是否完成
        """
        try:
            # 检查是否需要重连
            reconnected = self._reconnect_if_needed()
            # 准备更新字段
            update_fields = []
            params = []
            if status is not None:
                update_fields.append("status = %s")
                params.append(status)
            if current_stage is not None:
                update_fields.append("current_stage = %s")
                params.append(current_stage)
            if progress is not None:
                update_fields.append("progress = %s")
                params.append(progress)
            if current_file is not None:
                update_fields.append("current_file = %s")
                params.append(current_file)
            if message is not None:
                update_fields.append("message = %s")
                params.append(message)
            # 更新修改时间
            update_fields.append("update_time = %s")
            params.append(datetime.datetime.now())
            # 如果标记为完成，设置结束时间
            if completed is not None and completed:
                update_fields.append("end_time = %s")
                params.append(datetime.datetime.now())
                update_fields.append("completed = %s")
                params.append(1)
            elif completed is not None:
                update_fields.append("completed = %s")
                params.append(0)
            if not update_fields:
                logging.warning(f"没有提供任何更新字段，任务 {task_id} 的状态未更新")
                return False
            # 构建SQL语句
            sql = f"UPDATE ProcessingStatus SET {', '.join(update_fields)} WHERE task_id = %s"
            params.append(task_id)
            # 使用重试机制执行更新
            max_retries = 3
            retry_count = 0
            while retry_count < max_retries:
                try:
                    # 检查连接状态
                    if retry_count > 0:
                        reconnected = self._reconnect_if_needed()
                        if reconnected:
                            logging.info(f"第 {retry_count+1} 次尝试重连成功，准备更新任务状态")
                    # 执行更新
                    self.cursor.execute(sql, tuple(params))
                    self.conn.commit()
                    # 检查更新是否成功
                    if self.cursor.rowcount > 0:
                        return True
                    else:
                        logging.warning(f"未找到任务 {task_id} 或状态未发生变化")
                        # 如果任务不存在，尝试创建
                        if not self._check_task_exists(task_id):
                            logging.info(f"任务 {task_id} 不存在，正在创建新任务记录")
                            return self._create_processing_task(task_id, status, current_stage, progress, 
                                                               current_file, message, completed)
                        return False
                except mysql.connector.Error as err:
                    retry_count += 1
                    logging.error(f"更新任务状态时出错 (尝试 {retry_count}/{max_retries}): {str(err)}")
                    if retry_count < max_retries:
                        logging.info(f"5秒后重试更新任务状态...")
                        time.sleep(5)
                    else:
                        logging.error(f"已达到最大重试次数，无法更新任务 {task_id} 的状态")
                        return False
            return False
        except Exception as e:
            logging.error(f"更新任务 {task_id} 的处理状态时发生错误: {str(e)}")
            return False

    def get_processing_status(self, task_id):
        """
        获取任务处理状态
        
        参数:
            task_id: 任务ID
            
        返回:
            dict: 包含任务状态信息的字典，如果任务不存在则返回None
        """
        try:
            # 检查是否需要重连
            reconnected = self._reconnect_if_needed()
            
            # 使用重试机制执行查询
            max_retries = 3
            retry_count = 0
            
            while retry_count < max_retries:
                try:
                    # 检查连接状态
                    if retry_count > 0:
                        reconnected = self._reconnect_if_needed()
                        if reconnected:
                            logging.info(f"第 {retry_count+1} 次尝试重连成功，准备获取任务状态")
                    
                    # 执行查询
                    sql = """
                        SELECT 
                            task_id, status, current_stage, progress, 
                            current_file, message, start_time, update_time, end_time
                        FROM 
                            ProcessingStatus 
                        WHERE 
                            task_id = %s
                        """
                    self.cursor.execute(sql, (task_id,))
                    row = self.cursor.fetchone()
                    
                    if not row:
                        logging.warning(f"未找到任务: {task_id}")
                        return None
                    
                    # 获取字段名
                    field_names = [desc[0] for desc in self.cursor.description]
                    
                    # 将查询结果转换为字典
                    result = {}
                    for i, field in enumerate(field_names):
                        # 将日期时间字段转换为字符串
                        if isinstance(row[i], (datetime.datetime, datetime.date)):
                            result[field] = row[i].isoformat()
                        else:
                            result[field] = row[i]
                    
                    # 计算完成状态
                    result['completed'] = result['end_time'] is not None
                    
                    return result
                
                except mysql.connector.Error as err:
                    retry_count += 1
                    logging.error(f"获取任务状态时出错 (尝试 {retry_count}/{max_retries}): {str(err)}")
                    
                    if retry_count < max_retries:
                        logging.info(f"5秒后重试获取任务状态...")
                        time.sleep(5)
                    else:
                        logging.error(f"已达到最大重试次数，无法获取任务 {task_id} 的状态")
                        return None
            
            return None
                
        except Exception as e:
            logging.error(f"获取任务 {task_id} 的处理状态时发生错误: {str(e)}")
            return None
            
    def _create_processing_task(self, task_id, status=None, current_stage=None, progress=None, 
                               current_file=None, message=None, completed=None):
        """
        创建新的任务处理记录
        
        参数与update_processing_status相同
        """
        try:
            # 检查是否需要重连
            self._reconnect_if_needed()
            
            now = datetime.datetime.now()
            
            # 准备插入字段和值
            fields = ["task_id", "start_time", "update_time"]
            values = [task_id, now, now]
            
            if status is not None:
                fields.append("status")
                values.append(status)
            
            if current_stage is not None:
                fields.append("current_stage")
                values.append(current_stage)
            
            if progress is not None:
                fields.append("progress")
                values.append(progress)
            
            if current_file is not None:
                fields.append("current_file")
                values.append(current_file)
            
            if message is not None:
                fields.append("message")
                values.append(message)
            
            if completed is not None and completed:
                fields.append("end_time")
                values.append(now)
                fields.append("completed")
                values.append(1)
            elif completed is not None:
                fields.append("completed")
                values.append(0)
            
            # 构建SQL语句
            placeholders = ", ".join(["%s"] * len(fields))
            sql = f"INSERT INTO ProcessingStatus ({', '.join(fields)}) VALUES ({placeholders})"
            
            # 执行插入
            self.cursor.execute(sql, tuple(values))
            self.conn.commit()
            
            logging.info(f"成功创建任务 {task_id} 的处理记录")
            return True
                
        except Exception as e:
            logging.error(f"创建任务处理记录时出错: {str(e)}")
            return False
    
    def get_entity_by_id(self, entity_id):
        """
        通过ID获取任意类型的实体
        
        Args:
            entity_id (str): 实体ID
            
        Returns:
            dict: 实体信息字典
        """
        try:
            # 首先检查是否是算法实体
            logging.info(f"正在获取实体 {entity_id}")
            self.cursor.execute('SELECT * FROM Algorithms WHERE algorithm_id = %s', (entity_id,))
            row = self.cursor.fetchone()
            if row:
            # 解析JSON字符串为Python对象
                try:
                    authors = json.loads(row[4]) if row[4] else []
                except:
                    authors = []
                try:
                    dataset = json.loads(row[6]) if row[6] else []
                except:
                    dataset = []
                try:
                    metrics = json.loads(row[7]) if row[7] else []
                except:
                    metrics = []
                try:
                    arch_components = json.loads(row[8]) if row[8] else []
                except:
                    arch_components = []
                try:
                    arch_connections = json.loads(row[9]) if row[9] else []
                except:
                    arch_connections = []
                try:
                    arch_mechanisms = json.loads(row[10]) if row[10] else []
                except:
                    arch_mechanisms = []
                try:
                    meth_training = json.loads(row[11]) if row[11] else []
                except:
                    meth_training = []
                try:
                    meth_params = json.loads(row[12]) if row[12] else []
                except:
                    meth_params = []
                try:
                    feature_processing = json.loads(row[13]) if row[13] else []
                except:
                    feature_processing = []
                # 构建算法实体对象
                entity = {
                    'algorithm_entity': {
                        'algorithm_id': row[0],
                            'entity_id': row[0],
                        'name': row[1],
                        'title': row[2],
                        'year': row[3],
                        'authors': authors,
                        'task': row[5],
                        'dataset': dataset,
                        'metrics': metrics,
                        'architecture': {
                            'components': arch_components,
                            'connections': arch_connections,
                            'mechanisms': arch_mechanisms
                        },
                        'methodology': {
                            'training_strategy': meth_training,
                            'parameter_tuning': meth_params
                        },
                        'feature_processing': feature_processing,
                            'entity_type': 'Algorithm'
                        }
                    }
                # 获取演化关系
                relations = self._get_entity_relations(entity_id)
                entity['algorithm_entity']['evolution_relations'] = relations
                return entity
            # 如果不是算法实体，检查是否是数据集实体
            self.cursor.execute('SELECT * FROM Datasets WHERE dataset_id = %s', (entity_id,))
            row = self.cursor.fetchone()
            
            if row:
                # 转换数据集为标准格式
                dataset = {}
                for i, col in enumerate(self.cursor.description):
                    dataset[col[0]] = row[i]
                
                # 解析可能的JSON字段
                if 'creators' in dataset and dataset['creators']:
                    try:
                        dataset['creators'] = json.loads(dataset['creators'])
                    except:
                        # 如果不是有效的JSON，保持原样
                        pass
                
                # 确保entity_id字段存在
                dataset['entity_id'] = dataset['dataset_id']
                dataset['entity_type'] = 'Dataset'
            
            # 获取演化关系
                relations = self._get_entity_relations(entity_id)
                dataset['evolution_relations'] = relations
                
                return {'dataset_entity': dataset}
                
            # 如果不是数据集实体，检查是否是评估指标实体
            self.cursor.execute('SELECT * FROM Metrics WHERE metric_id = %s', (entity_id,))
            row = self.cursor.fetchone()
            
            if row:
                # 转换评估指标为标准格式
                metric = {}
                for i, col in enumerate(self.cursor.description):
                    metric[col[0]] = row[i]
                
                # 确保entity_id字段存在
                metric['entity_id'] = metric['metric_id']
                metric['entity_type'] = 'Metric'
                
                # 获取演化关系
                relations = self._get_entity_relations(entity_id)
                metric['evolution_relations'] = relations
                
                return {'metric_entity': metric}
            
            # 如果没有找到任何实体，返回None
            return None
            
        except Exception as e:
            logging.error(f"通过ID获取实体时出错: {str(e)}")
            import traceback
            logging.error(traceback.format_exc())
            return None
    
    def init_db(self):
        """初始化数据库"""
        self._connect_mysql()
        
        logging.info("MySQL数据库初始化完成")
        return True

    def _get_entity_relations(self, entity_id):
        """
        获取指定实体的演化关系
        
        Args:
            entity_id (str): 实体ID
            
        Returns:
            List[Dict]: 演化关系列表
        """
        try:
            logging.info(f"正在获取实体 {entity_id} 的演化关系")
            
            # 查询指向该实体的关系
            self.cursor.execute(
                """
                SELECT relation_id, from_entity, to_entity, relation_type, 
                       structure, detail, evidence, confidence,
                       from_entity_type, to_entity_type 
                FROM EvolutionRelations WHERE to_entity = %s
                """,
                (entity_id,)
            )
            rows = self.cursor.fetchall()
            
            relations = []
            column_names = ['relation_id', 'from_entity', 'to_entity', 'relation_type', 
                            'structure', 'detail', 'evidence', 'confidence', 
                            'from_entity_type', 'to_entity_type']
            
            for row in rows:
                relation = {}
                for i, col in enumerate(column_names):
                    relation[col] = row[i]
                
                if col == 'confidence' and relation[col] is not None:
                    relation[col] = float(relation[col])
                
                relations.append(relation)
            
            logging.info(f"实体 {entity_id} 有 {len(relations)} 个演化关系")
            return relations
            
        except Exception as e:
            logging.error(f"获取实体 {entity_id} 的演化关系时出错: {str(e)}")
            return []

    def delete_entity(self, entity_id):
        """
        删除指定的算法实体及其相关关系
        
        Args:
            entity_id (str): 实体ID
            
        Returns:
            bool: 操作是否成功
        """
        try:
            # 先删除与此实体相关的所有关系
            self.cursor.execute('DELETE FROM EvolutionRelations WHERE from_entity = %s OR to_entity = %s', 
                              (entity_id, entity_id))
            
            # 然后删除实体本身
            self.cursor.execute('DELETE FROM Algorithms WHERE algorithm_id = %s', (entity_id,))
            
            # 检查是否有行被删除
            rows_affected = self.cursor.rowcount
            
            self.conn.commit()
            
            return rows_affected > 0
        except Exception as e:
            logging.error(f"删除实体 {entity_id} 时出错: {str(e)}")
            self.conn.rollback()
            return False

    def store_algorithm_entity(self, entity):
        """
        存储算法实体到MySQL数据库
        
        Args:
            entity (dict): 要存储的算法实体数据
            
        Returns:
            bool: 是否成功存储
        """
        try:
            entity_type=None
            # 支持嵌套格式的实体
            if 'algorithm_entity' in entity:
                actual_entity = entity['algorithm_entity']
                entity_type = actual_entity.get('entity_type', 'Algorithm')
                if 'algorithm_id' in actual_entity:
                    actual_entity['entity_id'] = actual_entity['algorithm_id']
            elif 'dataset_entity' in entity:
                actual_entity = entity['dataset_entity']
                entity_type = actual_entity.get('entity_type', 'Dataset')
                if 'dataset_id' in actual_entity:
                    actual_entity['entity_id'] = actual_entity['dataset_id']
            elif 'metric_entity' in entity:
                actual_entity = entity['metric_entity']
                entity_type = actual_entity.get('entity_type', 'Metric')
                if 'metric_id' in actual_entity:
                    actual_entity['entity_id'] = actual_entity['metric_id']
            else:
                # 支持直接格式的实体
                actual_entity = entity
            
            # 确保ID存在且一致
            if 'algorithm_id' in actual_entity and 'entity_id' not in actual_entity:
                actual_entity['entity_id'] = actual_entity['algorithm_id'] 
            if 'entity_id' in actual_entity and 'algorithm_id' not in actual_entity:
                actual_entity['algorithm_id'] = actual_entity['entity_id']
            
            entity_id = actual_entity.get('entity_id', '')
            if not entity_id:
                logging.error("实体ID不能为空")
                return False
            
            logging.info(f"存储实体: {entity_id} (类型: {entity_type})")
            
            if entity_type == 'Algorithm':
                # 存储算法实体
                self._store_algorithm_mysql(actual_entity)
            elif entity_type == 'Dataset':
                # 存储数据集实体
                self._store_dataset_mysql(actual_entity)
            elif entity_type == 'Metric':
                # 存储评价指标实体
                self._store_metric_mysql(actual_entity)
            else:
                logging.warning(f"未知实体类型: {entity_type}，跳过存储")
                return False
            
            return True
            
        except Exception as e:
            logging.error(f"存储实体时出错: {str(e)}")
            return False

    def _store_algorithm_mysql(self, entity):
        """将算法实体存储到MySQL数据库"""
        try:
            # 从实体中提取相关字段
            algorithm_id = entity.get('entity_id', '') or entity.get('algorithm_id', '')
            name = entity.get('name', '')
            title = entity.get('title', '')
            year = str(entity.get('year', ''))
            
            # 处理列表类字段，确保是JSON字符串
            authors = entity.get('authors', [])
            if not isinstance(authors, list):
                authors = [authors] if authors else []
            authors = json.dumps(authors, ensure_ascii=False)
            
            # 处理task字段，可能是字符串或列表
            task_value = entity.get('task', entity.get('tasks', ''))
            if isinstance(task_value, list):
                task = ", ".join(str(t) for t in task_value)
            else:
                task = str(task_value)
            
            # 处理dataset字段，兼容dataset和datasets两种命名
            dataset_value = entity.get('dataset', entity.get('datasets', []))
            if not isinstance(dataset_value, list):
                dataset_value = [dataset_value] if dataset_value else []
            dataset = json.dumps(dataset_value, ensure_ascii=False)
            
            # 处理metrics字段，兼容metric和metrics两种命名
            metrics_value = entity.get('metrics', entity.get('metric', []))
            if not isinstance(metrics_value, list):
                metrics_value = [metrics_value] if metrics_value else []
            metrics = json.dumps(metrics_value, ensure_ascii=False)
            
            # 提取架构信息
            architecture = entity.get('architecture', {})
            if not isinstance(architecture, dict):
                architecture = {}
            
            # 处理architecture的components字段
            components = architecture.get('components', [])
            if not isinstance(components, list):
                components = [components] if components else []
            arch_components = json.dumps(components, ensure_ascii=False)
            
            # 处理architecture的connections字段
            connections = architecture.get('connections', [])
            if not isinstance(connections, list):
                connections = [connections] if connections else []
            arch_connections = json.dumps(connections, ensure_ascii=False)
            
            # 处理architecture的mechanisms字段
            mechanisms = architecture.get('mechanisms', [])
            if not isinstance(mechanisms, list):
                mechanisms = [mechanisms] if mechanisms else []
            arch_mechanisms = json.dumps(mechanisms, ensure_ascii=False)
                
            # 提取方法学信息
            methodology = entity.get('methodology', {})
            if not isinstance(methodology, dict):
                methodology = {}
            
            # 处理methodology的training_strategy字段
            training = methodology.get('training_strategy', [])
            if not isinstance(training, list):
                training = [training] if training else []
            training_strategy = json.dumps(training, ensure_ascii=False)
            
            # 处理methodology的parameter_tuning字段
            tuning = methodology.get('parameter_tuning', [])
            if not isinstance(tuning, list):
                tuning = [tuning] if tuning else []
            parameter_tuning = json.dumps(tuning, ensure_ascii=False)
            
            # 提取特征处理信息
            feature_proc = entity.get('feature_processing', [])
            if not isinstance(feature_proc, list):
                feature_proc = [feature_proc] if feature_proc else []
            feature_processing = json.dumps(feature_proc, ensure_ascii=False)
            
            # 检查是否已存在
            self.cursor.execute('SELECT COUNT(*) FROM Algorithms WHERE algorithm_id = %s', (algorithm_id,))
            if self.cursor.fetchone()[0] > 0:
                # 更新现有记录
                sql = '''
                    UPDATE Algorithms SET
                        name = %s, title = %s, year = %s, authors = %s,
                        task = %s, dataset = %s, metrics = %s,
                        architecture_components = %s, architecture_connections = %s,
                        architecture_mechanisms = %s, methodology_training_strategy = %s,
                        methodology_parameter_tuning = %s, feature_processing = %s,
                        entity_type = %s
                    WHERE algorithm_id = %s
                    '''
                self.cursor.execute(sql, (
                        name, title, year, authors,
                        task, dataset, metrics,
                        arch_components, arch_connections,
                        arch_mechanisms, training_strategy,
                        parameter_tuning, feature_processing,
                        'Algorithm', algorithm_id
                    ))
                logging.info(f"更新算法: {algorithm_id}")
            else:
                # 创建新记录
                sql = '''
                INSERT INTO Algorithms (
                    algorithm_id, name, title, year, authors,
                    task, dataset, metrics,
                    architecture_components, architecture_connections,
                    architecture_mechanisms, methodology_training_strategy,
                    methodology_parameter_tuning, feature_processing, entity_type
                ) VALUES (
                    %s, %s, %s, %s, %s,
                    %s, %s, %s,
                    %s, %s,
                    %s, %s,
                    %s, %s, %s
                )
                '''
                self.cursor.execute(sql, (
                    algorithm_id, name, title, year, authors,
                    task, dataset, metrics,
                    arch_components, arch_connections,
                    arch_mechanisms, training_strategy,
                    parameter_tuning, feature_processing, 'Algorithm'
                ))
                logging.info(f"存储算法: {algorithm_id}")
            
            self.conn.commit()
            return True
        except Exception as e:
            self.conn.rollback()
            logging.error(f"存储算法时出错: {str(e)}")
            return False
    
    def _store_dataset_mysql(self, entity):
        """将数据集实体存储到MySQL数据库"""
        try:
            # 从实体中提取相关字段
            dataset_id = entity.get('entity_id', '')
            name = entity.get('name', '')
            description = entity.get('description', '')
            domain = entity.get('domain', '')
            
            # 处理size字段，确保是整数
            size_value = entity.get('size', 0)
            try:
                # 尝试转换为整数
                if isinstance(size_value, str) and size_value.strip().lower() in ['未明确', '未知', 'unknown', 'unspecified', '']:
                    size = 0
                else:
                    size = int(size_value)
            except (ValueError, TypeError):
                # 如果转换失败，使用默认值0
                logging.warning(f"无法将数据集大小'{size_value}'转换为整数，使用默认值0")
                size = 0
                
            year = str(entity.get('year', ''))
            
            # 处理creators字段，确保是JSON字符串
            creators_value = entity.get('creators', [])
            if not isinstance(creators_value, list):
                creators_value = [creators_value] if creators_value else []
            creators = json.dumps(creators_value, ensure_ascii=False)
            
            # 检查是否已存在
            self.cursor.execute('SELECT COUNT(*) FROM Datasets WHERE dataset_id = %s', (dataset_id,))
            if self.cursor.fetchone()[0] > 0:
                # 更新现有记录
                sql = '''
                    UPDATE Datasets SET
                        name = %s, description = %s, domain = %s,
                        size = %s, year = %s, creators = %s,
                        entity_type = %s
                    WHERE dataset_id = %s
                    '''
                self.cursor.execute(sql, (
                        name, description, domain,
                        size, year, creators,
                        'Dataset', dataset_id
                    ))
                logging.info(f"更新数据集: {dataset_id}")
            else:
                # 创建新记录
                sql = '''
                INSERT INTO Datasets (
                    dataset_id, name, description, domain,
                    size, year, creators, entity_type
                ) VALUES (
                    %s, %s, %s, %s,
                    %s, %s, %s, %s
                )
                '''
                self.cursor.execute(sql, (
                    dataset_id, name, description, domain,
                    size, year, creators, 'Dataset'
                ))
                logging.info(f"存储数据集: {dataset_id}")
            
            self.conn.commit()
            return True
        except Exception as e:
            self.conn.rollback()
            logging.error(f"存储数据集时出错: {str(e)}")
            return False

    def _store_metric_mysql(self, entity):
        """将评价指标实体存储到MySQL数据库"""
        try:
            # 从实体中提取相关字段
            metric_id = entity.get('entity_id', '')
            name = entity.get('name', '')
            description = entity.get('description', '')
            category = entity.get('category', '')
            formula = entity.get('formula', '')
            
            # 确保所有字段都是字符串类型
            name = str(name) if name else ''
            description = str(description) if description else ''
            category = str(category) if category else ''
            formula = str(formula) if formula else ''
            
            # 检查是否已存在
            self.cursor.execute('SELECT COUNT(*) FROM Metrics WHERE metric_id = %s', (metric_id,))
            if self.cursor.fetchone()[0] > 0:
                # 更新现有记录
                sql = '''
                UPDATE Metrics SET
                    name = %s, description = %s, category = %s,
                    formula = %s, entity_type = %s
                WHERE metric_id = %s
                '''
                self.cursor.execute(sql, (
                    name, description, category,
                    formula, 'Metric', metric_id
                ))
                logging.info(f"更新评价指标: {metric_id}")
            else:
                # 创建新记录
                sql = '''
                INSERT INTO Metrics (
                    metric_id, name, description, category,
                    formula, entity_type
                ) VALUES (
                    %s, %s, %s, %s,
                    %s, %s
                )
                '''
                self.cursor.execute(sql, (
                    metric_id, name, description, category,
                    formula, 'Metric'
                ))
                logging.info(f"存储评价指标: {metric_id}")
            
            self.conn.commit()
            return True
        except Exception as e:
            self.conn.rollback()
            logging.error(f"存储评价指标时出错: {str(e)}")
            return False

    def get_all_datasets(self):
        """获取所有数据集实体"""
        max_retries = 3
        retry_count = 0
        while retry_count < max_retries:
            try:
                # 检查连接状态
                if not self._reconnect_if_needed():
                    logging.error("无法建立MySQL连接，等待5秒后重试")
                    retry_count += 1
                    time.sleep(5)
                    continue
                self.cursor.execute('SELECT * FROM Datasets')
                result = self.cursor.fetchall()
                if result:
                    # 转换结果为字典列表
                    datasets = []
                    for row in result:
                        dataset = {}
                        for i, col in enumerate(self.cursor.description):
                            dataset[col[0]] = row[i]
                        datasets.append(dataset)
                    return datasets
                return []
            except (mysql.connector.Error, MySQLError) as e:
                retry_count += 1
                logging.error(f"获取数据集列表时出错 (尝试 {retry_count}/{max_retries}): {str(e)}")
                time.sleep(5)
                self._connect_mysql()
                if retry_count >= max_retries:
                    logging.error("重试次数已达上限，无法获取数据集列表")
                    break
            except Exception as e:
                logging.error(f"获取数据集列表时出错: {str(e)}")
                import traceback
                logging.error(traceback.format_exc())
                break
        return []

    def get_all_metrics(self):
        """获取所有评估指标实体"""
        max_retries = 3
        retry_count = 0
        while retry_count < max_retries:
            try:
                # 检查连接状态
                if not self._reconnect_if_needed():
                    logging.error("无法建立MySQL连接，等待5秒后重试")
                    retry_count += 1
                    time.sleep(5)
                    continue
                self.cursor.execute('SELECT * FROM Metrics')
                result = self.cursor.fetchall()
                if result:
                    # 转换结果为字典列表
                    metrics = []
                    for row in result:
                        metric = {}
                        for i, col in enumerate(self.cursor.description):
                            metric[col[0]] = row[i]
                        metrics.append(metric)
                    return metrics
                return []
            except (mysql.connector.Error, MySQLError) as e:
                retry_count += 1
                logging.error(f"获取评估指标列表时出错 (尝试 {retry_count}/{max_retries}): {str(e)}")
                time.sleep(5)
                self._connect_mysql()
                if retry_count >= max_retries:
                    logging.error("重试次数已达上限，无法获取评估指标列表")
                    break
            except Exception as e:
                logging.error(f"获取评估指标列表时出错: {str(e)}")
                import traceback
                logging.error(traceback.format_exc())
                break
        return []

    def get_dataset_by_id(self, dataset_id):
        """获取指定ID的数据集详细信息"""
        try:
            sql = 'SELECT * FROM Datasets WHERE dataset_id = %s'
            self.cursor.execute(sql, (dataset_id,))
            result = self.cursor.fetchone()
            if result:
                # 转换结果为字典
                dataset = {}
                for i, col in enumerate(self.cursor.description):
                    dataset[col[0]] = result[i]
                return dataset
            return None
        except Exception as e:
            logging.error(f"获取数据集详细信息时出错: {str(e)}")
            return None
            
    def get_metric_by_id(self, metric_id):
        """获取指定ID的评估指标详细信息"""
        try:
            sql = 'SELECT * FROM Metrics WHERE metric_id = %s'
            self.cursor.execute(sql, (metric_id,))
            result = self.cursor.fetchone()
            if result:
                # 转换结果为字典
                metric = {}
                for i, col in enumerate(self.cursor.description):
                    metric[col[0]] = result[i]
                return metric
            return None
        except Exception as e:
            logging.error(f"获取评估指标详细信息时出错: {str(e)}")
            return None
            
    def get_relation_types(self):
        """获取系统中所有的演化关系类型"""
        try:
            self.cursor.execute('SELECT DISTINCT relation_type FROM EvolutionRelations')
            result = self.cursor.fetchall()
            if result:
                return [row[0] for row in result]
            return ["Improve", "Optimize", "Extend", "Replace", "Use"]
        except Exception as e:
            logging.error(f"获取关系类型列表时出错: {str(e)}")
            return ["Improve", "Optimize", "Extend", "Replace", "Use"]

    def get_incoming_relations(self, entity_id):
        """获取指向指定实体的关系（传入关系）"""
        try:
            query = """
            SELECT relation_id, from_entity, to_entity, relation_type, 
                   structure, detail, evidence, confidence,
                   from_entity_type, to_entity_type 
            FROM EvolutionRelations 
            WHERE to_entity = %s
            """
            self.cursor.execute(query, (entity_id,))
            rows = self.cursor.fetchall()
            relations = []
            column_names = [description[0] for description in self.cursor.description]
            for row in rows:
                relation = {}
                for i, col in enumerate(column_names):
                    relation[col] = row[i]
                if col == 'confidence' and relation[col] is not None:
                    relation[col] = float(relation[col])
                relations.append(relation)
            return relations
        except Exception as e:
            logging.error(f"获取传入关系时出错: {str(e)}")
            return []

    def get_outgoing_relations(self, entity_id):
        """获取从指定实体出发的所有关系"""
        if self.db_type == 'mysql':
            return self._get_outgoing_relations_mysql(entity_id)
        else:
            return []
            
    def _get_outgoing_relations_mysql(self, entity_id):
        """获取从指定实体出发的关系（传出关系）"""
        try:
            query = """
            SELECT relation_id, from_entity, to_entity, relation_type, 
                   structure, detail, evidence, confidence,
                   from_entity_type, to_entity_type  
            FROM EvolutionRelations 
            WHERE from_entity = %s
            """
            self.cursor.execute(query, (entity_id,))
            rows = self.cursor.fetchall()
            relations = []
            column_names = [description[0] for description in self.cursor.description]
            for row in rows:
                relation = {}
                for i, col in enumerate(column_names):
                    relation[col] = row[i]
                if col == 'confidence' and relation[col] is not None:
                    relation[col] = float(relation[col])
                relations.append(relation)
            return relations
        except Exception as e:
            logging.error(f"获取传出关系时出错: {str(e)}")
            return []
            
    def clear_all_data(self):
        """清除所有数据库中的数据"""
        if self.db_type == 'mysql':
            return self._clear_all_data_mysql()
        else:
            logging.warning("不支持的数据库类型")
            
    def _clear_all_data_mysql(self):
        """清除MySQL数据库中的所有数据"""
        try:
            # 清空各个表中的数据
            tables = ['EvolutionRelations', 'Algorithms', 'Datasets', 'Metrics', 'ProcessingStatus']
            
            for table in tables:
                self.cursor.execute(f"TRUNCATE TABLE {table}")
                logging.info(f"已清空表 {table}")
                
            self.conn.commit()
            
            logging.info("成功清除所有数据库数据")
            return True
        except Exception as e:
            logging.error(f"清除数据库数据时出错: {str(e)}")
            self.conn.rollback()
            raise

    def _check_task_exists(self, task_id):
        """检查任务是否存在"""
        try:
            self.cursor.execute('SELECT * FROM ProcessingStatus WHERE task_id = %s', (task_id,))
            return self.cursor.fetchone() is not None
        except Exception as e:
            logging.error(f"检查任务存在时出错: {str(e)}")
            return False

    def get_entities_by_task(self, task_id):
        """
        根据任务ID获取相关实体
        
        Args:
            task_id (str): 任务ID
            
        Returns:
            list: 相关实体列表
        """
        try:
            # 检查是否需要重连
            self._reconnect_if_needed()
            
            # 在当前数据库结构中，实体表中没有直接与任务ID关联的字段
            # 查看ProcessingStatus表中是否有包含该任务ID的记录
            task_sql = "SELECT * FROM ProcessingStatus WHERE task_id = %s"
            self.cursor.execute(task_sql, (task_id,))
            task_row = self.cursor.fetchone()
            
            if not task_row:
                logging.warning(f"未找到任务ID: {task_id}")
                # 如果找不到任务ID，返回空列表
                return []
            
            # 如果找到任务，返回所有实体（由于目前没有实体与任务的直接关联）
            logging.info(f"找到任务ID: {task_id}，返回所有实体")
            return self.get_all_entities()
            
        except Exception as e:
            logging.error(f"获取任务 {task_id} 的实体时出错: {str(e)}")
            logging.error(traceback.format_exc())
            return []

    def get_relations_by_task(self, task_id):
        """
        根据任务ID获取相关关系
        
        Args:
            task_id (str): 任务ID
            
        Returns:
            list: 相关关系列表
        """
        try:
            # 检查是否需要重连
            self._reconnect_if_needed()
            
            # 在当前数据库结构中，关系表中没有直接与任务ID关联的字段
            # 查看ProcessingStatus表中是否有包含该任务ID的记录
            task_sql = "SELECT * FROM ProcessingStatus WHERE task_id = %s"
            self.cursor.execute(task_sql, (task_id,))
            task_row = self.cursor.fetchone()
            
            if not task_row:
                logging.warning(f"未找到任务ID: {task_id}")
                # 如果找不到任务ID，返回空列表
                return []
            
            # 如果找到任务，返回所有关系（由于目前没有关系与任务的直接关联）
            logging.info(f"找到任务ID: {task_id}，返回所有关系")
            return self.get_all_relations()
            
        except Exception as e:
            logging.error(f"获取任务 {task_id} 的关系时出错: {str(e)}")
            logging.error(traceback.format_exc())
            return []

    def get_comparison_history(self, limit=20):
        """
        获取比较分析的历史任务记录
        
        Args:
            limit (int): 最大返回记录数
            
        Returns:
            list: 任务记录列表
        """
        try:
            # 检查是否需要重连
            self._reconnect_if_needed()
            
            # 查询与比较分析相关的任务
            query = """
            SELECT task_id, task_name, status, current_stage, progress, message, 
                   start_time, update_time, end_time, completed
            FROM ProcessingStatus
            WHERE task_name LIKE '%比较%' OR task_name LIKE '%比对%' OR task_name LIKE '%Comparison%' 
                  OR task_name LIKE '%Compare%' OR status = 'completed'
            ORDER BY start_time DESC
            LIMIT %s
            """
            
            self.cursor.execute(query, (limit,))
            rows = self.cursor.fetchall()
            
            # 如果没有找到任何结果，查询最近的任务记录
            if not rows:
                fallback_query = """
                SELECT task_id, task_name, status, current_stage, progress, message, 
                       start_time, update_time, end_time, completed
                FROM ProcessingStatus
                ORDER BY start_time DESC
                LIMIT %s
                """
                self.cursor.execute(fallback_query, (limit,))
                rows = self.cursor.fetchall()
                logging.info(f"未找到特定比较任务，返回最近的{limit}条任务记录")
            
            tasks = []
            # 获取字段名
            column_names = [desc[0] for desc in self.cursor.description]
            
            for row in rows:
                task = {}
                for i, col_name in enumerate(column_names):
                    # 处理日期时间类型
                    if isinstance(row[i], (datetime.datetime, datetime.date)):
                        task[col_name] = row[i].strftime('%Y-%m-%d %H:%M:%S')
                    # 处理completed布尔值
                    elif col_name == 'completed':
                        task[col_name] = row[i] == 1 if row[i] is not None else False
                    # 处理progress浮点值
                    elif col_name == 'progress':
                        task[col_name] = float(row[i]) if row[i] is not None else 0
                    # 其他字段
                    else:
                        task[col_name] = row[i] if row[i] is not None else ""
                
                tasks.append(task)
            
            logging.info(f"获取到 {len(tasks)} 条比较分析历史记录")
            return tasks
            
        except Exception as e:
            logging.error(f"获取比较分析历史记录时出错: {str(e)}")
            logging.error(traceback.format_exc())
            return []

# 创建数据库管理器实例
db_manager = DatabaseManager()