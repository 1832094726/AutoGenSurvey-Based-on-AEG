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
        
        # 在启动时先清理可能存在的连接池
        if hasattr(mysql.connector, '_CONNECTION_POOLS'):
            try:
                mysql.connector._CONNECTION_POOLS = {}
                logging.info("应用启动：已清理所有旧连接池")
            except Exception as e:
                logging.warning(f"清理连接池时出错: {str(e)}")
        
        # 创建新连接，应用启动时只建立一次连接
        self._connect_mysql()
        
        # 检查并初始化表
        self._check_and_init_tables()
    def _connect_mysql(self):
        """
        连接到MySQL数据库，使用单一连接，不使用连接池
        """
        try:
            # 只有在连接不存在或已关闭时才创建新连接
            if self.conn is not None and self.conn.is_connected():
                logging.info("MySQL连接已存在且有效，无需重新连接")
                return True
            
            # 先完全清理现有连接和游标
            if hasattr(self, 'cursor') and self.cursor:
                try:
                    self.cursor.close()
                    logging.info("成功关闭旧游标")
                except Exception as e:
                    logging.warning(f"关闭游标时出错: {str(e)}")
                self.cursor = None
                
            if hasattr(self, 'conn') and self.conn:
                try:
                    self.conn.close()
                    logging.info("成功关闭旧连接")
                except Exception as e:
                    logging.warning(f"关闭旧连接时出错: {str(e)}")
                self.conn = None
            
            # 全局禁用连接池以彻底解决pool exhausted问题
            if hasattr(mysql.connector, '_CONNECTION_POOLS'):
                # 清空所有连接池
                try:
                    # 尝试直接重置连接池字典
                    mysql.connector._CONNECTION_POOLS = {}
                    logging.info("成功清理所有连接池")
                except Exception as e:
                    logging.warning(f"清理连接池时出错: {str(e)}")
            
            logging.info("开始创建新的数据库连接...")
            # 使用持久连接，禁用连接池
            self.conn = mysql.connector.connect(
                host=Config.MYSQL_HOST,
                port=Config.MYSQL_PORT,
                user=Config.MYSQL_USER,
                password=Config.MYSQL_PASSWORD,
                database=Config.MYSQL_DB,
                charset='utf8mb4',
                use_pure=True,  # 使用纯Python实现，增加稳定性
                connection_timeout=300,  # 增加连接超时时间到5分钟
                autocommit=True,  # 使用自动提交避免事务问题
                pool_size=1,  # 最小连接池设置
                pool_reset_session=True,  # 重置会话
                get_warnings=True,
                raise_on_warnings=False,  # 不对警告抛出异常
            )
            
            # 使用字典游标以保持一致性，设置buffered=True避免未读结果错误
            self.cursor = self.conn.cursor(dictionary=True, buffered=True)
            
            
            logging.info("MySQL数据库连接成功，设置为长期保持连接")
            return True
        except Exception as e:
            logging.error(f"连接MySQL数据库时发生错误: {str(e)}")
            import traceback
            logging.error(traceback.format_exc())
            # 确保连接和游标为空
            self.conn = None
            self.cursor = None
            raise

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
                        # 由于使用了字典游标，row已经是字典，直接复制
                        if isinstance(row, dict):
                            dataset = row.copy()
                        else:
                            # 如果不是字典，则按列名映射
                            for i, col in enumerate(self.cursor.description):
                                colname = col[0]
                                dataset[colname] = row[i]
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
                        # 由于使用了字典游标，row已经是字典，直接复制
                        if isinstance(row, dict):
                            metric = row.copy()
                        else:
                            # 如果不是字典，则按列名映射
                            for i, col in enumerate(self.cursor.description):
                                colname = col[0]
                                metric[colname] = row[i]
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
                    if isinstance(row, dict):
                        relation_dict = row.copy()
                    else:
                        relation_dict = dict(zip(column_names, row))
                    
                    # 确保confidence是浮点数
                    if 'confidence' in relation_dict and relation_dict['confidence'] is not None:
                        try:
                            # 添加更多调试信息
                            confidence_val = relation_dict['confidence']
                            logging.debug(f"转换confidence值: {confidence_val}, 类型: {type(confidence_val)}")
                            if isinstance(confidence_val, str) and confidence_val.lower() == 'confidence':
                                relation_dict['confidence'] = 0.5  # 默认值
                                logging.warning(f"发现'confidence'作为值，已替换为默认值0.5")
                            else:
                                relation_dict['confidence'] = float(confidence_val)
                        except (ValueError, TypeError) as e:
                            logging.warning(f"无法将confidence值'{relation_dict['confidence']}'转换为浮点数: {str(e)}，设为默认值0.5")
                            relation_dict['confidence'] = 0.5  # 设置默认值
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
        """
        连接到MySQL数据库，使用单一连接，不使用连接池
        """
        try:
            # 先完全清理现有连接和游标
            if hasattr(self, 'cursor') and self.cursor:
                try:
                    self.cursor.close()
                except Exception as e:
                    logging.warning(f"关闭游标时出错: {str(e)}")
                self.cursor = None
                
            if hasattr(self, 'conn') and self.conn:
                try:
                    self.conn.close()
                except Exception as e:
                    logging.warning(f"关闭旧连接时出错: {str(e)}")
                self.conn = None
            
            # 使用全新连接，确保不使用连接池
            self.conn = mysql.connector.connect(
                host=Config.MYSQL_HOST,
                port=Config.MYSQL_PORT,
                user=Config.MYSQL_USER,
                password=Config.MYSQL_PASSWORD,
                database=Config.MYSQL_DB,
                charset='utf8mb4',
                use_pure=True,  # 使用纯Python实现，增加稳定性
                connection_timeout=300,  # 增加连接超时时间到5分钟
                autocommit=True,  # 使用自动提交避免事务问题
                pool_size=1,  # 最小连接池设置
                pool_reset_session=True,  # 重置会话
                get_warnings=True,
                raise_on_warnings=False,  # 不对警告抛出异常
                # 增加这些参数以防止连接超时断开
                time_zone='+00:00',
                sql_mode='TRADITIONAL'
            )
            
            # 使用字典游标以保持一致性
            self.cursor = self.conn.cursor(dictionary=True, buffered=True)
            
            # 设置会话超时时间
            self.cursor.execute("SET SESSION wait_timeout=86400")  # 24小时
            self.cursor.execute("SET SESSION interactive_timeout=86400")  # 24小时
            self.cursor.execute("SET SESSION net_read_timeout=3600")  # 1小时
            self.cursor.execute("SET SESSION net_write_timeout=3600")  # 1小时
            
            logging.info("MySQL数据库连接成功")
            return True
        except Exception as e:
            logging.error(f"连接MySQL数据库时发生错误: {str(e)}")
            import traceback
            logging.error(traceback.format_exc())
            # 确保连接和游标为空
            self.conn = None
            self.cursor = None
            raise
    
    def _reconnect_if_needed(self):
        """
        检查MySQL连接状态并在需要时重连
        
        Returns:
            bool: 如果连接正常或重连成功则返回True，否则返回False
        """
        try:
            # 检查连接是否存在
            if self.conn is None or self.cursor is None:
                logging.warning("MySQL连接或游标不存在，创建新连接")
                self._connect_mysql()
                logging.info("MySQL连接创建成功")
                return True
                
            # 只检查连接是否已断开，不主动关闭和重建连接
            try:
                if not self.conn.is_connected():
                    logging.warning("MySQL连接已断开，正在重新连接")
                    self._connect_mysql()
                    logging.info("MySQL重新连接成功")
                    return True
            except Exception as conn_error:
                logging.warning(f"检查连接状态时出错: {str(conn_error)}，尝试重新连接")
                self._connect_mysql()
                return True
                
            # 不再执行ping操作，减少不必要的数据库交互
            
            # 验证连接可用性，但不执行查询，减少不必要的数据库操作
            if hasattr(self.cursor, 'connection') and self.cursor.connection is None:
                logging.warning("MySQL游标连接无效，重新创建")
                # 只关闭并重建游标，保留连接
                try:
                    self.cursor.close()
                except:
                    pass
                # 创建新游标
                self.cursor = self.conn.cursor(dictionary=True, buffered=True)
                logging.info("MySQL游标重新创建成功")
                
            return True
                
        except Exception as e:
            logging.error(f"检查MySQL连接状态时出错: {str(e)}")
            
            # 不再重试多次，直接尝试连接一次
            try:
                # 只有在确实需要时才创建新连接
                if self.conn is None or self.cursor is None or not self.conn.is_connected():
                    self._connect_mysql()
                    logging.info("MySQL重新连接成功")
                    return True
            except Exception as reconnect_error:
                logging.error(f"重新连接MySQL时出错: {str(reconnect_error)}")
                # 重置连接和游标为None，强制下次调用重新创建
                self.conn = None
                self.cursor = None
                return False
        
            return self.conn is not None and self.cursor is not None
        
    def __del__(self):
        """析构函数，关闭数据库连接"""
        # 安全地关闭游标和连接
        try:
            if hasattr(self, 'cursor') and self.cursor:
                self.cursor.close()
        except Exception as e:
            pass
            
        try:
            if hasattr(self, 'conn') and self.conn:
                self.conn.close()
        except Exception as e:
            pass
    
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
                    entity_type VARCHAR(50) DEFAULT 'Algorithm',
                    task_id VARCHAR(255),
                    source VARCHAR(50) DEFAULT '未知'
                )
                ''')
                logging.info("创建MySQL Algorithms表")
            else:
                # 检查task_id字段是否存在，如果不存在则添加
                self.cursor.execute("SHOW COLUMNS FROM Algorithms LIKE 'task_id'")
                if not self.cursor.fetchone():
                    self.cursor.execute("ALTER TABLE Algorithms ADD COLUMN task_id VARCHAR(255)")
                    logging.info("向Algorithms表添加task_id字段")
                
                # 检查source字段是否存在，如果不存在则添加
                self.cursor.execute("SHOW COLUMNS FROM Algorithms LIKE 'source'")
                if not self.cursor.fetchone():
                    self.cursor.execute("ALTER TABLE Algorithms ADD COLUMN source VARCHAR(50) DEFAULT '未知'")
                    logging.info("向Algorithms表添加source字段")
            
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
                    entity_type VARCHAR(50) DEFAULT 'Dataset',
                    task_id VARCHAR(255),
                    source VARCHAR(50) DEFAULT '未知'
                )
                ''')
                logging.info("创建MySQL Datasets表")
            else:
                # 检查task_id字段是否存在，如果不存在则添加
                self.cursor.execute("SHOW COLUMNS FROM Datasets LIKE 'task_id'")
                if not self.cursor.fetchone():
                    self.cursor.execute("ALTER TABLE Datasets ADD COLUMN task_id VARCHAR(255)")
                    logging.info("向Datasets表添加task_id字段")
                
                # 检查source字段是否存在，如果不存在则添加
                self.cursor.execute("SHOW COLUMNS FROM Datasets LIKE 'source'")
                if not self.cursor.fetchone():
                    self.cursor.execute("ALTER TABLE Datasets ADD COLUMN source VARCHAR(50) DEFAULT '未知'")
                    logging.info("向Datasets表添加source字段")
            
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
                    entity_type VARCHAR(50) DEFAULT 'Metric',
                    task_id VARCHAR(255),
                    source VARCHAR(50) DEFAULT '未知'
                )
                ''')
                logging.info("创建MySQL Metrics表")
            else:
                # 检查task_id字段是否存在，如果不存在则添加
                self.cursor.execute("SHOW COLUMNS FROM Metrics LIKE 'task_id'")
                if not self.cursor.fetchone():
                    self.cursor.execute("ALTER TABLE Metrics ADD COLUMN task_id VARCHAR(255)")
                    logging.info("向Metrics表添加task_id字段")
                
                # 检查source字段是否存在，如果不存在则添加
                self.cursor.execute("SHOW COLUMNS FROM Metrics LIKE 'source'")
                if not self.cursor.fetchone():
                    self.cursor.execute("ALTER TABLE Metrics ADD COLUMN source VARCHAR(50) DEFAULT '未知'")
                    logging.info("向Metrics表添加source字段")
                    
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
                    from_entity_relation_type VARCHAR(50),
                    to_entity_relation_type VARCHAR(50),
                    problem_addressed TEXT,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
                    task_id VARCHAR(255),
                    source VARCHAR(50) DEFAULT '未知'
                )
                ''')
                logging.info("创建MySQL EvolutionRelations表")
            else:
                # 检查task_id字段是否存在，如果不存在则添加
                self.cursor.execute("SHOW COLUMNS FROM EvolutionRelations LIKE 'task_id'")
                if not self.cursor.fetchone():
                    self.cursor.execute("ALTER TABLE EvolutionRelations ADD COLUMN task_id VARCHAR(255)")
                    logging.info("向EvolutionRelations表添加task_id字段")
                    
                # 检查source字段是否存在，如果不存在则添加
                self.cursor.execute("SHOW COLUMNS FROM EvolutionRelations LIKE 'source'")
                if not self.cursor.fetchone():
                    self.cursor.execute("ALTER TABLE EvolutionRelations ADD COLUMN source VARCHAR(50) DEFAULT '未知'")
                    logging.info("向EvolutionRelations表添加source字段")
                
                # 检查problem_addressed字段是否存在，如果不存在则添加
                self.cursor.execute("SHOW COLUMNS FROM EvolutionRelations LIKE 'problem_addressed'")
                if not self.cursor.fetchone():
                    self.cursor.execute("ALTER TABLE EvolutionRelations ADD COLUMN problem_addressed TEXT")
                    logging.info("向EvolutionRelations表添加problem_addressed字段")
                
                # 检查from_entity_relation_type字段是否存在，如果不存在则添加
                self.cursor.execute("SHOW COLUMNS FROM EvolutionRelations LIKE 'from_entity_relation_type'")
                if not self.cursor.fetchone():
                    self.cursor.execute("ALTER TABLE EvolutionRelations ADD COLUMN from_entity_relation_type VARCHAR(50)")
                    logging.info("向EvolutionRelations表添加from_entity_relation_type字段")
                
                # 检查to_entity_relation_type字段是否存在，如果不存在则添加
                self.cursor.execute("SHOW COLUMNS FROM EvolutionRelations LIKE 'to_entity_relation_type'")
                if not self.cursor.fetchone():
                    self.cursor.execute("ALTER TABLE EvolutionRelations ADD COLUMN to_entity_relation_type VARCHAR(50)")
                    logging.info("向EvolutionRelations表添加to_entity_relation_type字段")
            
            self.conn.commit()
        except (mysql.connector.Error, MySQLError) as e:
            logging.error(f"初始化数据库表时出错: {str(e)}")
            self.conn.rollback()
        except Exception as e:
            logging.error(f"初始化数据库表时出错: {str(e)}")
            import traceback
            logging.error(traceback.format_exc())
            self.conn.rollback()
    
    
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
        
    def store_algorithm_relation(self, relation, task_id=None):
        """
        将单条实体演化关系存储到数据库
        
        Args:
            relation (dict): 演化关系数据，支持两种格式：
                1. 数据库格式：{from_entity, to_entity, relation_type, ...}
                2. API格式：{from_entities, to_entities, relation_type, ...}
            task_id (str, optional): 关联的任务ID
            
        Returns:
            bool: 是否成功存储
        """
        try:
            # 验证关系数据
            if not isinstance(relation, dict):
                logging.error(f"关系数据格式错误: {relation}")
                return False
                
            # 获取来源信息
            source = relation.get('source', '未知')
            
            # 检查是否是数据库格式（from_entity/to_entity）
            if "from_entity" in relation and "to_entity" in relation:
                # 直接使用数据库格式
                return self._store_relation_mysql(relation, task_id)
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
                        "to_entity_type": to_entity_type,
                        "source": source  # 添加来源字段
                    }
                    # 调用数据库存储方法
                    if self._store_relation_mysql(relation_data, task_id):
                        success_count += 1
            logging.info(f"成功存储 {success_count}/{relation_count} 条演化关系，任务ID: {task_id}")
            return success_count > 0
        except Exception as e:
            logging.error(f"存储演化关系时出错: {str(e)}")
            import traceback
            logging.error(traceback.format_exc())
            return False
    
    def _store_relation_mysql(self, relation_data, task_id=None):
        """
        将实体演化关系存储到MySQL数据库
        
        Args:
            relation_data (dict): 演化关系数据
            task_id (str, optional): 关联的任务ID
            
        Returns:
            bool: 是否成功存储
        """
        try:
            # 准备数据
            from_entity = relation_data["from_entity"]
            to_entity = relation_data["to_entity"]
            relation_type = relation_data["relation_type"]
            structure = relation_data.get("structure", "")
            detail = relation_data.get("detail", "")
            evidence = relation_data.get("evidence", "")
            confidence = relation_data.get("confidence", 0.0)
            from_entity_type = relation_data.get("from_entity_type", "Algorithm")
            to_entity_type = relation_data.get("to_entity_type", "Algorithm")
            source = relation_data.get("source", "未知")  # 获取来源字段，默认为"未知"
            
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
                    from_entity_type = %s, to_entity_type = %s, task_id = %s, source = %s
                WHERE from_entity = %s AND to_entity = %s AND relation_type = %s
                """
                self.cursor.execute(
                    update_sql, 
                    (structure, detail, evidence, confidence, 
                     from_entity_type, to_entity_type, task_id, source,
                     from_entity, to_entity, relation_type)
                )
                logging.info(f"更新演化关系: {from_entity} -> {to_entity} ({relation_type}), 任务ID: {task_id}, 来源: {source}")
            else:
                # 关系不存在，执行插入
                insert_sql = """
                INSERT INTO EvolutionRelations (
                    from_entity, to_entity, relation_type, 
                    structure, detail, evidence, confidence, 
                    from_entity_type, to_entity_type, task_id, source
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """
                self.cursor.execute(
                    insert_sql, 
                    (from_entity, to_entity, relation_type, 
                     structure, detail, evidence, confidence,
                     from_entity_type, to_entity_type, task_id, source)
                )
                logging.info(f"创建新演化关系: {from_entity} -> {to_entity} ({relation_type}), 任务ID: {task_id}, 来源: {source}")
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
            
            # 安全获取COUNT(*)值，同时支持字典和元组两种返回格式
            if result:
                if isinstance(result, dict):
                    # 字典格式，通过键名访问
                    count = result.get('COUNT(*)', 0)
                else:
                    # 元组格式，通过索引访问
                    count = result[0] if len(result) > 0 else 0
                
                return count > 0
            return False
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
                        'entity_id': entity_dict['algorithm_id'],
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
                        'entity_type': 'Algorithm',
                        'source': entity_dict.get('source', '未知')  # 添加来源字段
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
        max_retries = 3
        retry_count = 0
        
        while retry_count < max_retries:
            try:
                # 检查是否需要重连
                reconnected = self._reconnect_if_needed()
                if reconnected and retry_count > 0:
                    logging.info(f"第 {retry_count+1} 次尝试：数据库连接已重新建立")
                
                # 首先检查是否是算法实体
                logging.info(f"正在获取实体 {entity_id}")
                self.cursor.execute('SELECT * FROM Algorithms WHERE algorithm_id = %s', (entity_id,))
                row = self.cursor.fetchone()
                if row:
                    # 解析JSON字符串为Python对象
                    try:
                        authors = json.loads(row['authors']) if row['authors'] else []
                    except:
                        authors = []
                    try:
                        dataset = json.loads(row['dataset']) if row['dataset'] else []
                    except:
                        dataset = []
                    try:
                        metrics = json.loads(row['metrics']) if row['metrics'] else []
                    except:
                        metrics = []
                    try:
                        arch_components = json.loads(row['architecture_components']) if row['architecture_components'] else []
                    except:
                        arch_components = []
                    try:
                        arch_connections = json.loads(row['architecture_connections']) if row['architecture_connections'] else []
                    except:
                        arch_connections = []
                    try:
                        arch_mechanisms = json.loads(row['architecture_mechanisms']) if row['architecture_mechanisms'] else []
                    except:
                        arch_mechanisms = []
                    try:
                        meth_training = json.loads(row['methodology_training_strategy']) if row['methodology_training_strategy'] else []
                    except:
                        meth_training = []
                    try:
                        meth_params = json.loads(row['methodology_parameter_tuning']) if row['methodology_parameter_tuning'] else []
                    except:
                        meth_params = []
                    try:
                        feature_processing = json.loads(row['feature_processing']) if row['feature_processing'] else []
                    except:
                        feature_processing = []
                    # 构建算法实体对象
                    entity = {
                        'algorithm_entity': {
                            'algorithm_id': row['algorithm_id'],
                            'entity_id': row['algorithm_id'],
                            'name': row['name'],
                            'title': row['title'],
                            'year': row['year'],
                            'authors': authors,
                            'task': row['task'],
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
                            'entity_type': 'Algorithm',
                            'source': row['source'] if 'source' in row else '未知'  # 获取source字段
                        }
                    }
                    # 获取演化关系
                    try:
                        relations = self._get_entity_relations(entity_id)
                        entity['algorithm_entity']['evolution_relations'] = relations
                    except Exception as rel_err:
                        logging.error(f"获取实体 {entity_id} 的关系时出错: {str(rel_err)}")
                        entity['algorithm_entity']['evolution_relations'] = []
                    
                    return entity
                
                # 如果不是算法实体，检查是否是数据集实体
                self.cursor.execute('SELECT * FROM Datasets WHERE dataset_id = %s', (entity_id,))
                row = self.cursor.fetchone()
                
                if row:
                    # 转换数据集为标准格式
                    dataset = {}
                    for i, col in enumerate(self.cursor.description):
                        dataset[col[0]] = row[col[0]]
                    
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
                    try:
                        relations = self._get_entity_relations(entity_id)
                        dataset['evolution_relations'] = relations
                    except Exception as rel_err:
                        logging.error(f"获取数据集 {entity_id} 的关系时出错: {str(rel_err)}")
                        dataset['evolution_relations'] = []
                    
                    return {'dataset_entity': dataset}
                    
                # 如果不是数据集实体，检查是否是评估指标实体
                self.cursor.execute('SELECT * FROM Metrics WHERE metric_id = %s', (entity_id,))
                row = self.cursor.fetchone()
                
                if row:
                    # 转换评估指标为标准格式
                    metric = {}
                    for i, col in enumerate(self.cursor.description):
                        metric[col[0]] = row[col[0]]
                    
                    # 确保entity_id字段存在
                    metric['entity_id'] = metric['metric_id']
                    metric['entity_type'] = 'Metric'
                    
                    # 获取演化关系
                    try:
                        relations = self._get_entity_relations(entity_id)
                        metric['evolution_relations'] = relations
                    except Exception as rel_err:
                        logging.error(f"获取评价指标 {entity_id} 的关系时出错: {str(rel_err)}")
                        metric['evolution_relations'] = []
                    
                    return {'metric_entity': metric}
                
                # 如果没有找到任何实体且这是最后一次重试
                if retry_count == max_retries - 1:
                    logging.warning(f"经过 {max_retries} 次尝试，未找到实体 {entity_id}")
                    return None
                
                # 增加重试计数并继续
                retry_count += 1
                time.sleep(1)
                
            except mysql.connector.errors.OperationalError as e:
                retry_count += 1
                logging.error(f"获取实体 {entity_id} 时数据库连接错误 (尝试 {retry_count}/{max_retries}): {str(e)}")
                
                if retry_count < max_retries:
                    # 等待后重试
                    time.sleep(2 * retry_count)
                    try:
                        # 尝试重新连接
                        self._connect_mysql()
                        logging.info(f"第 {retry_count} 次重试: 数据库重新连接成功")
                    except Exception as conn_err:
                        logging.error(f"尝试重新连接数据库时出错: {str(conn_err)}")
                else:
                    # 所有重试都失败
                    logging.error(f"经过 {max_retries} 次重试后，无法获取实体 {entity_id}")
                    return None
                
            except Exception as e:
                logging.error(f"通过ID获取实体时出错: {str(e)}")
                import traceback
                logging.error(traceback.format_exc())
                return None
            
        # 如果所有重试都失败
        return None

    def _get_entity_relations(self, entity_id):
        """
        获取指定实体的演化关系
        
        Args:
            entity_id (str): 实体ID
            
        Returns:
            List[Dict]: 演化关系列表
        """
        max_retries = 3
        retry_count = 0
        
        while retry_count < max_retries:
            try:
                # 检查是否需要重连
                reconnected = self._reconnect_if_needed()
                if reconnected and retry_count > 0:
                    logging.info(f"第 {retry_count+1} 次尝试：数据库连接已重新建立")
                
                logging.info(f"正在获取实体 {entity_id} 的演化关系")
                
                # 查询指向该实体的关系
                self.cursor.execute(
                    """
                    SELECT relation_id, from_entity, to_entity, relation_type, 
                           structure, detail, evidence, confidence,
                           from_entity_type, to_entity_type, source
                    FROM EvolutionRelations 
                    WHERE to_entity = %s
                    """,
                    (entity_id,)
                )
                rows = self.cursor.fetchall()
                
                relations = []
                
                for row in rows:
                    relation = {
                        'relation_id': row['relation_id'],
                        'from_entity': row['from_entity'],
                        'to_entity': row['to_entity'],
                        'relation_type': row['relation_type'],
                        'structure': row.get('structure', ''),
                        'detail': row.get('detail', ''),
                        'evidence': row.get('evidence', ''),
                        'confidence': float(row['confidence']) if row['confidence'] is not None else 0.0,
                        'from_entity_type': row.get('from_entity_type', 'Algorithm'),
                        'to_entity_type': row.get('to_entity_type', 'Algorithm'),
                        'source': row.get('source', '未知')
                    }
                    relations.append(relation)
                
                logging.info(f"实体 {entity_id} 有 {len(relations)} 个演化关系")
                return relations
                
            except mysql.connector.errors.OperationalError as e:
                retry_count += 1
                logging.error(f"获取实体 {entity_id} 的演化关系时数据库连接错误 (尝试 {retry_count}/{max_retries}): {str(e)}")
                
                if retry_count < max_retries:
                    # 等待后重试
                    time.sleep(2 * retry_count)
                    try:
                        # 尝试重新连接
                        self._connect_mysql()
                        logging.info(f"第 {retry_count} 次重试: 数据库重新连接成功")
                    except Exception as conn_err:
                        logging.error(f"尝试重新连接数据库时出错: {str(conn_err)}")
                else:
                    # 所有重试都失败
                    logging.error(f"经过 {max_retries} 次重试后，无法获取实体 {entity_id} 的演化关系")
                    return []
                
            except Exception as e:
                logging.error(f"获取实体 {entity_id} 的演化关系时出错: {str(e)}")
                return []
                
        # 如果所有重试都失败
        return []
    
    def init_db(self):
        """初始化数据库"""
        self._connect_mysql()
        
        logging.info("MySQL数据库初始化完成")
        return True

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

    def store_algorithm_entity(self, entity, task_id=None):
        """
        存储算法实体到MySQL数据库
        
        Args:
            entity (dict): 要存储的算法实体数据
            task_id (str, optional): 关联的任务ID
            
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
                # 继承外层实体的source属性
                if 'source' in entity and 'source' not in actual_entity:
                    actual_entity['source'] = entity['source']
            elif 'dataset_entity' in entity:
                actual_entity = entity['dataset_entity']
                entity_type = actual_entity.get('entity_type', 'Dataset')
                if 'dataset_id' in actual_entity:
                    actual_entity['entity_id'] = actual_entity['dataset_id']
                # 继承外层实体的source属性
                if 'source' in entity and 'source' not in actual_entity:
                    actual_entity['source'] = entity['source']
            elif 'metric_entity' in entity:
                actual_entity = entity['metric_entity']
                entity_type = actual_entity.get('entity_type', 'Metric')
                if 'metric_id' in actual_entity:
                    actual_entity['entity_id'] = actual_entity['metric_id']
                # 继承外层实体的source属性
                if 'source' in entity and 'source' not in actual_entity:
                    actual_entity['source'] = entity['source']
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
                
            # 确保source字段存在
            if 'source' not in actual_entity:
                # 尝试从task_id推断来源
                if task_id and '_review' in task_id:
                    actual_entity['source'] = '综述'
                elif task_id and '_citation' in task_id:
                    actual_entity['source'] = '引文'
                else:
                    actual_entity['source'] = '未知'
            
            logging.info(f"存储实体: {entity_id} (类型: {entity_type}, 任务ID: {task_id}, 来源: {actual_entity.get('source', '未知')})")
            
            if entity_type == 'Algorithm':
                # 存储算法实体
                self._store_algorithm_mysql(actual_entity, task_id)
            elif entity_type == 'Dataset':
                # 存储数据集实体
                self._store_dataset_mysql(actual_entity, task_id)
            elif entity_type == 'Metric':
                # 存储评价指标实体
                self._store_metric_mysql(actual_entity, task_id)
            else:
                logging.warning(f"未知实体类型: {entity_type}，跳过存储")
                return False
            
            return True
            
        except Exception as e:
            logging.error(f"存储实体时出错: {str(e)}")
            return False

    def _store_algorithm_mysql(self, entity, task_id=None):
        """
        将算法实体存储到MySQL数据库
        
        Args:
            entity (dict): 算法实体数据
            task_id (str, optional): 关联的任务ID
        
        Returns:
            bool: 是否成功存储
        """
        try:
            # 从实体中提取相关字段
            algorithm_id = entity.get('algorithm_id', '')
            name = entity.get('name', '')
            title = entity.get('title', '')
            year = str(entity.get('year', ''))
            
            # 获取来源信息，默认为"未知"
            source = entity.get('source', '未知')
            
            # 处理authors字段，确保是JSON字符串
            authors_value = entity.get('authors', [])
            if not isinstance(authors_value, list):
                authors_value = [authors_value] if authors_value else []
            authors = json.dumps(authors_value, ensure_ascii=False)
            
            # 处理task字段，确保是JSON字符串
            task_value = entity.get('task', [])
            if not isinstance(task_value, list):
                task_value = [task_value] if task_value else []
            task = json.dumps(task_value, ensure_ascii=False)
            
            # 处理dataset字段，确保是JSON字符串
            dataset_value = entity.get('dataset', [])
            if not isinstance(dataset_value, list):
                dataset_value = [dataset_value] if dataset_value else []
            dataset = json.dumps(dataset_value, ensure_ascii=False)
            
            # 处理metrics字段，确保是JSON字符串
            metrics_value = entity.get('metrics', [])
            if not isinstance(metrics_value, list):
                metrics_value = [metrics_value] if metrics_value else []
            metrics = json.dumps(metrics_value, ensure_ascii=False)
            
            # 处理architecture相关字段
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
                
            # 处理methodology相关字段
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
            result = self.cursor.fetchone()
            exists = False
            
            # 安全地获取COUNT(*)的值，同时支持字典和元组两种访问方式
            if result:
                if isinstance(result, dict):
                    # 字典形式访问，使用列名
                    count = result.get('COUNT(*)', 0)
                else:
                    # 元组形式访问，使用索引
                    count = result[0] if len(result) > 0 else 0
                
                exists = count > 0
            
            if exists:
                # 更新现有记录
                sql = '''
                    UPDATE Algorithms SET
                        name = %s, title = %s, year = %s, authors = %s,
                        task = %s, dataset = %s, metrics = %s,
                        architecture_components = %s, architecture_connections = %s,
                        architecture_mechanisms = %s, methodology_training_strategy = %s,
                        methodology_parameter_tuning = %s, feature_processing = %s,
                        entity_type = %s, task_id = %s, source = %s
                    WHERE algorithm_id = %s
                    '''
                self.cursor.execute(sql, (
                        name, title, year, authors,
                        task, dataset, metrics,
                        arch_components, arch_connections,
                        arch_mechanisms, training_strategy,
                        parameter_tuning, feature_processing,
                        'Algorithm', task_id, source, algorithm_id
                    ))
                logging.info(f"更新算法: {algorithm_id}, 任务ID: {task_id}, 来源: {source}")
            else:
                # 创建新记录
                sql = '''
                INSERT INTO Algorithms (
                    algorithm_id, name, title, year, authors,
                    task, dataset, metrics,
                    architecture_components, architecture_connections,
                    architecture_mechanisms, methodology_training_strategy,
                    methodology_parameter_tuning, feature_processing, entity_type, task_id, source
                ) VALUES (
                    %s, %s, %s, %s, %s,
                    %s, %s, %s,
                    %s, %s,
                    %s, %s,
                    %s, %s, %s, %s, %s
                )
                '''
                self.cursor.execute(sql, (
                    algorithm_id, name, title, year, authors,
                    task, dataset, metrics,
                    arch_components, arch_connections,
                    arch_mechanisms, training_strategy,
                    parameter_tuning, feature_processing, 'Algorithm', task_id, source
                ))
                logging.info(f"存储算法: {algorithm_id}, 任务ID: {task_id}, 来源: {source}")
            
            self.conn.commit()
            return True
        except Exception as e:
            self.conn.rollback()
            logging.error(f"存储算法时出错: {str(e)}")
            return False
    
    def _store_dataset_mysql(self, entity, task_id=None):
        """
        将数据集实体存储到MySQL数据库
        
        Args:
            entity (dict): 数据集实体数据
            task_id (str, optional): 关联的任务ID
            
        Returns:
            bool: 是否成功存储
        """
        try:
            # 从实体中提取相关字段
            dataset_id = entity.get('entity_id', '')
            name = entity.get('name', '')
            description = entity.get('description', '')
            domain = entity.get('domain', '')
            
            # 获取来源信息，默认为"未知"
            source = entity.get('source', '未知')
            
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
            result = self.cursor.fetchone()
            exists = False
            
            # 安全地获取COUNT(*)的值
            if result:
                if isinstance(result, dict):
                    # 字典形式访问，使用列名
                    count = result.get('COUNT(*)', 0)
                else:
                    # 元组形式访问，使用索引
                    count = result[0] if len(result) > 0 else 0
                
                exists = count > 0
            
            if exists:
                # 更新现有记录
                sql = '''
                    UPDATE Datasets SET
                        name = %s, description = %s, domain = %s,
                        size = %s, year = %s, creators = %s,
                        entity_type = %s, task_id = %s, source = %s
                    WHERE dataset_id = %s
                    '''
                self.cursor.execute(sql, (
                        name, description, domain,
                        size, year, creators,
                        'Dataset', task_id, source, dataset_id
                    ))
                logging.info(f"更新数据集: {dataset_id}, 任务ID: {task_id}, 来源: {source}")
            else:
                # 创建新记录
                sql = '''
                INSERT INTO Datasets (
                    dataset_id, name, description, domain,
                    size, year, creators, entity_type, task_id, source
                ) VALUES (
                    %s, %s, %s, %s,
                    %s, %s, %s, %s, %s, %s
                )
                '''
                self.cursor.execute(sql, (
                    dataset_id, name, description, domain,
                    size, year, creators, 'Dataset', task_id, source
                ))
                logging.info(f"存储数据集: {dataset_id}, 任务ID: {task_id}, 来源: {source}")
            
            self.conn.commit()
            return True
        except Exception as e:
            self.conn.rollback()
            logging.error(f"存储数据集时出错: {str(e)}")
            return False

    def _store_metric_mysql(self, entity, task_id=None):
        """
        将评价指标实体存储到MySQL数据库
        
        Args:
            entity (dict): 评价指标实体数据
            task_id (str, optional): 关联的任务ID
            
        Returns:
            bool: 是否成功存储
        """
        try:
            # 从实体中提取相关字段
            metric_id = entity.get('entity_id', '')
            name = entity.get('name', '')
            description = entity.get('description', '')
            category = entity.get('category', '')
            formula = entity.get('formula', '')
            
            # 获取来源信息，默认为"未知"
            source = entity.get('source', '未知')
            
            # 确保所有字段都是字符串类型
            name = str(name) if name else ''
            description = str(description) if description else ''
            category = str(category) if category else ''
            formula = str(formula) if formula else ''
            
            # 检查是否已存在
            self.cursor.execute('SELECT COUNT(*) FROM Metrics WHERE metric_id = %s', (metric_id,))
            result = self.cursor.fetchone()
            exists = False
            
            # 安全地获取COUNT(*)的值
            if result:
                if isinstance(result, dict):
                    # 字典形式访问，使用列名
                    count = result.get('COUNT(*)', 0)
                else:
                    # 元组形式访问，使用索引
                    count = result[0] if len(result) > 0 else 0
                
                exists = count > 0
            
            if exists:
                # 更新现有记录
                sql = '''
                UPDATE Metrics SET
                    name = %s, description = %s, category = %s,
                    formula = %s, entity_type = %s, task_id = %s, source = %s
                WHERE metric_id = %s
                '''
                self.cursor.execute(sql, (
                    name, description, category,
                    formula, 'Metric', task_id, source, metric_id
                ))
                logging.info(f"更新评价指标: {metric_id}, 任务ID: {task_id}, 来源: {source}")
            else:
                # 创建新记录
                sql = '''
                INSERT INTO Metrics (
                    metric_id, name, description, category,
                    formula, entity_type, task_id, source
                ) VALUES (
                    %s, %s, %s, %s,
                    %s, %s, %s, %s
                )
                '''
                self.cursor.execute(sql, (
                    metric_id, name, description, category,
                    formula, 'Metric', task_id, source
                ))
                logging.info(f"存储评价指标: {metric_id}, 任务ID: {task_id}, 来源: {source}")
            
            self.conn.commit()
            return True
        except Exception as e:
            self.conn.rollback()
            logging.error(f"存储评价指标时出错: {str(e)}")
            return False


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
            FROMEvolutionRelations
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
            FROMEvolutionRelations
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
        max_retries = 5  # 增加最大重试次数
        retry_count = 0
        connection_errors = []
        
        while retry_count < max_retries:
            try:
                # 检查是否需要重连
                reconnected = self._reconnect_if_needed()
                if reconnected:
                    logging.info(f"第 {retry_count+1} 次尝试：数据库连接已重新建立")
                
                # 先查询ProcessingStatus表确认任务是否存在
                try:
                    task_sql = "SELECT * FROM ProcessingStatus WHERE task_id = %s"
                    self.cursor.execute(task_sql, (task_id,))
                    task_row = self.cursor.fetchone()
                    
                    if not task_row and retry_count == 0:
                        logging.warning(f"未找到任务ID: {task_id}")
                        # 如果找不到任务ID，返回空列表
                        return []
                except Exception as e:
                    logging.error(f"查询任务状态时出错: {str(e)}")
                    # 任务查询出错，但继续尝试查询实体
                
                all_entities = []
                
                # 查询算法实体
                self.cursor.execute("SELECT * FROM Algorithms WHERE task_id = %s", (task_id,))
                algorithm_rows = self.cursor.fetchall()
                
                # 处理算法实体
                for row in algorithm_rows:
                    # 处理JSON字段
                    try:
                        authors = json.loads(row['authors']) if row['authors'] else []
                    except:
                        authors = []
                    try:
                        dataset = json.loads(row['dataset']) if row['dataset'] else []
                    except:
                        dataset = []
                    try:
                        metrics = json.loads(row['metrics']) if row['metrics'] else []
                    except:
                        metrics = []
                    try:
                        arch_components = json.loads(row['architecture_components']) if row['architecture_components'] else []
                    except:
                        arch_components = []
                    try:
                        arch_connections = json.loads(row['architecture_connections']) if row['architecture_connections'] else []
                    except:
                        arch_connections = []
                    try:
                        arch_mechanisms = json.loads(row['architecture_mechanisms']) if row['architecture_mechanisms'] else []
                    except:
                        arch_mechanisms = []
                    try:
                        meth_training = json.loads(row['methodology_training_strategy']) if row['methodology_training_strategy'] else []
                    except:
                        meth_training = []
                    try:
                        meth_params = json.loads(row['methodology_parameter_tuning']) if row['methodology_parameter_tuning'] else []
                    except:
                        meth_params = []
                    try:
                        feature_processing = json.loads(row['feature_processing']) if row['feature_processing'] else []
                    except:
                        feature_processing = []
                    
                    # 获取来源或使用默认值
                    source = row.get('source', '未知')
                    
                    # 构建规范化的实体对象
                    algorithm_entity = {
                        'algorithm_id': row['algorithm_id'],
                        'entity_id': row['algorithm_id'],
                        'name': row['name'],
                        'title': row.get('title', ''),
                        'year': row.get('year', ''),
                        'authors': authors,
                        'task': row.get('task', ''),
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
                        'entity_type': 'Algorithm',
                        'task_id': row.get('task_id', task_id),
                        'source': source
                    }
                    all_entities.append({'algorithm_entity': algorithm_entity})
                
                # 查询数据集实体
                self.cursor.execute("SELECT * FROM Datasets WHERE task_id = %s", (task_id,))
                dataset_rows = self.cursor.fetchall()
                
                # 处理数据集实体
                for row in dataset_rows:
                    # 处理JSON字段
                    try:
                        creators = json.loads(row['creators']) if row['creators'] else []
                    except:
                        creators = []
                    
                    # 构建规范化的数据集实体
                    dataset_entity = {
                        'dataset_id': row['dataset_id'],
                        'entity_id': row['dataset_id'],
                        'name': row['name'],
                        'description': row.get('description', ''),
                        'domain': row.get('domain', ''),
                        'size': row.get('size', 0),
                        'year': row.get('year', ''),
                        'creators': creators,
                        'entity_type': 'Dataset',
                        'task_id': row.get('task_id', task_id),
                        'source': row.get('source', '未知')
                    }
                    all_entities.append({'dataset_entity': dataset_entity})
                
                # 查询评价指标实体
                self.cursor.execute("SELECT * FROM Metrics WHERE task_id = %s", (task_id,))
                metric_rows = self.cursor.fetchall()
                
                # 处理评价指标实体
                for row in metric_rows:
                    # 构建规范化的指标实体
                    metric_entity = {
                        'metric_id': row['metric_id'],
                        'entity_id': row['metric_id'],
                        'name': row['name'],
                        'description': row.get('description', ''),
                        'category': row.get('category', ''),
                        'formula': row.get('formula', ''),
                        'entity_type': 'Metric',
                        'task_id': row.get('task_id', task_id),
                        'source': row.get('source', '未知')
                    }
                    all_entities.append({'metric_entity': metric_entity})
                
                # 如果没有找到任何实体，记录警告
                if not all_entities and retry_count == 0:
                    logging.warning(f"未找到与任务 {task_id} 关联的实体")
                
                if all_entities:
                    logging.info(f"找到 {len(all_entities)} 个与任务 {task_id} 关联的实体")
                    return all_entities
                
                # 如果是最后一次重试仍未找到实体
                if retry_count == max_retries - 1 and not all_entities:
                    logging.warning(f"经过 {max_retries} 次尝试，仍未找到与任务 {task_id} 关联的实体")
                    return []
                
                # 未找到实体但还有重试机会，增加重试计数
                retry_count += 1
                time.sleep(1)  # 短暂等待后重试
                
            except mysql.connector.errors.OperationalError as e:
                retry_count += 1
                error_message = str(e)
                connection_errors.append(error_message)
                logging.error(f"获取任务 {task_id} 的实体时数据库连接错误 (尝试 {retry_count}/{max_retries}): {error_message}")
                
                if retry_count < max_retries:
                    # 等待几秒后重试
                    time.sleep(2 * retry_count)  # 随着重试次数增加等待时间
                    try:
                        # 强制重新连接
                        self._connect_mysql()
                        logging.info(f"第 {retry_count} 次重试: 数据库重新连接成功")
                    except Exception as connect_error:
                        logging.error(f"尝试重新连接数据库时出错: {str(connect_error)}")
                else:
                    # 所有重试都失败，记录最后的错误
                    logging.error(f"经过 {max_retries} 次重试后，无法获取任务 {task_id} 的实体")
                    logging.error(f"遇到的连接错误: {', '.join(connection_errors)}")
                    return []
                    
            except Exception as e:
                logging.error(f"获取任务 {task_id} 的实体时出错: {str(e)}")
                logging.error(traceback.format_exc())
                
                # 对于非连接错误，立即返回
                return []
                
        # 如果所有重试都失败，返回空列表
        return []

    def get_relations_by_task(self, task_id):
        """
        获取与特定任务ID关联的关系
        
        Args:
            task_id (str): 任务ID
            
        Returns:
            list: 关系列表
        """
        max_retries = 5  # 增加最大重试次数
        retry_count = 0
        connection_errors = []
        
        while retry_count < max_retries:
            try:
                # 检查是否需要重连
                reconnected = self._reconnect_if_needed()
                if reconnected:
                    logging.info(f"第 {retry_count+1} 次尝试：数据库连接已重新建立")
                
                # 检查任务是否存在
                try:
                    check_sql = "SELECT task_id FROM ProcessingStatus WHERE task_id = %s"
                    self.cursor.execute(check_sql, (task_id,))
                    task_row = self.cursor.fetchone()
                    
                    if not task_row and retry_count == 0:
                        logging.warning(f"未找到任务ID: {task_id}")
                        # 如果找不到任务ID，返回空列表
                        return []
                except Exception as e:
                    logging.error(f"查询任务状态时出错: {str(e)}")
                    # 任务查询出错，但继续尝试查询关系
                
                # 查询与任务ID相关联的关系
                relations_sql = """
                SELECT relation_id, from_entity, to_entity, relation_type, structure, 
                       detail, evidence, confidence, from_entity_type, to_entity_type,
                       task_id, source
                FROM EvolutionRelations
                WHERE task_id = %s
                """
                self.cursor.execute(relations_sql, (task_id,))
                rows = self.cursor.fetchall()
                
                relations = []
                if not rows and retry_count == 0:
                    logging.warning(f"未找到与任务 {task_id} 关联的关系记录")
                    
                # 如果是最后一次重试仍未找到关系
                if retry_count == max_retries - 1 and not rows:
                    logging.warning(f"经过 {max_retries} 次尝试，仍未找到与任务 {task_id} 关联的关系")
                    return []
                
                # 处理查询结果
                for row in rows:
                    relation = {
                        'relation_id': row['relation_id'],
                        'from_entity': row['from_entity'],
                        'to_entity': row['to_entity'],
                        'relation_type': row['relation_type'],
                        'structure': row.get('structure', ''),
                        'detail': row.get('detail', ''),
                        'evidence': row.get('evidence', ''),
                        'confidence': float(row['confidence']) if row['confidence'] is not None else 0.0,
                        'from_entity_type': row.get('from_entity_type', 'Algorithm'),
                        'to_entity_type': row.get('to_entity_type', 'Algorithm'),
                        'task_id': row.get('task_id', task_id),
                        'source': row.get('source', '未知')
                    }
                    relations.append(relation)
                
                if relations:
                    logging.info(f"找到 {len(relations)} 个与任务 {task_id} 关联的关系记录")
                    return relations
                
                # 未找到关系但还有重试机会，增加重试计数
                retry_count += 1
                time.sleep(1)  # 短暂等待后重试
                
            except mysql.connector.errors.OperationalError as e:
                retry_count += 1
                error_message = str(e)
                connection_errors.append(error_message)
                logging.error(f"获取任务 {task_id} 的关系时数据库连接错误 (尝试 {retry_count}/{max_retries}): {error_message}")
                
                if retry_count < max_retries:
                    # 等待几秒后重试
                    time.sleep(2 * retry_count)  # 随着重试次数增加等待时间
                    try:
                        # 强制重新连接
                        self._connect_mysql()
                        logging.info(f"第 {retry_count} 次重试: 数据库重新连接成功")
                    except Exception as connect_error:
                        logging.error(f"尝试重新连接数据库时出错: {str(connect_error)}")
                else:
                    # 所有重试都失败，记录最后的错误
                    logging.error(f"经过 {max_retries} 次重试后，无法获取任务 {task_id} 的关系")
                    logging.error(f"遇到的连接错误: {', '.join(connection_errors)}")
                    return []
                
            except Exception as e:
                logging.error(f"获取任务 {task_id} 的关系时出错: {str(e)}")
                logging.error(traceback.format_exc())
                
                # 对于非连接错误，立即返回
                return []
                
        # 如果所有重试都失败，返回空列表
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
                # 修复：不再使用枚举索引访问row，直接通过字段名访问
                for col_name in column_names:
                    # 处理日期时间类型
                    if isinstance(row[col_name], (datetime.datetime, datetime.date)):
                        task[col_name] = row[col_name].strftime('%Y-%m-%d %H:%M:%S')
                    # 处理completed布尔值
                    elif col_name == 'completed':
                        task[col_name] = row[col_name] == 1 if row[col_name] is not None else False
                    # 处理progress浮点值
                    elif col_name == 'progress':
                        task[col_name] = float(row[col_name]) if row[col_name] is not None else 0
                    # 其他字段
                    else:
                        task[col_name] = row[col_name] if row[col_name] is not None else ""
                
                tasks.append(task)
            
            logging.info(f"获取到 {len(tasks)} 条比较分析历史记录")
            return tasks
            
        except Exception as e:
            logging.error(f"获取比较分析历史记录时出错: {str(e)}")
            logging.error(traceback.format_exc())
            return []

    def save_entities_and_relations(self, entities, relations, task_id=None):
        """
        保存实体和关系到数据库
        
        Args:
            entities (list): 要保存的实体列表
            relations (list): 要保存的关系列表
            task_id (str, optional): 关联的任务ID
        """
        if not entities and not relations:
            logging.warning("没有实体和关系需要保存")
            return
            
        # 保存实体
        entity_count = 0
        for entity in entities:
            if entity:
                self.store_algorithm_entity(entity, task_id)
                entity_count += 1
        
        # 保存关系
        relation_count = 0
        for relation in relations:
            if relation:
                self.store_algorithm_relation(relation, task_id)
                relation_count += 1
        
        logging.info(f"已保存 {entity_count} 个实体和 {relation_count} 个关系到数据库，任务ID: {task_id}")

    def get_relation_by_id(self, relation_id):
        """
        通过ID获取关系详情
        
        Args:
            relation_id (str/int): 关系ID
            
        Returns:
            dict: 关系详情，如果未找到则返回None
        """
        max_retries = 3
        retry_count = 0
        
        while retry_count < max_retries:
            try:
                # 检查是否需要重连
                reconnected = self._reconnect_if_needed()
                if reconnected and retry_count > 0:
                    logging.info(f"第 {retry_count+1} 次尝试：数据库连接已重新建立")
            
                # 查询关系
                sql = """
                    SELECT relation_id, from_entity, to_entity, relation_type, structure, 
                          detail, evidence, confidence, from_entity_type, to_entity_type,
                          created_at, updated_at, task_id, source
                    FROM EvolutionRelations 
                    WHERE relation_id = %s
                    """
                self.cursor.execute(sql, (relation_id,))
                row = self.cursor.fetchone()
                
                if not row:
                    logging.warning(f"未找到关系: {relation_id}")
                    return None
                
                # 构建关系字典
                relation = {
                    'relation_id': row['relation_id'],
                    'from_entity': row['from_entity'],
                    'to_entity': row['to_entity'],
                    'relation_type': row['relation_type'],
                    'structure': row.get('structure', ''),
                    'detail': row.get('detail', ''),
                    'evidence': row.get('evidence', ''),
                    'confidence': float(row['confidence']) if row['confidence'] is not None else 0.0,
                    'from_entity_type': row.get('from_entity_type', 'Algorithm'),
                    'to_entity_type': row.get('to_entity_type', 'Algorithm'),
                    'source': row.get('source', '未知')
                }
                
                    # 处理日期时间类型
                if 'created_at' in row and row['created_at']:
                    relation['created_at'] = row['created_at'].strftime('%Y-%m-%d %H:%M:%S')
                if 'updated_at' in row and row['updated_at']:
                    relation['updated_at'] = row['updated_at'].strftime('%Y-%m-%d %H:%M:%S')
                
                # 补充关系信息
                try:
                    # 获取来源实体信息
                    from_entity = self.get_entity_by_id(relation['from_entity'])
                    if from_entity:
                        if 'algorithm_entity' in from_entity:
                            relation['from_entity_name'] = from_entity['algorithm_entity'].get('name', '')
                        elif 'dataset_entity' in from_entity:
                            relation['from_entity_name'] = from_entity['dataset_entity'].get('name', '')
                        elif 'metric_entity' in from_entity:
                            relation['from_entity_name'] = from_entity['metric_entity'].get('name', '')
                    
                    # 获取目标实体信息
                    to_entity = self.get_entity_by_id(relation['to_entity'])
                    if to_entity:
                        if 'algorithm_entity' in to_entity:
                            relation['to_entity_name'] = to_entity['algorithm_entity'].get('name', '')
                        elif 'dataset_entity' in to_entity:
                            relation['to_entity_name'] = to_entity['dataset_entity'].get('name', '')
                        elif 'metric_entity' in to_entity:
                            relation['to_entity_name'] = to_entity['metric_entity'].get('name', '')
                except Exception as e:
                    logging.warning(f"获取关系 {relation_id} 的相关实体信息时出错: {str(e)}")
                
                logging.info(f"成功获取关系: {relation_id}, 类型: {relation.get('relation_type')}, 来源: {relation.get('source', '未知')}")
                return relation
                
            except mysql.connector.errors.OperationalError as e:
                retry_count += 1
                logging.error(f"获取关系 {relation_id} 时数据库连接错误 (尝试 {retry_count}/{max_retries}): {str(e)}")
                
                if retry_count < max_retries:
                    # 等待后重试
                    time.sleep(2 * retry_count)
                    try:
                        # 尝试重新连接
                        self._connect_mysql()
                        logging.info(f"第 {retry_count} 次重试: 数据库重新连接成功")
                    except Exception as conn_err:
                        logging.error(f"尝试重新连接数据库时出错: {str(conn_err)}")
                else:
                    # 所有重试都失败
                    logging.error(f"经过 {max_retries} 次重试后，无法获取关系 {relation_id}")
                    return None
                    
            except Exception as e:
                logging.error(f"获取关系 {relation_id} 时出错: {str(e)}")
                logging.error(traceback.format_exc())
                return None
                
        # 如果所有重试都失败
        return None

# 创建数据库管理器实例
db_manager = DatabaseManager()