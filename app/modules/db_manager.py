import os
import logging
import json
import datetime
import time  # 添加time模块用于睡眠
import mysql.connector  # 导入MySQL连接器包
import traceback
from mysql.connector import Error as MySQLError  # 导入MySQL错误类型便于捕获
from app.config import Config
from app.modules.db_pool import db_utils

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class MySQLConnectionPool:
    """MySQL数据库连接池实现"""
    
    def __init__(self, pool_size=5, max_connections=10):
        """初始化连接池
        
        Args:
            pool_size (int): 初始连接池大小
            max_connections (int): 最大连接数
        """
        self.pool_size = pool_size
        self.max_connections = max_connections
        self.connections = []
        self.in_use = set()
        self.init_pool()
        
    def init_pool(self):
        """初始化连接池，创建初始连接"""
        for _ in range(self.pool_size):
            try:
                conn = self._create_connection()
                if conn:
                    self.connections.append(conn)
            except Exception as e:
                logging.error(f"初始化连接池时创建连接失败: {str(e)}")
                
        logging.info(f"初始化连接池完成，当前连接数: {len(self.connections)}")
    
    def _create_connection(self):
        """创建新的数据库连接"""
        try:
            conn = mysql.connector.connect(
                host=Config.MYSQL_HOST,
                port=Config.MYSQL_PORT,
                user=Config.MYSQL_USER,
                password=Config.MYSQL_PASSWORD,
                database=Config.MYSQL_DB,
                charset='utf8mb4',
                use_pure=True,  # 使用纯Python实现，增加稳定性
                connection_timeout=300,  # 增加连接超时时间到5分钟
                autocommit=True,  # 使用自动提交避免事务问题
                get_warnings=True,
                raise_on_warnings=False,  # 不对警告抛出异常
                time_zone='+00:00',
                sql_mode='TRADITIONAL'
            )
            
            # 设置会话超时时间
            cursor = conn.cursor()
            cursor.execute("SET SESSION wait_timeout=86400")  # 24小时
            cursor.execute("SET SESSION interactive_timeout=86400")  # 24小时
            cursor.execute("SET SESSION net_read_timeout=3600")  # 1小时
            cursor.execute("SET SESSION net_write_timeout=3600")  # 1小时
            cursor.close()
            
            return conn
        except Exception as e:
            logging.error(f"创建数据库连接失败: {str(e)}")
            return None
    
    def get_connection(self):
        """从连接池获取一个连接
        
        Returns:
            connection: 数据库连接
        """
        # 先检查现有空闲连接
        for conn in self.connections:
            if conn not in self.in_use:
                try:
                    # 验证连接是否有效
                    if conn.is_connected():
                        self.in_use.add(conn)
                        return conn
                    else:
                        # 无效连接，从池中移除
                        self.connections.remove(conn)
                except:
                    # 连接出错，从池中移除
                    if conn in self.connections:
                        self.connections.remove(conn)
        
        # 如果没有可用连接且未达到最大连接数，创建新连接
        if len(self.connections) < self.max_connections:
            conn = self._create_connection()
            if conn:
                self.connections.append(conn)
                self.in_use.add(conn)
                return conn
        
        # 如果所有连接都在使用中且达到最大连接数，等待并重试
        for _ in range(3):  # 最多重试3次
            time.sleep(1)
            # 再次检查是否有连接被释放
            for conn in self.connections:
                if conn not in self.in_use:
                    try:
                        if conn.is_connected():
                            self.in_use.add(conn)
                            return conn
                    except:
                        # 连接出错，从池中移除
                        if conn in self.connections:
                            self.connections.remove(conn)
        
        # 如果仍然没有可用连接，作为最后手段，强制创建一个新连接
        logging.warning("连接池已满但仍需要连接，创建临时连接")
        conn = self._create_connection()
        if conn:
            # 这个连接不加入池中，使用后会被关闭
            self.in_use.add(conn)
            return conn
            
        raise Exception("无法获取数据库连接，连接池已满且无法创建新连接")
    
    def release_connection(self, conn):
        """释放连接回连接池
        
        Args:
            conn: 数据库连接
        """
        if conn in self.in_use:
            self.in_use.remove(conn)
            # 检查连接是否仍然有效
            try:
                if conn.is_connected():
                    # 有效连接保留在池中
                    pass
                else:
                    # 无效连接从池中移除
                    if conn in self.connections:
                        self.connections.remove(conn)
                    conn.close()
            except:
                # 连接出错，从池中移除
                if conn in self.connections:
                    self.connections.remove(conn)
                try:
                    conn.close()
                except:
                    pass
        elif conn in self.connections:
            # 如果连接在池中但不在使用中，什么都不做
            pass
        else:
            # 如果是临时连接（不在池中），直接关闭
            try:
                conn.close()
            except:
                pass
    
    def close_all(self):
        """关闭所有连接"""
        for conn in self.connections:
            try:
                conn.close()
            except:
                pass
        self.connections = []
        self.in_use = set()
        logging.info("已关闭所有数据库连接")

class DatabaseManager:
    """数据库管理器，处理算法实体和关系的存储与读取"""
    
    def __init__(self):
        """初始化数据库连接"""
        self.db_type = 'mysql'  # 固定为MySQL
        # 初始化连接池
        self.pool = MySQLConnectionPool(pool_size=3, max_connections=10)
        
        # 在启动时先清理可能存在的连接池
        if hasattr(mysql.connector, '_CONNECTION_POOLS'):
            try:
                mysql.connector._CONNECTION_POOLS = {}
                logging.info("应用启动：已清理所有旧连接池")
            except Exception as e:
                logging.warning(f"清理连接池时出错: {str(e)}")
        
        # 检查并初始化表
        self._check_and_init_tables()
        
    def _execute_query(self, query, params=None, fetch_one=False, commit=False):
        """执行SQL查询并处理结果
        
        Args:
            query (str): SQL查询语句
            params (tuple/dict): 查询参数
            fetch_one (bool): 是否只获取一条结果
            commit (bool): 是否提交事务
            
        Returns:
            查询结果或影响的行数
        """
        conn = None
        cursor = None
        try:
            conn = self.pool.get_connection()
            cursor = conn.cursor(dictionary=True, buffered=True)
            
            if params:
                cursor.execute(query, params)
            else:
                cursor.execute(query)
                
            if commit:
                conn.commit()
                return cursor.rowcount
                
            if fetch_one:
                return cursor.fetchone()
            else:
                return cursor.fetchall()
                
        except Exception as e:
            if commit and conn:
                try:
                    conn.rollback()
                except:
                    pass
            logging.error(f"执行查询出错: {str(e)}")
            logging.error(f"查询语句: {query}")
            logging.error(f"参数: {params}")
            logging.error(traceback.format_exc())
            raise
        finally:
            if cursor:
                try:
                    cursor.close()
                except:
                    pass
            if conn:
                self.pool.release_connection(conn)
                
    def _check_and_init_tables(self):
        """检查表是否存在，并在需要时初始化表"""
        self._check_and_init_tables_mysql()
    
    def _check_and_init_tables_mysql(self):
        """检查并初始化MySQL表"""
        conn = None
        cursor = None
        try:
            conn = self.pool.get_connection()
            cursor = conn.cursor(dictionary=True)
            
            # 检查算法表是否存在
            cursor.execute("SHOW TABLES LIKE 'Algorithms'")
            if not cursor.fetchone():
                # 创建算法表
                cursor.execute('''
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
                cursor.execute("SHOW COLUMNS FROM Algorithms LIKE 'task_id'")
                if not cursor.fetchone():
                    cursor.execute("ALTER TABLE Algorithms ADD COLUMN task_id VARCHAR(255)")
                    logging.info("向Algorithms表添加task_id字段")
                
                # 检查source字段是否存在，如果不存在则添加
                cursor.execute("SHOW COLUMNS FROM Algorithms LIKE 'source'")
                if not cursor.fetchone():
                    cursor.execute("ALTER TABLE Algorithms ADD COLUMN source VARCHAR(50) DEFAULT '未知'")
                    logging.info("向Algorithms表添加source字段")
            
            # 检查数据集表是否存在
            cursor.execute("SHOW TABLES LIKE 'Datasets'")
            if not cursor.fetchone():
                # 创建数据集表
                cursor.execute('''
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
                cursor.execute("SHOW COLUMNS FROM Datasets LIKE 'task_id'")
                if not cursor.fetchone():
                    cursor.execute("ALTER TABLE Datasets ADD COLUMN task_id VARCHAR(255)")
                    logging.info("向Datasets表添加task_id字段")
                
                # 检查source字段是否存在，如果不存在则添加
                cursor.execute("SHOW COLUMNS FROM Datasets LIKE 'source'")
                if not cursor.fetchone():
                    cursor.execute("ALTER TABLE Datasets ADD COLUMN source VARCHAR(50) DEFAULT '未知'")
                    logging.info("向Datasets表添加source字段")
            
            # 检查指标表是否存在
            cursor.execute("SHOW TABLES LIKE 'Metrics'")
            if not cursor.fetchone():
                # 创建指标表
                cursor.execute('''
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
                cursor.execute("SHOW COLUMNS FROM Metrics LIKE 'task_id'")
                if not cursor.fetchone():
                    cursor.execute("ALTER TABLE Metrics ADD COLUMN task_id VARCHAR(255)")
                    logging.info("向Metrics表添加task_id字段")
                
                # 检查source字段是否存在，如果不存在则添加
                cursor.execute("SHOW COLUMNS FROM Metrics LIKE 'source'")
                if not cursor.fetchone():
                    cursor.execute("ALTER TABLE Metrics ADD COLUMN source VARCHAR(50) DEFAULT '未知'")
                    logging.info("向Metrics表添加source字段")
                    
            # 检查关系表是否存在
            cursor.execute("SHOW TABLES LIKE 'EvolutionRelations'")
            if not cursor.fetchone():
                # 创建关系表，确保relation_type非空
                cursor.execute('''
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
                cursor.execute("SHOW COLUMNS FROM EvolutionRelations LIKE 'task_id'")
                if not cursor.fetchone():
                    cursor.execute("ALTER TABLE EvolutionRelations ADD COLUMN task_id VARCHAR(255)")
                    logging.info("向EvolutionRelations表添加task_id字段")
                    
                # 检查source字段是否存在，如果不存在则添加
                cursor.execute("SHOW COLUMNS FROM EvolutionRelations LIKE 'source'")
                if not cursor.fetchone():
                    cursor.execute("ALTER TABLE EvolutionRelations ADD COLUMN source VARCHAR(50) DEFAULT '未知'")
                    logging.info("向EvolutionRelations表添加source字段")
                
                # 检查problem_addressed字段是否存在，如果不存在则添加
                cursor.execute("SHOW COLUMNS FROM EvolutionRelations LIKE 'problem_addressed'")
                if not cursor.fetchone():
                    cursor.execute("ALTER TABLE EvolutionRelations ADD COLUMN problem_addressed TEXT")
                    logging.info("向EvolutionRelations表添加problem_addressed字段")
                
                # 检查from_entity_relation_type字段是否存在，如果不存在则添加
                cursor.execute("SHOW COLUMNS FROM EvolutionRelations LIKE 'from_entity_relation_type'")
                if not cursor.fetchone():
                    cursor.execute("ALTER TABLE EvolutionRelations ADD COLUMN from_entity_relation_type VARCHAR(50)")
                    logging.info("向EvolutionRelations表添加from_entity_relation_type字段")
                
                # 检查to_entity_relation_type字段是否存在，如果不存在则添加
                cursor.execute("SHOW COLUMNS FROM EvolutionRelations LIKE 'to_entity_relation_type'")
                if not cursor.fetchone():
                    cursor.execute("ALTER TABLE EvolutionRelations ADD COLUMN to_entity_relation_type VARCHAR(50)")
                    logging.info("向EvolutionRelations表添加to_entity_relation_type字段")
            
            # 检查处理状态表是否存在
            cursor.execute("SHOW TABLES LIKE 'ProcessingStatus'")
            if not cursor.fetchone():
                # 创建处理状态表
                cursor.execute('''
                CREATE TABLE IF NOT EXISTS ProcessingStatus (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    task_id VARCHAR(255) UNIQUE NOT NULL,
                    task_name VARCHAR(255),
                    status VARCHAR(50),
                    current_stage VARCHAR(255),
                    progress FLOAT DEFAULT 0,
                    current_file VARCHAR(255),
                    message TEXT,
                    start_time DATETIME,
                    update_time DATETIME,
                    end_time DATETIME,
                    completed TINYINT(1) DEFAULT 0
                )
                ''')
                logging.info("创建MySQL ProcessingStatus表")
            
            conn.commit()
        except (mysql.connector.Error, MySQLError) as e:
            logging.error(f"初始化数据库表时出错: {str(e)}")
            if conn:
                try:
                    conn.rollback()
                except:
                    pass
        except Exception as e:
            logging.error(f"初始化数据库表时出错: {str(e)}")
            logging.error(traceback.format_exc())
            if conn:
                try:
                    conn.rollback()
                except:
                    pass
        finally:
            if cursor:
                try:
                    cursor.close()
                except:
                    pass
            if conn:
                self.pool.release_connection(conn)

    def __del__(self):
        """析构函数，关闭数据库连接池"""
        try:
            if hasattr(self, 'pool'):
                self.pool.close_all()
        except Exception as e:
            pass

    def get_all_datasets(self):
        """获取所有数据集实体"""
        max_retries = 3
        retry_count = 0
        while retry_count < max_retries:
            try:
                query = 'SELECT * FROM Datasets'
                result = db_utils.select_all(query)
                
                if result:
                    return result
                return []
            except Exception as e:
                retry_count += 1
                logging.error(f"获取数据集列表时出错 (尝试 {retry_count}/{max_retries}): {str(e)}")
                time.sleep(5)
                if retry_count >= max_retries:
                    logging.error("重试次数已达上限，无法获取数据集列表")
                    break
        return []

    def get_all_metrics(self):
        """获取所有评估指标实体"""
        max_retries = 3
        retry_count = 0
        while retry_count < max_retries:
            try:
                query = 'SELECT * FROM Metrics'
                result = db_utils.select_all(query)
                
                if result:
                    return result
                return []
            except Exception as e:
                retry_count += 1
                logging.error(f"获取评估指标列表时出错 (尝试 {retry_count}/{max_retries}): {str(e)}")
                time.sleep(5)
                if retry_count >= max_retries:
                    logging.error("重试次数已达上限，无法获取评估指标列表")
                    break
        return []

    def _get_all_relations_mysql(self):
        """从MySQL获取所有演化关系"""
        max_retries = 3
        retry_count = 0
        while retry_count < max_retries:
            try:
                query = """
                SELECT relation_id, from_entity, to_entity, relation_type, structure, 
                       detail, evidence, confidence, from_entity_type, to_entity_type
                FROM EvolutionRelations
                """
                rows = db_utils.select_all(query)
                
                relations = []
                logging.warning("从MySQL获取到 %d 行关系数据", len(rows))
                
                for row in rows:
                    relation_dict = row
                    
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
            except Exception as e:
                retry_count += 1
                logging.error(f"从MySQL获取关系时出错 (尝试 {retry_count}/{max_retries}): {str(e)}")
                # 如果是连接问题，等待后重试
                time.sleep(5)
                if retry_count >= max_retries:
                    logging.error("重试次数已达上限，无法从数据库获取关系")
                    break
        return []

    def get_processing_status(self, task_id):
        """
        获取任务的处理状态
        
        Args:
            task_id (str): 任务ID
            
        Returns:
            dict: 包含任务状态的字典
        """
        try:
            # 查询任务状态
            sql = """
            SELECT id, task_id, task_name, status, progress, current_stage, message, 
                   start_time, update_time, end_time, completed 
            FROM ProcessingStatus 
            WHERE task_id = %s
            """
            result = db_utils.select_one(sql, (task_id,))
            
            if not result:
                logging.warning(f"未找到任务ID为 {task_id} 的处理状态")
                return None
            
            # 确保message字段是字符串
            if 'message' in result and result['message'] is not None:
                if isinstance(result['message'], (bytearray, bytes)):
                    try:
                        result['message'] = result['message'].decode('utf-8')
                    except UnicodeDecodeError:
                        result['message'] = str(result['message'])
            
            # 尝试将message解析为JSON（如果是JSON字符串）
            if 'message' in result and result['message'] and isinstance(result['message'], str):
                try:
                    message_json = json.loads(result['message'])
                    result['data'] = message_json
                except (json.JSONDecodeError, TypeError):
                    # 如果无法解析为JSON，保持原样
                    pass
                    
            # 处理日期时间字段
            for date_field in ['start_time', 'update_time', 'end_time']:
                if date_field in result and result[date_field] is not None:
                    if isinstance(result[date_field], (datetime.datetime, datetime.date)):
                        result[date_field] = result[date_field].isoformat()
                    
            return result
            
        except Exception as e:
            logging.error(f"获取任务 {task_id} 的处理状态时发生错误: {str(e)}")
            logging.error(traceback.format_exc())
            return None

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
                        params = (
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
                        )
                        db_utils.insert_one(sql, params)
                
                logging.info(f"已将 {len(entities)} 个实体存储到MySQL数据库")
                return
            except Exception as e:
                retry_count += 1
                logging.error(f"存储实体到MySQL时出错 (尝试 {retry_count}/{max_retries}): {str(e)}")
                # 如果是连接问题，等待后重试
                time.sleep(5)
                if retry_count >= max_retries:
                    logging.error("重试次数已达上限，无法存储实体到数据库")
                    break
        
        logging.error(f"无法将 {len(entities)} 个实体存储到MySQL数据库")

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
            
            # 首先检查是否存在相同的关系
            check_sql = """
            SELECT relation_id FROM EvolutionRelations 
            WHERE from_entity = %s AND to_entity = %s AND relation_type = %s
            """
            result = db_utils.select_one(check_sql, (from_entity, to_entity, relation_type))
            
            if result:
                # 关系已存在，执行更新
                update_sql = """
                UPDATE EvolutionRelations 
                SET structure = %s, detail = %s, evidence = %s, confidence = %s,
                    from_entity_type = %s, to_entity_type = %s, task_id = %s, source = %s
                WHERE from_entity = %s AND to_entity = %s AND relation_type = %s
                """
                params = (
                    structure, detail, evidence, confidence, 
                    from_entity_type, to_entity_type, task_id, source,
                    from_entity, to_entity, relation_type
                )
                db_utils.update_one(update_sql, params)
                logging.info(f"更新演化关系: {from_entity} -> {to_entity} ({relation_type}), 任务ID: {task_id}, 来源: {source}")
            else:
                # 关系不存在，执行插入
                insert_sql = """
                INSERT INTO EvolutionRelations (
                    from_entity, to_entity, relation_type, 
                    structure, detail, evidence, confidence, 
                    from_entity_type, to_entity_type, task_id, source
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """
                params = (
                    from_entity, to_entity, relation_type, 
                    structure, detail, evidence, confidence,
                    from_entity_type, to_entity_type, task_id, source
                )
                db_utils.insert_one(insert_sql, params)
                logging.info(f"创建新演化关系: {from_entity} -> {to_entity} ({relation_type}), 任务ID: {task_id}, 来源: {source}")
            
            return True
        except Exception as e:
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
            check_sql = f"SELECT COUNT(*) as count FROM {table_name} WHERE {id_column} = %s"
            result = db_utils.select_one(check_sql, (entity_id,))
            
            if result and 'count' in result:
                return result['count'] > 0
            return False
        except Exception as e:
            logging.error(f"检查实体 {entity_id} 是否存在时出错: {str(e)}")
            return False

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

    def _get_all_entities_mysql(self):
        """从MySQL数据库获取所有算法实体"""
        max_retries = 3
        retry_count = 0
        
        while retry_count < max_retries:
            try:
                # 查询所有算法实体
                query = "SELECT * FROM Algorithms"
                rows = db_utils.select_all(query)
                
                entities = []
                logging.warning("从MySQL获取到 %d 行数据", len(rows))
                
                for row in rows:
                    entity_dict = row
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
            except Exception as e:
                retry_count += 1
                logging.error(f"从MySQL获取实体时出错 (尝试 {retry_count}/{max_retries}): {str(e)}")
                # 如果是连接问题，等待后重试
                time.sleep(5)
                if retry_count >= max_retries:
                    logging.error("重试次数已达上限，无法从数据库获取实体")
                    break
                    
        return []

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
            # 检查实体是否存在
            check_sql = 'SELECT COUNT(*) as count FROM Algorithms WHERE algorithm_id = %s'
            result = db_utils.select_one(check_sql, (entity_id,))
            
            if not result or result['count'] == 0:
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
            db_utils.update_one(update_sql, params)
            
            return True
        except Exception as e:
            logging.error(f"更新实体 {entity_id} 时出错: {str(e)}")
            return False
    
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
            delete_rel_sql = 'DELETE FROM EvolutionRelations WHERE from_entity = %s OR to_entity = %s'
            db_utils.execute(delete_rel_sql, (entity_id, entity_id))
            
            # 然后删除实体本身
            delete_entity_sql = 'DELETE FROM Algorithms WHERE algorithm_id = %s'
            result = db_utils.execute(delete_entity_sql, (entity_id,))
            
            # 检查是否有行被删除
            return result > 0
        except Exception as e:
            logging.error(f"删除实体 {entity_id} 时出错: {str(e)}")
            return False
    
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
            
            insert_sql = '''
            INSERT INTO ProcessingStatus 
            (task_id, status, current_stage, progress, message, start_time, update_time)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            ON DUPLICATE KEY UPDATE
            status = VALUES(status),
            current_stage = VALUES(current_stage),
            progress = VALUES(progress),
            message = VALUES(message),
            update_time = VALUES(update_time)
            '''
            
            db_utils.insert_one(insert_sql, (
                task_id, 'waiting', '初始化', 0.0, f'创建任务: {task_name}', now, now
            ))
            
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
            
            # 执行更新
            result = db_utils.update_one(sql, tuple(params))
            
            # 检查更新是否成功
            if result > 0:
                return True
            else:
                logging.warning(f"未找到任务 {task_id} 或状态未发生变化")
                # 如果任务不存在，尝试创建
                if not self._check_task_exists(task_id):
                    logging.info(f"任务 {task_id} 不存在，正在创建新任务记录")
                    return self._create_processing_task(task_id, status, current_stage, progress, 
                                                       current_file, message, completed)
                return False
        except Exception as e:
            logging.error(f"更新任务 {task_id} 的处理状态时发生错误: {str(e)}")
            return False
    
    def _create_processing_task(self, task_id, status=None, current_stage=None, progress=None, 
                               current_file=None, message=None, completed=None):
        """
        创建新的任务处理记录
        
        参数与update_processing_status相同
        """
        try:
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
            db_utils.insert_one(sql, tuple(values))
            
            logging.info(f"成功创建任务 {task_id} 的处理记录")
            return True
                
        except Exception as e:
            logging.error(f"创建任务处理记录时出错: {str(e)}")
            return False
    
    def _check_task_exists(self, task_id):
        """检查任务是否存在"""
        try:
            check_sql = 'SELECT COUNT(*) as count FROM ProcessingStatus WHERE task_id = %s'
            result = db_utils.select_one(check_sql, (task_id,))
            return result and result['count'] > 0
        except Exception as e:
            logging.error(f"检查任务存在时出错: {str(e)}")
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
                # 首先检查是否是算法实体
                logging.info(f"正在获取实体 {entity_id}")
                algo_sql = 'SELECT * FROM Algorithms WHERE algorithm_id = %s'
                row = db_utils.select_one(algo_sql, (entity_id,))
                
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
                dataset_sql = 'SELECT * FROM Datasets WHERE dataset_id = %s'
                row = db_utils.select_one(dataset_sql, (entity_id,))
                
                if row:
                    # 确保entity_id字段存在
                    row['entity_id'] = row['dataset_id']
                    row['entity_type'] = 'Dataset'
                    
                    # 解析可能的JSON字段
                    if 'creators' in row and row['creators']:
                        try:
                            row['creators'] = json.loads(row['creators'])
                        except:
                            # 如果不是有效的JSON，保持原样
                            pass
                
                    # 获取演化关系
                    try:
                        relations = self._get_entity_relations(entity_id)
                        row['evolution_relations'] = relations
                    except Exception as rel_err:
                        logging.error(f"获取数据集 {entity_id} 的关系时出错: {str(rel_err)}")
                        row['evolution_relations'] = []
                    
                    return {'dataset_entity': row}
                    
                # 如果不是数据集实体，检查是否是评估指标实体
                metric_sql = 'SELECT * FROM Metrics WHERE metric_id = %s'
                row = db_utils.select_one(metric_sql, (entity_id,))
                
                if row:
                    # 确保entity_id字段存在
                    row['entity_id'] = row['metric_id']
                    row['entity_type'] = 'Metric'
                    
                    # 获取演化关系
                    try:
                        relations = self._get_entity_relations(entity_id)
                        row['evolution_relations'] = relations
                    except Exception as rel_err:
                        logging.error(f"获取评价指标 {entity_id} 的关系时出错: {str(rel_err)}")
                        row['evolution_relations'] = []
                    
                    return {'metric_entity': row}
                
                # 如果没有找到任何实体且这是最后一次重试
                if retry_count == max_retries - 1:
                    logging.warning(f"经过 {max_retries} 次尝试，未找到实体 {entity_id}")
                    return None
                
                # 增加重试计数并继续
                retry_count += 1
                time.sleep(1)
                
            except Exception as e:
                retry_count += 1
                logging.error(f"通过ID获取实体时出错: {str(e)}")
                logging.error(traceback.format_exc())
                
                if retry_count < max_retries:
                    # 等待后重试
                    time.sleep(2 * retry_count)
                else:
                    # 所有重试都失败
                    logging.error(f"经过 {max_retries} 次重试后，无法获取实体 {entity_id}")
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
                logging.info(f"正在获取实体 {entity_id} 的演化关系")
                
                # 查询指向该实体的关系
                sql = """
                SELECT relation_id, from_entity, to_entity, relation_type, 
                       structure, detail, evidence, confidence,
                       from_entity_type, to_entity_type, source
                FROM EvolutionRelations 
                WHERE to_entity = %s
                """
                rows = db_utils.select_all(sql, (entity_id,))
                
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
                
            except Exception as e:
                retry_count += 1
                logging.error(f"获取实体 {entity_id} 的演化关系时出错: {str(e)}")
                
                if retry_count < max_retries:
                    # 等待后重试
                    time.sleep(2 * retry_count)
                else:
                    # 所有重试都失败
                    logging.error(f"经过 {max_retries} 次重试后，无法获取实体 {entity_id} 的演化关系")
                    return []
                
        # 如果所有重试都失败
        return []

    def get_dataset_by_id(self, dataset_id):
        """获取指定ID的数据集详细信息"""
        try:
            sql = 'SELECT * FROM Datasets WHERE dataset_id = %s'
            result = db_utils.select_one(sql, (dataset_id,))
            if result:
                return result
            return None
        except Exception as e:
            logging.error(f"获取数据集详细信息时出错: {str(e)}")
            return None
            
    def get_metric_by_id(self, metric_id):
        """获取指定ID的评估指标详细信息"""
        try:
            sql = 'SELECT * FROM Metrics WHERE metric_id = %s'
            result = db_utils.select_one(sql, (metric_id,))
            if result:
                return result
            return None
        except Exception as e:
            logging.error(f"获取评估指标详细信息时出错: {str(e)}")
            return None
            
    def get_relation_types(self):
        """获取系统中所有的演化关系类型"""
        try:
            sql = 'SELECT DISTINCT relation_type FROM EvolutionRelations'
            result = db_utils.select_all(sql)
            if result:
                return [row['relation_type'] for row in result]
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
            rows = db_utils.select_all(query, (entity_id,))
            
            relations = []
            for row in rows:
                relation = row.copy()
                if 'confidence' in relation and relation['confidence'] is not None:
                    relation['confidence'] = float(relation['confidence'])
                relations.append(relation)
            return relations
        except Exception as e:
            logging.error(f"获取传入关系时出错: {str(e)}")
            return []

    def get_outgoing_relations(self, entity_id):
        """获取从指定实体出发的所有关系"""
        return self._get_outgoing_relations_mysql(entity_id)
            
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
            rows = db_utils.select_all(query, (entity_id,))
            
            relations = []
            for row in rows:
                relation = row.copy()
                if 'confidence' in relation and relation['confidence'] is not None:
                    relation['confidence'] = float(relation['confidence'])
                relations.append(relation)
            return relations
        except Exception as e:
            logging.error(f"获取传出关系时出错: {str(e)}")
            return []

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
        
        db_utils.insert_one(sql, (
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
        check_sql = 'SELECT COUNT(*) as count FROM EvolutionRelations WHERE relation_id = %s'
        result = db_utils.select_one(check_sql, (relation_id,))
        if not result or result['count'] == 0:
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
        db_utils.update_one(update_sql, params)
        
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
        sql = 'DELETE FROM EvolutionRelations WHERE relation_id = %s'
        db_utils.execute(sql, (relation_id,))
        return True
            
    def clear_all_data(self):
        """清除所有数据库中的数据"""
        return self._clear_all_data_mysql()
            
    def _clear_all_data_mysql(self):
        """清除MySQL数据库中的所有数据"""
        try:
            # 清空各个表中的数据
            tables = ['EvolutionRelations', 'Algorithms', 'Datasets', 'Metrics', 'ProcessingStatus']
            
            for table in tables:
                db_utils.execute(f"TRUNCATE TABLE {table}")
                logging.info(f"已清空表 {table}")
            
            logging.info("成功清除所有数据库数据")
            return True
        except Exception as e:
            logging.error(f"清除数据库数据时出错: {str(e)}")
            raise

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
        
        while retry_count < max_retries:
            try:
                # 先查询ProcessingStatus表确认任务是否存在
                try:
                    task_sql = "SELECT * FROM ProcessingStatus WHERE task_id = %s"
                    task_row = db_utils.select_one(task_sql, (task_id,))
                    
                    if not task_row and retry_count == 0:
                        logging.warning(f"未找到任务ID: {task_id}")
                        # 如果找不到任务ID，返回空列表
                        return []
                except Exception as e:
                    logging.error(f"查询任务状态时出错: {str(e)}")
                    # 任务查询出错，但继续尝试查询实体
                
                all_entities = []
                
                # 查询算法实体
                algo_sql = "SELECT * FROM Algorithms WHERE task_id = %s"
                algorithm_rows = db_utils.select_all(algo_sql, (task_id,))
                
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
                dataset_sql = "SELECT * FROM Datasets WHERE task_id = %s"
                dataset_rows = db_utils.select_all(dataset_sql, (task_id,))
                
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
                metric_sql = "SELECT * FROM Metrics WHERE task_id = %s"
                metric_rows = db_utils.select_all(metric_sql, (task_id,))
                
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
                
            except Exception as e:
                retry_count += 1
                logging.error(f"获取任务 {task_id} 的实体时出错: {str(e)}")
                logging.error(traceback.format_exc())
                
                if retry_count < max_retries:
                    # 等待几秒后重试
                    time.sleep(2 * retry_count)  # 随着重试次数增加等待时间
                else:
                    # 所有重试都失败
                    logging.error(f"经过 {max_retries} 次重试后，无法获取任务 {task_id} 的实体")
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
        
        while retry_count < max_retries:
            try:
                # 检查任务是否存在
                try:
                    check_sql = "SELECT task_id FROM ProcessingStatus WHERE task_id = %s"
                    task_row = db_utils.select_one(check_sql, (task_id,))
                    
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
                rows = db_utils.select_all(relations_sql, (task_id,))
                
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
                
            except Exception as e:
                retry_count += 1
                logging.error(f"获取任务 {task_id} 的关系时出错: {str(e)}")
                logging.error(traceback.format_exc())
                
                if retry_count < max_retries:
                    # 等待几秒后重试
                    time.sleep(2 * retry_count)  # 随着重试次数增加等待时间
                else:
                    # 所有重试都失败
                    logging.error(f"经过 {max_retries} 次重试后，无法获取任务 {task_id} 的关系")
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
            
            rows = db_utils.select_all(query, (limit,))
            
            # 如果没有找到任何结果，查询最近的任务记录
            if not rows:
                fallback_query = """
                SELECT task_id, task_name, status, current_stage, progress, message, 
                       start_time, update_time, end_time, completed
                FROM ProcessingStatus
                ORDER BY start_time DESC
                LIMIT %s
                """
                rows = db_utils.select_all(fallback_query, (limit,))
                logging.info(f"未找到特定比较任务，返回最近的{limit}条任务记录")
            
            tasks = []
            
            for row in rows:
                task = row.copy()
                # 处理日期时间类型
                for col_name in ['start_time', 'update_time', 'end_time']:
                    if col_name in task and task[col_name] and isinstance(task[col_name], (datetime.datetime, datetime.date)):
                        task[col_name] = task[col_name].strftime('%Y-%m-%d %H:%M:%S')
                # 处理completed布尔值
                if 'completed' in task:
                    task['completed'] = task['completed'] == 1 if task['completed'] is not None else False
                # 处理progress浮点值
                if 'progress' in task and task['progress'] is not None:
                    task['progress'] = float(task['progress'])
                
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
                # 查询关系
                sql = """
                    SELECT relation_id, from_entity, to_entity, relation_type, structure, 
                          detail, evidence, confidence, from_entity_type, to_entity_type,
                          created_at, updated_at, task_id, source
                    FROM EvolutionRelations 
                    WHERE relation_id = %s
                    """
                row = db_utils.select_one(sql, (relation_id,))
                
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
                
            except Exception as e:
                retry_count += 1
                logging.error(f"获取关系 {relation_id} 时出错: {str(e)}")
                
                if retry_count < max_retries:
                    # 等待后重试
                    time.sleep(2 * retry_count)
                else:
                    # 所有重试都失败
                    logging.error(f"经过 {max_retries} 次重试后，无法获取关系 {relation_id}")
                    return None
                
        # 如果所有重试都失败
        return None
        
    def init_db(self):
        """初始化数据库"""
        self._check_and_init_tables()
        logging.info("MySQL数据库初始化完成")
        return True
        
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
                return self._store_algorithm_mysql(actual_entity, task_id)
            elif entity_type == 'Dataset':
                # 存储数据集实体
                return self._store_dataset_mysql(actual_entity, task_id)
            elif entity_type == 'Metric':
                # 存储评价指标实体
                return self._store_metric_mysql(actual_entity, task_id)
            else:
                logging.warning(f"未知实体类型: {entity_type}，跳过存储")
                return False
                
        except Exception as e:
            logging.error(f"存储实体时出错: {str(e)}")
            logging.error(traceback.format_exc())
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
            # 处理嵌套结构
            if 'algorithm_entity' in entity:
                entity_data = entity['algorithm_entity']
                # 确保source字段正确传递
                if 'source' not in entity_data and 'source' in entity:
                    entity_data['source'] = entity['source']
            else:
                entity_data = entity
            
            # 从实体中提取相关字段
            algorithm_id = entity_data.get('algorithm_id', '')
            name = entity_data.get('name', '')
            title = entity_data.get('title', '')
            year = str(entity_data.get('year', ''))
            
            # 获取来源信息，默认为"未知"
            source = entity_data.get('source', '未知')
            
            # 处理authors字段，确保是JSON字符串
            authors_value = entity_data.get('authors', [])
            if not isinstance(authors_value, list):
                authors_value = [authors_value] if authors_value else []
            authors = json.dumps(authors_value, ensure_ascii=False)
            
            # 处理task字段，确保是JSON字符串
            task_value = entity_data.get('task', [])
            if not isinstance(task_value, list):
                task_value = [task_value] if task_value else []
            task = json.dumps(task_value, ensure_ascii=False)
            
            # 处理dataset字段，确保是JSON字符串
            dataset_value = entity_data.get('dataset', [])
            if not isinstance(dataset_value, list):
                dataset_value = [dataset_value] if dataset_value else []
            dataset = json.dumps(dataset_value, ensure_ascii=False)
            
            # 处理metrics字段，确保是JSON字符串
            metrics_value = entity_data.get('metrics', [])
            if not isinstance(metrics_value, list):
                metrics_value = [metrics_value] if metrics_value else []
            metrics = json.dumps(metrics_value, ensure_ascii=False)
            
            # 处理architecture相关字段
            architecture = entity_data.get('architecture', {})
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
            methodology = entity_data.get('methodology', {})
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
            feature_proc = entity_data.get('feature_processing', [])
            if not isinstance(feature_proc, list):
                feature_proc = [feature_proc] if feature_proc else []
            feature_processing = json.dumps(feature_proc, ensure_ascii=False)
            
            # 检查是否已存在
            check_sql = 'SELECT COUNT(*) as count FROM Algorithms WHERE algorithm_id = %s'
            result = db_utils.select_one(check_sql, (algorithm_id,))
            exists = result and result['count'] > 0
            
            if exists:
                # 更新现有记录
                update_sql = '''
                    UPDATE Algorithms SET
                        name = %s, title = %s, year = %s, authors = %s,
                        task = %s, dataset = %s, metrics = %s,
                        architecture_components = %s, architecture_connections = %s,
                        architecture_mechanisms = %s, methodology_training_strategy = %s,
                        methodology_parameter_tuning = %s, feature_processing = %s,
                        entity_type = %s, task_id = %s, source = %s
                    WHERE algorithm_id = %s
                    '''
                db_utils.update_one(update_sql, (
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
                insert_sql = '''
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
                db_utils.insert_one(insert_sql, (
                    algorithm_id, name, title, year, authors,
                    task, dataset, metrics,
                    arch_components, arch_connections,
                    arch_mechanisms, training_strategy,
                    parameter_tuning, feature_processing, 'Algorithm', task_id, source
                ))
                logging.info(f"存储算法: {algorithm_id}, 任务ID: {task_id}, 来源: {source}")
            
            return True
        except Exception as e:
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
            # 处理嵌套结构
            if 'dataset_entity' in entity:
                entity_data = entity['dataset_entity']
                # 确保source字段正确传递
                if 'source' not in entity_data and 'source' in entity:
                    entity_data['source'] = entity['source']
            else:
                entity_data = entity
            
            # 从实体中提取相关字段
            dataset_id = entity_data.get('entity_id', '')
            name = entity_data.get('name', '')
            description = entity_data.get('description', '')
            domain = entity_data.get('domain', '')
            
            # 获取来源信息，默认为"未知"
            source = entity_data.get('source', '未知')
            
            # 处理size字段，确保是整数
            size_value = entity_data.get('size', 0)
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
                
            year = str(entity_data.get('year', ''))
            
            # 处理creators字段，确保是JSON字符串
            creators_value = entity_data.get('creators', [])
            if not isinstance(creators_value, list):
                creators_value = [creators_value] if creators_value else []
            creators = json.dumps(creators_value, ensure_ascii=False)
            
            # 检查是否已存在
            check_sql = 'SELECT COUNT(*) as count FROM Datasets WHERE dataset_id = %s'
            result = db_utils.select_one(check_sql, (dataset_id,))
            exists = result and result['count'] > 0
            
            if exists:
                # 更新现有记录
                update_sql = '''
                    UPDATE Datasets SET
                        name = %s, description = %s, domain = %s,
                        size = %s, year = %s, creators = %s,
                        entity_type = %s, task_id = %s, source = %s
                    WHERE dataset_id = %s
                    '''
                db_utils.update_one(update_sql, (
                        name, description, domain,
                        size, year, creators,
                        'Dataset', task_id, source, dataset_id
                    ))
                logging.info(f"更新数据集: {dataset_id}, 任务ID: {task_id}, 来源: {source}")
            else:
                # 创建新记录
                insert_sql = '''
                INSERT INTO Datasets (
                    dataset_id, name, description, domain,
                    size, year, creators, entity_type, task_id, source
                ) VALUES (
                    %s, %s, %s, %s,
                    %s, %s, %s, %s, %s, %s
                )
                '''
                db_utils.insert_one(insert_sql, (
                    dataset_id, name, description, domain,
                    size, year, creators, 'Dataset', task_id, source
                ))
                logging.info(f"存储数据集: {dataset_id}, 任务ID: {task_id}, 来源: {source}")
            
            return True
        except Exception as e:
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
            # 处理嵌套结构
            if 'metric_entity' in entity:
                entity_data = entity['metric_entity']
                # 确保source字段正确传递
                if 'source' not in entity_data and 'source' in entity:
                    entity_data['source'] = entity['source']
            else:
                entity_data = entity
                
            # 从实体中提取相关字段
            metric_id = entity_data.get('metric_id', '') or entity_data.get('entity_id', '')
            name = entity_data.get('name', '')
            description = entity_data.get('description', '')
            category = entity_data.get('category', '')
            formula = entity_data.get('formula', '')
            
            # 获取来源信息，默认为"未知"
            source = entity_data.get('source', '未知')
            
            # 确保所有字段都是字符串类型
            name = str(name) if name else ''
            description = str(description) if description else ''
            category = str(category) if category else ''
            formula = str(formula) if formula else ''
            
            # 处理range字段
            range_value = entity_data.get('range', '')
            range_value = str(range_value) if range_value else ''
            
            # 处理tasks字段，确保是JSON字符串
            tasks_value = entity_data.get('tasks', [])
            if not isinstance(tasks_value, list):
                tasks_value = [tasks_value] if tasks_value else []
            tasks = json.dumps(tasks_value, ensure_ascii=False)
            
            # 检查是否已存在
            check_sql = 'SELECT COUNT(*) as count FROM Metrics WHERE metric_id = %s'
            result = db_utils.select_one(check_sql, (metric_id,))
            exists = result and result['count'] > 0
            
            if exists:
                # 更新现有记录
                update_sql = '''
                UPDATE Metrics SET
                    name = %s, description = %s, category = %s,
                    formula = %s, entity_type = %s, task_id = %s, source = %s,
                    range = %s, tasks = %s
                WHERE metric_id = %s
                '''
                db_utils.update_one(update_sql, (
                    name, description, category,
                    formula, 'Metric', task_id, source,
                    range_value, tasks, metric_id
                ))
                logging.info(f"更新评价指标: {metric_id}, 任务ID: {task_id}, 来源: {source}")
            else:
                # 创建新记录
                insert_sql = '''
                INSERT INTO Metrics (
                    metric_id, name, description, category,
                    formula, entity_type, task_id, source,
                    range, tasks
                ) VALUES (
                    %s, %s, %s, %s,
                    %s, %s, %s, %s,
                    %s, %s
                )
                '''
                db_utils.insert_one(insert_sql, (
                    metric_id, name, description, category,
                    formula, 'Metric', task_id, source,
                    range_value, tasks
                ))
                logging.info(f"存储评价指标: {metric_id}, 任务ID: {task_id}, 来源: {source}")
            
            return True
        except Exception as e:
            logging.error(f"存储评价指标时出错: {str(e)}")
            return False

# 创建数据库管理器实例
db_manager = DatabaseManager()