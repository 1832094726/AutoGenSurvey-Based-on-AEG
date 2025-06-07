"""
数据库连接池操作工具类
"""

import logging
import mysql.connector
import time
from app.config import Config

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

# 数据库操作辅助类
class DBUtils:
    """数据库操作通用工具类"""
    
    def __init__(self, pool_size=3, max_connections=10):
        """初始化数据库工具类
        
        Args:
            pool_size (int): 连接池初始大小
            max_connections (int): 最大连接数
        """
        self.pool = MySQLConnectionPool(pool_size, max_connections)
    
    def execute_query(self, query, params=None, fetch_one=False, commit=False):
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
            import traceback
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
    
    def select_one(self, sql, args=None):
        """查询一条数据
        
        Args:
            sql (str): SQL查询语句
            args (tuple/dict): 查询参数
            
        Returns:
            dict: 查询结果，如果未找到则返回None
        """
        return self.execute_query(sql, args, fetch_one=True)
    
    def select_all(self, sql, args=None):
        """查询多条数据
        
        Args:
            sql (str): SQL查询语句
            args (tuple/dict): 查询参数
            
        Returns:
            list: 查询结果列表
        """
        return self.execute_query(sql, args, fetch_one=False)
    
    def insert_one(self, sql, args=None):
        """插入一条数据
        
        Args:
            sql (str): SQL插入语句
            args (tuple/dict): 插入参数
            
        Returns:
            int: 影响的行数
        """
        return self.execute_query(sql, args, commit=True)
    
    def insert_one_pk(self, sql, args=None):
        """插入一条数据并返回主键ID
        
        Args:
            sql (str): SQL插入语句
            args (tuple/dict): 插入参数
            
        Returns:
            int: 新插入记录的主键ID
        """
        conn = None
        cursor = None
        try:
            conn = self.pool.get_connection()
            cursor = conn.cursor(dictionary=True, buffered=True)
            
            if args:
                cursor.execute(sql, args)
            else:
                cursor.execute(sql)
                
            conn.commit()
            return cursor.lastrowid
                
        except Exception as e:
            if conn:
                try:
                    conn.rollback()
                except:
                    pass
            logging.error(f"插入数据并获取主键时出错: {str(e)}")
            logging.error(f"查询语句: {sql}")
            logging.error(f"参数: {args}")
            import traceback
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
    
    def delete_one(self, sql, args=None):
        """删除数据
        
        Args:
            sql (str): SQL删除语句
            args (tuple/dict): 删除参数
            
        Returns:
            int: 影响的行数
        """
        return self.execute_query(sql, args, commit=True)
    
    def update_one(self, sql, args=None):
        """更新数据
        
        Args:
            sql (str): SQL更新语句
            args (tuple/dict): 更新参数
            
        Returns:
            int: 影响的行数
        """
        return self.execute_query(sql, args, commit=True)
    
    def close(self):
        """关闭连接池"""
        if hasattr(self, 'pool'):
            self.pool.close_all()

# 创建单例实例
db_utils = DBUtils(pool_size=3, max_connections=10) 