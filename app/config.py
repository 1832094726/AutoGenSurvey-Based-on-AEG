import os
import json

class Config:
    """基础配置类"""
    # 环境配置
    ENV = os.getenv('ENV', 'development')  # 默认为开发环境
    DEBUG = ENV == 'development'  # 在开发环境中启用调试
    LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')  # 日志级别
    
    # 项目路径
    BASE_DIR = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
    
    # 数据库配置
    DATABASE_PATH = os.path.join('./data', 'db', 'algknowledge.db')  # SQLite数据库路径
    
    # 上传文件配置
    UPLOAD_FOLDER = os.path.join('./data', 'uploads')  # 文件上传目录
    TEMP_FOLDER = os.path.join('./data', 'temp')
    ALLOWED_EXTENSIONS = {'pdf', 'txt', 'csv', 'xlsx', 'json'}  # 允许上传的文件类型
    MAX_CONTENT_LENGTH = 100 * 1024 * 1024  # 最大文件上传大小（100MB）
    
    # 缓存配置
    CACHE_DIR = os.path.join('./data', 'cache')  # 缓存目录
    CACHE_MAX_SIZE = 500 * 1024 * 1024  # 缓存目录最大大小限制（500MB）
    CACHE_CHECK_INTERVAL = 86400  # 缓存检查间隔（秒），默认为1天
    CACHE_MAX_AGE = 30 * 86400  # 缓存文件最大保存时间（秒），默认为30天
    
    # API配置
    API_PREFIX = '/api/v1'  # API前缀
    DISABLE_AI = os.environ.get('DISABLE_AI', 'False').lower() == 'true'  # 是否禁用AI功能
    # 安全配置
    SECRET_KEY = os.getenv('SECRET_KEY', 'dev_secret_key')  # 密钥
    SECURITY_PASSWORD_SALT = os.getenv('SECURITY_PASSWORD_SALT', 'dev_password_salt')  # 密码盐值
    
    # 模型配置
    QWEN_API_KEY = os.getenv('QWEN_API_KEY', 'sk-0fe80fc99c3045dfaa4c2921910245c1')  # 千问API密钥
    QWEN_BASE_URL = os.getenv('QWEN_BASE_URL', 'https://dashscope.aliyuncs.com/compatible-mode/v1')
    QWEN_MODEL = os.getenv('QWEN_MODEL', 'qwen-long')  # 千问模型
    DEFAULT_MODEL = os.environ.get('DEFAULT_MODEL', 'qwen-long')  # 默认使用千问长文本模型
    
    # 新增模型API密钥配置
    ANTHROPIC_API_KEY = os.environ.get('ANTHROPIC_API_KEY', 'sk-ZHVYXikfYSPi4eq4Qi8W7XPjZx6r90bQcV2hgI454Bi5DwjN')  # Claude API密钥
    OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY', 'sk-ZHVYXikfYSPi4eq4Qi8W7XPjZx6r90bQcV2hgI454Bi5DwjN')  # OpenAI API密钥
    GEMINI_API_KEY = os.environ.get('GEMINI_API_KEY', 'sk-ZHVYXikfYSPi4eq4Qi8W7XPjZx6r90bQcV2hgI454Bi5DwjN')  # Gemini API密钥
    
    # 模型基础URL配置
    ANTHROPIC_BASE_URL = os.environ.get('ANTHROPIC_BASE_URL', 'https://api.shubiaobiao.com/v1')
    OPENAI_BASE_URL = os.environ.get('OPENAI_BASE_URL', 'https://api.shubiaobiao.com/v1')
    GEMINI_BASE_URL = os.environ.get('GEMINI_BASE_URL', 'https://api.shubiaobiao.com/v1')
    
    # JSON响应大小限制配置
    MAX_JSON_RESPONSE_SIZE = 8192000  # 最大JSON响应大小（字符数）
    
    # 处理并发配置
    MAX_WORKERS = 40  # 最大工作线程数

    # 重试配置
    MAX_RETRY_ATTEMPTS = 3  # 最大重试次数
    RETRY_DELAY = 5  # 重试延迟（秒）
    
    # 算法处理配置
    ALGORITHM_BATCH_SIZE = 10  # 每批处理的算法数量
    
    # 基础配置
    TESTING = False
    SECRET_KEY = os.environ.get('SECRET_KEY', 'dev-key')
    
    # 文件路径配置
    ROOT_DIR = os.path.dirname(BASE_DIR)
    UPLOAD_DIR = os.path.join('./data', 'pdfs')  # 上传PDF文件存储目录
    OUTPUT_FOLDER = os.path.join('./data')  # 输出文件夹
    GRAPH_DATA_DIR = os.path.join('./data', 'graph')  # 图数据存储目录
    LOG_DIR = os.path.join('./data', 'logs')  # 日志存储目录
    CITED_PAPERS_DIR = os.path.join('./data', 'cited_papers')  # 引用文献存储目录
    
    # DeepSeek大模型API配置
    DEEPSEEK_API_KEY = os.environ.get('DEEPSEEK_API_KEY', 'your-deepseek-api-key')
    DEEPSEEK_API_BASE = os.environ.get('DEEPSEEK_API_BASE', 'https://api.deepseek.com')
    DEEPSEEK_MODEL = os.environ.get('DEEPSEEK_MODEL', 'deepseek-chat')
    DEEPSEEK_EMBEDDING_MODEL = os.environ.get('DEEPSEEK_EMBEDDING_MODEL', 'deepseek-embedding')
    
    # 数据库配置 - 只使用MySQL
    DB_TYPE = 'mysql'  # 固定为MySQL
    
    # MySQL配置
    MYSQL_HOST = os.environ.get('MYSQL_HOST', 'localhost')
    MYSQL_PORT = int(os.environ.get('MYSQL_PORT', 3306))
    MYSQL_USER = os.environ.get('MYSQL_USER', 'root')
    MYSQL_PASSWORD = os.environ.get('MYSQL_PASSWORD', '123456')
    MYSQL_DB = os.environ.get('MYSQL_DB', 'algorithm_evolution')

    # 新添加的OpenAI API相关配置
    OPENAI_MODEL = os.environ.get('OPENAI_MODEL', 'gpt-3.5-turbo')

    @classmethod
    def init_app(cls, app):
        """初始化应用程序配置"""
        # 确保必要的目录存在
        os.makedirs(cls.UPLOAD_FOLDER, exist_ok=True)
        os.makedirs(cls.CACHE_DIR, exist_ok=True)
        os.makedirs(os.path.dirname(cls.DATABASE_PATH), exist_ok=True)
        
        # 配置日志
        import logging
        logging.basicConfig(
            level=getattr(logging, cls.LOG_LEVEL),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # 定期清理过期缓存
        if not cls.DEBUG:  # 在非开发环境中启用定期清理
            cls._schedule_cache_cleanup()
    
    @classmethod
    def _schedule_cache_cleanup(cls):
        """安排定期清理缓存任务"""
        import threading
        import time
        
        def cleanup_task():
            while True:
                try:
                    cls._clean_cache()
                    time.sleep(cls.CACHE_CHECK_INTERVAL)
                except Exception as e:
                    import logging
                    logging.error(f"缓存清理失败: {str(e)}")
                    # 发生错误时延迟一小时后重试
                    time.sleep(3600)
        
        # 启动清理线程
        cleanup_thread = threading.Thread(target=cleanup_task, daemon=True)
        cleanup_thread.start()
    
    @classmethod
    def _clean_cache(cls):
        """清理过期和超大缓存文件"""
        import logging
        import time
        import os
        
        try:
            now = time.time()
            total_size = 0
            files_info = []
            
            # 收集所有缓存文件信息
            for file_name in os.listdir(cls.CACHE_DIR):
                file_path = os.path.join(cls.CACHE_DIR, file_name)
                if os.path.isfile(file_path):
                    stats = os.stat(file_path)
                    total_size += stats.st_size
                    files_info.append({
                        'path': file_path,
                        'size': stats.st_size,
                        'mtime': stats.st_mtime,
                        'atime': stats.st_atime
                    })
            
            # 如果缓存总大小超过限制，则按访问时间排序删除最旧的文件
            if total_size > cls.CACHE_MAX_SIZE:
                logging.info(f"缓存总大小 ({total_size / 1024 / 1024:.2f} MB) 超过限制 ({cls.CACHE_MAX_SIZE / 1024 / 1024:.2f} MB)，开始清理...")
                # 按访问时间排序，最旧的在前面
                files_info.sort(key=lambda x: x['atime'])
                
                # 删除最旧的文件，直到缓存大小低于限制
                for file_info in files_info:
                    if total_size <= cls.CACHE_MAX_SIZE * 0.9:  # 保留10%的缓冲区
                        break
                    
                    try:
                        os.remove(file_info['path'])
                        total_size -= file_info['size']
                        logging.info(f"删除旧缓存文件: {file_info['path']} ({file_info['size'] / 1024 / 1024:.2f} MB)")
                    except Exception as e:
                        logging.error(f"删除缓存文件失败: {file_info['path']}, 错误: {str(e)}")
            
            # 删除超过最大保存时间的文件
            for file_info in files_info:
                if now - file_info['mtime'] > cls.CACHE_MAX_AGE:
                    try:
                        os.remove(file_info['path'])
                        logging.info(f"删除过期缓存文件: {file_info['path']}, 超过 {cls.CACHE_MAX_AGE / 86400:.1f} 天")
                    except Exception as e:
                        if os.path.exists(file_info['path']):  # 检查文件是否还存在，可能已被之前的清理步骤删除
                            logging.error(f"删除过期缓存文件失败: {file_info['path']}, 错误: {str(e)}")
            
            # 检查和修复不完整的缓存文件
            cls._check_incomplete_cache_files()
            
            logging.info("缓存清理完成")
        except Exception as e:
            logging.exception(f"缓存清理过程中发生错误: {str(e)}")
    
    @classmethod
    def _check_incomplete_cache_files(cls):
        """检查并修复不完整的缓存文件"""
        import logging
        import os
        import json
        import datetime
        
        try:
            incomplete_count = 0
            fixed_count = 0
            
            for file_name in os.listdir(cls.CACHE_DIR):
                if file_name.endswith('.json'):
                    file_path = os.path.join(cls.CACHE_DIR, file_name)
                    
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            try:
                                data = json.load(f)
                                
                                # 检查是否有error字段但没有completed字段
                                if 'error' in data and 'completed' not in data:
                                    data['completed'] = False
                                    data['timestamp'] = datetime.datetime.now().isoformat()
                                    
                                    with open(file_path, 'w', encoding='utf-8') as fw:
                                        json.dump(data, fw, ensure_ascii=False, indent=2)
                                    
                                    fixed_count += 1
                                    logging.info(f"修复了不完整的缓存文件: {file_path}")
                            except json.JSONDecodeError:
                                # 文件内容不是有效的JSON
                                logging.warning(f"缓存文件 {file_path} 不是有效的JSON，将被标记为不完整")
                                with open(file_path, 'w', encoding='utf-8') as fw:
                                    json.dump({
                                        'error': 'Invalid JSON content',
                                        'completed': False,
                                        'timestamp': datetime.datetime.now().isoformat()
                                    }, fw, ensure_ascii=False, indent=2)
                                
                                incomplete_count += 1
                    except Exception as e:
                        logging.error(f"检查缓存文件 {file_path} 时出错: {str(e)}")
            
            if incomplete_count > 0 or fixed_count > 0:
                logging.info(f"缓存文件检查: 发现 {incomplete_count} 个无效文件，修复了 {fixed_count} 个不完整文件")
                
        except Exception as e:
            logging.exception(f"检查不完整缓存文件时出错: {str(e)}")

    # 新添加的加载和保存配置文件的方法
    @staticmethod
    def load_config(config_file='config.json'):
        """从JSON文件加载配置"""
        config_path = os.path.join(Config.BASE_DIR, config_file)
        if os.path.exists(config_path):
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    config_data = json.load(f)
                    
                    # 更新数据库配置
                    if 'mysql' in config_data:
                        mysql = config_data['mysql']
                        Config.MYSQL_HOST = mysql.get('host', Config.MYSQL_HOST)
                        Config.MYSQL_PORT = int(mysql.get('port', Config.MYSQL_PORT))
                        Config.MYSQL_USER = mysql.get('user', Config.MYSQL_USER)
                        Config.MYSQL_PASSWORD = mysql.get('password', Config.MYSQL_PASSWORD)
                        Config.MYSQL_DB = mysql.get('database', Config.MYSQL_DB)
                    
                    # 更新API配置
                    if 'api' in config_data:
                        api = config_data['api']
                        Config.OPENAI_API_KEY = api.get('openai_api_key', Config.OPENAI_API_KEY)
                        Config.OPENAI_MODEL = api.get('openai_model', Config.OPENAI_MODEL)
                        Config.QWEN_API_KEY = api.get('qwen_api_key', Config.QWEN_API_KEY)
                        Config.QWEN_BASE_URL = api.get('qwen_base_url', Config.QWEN_BASE_URL)
                    
                    print("配置已从文件加载: " + config_path)
            except Exception as e:
                print("加载配置文件时出错: " + str(e))
                
    @staticmethod
    def save_config(config_file='config.json'):
        """将配置保存到JSON文件"""
        config_path = os.path.join(Config.BASE_DIR, config_file)
        try:
            config_data = {
                'mysql': {
                    'host': Config.MYSQL_HOST,
                    'port': Config.MYSQL_PORT,
                    'user': Config.MYSQL_USER,
                    'password': Config.MYSQL_PASSWORD,
                    'database': Config.MYSQL_DB
                },
                'api': {
                    'openai_api_key': Config.OPENAI_API_KEY,
                    'openai_model': Config.OPENAI_MODEL,
                    'qwen_api_key': Config.QWEN_API_KEY,
                    'qwen_base_url': Config.QWEN_BASE_URL
                }
            }
            
            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(config_data, f, indent=4, ensure_ascii=False)
                
            print("配置已保存到文件: " + config_path)
                
        except Exception as e:
            print("保存配置文件时出错: " + str(e))


class DevelopmentConfig(Config):
    """开发环境配置"""
    DEBUG = True


class TestingConfig(Config):
    """测试环境配置"""
    DEBUG = True
    TESTING = True


class ProductionConfig(Config):
    """生产环境配置"""
    DEBUG = False


# 配置名称映射字典
config_by_name = {
    'development': DevelopmentConfig,
    'testing': TestingConfig,
    'production': ProductionConfig,
    'default': DevelopmentConfig
}

# 确保必要的目录存在
for directory in [Config.UPLOAD_FOLDER, Config.CITED_PAPERS_DIR, Config.OUTPUT_FOLDER, Config.GRAPH_DATA_DIR, Config.LOG_DIR, Config.CACHE_DIR]:
    if not os.path.exists(directory):
        os.makedirs(directory)

# 加载配置
Config.load_config() 