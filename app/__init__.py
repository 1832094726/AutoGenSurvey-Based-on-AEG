import os
import logging
from flask import Flask
from app.config import config_by_name
from app.api.combined_api import combined_api
# 以下导入保留用于备份，后续可以移除
# from app.api.entities import entity_api
# from app.api.relations import relation_api
# from app.api.graph import graph_api
# from app.api.graph_routes import graph_bp
# from app.api.entity import entity_bp
# from app.api.document import document_bp
# from app.api.evolution import evolution_bp
# from app.api.comparison import comparison_bp

def create_app(config_name='default'):
    """
    创建Flask应用实例
    
    Args:
        config_name: 配置名称，默认为'default'
        
    Returns:
        app: Flask应用实例
    """
    app = Flask(__name__)
    
    # 加载配置
    app.config.from_object(config_by_name[config_name])
    
    # 确保必要的文件夹存在
    ensure_dirs(app)
    
    # 注册蓝图
    register_blueprints(app)
    
    # 返回创建的应用
    return app

def ensure_dirs(app):
    """确保应用所需的目录已创建"""
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)
    os.makedirs(app.config['GRAPH_DATA_DIR'], exist_ok=True)
    os.makedirs(app.config['LOG_DIR'], exist_ok=True)

def register_blueprints(app):
    """注册所有蓝图"""
    # 注册合并后的API蓝图
    app.register_blueprint(combined_api, url_prefix='/api')
    
    # 以下注册保留用于备份，后续可以移除
    """
    # 注册API蓝图
    app.register_blueprint(entity_api, url_prefix='/api/entities')
    app.register_blueprint(relation_api, url_prefix='/api/relations')
    app.register_blueprint(graph_api, url_prefix='/api/graph')
    app.register_blueprint(graph_bp, url_prefix='/api/graph/v2')
    # 已创建的蓝图
    app.register_blueprint(entity_bp, url_prefix='/api/entity')
    app.register_blueprint(document_bp, url_prefix='/api/document')
    app.register_blueprint(evolution_bp, url_prefix='/api/evolution')
    app.register_blueprint(comparison_bp, url_prefix='/api/comparison')
    """
    
    # 导入并注册主蓝图 - 直接使用main.py中的蓝图
    from app.main import main as main_blueprint
    app.register_blueprint(main_blueprint)
    
    # 导入并注册routes蓝图
    try:
        from app.routes.main import routes_main
        app.register_blueprint(routes_main)
    except ImportError:
        logging.warning("无法导入routes_main蓝图") 