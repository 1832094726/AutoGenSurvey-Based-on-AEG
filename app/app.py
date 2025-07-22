def create_app(config_class=Config):
    """
    创建并配置Flask应用
    
    Args:
        config_class: 配置类，默认为Config
    
    Returns:
        Flask应用实例
    """
    app = Flask(__name__)
    app.config.from_object(config_class)
    
    # 初始化配置（包括缓存清理）
    config_class.init_app(app)
    
    # 配置CORS
    CORS(app)
    
    # 初始化扩展
    db.init_app(app)

    # 注册论文分析API（添加错误处理）
    try:
        from app.api.paper_analysis_api import paper_analysis_api
        app.register_blueprint(paper_analysis_api, url_prefix='/api')
        print("✅ 论文分析API蓝图注册成功")
    except Exception as e:
        print(f"❌ 论文分析API蓝图注册失败: {str(e)}")
        import traceback
        traceback.print_exc()
    # 注册蓝图
    from app.routes.main import main as main_blueprint
    app.register_blueprint(main_blueprint)
    
    from app.routes.api import api as api_blueprint
    app.register_blueprint(api_blueprint, url_prefix='/api/v1')
    
    # 创建必要的目录
    for directory in [app.config['UPLOAD_FOLDER'], app.config['CACHE_DIR']]:
        os.makedirs(directory, exist_ok=True)
        
    # 注册错误处理器
    register_error_handlers(app)
    
    # 运行应用前确保数据库表存在
    with app.app_context():
        db.create_all()
    
    return app 