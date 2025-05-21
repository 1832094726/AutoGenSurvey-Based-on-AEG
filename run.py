import argparse
from app import create_app

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='启动算法要素关系图生成系统')
    parser.add_argument('--config', default='development', help='配置名称: development, production')
    parser.add_argument('--host', default='0.0.0.0', help='主机地址')
    parser.add_argument('--port', type=int, default=5000, help='端口号')
    parser.add_argument('--debug', action='store_true', help='是否开启调试模式')
    return parser.parse_args()

if __name__ == '__main__':
    # 解析命令行参数
    args = parse_args()
    
    # 创建应用
    app = create_app(args.config)
    
    # 启动应用
    app.run(
        host=args.host,
        port=args.port,
        debug=args.debug or app.config.get('DEBUG', False)
    ) 