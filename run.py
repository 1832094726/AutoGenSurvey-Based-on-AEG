import traceback
import sys

# 禁用Matplotlib交互式后端，避免TkAgg问题
try:
    import matplotlib
    matplotlib.use('Agg')  # 使用非交互式后端
    print("已设置Matplotlib为非交互式后端Agg")
except ImportError:
    print("未发现Matplotlib，跳过后端配置")

try:
    import argparse
    print("成功导入 argparse")
except ImportError as e:
    print(f"导入 argparse 失败: {e}")
    sys.exit(1)

try:
    from app import create_app
    print("成功导入 create_app")
except ImportError as e:
    print(f"导入 create_app 失败: {e}")
    traceback.print_exc()
    sys.exit(1)

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='启动算法要素关系图生成系统')
    parser.add_argument('--config', default='development', help='配置名称: development, production')
    parser.add_argument('--host', default='0.0.0.0', help='主机地址')
    parser.add_argument('--port', type=int, default=5000, help='端口号')
    parser.add_argument('--debug', action='store_true', help='是否开启调试模式')
    parser.add_argument('--safe', action='store_true', help='安全模式 (仅测试创建应用)')
    parser.add_argument('--check-imports', action='store_true', help='仅检查导入是否成功')
    return parser.parse_args()

if __name__ == '__main__':
    print("开始执行 run.py")
    
    try:
        # 解析命令行参数
        args = parse_args()
        print(f"成功解析命令行参数: {args}")
        
        # 如果仅检查导入
        if args.check_imports:
            print("导入检查成功，所有模块可正常导入")
            sys.exit(0)
            
        try:
            print(f"正在尝试创建应用实例，配置模式: {args.config}")
            # 创建应用
            app = create_app(args.config)
            print("成功创建应用实例")
            
            # 如果是安全模式，只创建应用实例，不启动服务器
            if args.safe:
                print("安全模式: 应用实例创建成功，不启动服务器")
                sys.exit(0)
                
            # 启动应用
            print(f"启动应用服务器，主机: {args.host}, 端口: {args.port}")
            app.run(
                host=args.host,
                port=args.port,
                debug=args.debug or app.config.get('DEBUG', False)
            )
        except Exception as e:
            print(f"应用启动过程中发生错误: {str(e)}")
            print("错误详情:")
            traceback.print_exc()
            sys.exit(1)
    except Exception as e:
        print(f"解析命令行参数时发生错误: {str(e)}")
        print("错误详情:")
        traceback.print_exc()
        sys.exit(1) 