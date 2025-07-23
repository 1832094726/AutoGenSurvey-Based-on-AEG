#!/usr/bin/env python3
"""
AutoSurvey集成环境验证脚本
检查所有依赖、配置和组件是否正确设置
"""

import sys
import os
import importlib
import subprocess
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def check_python_version():
    """检查Python版本"""
    print("🐍 检查Python版本...")
    
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print(f"❌ Python版本过低: {version.major}.{version.minor}")
        print("   需要Python 3.8或更高版本")
        return False
    
    print(f"✅ Python版本: {version.major}.{version.minor}.{version.micro}")
    return True

def check_required_packages():
    """检查必需的Python包"""
    print("\n📦 检查必需的Python包...")
    
    required_packages = [
        "flask",
        "asyncio",
        "sqlite3",
        "json",
        "datetime",
        "pathlib",
        "logging",
        "typing",
        "dataclasses",
        "hashlib",
        "tempfile",
        "shutil",
        "re"
    ]
    
    optional_packages = [
        ("pytest", "运行测试"),
        ("python-docx", "生成Word文档"),
        ("psutil", "系统监控"),
        ("requests", "HTTP请求")
    ]
    
    missing_required = []
    missing_optional = []
    
    # 检查必需包
    for package in required_packages:
        try:
            importlib.import_module(package)
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package} (必需)")
            missing_required.append(package)
    
    # 检查可选包
    for package, description in optional_packages:
        try:
            importlib.import_module(package.replace("-", "_"))
            print(f"✅ {package} ({description})")
        except ImportError:
            print(f"⚠️  {package} ({description}) - 可选")
            missing_optional.append((package, description))
    
    if missing_required:
        print(f"\n❌ 缺少必需包: {', '.join(missing_required)}")
        return False
    
    if missing_optional:
        print(f"\n⚠️  缺少可选包:")
        for package, desc in missing_optional:
            print(f"   • {package}: {desc}")
    
    return True

def check_project_structure():
    """检查项目结构"""
    print("\n📁 检查项目结构...")
    
    required_dirs = [
        "app",
        "app/modules",
        "app/routes",
        "app/templates",
        "app/static",
        "app/static/js",
        "app/static/css",
        "tests",
        "scripts"
    ]
    
    required_files = [
        "app/__init__.py",
        "app/config.py",
        "app/modules/autosurvey_integration.py",
        "app/modules/survey_generation_engine.py",
        "app/modules/survey_storage_manager.py",
        "app/modules/survey_formatter.py",
        "app/modules/lineage_description_generator.py",
        "app/routes/autosurvey_routes.py",
        "app/templates/autosurvey.html",
        "app/static/js/autosurvey.js"
    ]
    
    project_root = Path(__file__).parent.parent
    
    # 检查目录
    missing_dirs = []
    for dir_path in required_dirs:
        full_path = project_root / dir_path
        if full_path.exists():
            print(f"✅ {dir_path}/")
        else:
            print(f"❌ {dir_path}/")
            missing_dirs.append(dir_path)
    
    # 检查文件
    missing_files = []
    for file_path in required_files:
        full_path = project_root / file_path
        if full_path.exists():
            print(f"✅ {file_path}")
        else:
            print(f"❌ {file_path}")
            missing_files.append(file_path)
    
    if missing_dirs or missing_files:
        print(f"\n❌ 项目结构不完整")
        if missing_dirs:
            print(f"   缺少目录: {', '.join(missing_dirs)}")
        if missing_files:
            print(f"   缺少文件: {', '.join(missing_files)}")
        return False
    
    return True

def check_module_imports():
    """检查模块导入"""
    print("\n🔗 检查模块导入...")
    
    modules_to_test = [
        "app.modules.autosurvey_integration",
        "app.modules.survey_generation_engine", 
        "app.modules.survey_storage_manager",
        "app.modules.survey_formatter",
        "app.modules.lineage_description_generator",
        "app.routes.autosurvey_routes"
    ]
    
    import_errors = []
    
    for module_name in modules_to_test:
        try:
            importlib.import_module(module_name)
            print(f"✅ {module_name}")
        except ImportError as e:
            print(f"❌ {module_name}: {str(e)}")
            import_errors.append((module_name, str(e)))
        except Exception as e:
            print(f"⚠️  {module_name}: {str(e)}")
    
    if import_errors:
        print(f"\n❌ 模块导入失败:")
        for module, error in import_errors:
            print(f"   • {module}: {error}")
        return False
    
    return True

def check_database_setup():
    """检查数据库设置"""
    print("\n🗄️  检查数据库设置...")
    
    try:
        from app.modules.db_manager import DatabaseManager
        
        # 尝试创建数据库管理器实例
        db_manager = DatabaseManager()
        print("✅ 数据库管理器初始化成功")
        
        # 检查数据库连接
        if hasattr(db_manager, 'db_utils') and db_manager.db_utils:
            print("✅ 数据库工具可用")
        else:
            print("⚠️  数据库工具未配置")
        
        return True
        
    except ImportError:
        print("⚠️  数据库管理器模块未找到")
        return True  # 不是致命错误
    except Exception as e:
        print(f"⚠️  数据库设置检查失败: {str(e)}")
        return True  # 不是致命错误

def check_config_file():
    """检查配置文件"""
    print("\n⚙️  检查配置文件...")
    
    try:
        from app.config import Config
        
        config = Config()
        print("✅ 配置文件加载成功")
        
        # 检查关键配置项
        if hasattr(config, 'SECRET_KEY'):
            print("✅ SECRET_KEY 已配置")
        else:
            print("⚠️  SECRET_KEY 未配置")
        
        if hasattr(config, 'DATABASE_URL'):
            print("✅ DATABASE_URL 已配置")
        else:
            print("⚠️  DATABASE_URL 未配置")
        
        return True
        
    except ImportError as e:
        print(f"❌ 配置文件导入失败: {str(e)}")
        return False
    except Exception as e:
        print(f"⚠️  配置文件检查失败: {str(e)}")
        return True

def check_autosurvey_components():
    """检查AutoSurvey组件"""
    print("\n🤖 检查AutoSurvey组件...")
    
    try:
        from app.modules.autosurvey_integration import (
            TaskSelector, EntityRelationExtractor, DataFormatConverter,
            AutoSurveyConnector, AutoSurveyConfig, AlgorithmLineageAnalyzer
        )
        
        print("✅ 核心组件导入成功")
        
        # 测试组件初始化
        config = AutoSurveyConfig()
        print("✅ AutoSurvey配置初始化成功")
        
        converter = DataFormatConverter()
        print("✅ 数据格式转换器初始化成功")
        
        return True
        
    except ImportError as e:
        print(f"❌ AutoSurvey组件导入失败: {str(e)}")
        return False
    except Exception as e:
        print(f"⚠️  AutoSurvey组件检查失败: {str(e)}")
        return True

def check_storage_system():
    """检查存储系统"""
    print("\n💾 检查存储系统...")
    
    try:
        from app.modules.survey_storage_manager import SurveyStorageManager
        
        # 使用临时目录测试
        import tempfile
        with tempfile.TemporaryDirectory() as temp_dir:
            storage_manager = SurveyStorageManager(temp_dir)
            print("✅ 存储管理器初始化成功")
            
            # 测试基本操作
            stats = storage_manager.get_storage_stats()
            print("✅ 存储统计功能正常")
        
        return True
        
    except ImportError as e:
        print(f"❌ 存储系统导入失败: {str(e)}")
        return False
    except Exception as e:
        print(f"⚠️  存储系统检查失败: {str(e)}")
        return True

def check_external_dependencies():
    """检查外部依赖"""
    print("\n🔧 检查外部依赖...")
    
    # 检查LaTeX（用于PDF生成）
    try:
        result = subprocess.run(['pdflatex', '--version'], 
                              capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            print("✅ LaTeX (pdflatex) 可用")
        else:
            print("⚠️  LaTeX (pdflatex) 不可用 - PDF生成功能受限")
    except (subprocess.TimeoutExpired, FileNotFoundError):
        print("⚠️  LaTeX (pdflatex) 未安装 - PDF生成功能受限")
    
    # 检查其他可选工具
    optional_tools = [
        ("pandoc", "文档转换"),
        ("git", "版本控制")
    ]
    
    for tool, description in optional_tools:
        try:
            result = subprocess.run([tool, '--version'], 
                                  capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                print(f"✅ {tool} 可用 ({description})")
            else:
                print(f"⚠️  {tool} 不可用 ({description})")
        except (subprocess.TimeoutExpired, FileNotFoundError):
            print(f"⚠️  {tool} 未安装 ({description})")
    
    return True

def run_validation():
    """运行完整验证"""
    print("🔍 AutoSurvey集成环境验证\n")
    
    checks = [
        ("Python版本", check_python_version),
        ("必需包", check_required_packages),
        ("项目结构", check_project_structure),
        ("模块导入", check_module_imports),
        ("配置文件", check_config_file),
        ("数据库设置", check_database_setup),
        ("AutoSurvey组件", check_autosurvey_components),
        ("存储系统", check_storage_system),
        ("外部依赖", check_external_dependencies)
    ]
    
    results = []
    
    for check_name, check_func in checks:
        try:
            result = check_func()
            results.append((check_name, result))
        except Exception as e:
            print(f"❌ {check_name} 检查时出错: {str(e)}")
            results.append((check_name, False))
    
    # 输出总结
    print("\n" + "="*50)
    print("📋 验证结果总结:")
    
    passed = 0
    failed = 0
    
    for check_name, result in results:
        if result:
            print(f"✅ {check_name}")
            passed += 1
        else:
            print(f"❌ {check_name}")
            failed += 1
    
    print(f"\n通过: {passed}/{len(results)}")
    
    if failed == 0:
        print("\n🎉 所有检查通过！AutoSurvey集成环境配置正确")
        return True
    else:
        print(f"\n⚠️  有 {failed} 项检查未通过，请检查上述问题")
        return False

if __name__ == "__main__":
    success = run_validation()
    sys.exit(0 if success else 1)
