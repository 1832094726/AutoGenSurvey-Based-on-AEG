#!/usr/bin/env python3
"""
测试AutoSurvey依赖和环境
"""

import sys
import os

def test_imports():
    """测试关键模块导入"""
    print("=== 测试关键模块导入 ===")
    
    modules_to_test = [
        'numpy',
        'torch',
        'transformers', 
        'sentence_transformers',
        'sklearn',
        'faiss',
        'tinydb',
        'langchain'
    ]
    
    success_count = 0
    for module in modules_to_test:
        try:
            __import__(module)
            print(f"✅ {module} - 导入成功")
            success_count += 1
        except ImportError as e:
            print(f"❌ {module} - 导入失败: {str(e)}")
        except Exception as e:
            print(f"⚠️ {module} - 其他错误: {str(e)}")
    
    print(f"\n导入测试结果: {success_count}/{len(modules_to_test)} 成功")
    return success_count == len(modules_to_test)

def test_autosurvey_imports():
    """测试AutoSurvey特定导入"""
    print("\n=== 测试AutoSurvey特定导入 ===")
    
    # 添加AutoSurvey路径
    autosurvey_path = os.path.join(os.path.dirname(__file__), '..', 'AutoSurvey')
    sys.path.insert(0, autosurvey_path)
    
    try:
        from src.database import database
        print("✅ AutoSurvey database模块导入成功")
        return True
    except ImportError as e:
        print(f"❌ AutoSurvey database模块导入失败: {str(e)}")
        return False
    except Exception as e:
        print(f"⚠️ AutoSurvey database模块其他错误: {str(e)}")
        return False

def main():
    """主测试函数"""
    print("开始测试AutoSurvey依赖环境...")
    
    # 测试基础模块
    basic_test = test_imports()
    
    # 如果基础模块测试通过，则测试AutoSurvey特定模块
    autosurvey_test = False
    if basic_test:
        autosurvey_test = test_autosurvey_imports()
    
    print("\n=== 测试结果总结 ===")
    if basic_test and autosurvey_test:
        print("✅ 所有测试通过！AutoSurvey环境准备就绪")
        return True
    elif basic_test:
        print("⚠️ 基础模块正常，但AutoSurvey模块有问题")
        return False
    else:
        print("❌ 基础模块有问题，需要修复依赖")
        return False

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
