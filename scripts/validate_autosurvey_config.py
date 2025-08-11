#!/usr/bin/env python3
"""
验证AutoSurvey配置和环境
"""

import os
import sys
import subprocess

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.config import Config

def main():
    print("=== AutoSurvey配置验证 ===")
    
    # 检查配置
    print(f"1. AutoSurvey启用: {Config.AUTOSURVEY_ENABLED}")
    print(f"2. AutoSurvey路径: {Config.AUTOSURVEY_PATH}")
    print(f"3. 路径存在: {os.path.exists(Config.AUTOSURVEY_PATH)}")
    print(f"4. 输出路径: {Config.AUTOSURVEY_OUTPUT_PATH}")
    print(f"5. 数据库路径: {Config.AUTOSURVEY_DATABASE_PATH}")
    print(f"6. 默认模型: {Config.DEFAULT_MODEL}")
    print(f"7. Qwen API Key: {'已配置' if Config.QWEN_API_KEY else '未配置'}")
    print(f"8. Qwen Base URL: {Config.QWEN_BASE_URL}")
    
    # 检查AutoSurvey main.py
    main_py = os.path.join(Config.AUTOSURVEY_PATH, 'main.py')
    print(f"9. main.py存在: {os.path.exists(main_py)}")
    
    # 创建输出目录
    os.makedirs(Config.AUTOSURVEY_OUTPUT_PATH, exist_ok=True)
    print(f"10. 输出目录已创建: {os.path.exists(Config.AUTOSURVEY_OUTPUT_PATH)}")
    
    # 测试帮助命令
    if os.path.exists(main_py):
        print("\n=== 测试AutoSurvey帮助命令 ===")
        try:
            result = subprocess.run(
                ['python', 'main.py', '--help'],
                cwd=Config.AUTOSURVEY_PATH,
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if result.returncode == 0:
                print("✅ AutoSurvey帮助命令执行成功")
            else:
                print("❌ AutoSurvey帮助命令执行失败")
                if result.stderr:
                    print(f"错误: {result.stderr}")
        except Exception as e:
            print(f"❌ 执行失败: {str(e)}")
    
    print("\n=== 验证完成 ===")
    print("如果以上配置正确，AutoSurvey集成功能应该可以正常工作。")

if __name__ == '__main__':
    main()
