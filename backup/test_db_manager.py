import sys
import os
import logging
import json
import traceback
from datetime import datetime

# 创建日志目录
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)

# 配置日志到文件
log_file = os.path.join(log_dir, f"db_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)

# 添加当前目录到sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

def main():
    """测试数据库管理器的基本功能"""
    results = {
        "success": True,
        "errors": [],
        "test_results": {}
    }
    
    try:
        logging.info("开始测试数据库管理器")
        
        # 导入数据库管理器 - 放在函数内以便捕获导入错误
        try:
            from app.modules.db_manager import db_manager
            results["test_results"]["import"] = "成功"
        except Exception as e:
            error_msg = f"导入db_manager模块失败: {str(e)}"
            logging.error(error_msg)
            logging.error(traceback.format_exc())
            results["success"] = False
            results["errors"].append(error_msg)
            results["test_results"]["import"] = "失败"
            # 保存结果并退出
            save_results(results)
            return False
        
        # 测试连接
        try:
            if not hasattr(db_manager, 'conn'):
                error_msg = "数据库连接不存在"
                logging.error(error_msg)
                results["success"] = False
                results["errors"].append(error_msg)
                results["test_results"]["connection"] = "失败"
            else:
                logging.info("数据库连接正常")
                results["test_results"]["connection"] = "成功"
        except Exception as e:
            error_msg = f"检查数据库连接时出错: {str(e)}"
            logging.error(error_msg)
            logging.error(traceback.format_exc())
            results["success"] = False
            results["errors"].append(error_msg)
            results["test_results"]["connection"] = "失败"
        
        # 获取所有实体
        try:
            logging.info("获取所有实体...")
            entities = db_manager.get_all_entities()
            logging.info(f"成功获取 {len(entities)} 个实体")
            results["test_results"]["get_entities"] = {
                "status": "成功",
                "count": len(entities)
            }
        except Exception as e:
            error_msg = f"获取实体时出错: {str(e)}"
            logging.error(error_msg)
            logging.error(traceback.format_exc())
            results["success"] = False
            results["errors"].append(error_msg)
            results["test_results"]["get_entities"] = {
                "status": "失败",
                "error": str(e)
            }
        
        # 获取所有关系
        try:
            logging.info("获取所有关系...")
            relations = db_manager.get_all_relations()
            logging.info(f"成功获取 {len(relations)} 个关系")
            results["test_results"]["get_relations"] = {
                "status": "成功",
                "count": len(relations)
            }
        except Exception as e:
            error_msg = f"获取关系时出错: {str(e)}"
            logging.error(error_msg)
            logging.error(traceback.format_exc())
            results["success"] = False
            results["errors"].append(error_msg)
            results["test_results"]["get_relations"] = {
                "status": "失败",
                "error": str(e)
            }
        
        # 总结测试结果
        if results["success"]:
            logging.info("所有测试通过!")
        else:
            logging.error(f"测试失败，共有 {len(results['errors'])} 个错误")
            
        # 保存结果
        save_results(results)
        return results["success"]
        
    except Exception as e:
        error_msg = f"测试过程中出现意外错误: {str(e)}"
        logging.error(error_msg)
        logging.error(traceback.format_exc())
        results["success"] = False
        results["errors"].append(error_msg)
        save_results(results)
        return False

def save_results(results):
    """保存测试结果到JSON文件"""
    result_file = os.path.join(log_dir, f"test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    try:
        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        logging.info(f"测试结果已保存至: {result_file}")
    except Exception as e:
        logging.error(f"保存测试结果时出错: {str(e)}")

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 