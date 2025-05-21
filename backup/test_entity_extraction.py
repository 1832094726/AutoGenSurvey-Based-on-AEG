import sys
import logging
import json
import os

# 设置日志级别
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# 添加当前目录到sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def main():
    """测试db_manager的功能"""
    try:
        # 导入相关模块，不导入agents模块
        from app.modules.db_manager import db_manager
        
        # 连接数据库
        logging.info("测试数据库连接...")
        db_manager.init_db()
        logging.info("数据库连接成功!")
        
        # 测试关系处理
        logging.info("测试数据库存储关系...")
        relation_data = {
            "from_entity": "Test_Algorithm_DB",
            "to_entity": "Test_Dataset_DB",
            "relation_type": "Use",
            "structure": "Evaluation.Dataset",
            "detail": "测试使用关系",
            "evidence": "这是一个测试",
            "confidence": 0.95,
            "from_entity_type": "Algorithm",
            "to_entity_type": "Dataset"
        }
        
        success = db_manager._store_relation_mysql(relation_data)
        if success:
            logging.info("关系存储成功!")
        else:
            logging.error("关系存储失败!")
        
        # 测试算法存储
        algo_entity = {
            "entity_id": "Test_Algorithm_DB",
            "algorithm_id": "Test_Algorithm_DB",
            "entity_type": "Algorithm",
            "name": "TestAlgo",
            "year": 2023,
            "authors": ["测试作者1", "测试作者2"],
            "task": "测试任务",
            "dataset": ["测试数据集"],
            "metrics": ["测试指标"]
        }
        
        success = db_manager._store_algorithm_mysql(algo_entity)
        if success:
            logging.info("算法存储成功!")
        else:
            logging.error("算法存储失败!")
        
        logging.info("测试完成!")
        return True
        
    except Exception as e:
        logging.error(f"测试过程出错: {str(e)}")
        import traceback
        logging.error(traceback.format_exc())
        return False

if __name__ == "__main__":
    success = main()
    if success:
        logging.info("测试通过!")
    else:
        logging.error("测试失败!")
        sys.exit(1) 