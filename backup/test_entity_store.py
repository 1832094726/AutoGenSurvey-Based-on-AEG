import sys
import logging
import json
import os

# 设置日志级别
logging.basicConfig(level=logging.INFO)

# 添加当前目录到sys.path
sys.path.append('.')

# 导入所需模块
from app.modules.db_manager import db_manager

def main():
    # 测试存储算法实体
    algorithm = {
        "algorithm_id": "TestAlgo2023",
        "entity_type": "Algorithm",
        "name": "TestAlgorithm",
        "title": "A Test Algorithm for MWP",
        "year": 2023,
        "authors": ["Test Author", "Another Author"],
        "tasks": ["Math Word Problem Solving", "Natural Language Processing"],
        "datasets": ["Math23K", "Dolphin18K"],
        "metrics": ["Accuracy", "Precision", "Recall"],
        "architecture": {
            "components": ["Encoder", "Decoder", "Attention"],
            "connections": ["Bidirectional"],
            "mechanisms": ["GRU", "LSTM"]
        },
        "methodology": {
            "training_strategy": ["CrossEntropy", "Teacher Forcing"],
            "parameter_tuning": ["Adam", "Learning Rate Scheduling"]
        },
        "feature_processing": ["Tokenization", "Normalization", "Embedding"]
    }
    
    # 测试存储数据集实体
    dataset = {
        "entity_id": "TestDataset2023",
        "entity_type": "Dataset",
        "name": "Test Dataset",
        "description": "A test dataset for evaluation",
        "domain": "Mathematics",
        "size": 5000,
        "year": 2023,
        "creators": ["Test Creator", "Another Creator"]
    }
    
    # 测试存储指标实体
    metric = {
        "entity_id": "TestMetric2023",
        "entity_type": "Metric",
        "name": "Test Metric",
        "description": "A test metric for evaluation",
        "category": "Classification",
        "formula": "(TP)/(TP+FP)"
    }
    
    # 存储实体
    logging.info("开始存储算法实体...")
    success1 = db_manager.store_algorithm_entity(algorithm)
    logging.info(f"存储算法实体 {'成功' if success1 else '失败'}")
    
    logging.info("开始存储数据集实体...")
    success2 = db_manager.store_algorithm_entity(dataset)
    logging.info(f"存储数据集实体 {'成功' if success2 else '失败'}")
    
    logging.info("开始存储指标实体...")
    success3 = db_manager.store_algorithm_entity(metric)
    logging.info(f"存储指标实体 {'成功' if success3 else '失败'}")
    
    # 验证是否成功存储
    logging.info("验证存储结果...")
    entities = db_manager.get_all_entities()
    logging.info(f"总共获取到 {len(entities)} 个实体")
    
    # 打印实体ID及类型
    for entity_wrapper in entities:
        if 'algorithm_entity' in entity_wrapper:
            entity = entity_wrapper['algorithm_entity']
            logging.info(f"算法: {entity.get('algorithm_id')} - {entity.get('name')}")
        elif 'dataset_entity' in entity_wrapper:
            entity = entity_wrapper['dataset_entity']
            logging.info(f"数据集: {entity.get('dataset_id')} - {entity.get('name')}")
        elif 'metric_entity' in entity_wrapper:
            entity = entity_wrapper['metric_entity']
            logging.info(f"指标: {entity.get('metric_id')} - {entity.get('name')}")
    
    return True

if __name__ == "__main__":
    success = main()
    if success:
        print("测试成功!")
    else:
        print("测试失败!")
        sys.exit(1) 