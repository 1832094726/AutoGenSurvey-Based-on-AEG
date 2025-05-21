import sys
import logging
import json
import os

# 设置日志级别
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("entity_normalization_test.log"),
        logging.StreamHandler()
    ]
)

# 添加当前目录到sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def normalize_entities(entities):
    """模拟data_extraction.py中的实体规范化逻辑"""
    normalized_entities = []
    for entity in entities:
        # 处理旧结构（algorithm_entity、dataset_entity、metric_entity嵌套）
        if 'algorithm_entity' in entity:
            algo = entity['algorithm_entity']
            # 确保entity_id一致
            if 'algorithm_id' in algo:
                algo['entity_id'] = algo['algorithm_id']
            # 确保entity_type存在
            if 'entity_type' not in algo:
                algo['entity_type'] = 'Algorithm'
            normalized_entities.append(algo)
        elif 'dataset_entity' in entity:
            dataset = entity['dataset_entity']
            # 确保entity_id一致
            if 'dataset_id' in dataset:
                dataset['entity_id'] = dataset['dataset_id']
            # 确保entity_type存在
            if 'entity_type' not in dataset:
                dataset['entity_type'] = 'Dataset'
            normalized_entities.append(dataset)
        elif 'metric_entity' in entity:
            metric = entity['metric_entity']
            # 确保entity_id一致
            if 'metric_id' in metric:
                metric['entity_id'] = metric['metric_id']
            # 确保entity_type存在
            if 'entity_type' not in metric:
                metric['entity_type'] = 'Metric'
            normalized_entities.append(metric)
        # 处理新结构（直接对象）
        elif 'entity_id' in entity or 'algorithm_id' in entity or 'dataset_id' in entity or 'metric_id' in entity:
            # 统一ID字段
            if 'algorithm_id' in entity and 'entity_id' not in entity:
                entity['entity_id'] = entity['algorithm_id']
            if 'dataset_id' in entity and 'entity_id' not in entity:
                entity['entity_id'] = entity['dataset_id']
            if 'metric_id' in entity and 'entity_id' not in entity:
                entity['entity_id'] = entity['metric_id']
            
            if 'entity_id' in entity:
                if 'algorithm_id' not in entity and entity.get('entity_type') == 'Algorithm':
                    entity['algorithm_id'] = entity['entity_id']
                elif 'dataset_id' not in entity and entity.get('entity_type') == 'Dataset':
                    entity['dataset_id'] = entity['entity_id']
                elif 'metric_id' not in entity and entity.get('entity_type') == 'Metric':
                    entity['metric_id'] = entity['entity_id']
            
            # 确保entity_type存在
            if 'entity_type' not in entity:
                # 尝试根据ID推断类型
                if 'algorithm_id' in entity:
                    entity['entity_type'] = 'Algorithm'
                elif 'dataset_id' in entity:
                    entity['entity_type'] = 'Dataset'
                elif 'metric_id' in entity:
                    entity['entity_type'] = 'Metric'
                else:
                    entity['entity_type'] = 'Algorithm'  # 默认类型
            
            normalized_entities.append(entity)
        else:
            logging.warning(f"未知实体格式: {entity.keys()}")
    
    return normalized_entities

def main():
    """测试实体规范化逻辑"""
    try:
        # 创建测试数据
        test_entities = [
            {'algorithm_entity': {'name': '算法1', 'algorithm_id': 'algo1'}},
            {'dataset_entity': {'name': '数据集1', 'dataset_id': 'data1'}},
            {'metric_entity': {'name': '指标1', 'metric_id': 'metric1'}},
            {'algorithm_entity': {'name': '算法2', 'algorithm_id': 'algo2'}},
            {'entity_id': 'direct1', 'name': '直接实体1'},
            {'algorithm_id': 'algo3', 'name': '算法3'},
            {'dataset_id': 'data2', 'name': '数据集2'},
            {'metric_id': 'metric2', 'name': '指标2'},
            {'entity_id': 'multi1', 'entity_type': 'Algorithm', 'name': '多类型1'},
            {'entity_id': 'multi2', 'entity_type': 'Dataset', 'name': '多类型2'},
            {'entity_id': 'multi3', 'entity_type': 'Metric', 'name': '多类型3'}
        ]
        
        # 规范化实体
        normalized = normalize_entities(test_entities)
        
        # 检查结果
        logging.info(f"规范化后的实体数量: {len(normalized)}")
        
        # 打印规范化后的实体
        for i, entity in enumerate(normalized):
            logging.info(f"实体 {i+1}:")
            logging.info(f"  类型: {entity.get('entity_type', 'Unknown')}")
            logging.info(f"  entity_id: {entity.get('entity_id', 'missing')}")
            if entity.get('entity_type') == 'Algorithm':
                logging.info(f"  algorithm_id: {entity.get('algorithm_id', 'missing')}")
            elif entity.get('entity_type') == 'Dataset':
                logging.info(f"  dataset_id: {entity.get('dataset_id', 'missing')}")
            elif entity.get('entity_type') == 'Metric':
                logging.info(f"  metric_id: {entity.get('metric_id', 'missing')}")
            logging.info(f"  name: {entity.get('name', 'missing')}")
        
        # 按类型统计
        entity_types = {}
        for entity in normalized:
            entity_type = entity.get('entity_type', 'Unknown')
            if entity_type not in entity_types:
                entity_types[entity_type] = 0
            entity_types[entity_type] += 1
        
        logging.info(f"实体类型统计: {entity_types}")
        
        # 检查ID字段
        for entity in normalized:
            entity_type = entity.get('entity_type', 'Unknown')
            entity_id = entity.get('entity_id', 'missing')
            
            if entity_type == 'Algorithm':
                algo_id = entity.get('algorithm_id', 'missing')
                if entity_id != algo_id:
                    logging.warning(f"ID不匹配 - entity_id: {entity_id}, algorithm_id: {algo_id}")
            elif entity_type == 'Dataset':
                dataset_id = entity.get('dataset_id', 'missing')
                if entity_id != dataset_id:
                    logging.warning(f"ID不匹配 - entity_id: {entity_id}, dataset_id: {dataset_id}")
            elif entity_type == 'Metric':
                metric_id = entity.get('metric_id', 'missing')
                if entity_id != metric_id:
                    logging.warning(f"ID不匹配 - entity_id: {entity_id}, metric_id: {metric_id}")
        
        logging.info("规范化测试完成!")
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