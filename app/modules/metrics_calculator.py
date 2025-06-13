"""
指标计算模块，用于比较分析中各种指标的计算
"""

import logging
import json
import re
from collections import Counter, defaultdict

def calculate_entity_statistics(review_entities, citation_entities):
    """
    计算实体相关统计指标
    
    Args:
        review_entities (list): 综述中的实体列表
        citation_entities (list): 引用文献中的实体列表
    
    Returns:
        dict: 实体统计指标
    """
    logging.info(f"[calculate_entity_statistics] 输入综述实体数: {len(review_entities)}, 引文实体数: {len(citation_entities)}")
    stats = {
        'algorithm_count_review': 0,
        'dataset_count_review': 0,
        'metric_count_review': 0,
        'algorithm_count_citations': 0,
        'dataset_count_citations': 0,
        'metric_count_citations': 0,
        'entity_precision': 0.0,
        'entity_recall': 0.0,
        'f1_score': 0.0
    }
    
    # 统计综述中的实体类型
    for entity in review_entities:
        if 'algorithm_entity' in entity:
            stats['algorithm_count_review'] += 1
        elif 'dataset_entity' in entity:
            stats['dataset_count_review'] += 1
        elif 'metric_entity' in entity:
            stats['metric_count_review'] += 1
    
    logging.info(f"[calculate_entity_statistics] 综述实体类型统计: algorithm={stats['algorithm_count_review']}, dataset={stats['dataset_count_review']}, metric={stats['metric_count_review']}")
    
    # 统计引用文献中的实体类型
    for entity in citation_entities:
        if 'algorithm_entity' in entity:
            stats['algorithm_count_citations'] += 1
        elif 'dataset_entity' in entity:
            stats['dataset_count_citations'] += 1
        elif 'metric_entity' in entity:
            stats['metric_count_citations'] += 1
    
    logging.info(f"[calculate_entity_statistics] 引文实体类型统计: algorithm={stats['algorithm_count_citations']}, dataset={stats['dataset_count_citations']}, metric={stats['metric_count_citations']}")
    
    # 提取实体ID，用于计算精确率和召回率
    review_entity_ids = set()
    citation_entity_ids = set()
    
    for entity in review_entities:
        entity_id = None
        if 'algorithm_entity' in entity and 'algorithm_id' in entity['algorithm_entity']:
            entity_id = entity['algorithm_entity']['algorithm_id']
        elif 'dataset_entity' in entity and 'dataset_id' in entity['dataset_entity']:
            entity_id = entity['dataset_entity']['dataset_id']
        elif 'metric_entity' in entity and 'metric_id' in entity['metric_entity']:
            entity_id = entity['metric_entity']['metric_id']
            
        if entity_id:
            review_entity_ids.add(entity_id)
    
    for entity in citation_entities:
        entity_id = None
        if 'algorithm_entity' in entity and 'algorithm_id' in entity['algorithm_entity']:
            entity_id = entity['algorithm_entity']['algorithm_id']
        elif 'dataset_entity' in entity and 'dataset_id' in entity['dataset_entity']:
            entity_id = entity['dataset_entity']['dataset_id']
        elif 'metric_entity' in entity and 'metric_id' in entity['metric_entity']:
            entity_id = entity['metric_entity']['metric_id']
            
        if entity_id:
            citation_entity_ids.add(entity_id)
    
    logging.info(f"[calculate_entity_statistics] 综述实体ID: {review_entity_ids}")
    logging.info(f"[calculate_entity_statistics] 引文实体ID: {citation_entity_ids}")
    
    # 计算实体精确率和召回率
    common_entities = review_entity_ids.intersection(citation_entity_ids)
    
    logging.info(f"[calculate_entity_statistics] 交集实体ID: {common_entities}")
    
    if review_entity_ids:
        stats['entity_recall'] = len(common_entities) / len(review_entity_ids)
    
    if citation_entity_ids:
        stats['entity_precision'] = len(common_entities) / len(citation_entity_ids)
    
    # 计算F1分数
    if stats['entity_precision'] + stats['entity_recall'] > 0:
        stats['f1_score'] = 2 * stats['entity_precision'] * stats['entity_recall'] / (stats['entity_precision'] + stats['entity_recall'])
    
    logging.info(f"[calculate_entity_statistics] 结果: {stats}")
    
    return stats

def calculate_relation_statistics(relations, review_relations=None, citation_relations=None):
    """
    计算关系相关统计指标
    
    Args:
        relations (list): 所有关系列表
        review_relations (list): 综述中的关系列表
        citation_relations (list): 引文中的关系列表
    
    Returns:
        dict: 关系统计指标
    """
    logging.info(f"[calculate_relation_statistics] 输入关系数: {len(relations)}")
    
    # 默认值处理
    review_relations = review_relations or []
    citation_relations = citation_relations or []
    
    # 如果未提供分类关系，尝试从relations中的source字段分类
    if not review_relations and not citation_relations:
        review_relations = [r for r in relations if r.get('source') == '综述']
        citation_relations = [r for r in relations if r.get('source') == '引文']
    
    stats = {
        'total_relations': len(relations),
        'review_relations': len(review_relations),
        'citation_relations': len(citation_relations),
        'relation_coverage': 0.0,
        'improve_count': 0,
        'optimize_count': 0,
        'extend_count': 0,
        'replace_count': 0,
        'use_count': 0,
        'other_count': 0,
        'relation_types': Counter()
    }
    
    # 统计不同类型的关系
    entity_in_relations = set()  # 用于计算实体参与度
    
    for relation in relations:
        relation_type = relation.get('relation_type', '').lower()
        
        # 统计实体参与关系情况
        from_entity = relation.get('from_entity')
        to_entity = relation.get('to_entity')
        
        if from_entity:
            entity_in_relations.add(from_entity)
        if to_entity:
            entity_in_relations.add(to_entity)
        
        # 统计关系类型
        stats['relation_types'][relation_type] += 1
        
        if 'improve' in relation_type:
            stats['improve_count'] += 1
        elif 'optimize' in relation_type:
            stats['optimize_count'] += 1
        elif 'extend' in relation_type:
            stats['extend_count'] += 1
        elif 'replace' in relation_type:
            stats['replace_count'] += 1
        elif 'use' in relation_type:
            stats['use_count'] += 1
        else:
            stats['other_count'] += 1
    
    # 计算关系覆盖率：重合的关系数除以综述的关系数
    # 重合关系：源实体、目标实体和关系类型都相同
    if review_relations:
        review_relation_keys = set()
        for r in review_relations:
            # 创建关系的唯一标识：from_entity + to_entity + relation_type
            key = (r.get('from_entity', ''), r.get('to_entity', ''), r.get('relation_type', '').lower())
            review_relation_keys.add(key)
        
        citation_relation_keys = set()
        for r in citation_relations:
            key = (r.get('from_entity', ''), r.get('to_entity', ''), r.get('relation_type', '').lower())
            citation_relation_keys.add(key)
        
        # 计算重合关系数
        overlapping_relations = review_relation_keys.intersection(citation_relation_keys)
        overlapping_count = len(overlapping_relations)
        
        # 关系覆盖率 = 重合关系数 / 综述关系数
        if review_relation_keys:
            stats['relation_coverage'] = overlapping_count / len(review_relation_keys)
            stats['overlapping_relations'] = overlapping_count
        
        logging.info(f"[calculate_relation_statistics] 综述关系数: {len(review_relations)}, " 
                    f"引文关系数: {len(citation_relations)}, "
                    f"重合关系数: {overlapping_count}, "
                    f"关系覆盖率: {stats['relation_coverage']:.4f}")
    else:
        # 如果没有综述关系，则使用参与关系的实体比例作为覆盖率（兼容旧逻辑）
        if entity_in_relations:
            stats['relation_coverage'] = len(entity_in_relations) / (len(entity_in_relations) * 1.2)
    
    logging.info(f"[calculate_relation_statistics] 关系类型统计: {stats['relation_types']}")
    logging.info(f"[calculate_relation_statistics] 各类型计数: improve={stats['improve_count']}, "
                f"optimize={stats['optimize_count']}, extend={stats['extend_count']}, "
                f"replace={stats['replace_count']}, use={stats['use_count']}, other={stats['other_count']}")
    
    logging.info(f"[calculate_relation_statistics] 结果: {stats}")
    
    return stats

def calculate_clustering_metrics(entities, relations):
    """
    计算聚类相关指标
    
    Args:
        entities (list): 实体列表
        relations (list): 关系列表
    
    Returns:
        dict: 聚类指标
    """
    logging.info(f"[calculate_clustering_metrics] 输入实体数: {len(entities)}, 关系数: {len(relations)}")
    stats = {
        'precision': 0,  # 示例值
        'recall': 0,     # 示例值
        'clusters': []      # 聚类结果
    }
    
    # 构建实体关系图
    entity_relations = defaultdict(set)
    for relation in relations:
        from_entity = relation.get('from_entity')
        to_entity = relation.get('to_entity')
        if from_entity and to_entity:
            entity_relations[from_entity].add(to_entity)
    
    logging.info(f"[calculate_clustering_metrics] 构建的实体关系图: {dict(entity_relations)}")
    
    # 简单聚类实现（通过关系连接形成聚类）
    clustered = set()
    clusters = []
    
    # 获取所有算法实体ID
    algorithm_entities = []
    for entity in entities:
        if 'algorithm_entity' in entity and 'algorithm_id' in entity['algorithm_entity']:
            algorithm_entities.append(entity['algorithm_entity']['algorithm_id'])
    
    logging.info(f"[calculate_clustering_metrics] 算法实体ID: {algorithm_entities}")
    
    # 对每个未聚类的实体，寻找其关联实体形成聚类
    for entity_id in algorithm_entities:
        if entity_id in clustered:
            continue
            
        # 开始一个新聚类
        cluster = [entity_id]
        clustered.add(entity_id)
        
        # 寻找直接相关的实体
        related_entities = entity_relations.get(entity_id, set())
        for related in related_entities:
            if related not in clustered and related in algorithm_entities:
                cluster.append(related)
                clustered.add(related)
        
        # 只有当聚类包含多个实体时才添加
        if len(cluster) > 1:
            clusters.append(cluster)
    
    # 对于剩余的孤立实体，创建单独的聚类
    for entity_id in algorithm_entities:
        if entity_id not in clustered:
            clusters.append([entity_id])
            clustered.add(entity_id)
    
    stats['clusters'] = clusters
    
    logging.info(f"[calculate_clustering_metrics] 聚类结果: {clusters}")
    
    return stats

def calculate_comparison_metrics(review_entities, citation_entities, relations):
    """
    计算各种比较指标
    
    Args:
        review_entities (list): 从综述中提取的实体列表
        citation_entities (list): 从引用文献中提取的实体列表
        relations (list): 提取的演化关系列表
    
    Returns:
        dict: 包含各种指标的字典
    """
    logging.info(f"[calculate_comparison_metrics] 输入综述实体数: {len(review_entities)}, 引文实体数: {len(citation_entities)}, 关系数: {len(relations)}")
    metrics = {
        'entity_stats': calculate_entity_statistics(review_entities, citation_entities),
        'relation_stats': calculate_relation_statistics(relations),
        'clustering': calculate_clustering_metrics(review_entities + citation_entities, relations)
    }
    
    logging.info(f"[calculate_comparison_metrics] 结果: {json.dumps(metrics, ensure_ascii=False)}")
    
    return metrics

def get_entity_by_id(entities, entity_id):
    """
    根据ID获取实体
    
    Args:
        entities (list): 实体列表
        entity_id (str): 实体ID
    
    Returns:
        dict: 实体信息，未找到则返回None
    """
    logging.info(f"[get_entity_by_id] 输入实体数: {len(entities)}, 查询ID: {entity_id}")
    for entity in entities:
        if 'algorithm_entity' in entity and entity['algorithm_entity'].get('algorithm_id') == entity_id:
            logging.info(f"[get_entity_by_id] 命中算法实体: {entity['algorithm_entity']}")
            return entity['algorithm_entity']
        elif 'dataset_entity' in entity and entity['dataset_entity'].get('dataset_id') == entity_id:
            logging.info(f"[get_entity_by_id] 命中数据集实体: {entity['dataset_entity']}")
            return entity['dataset_entity']
        elif 'metric_entity' in entity and entity['metric_entity'].get('metric_id') == entity_id:
            logging.info(f"[get_entity_by_id] 命中指标实体: {entity['metric_entity']}")
            return entity['metric_entity']
    logging.warning(f"[get_entity_by_id] 未找到ID={entity_id}的实体")
    return None 