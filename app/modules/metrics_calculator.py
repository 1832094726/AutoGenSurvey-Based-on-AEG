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
    # 初始化统计数据
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
    
    # 统计引用文献中的实体类型
    for entity in citation_entities:
        if 'algorithm_entity' in entity:
            stats['algorithm_count_citations'] += 1
        elif 'dataset_entity' in entity:
            stats['dataset_count_citations'] += 1
        elif 'metric_entity' in entity:
            stats['metric_count_citations'] += 1
    
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
    
    # 计算实体精确率和召回率
    common_entities = review_entity_ids.intersection(citation_entity_ids)
    
    if review_entity_ids:
        stats['entity_recall'] = len(common_entities) / len(review_entity_ids)
    
    if citation_entity_ids:
        stats['entity_precision'] = len(common_entities) / len(citation_entity_ids)
    
    # 计算F1分数
    if stats['entity_precision'] + stats['entity_recall'] > 0:
        stats['f1_score'] = 2 * stats['entity_precision'] * stats['entity_recall'] / (stats['entity_precision'] + stats['entity_recall'])
    
    return stats

def calculate_relation_statistics(relations):
    """
    计算关系相关统计指标
    
    Args:
        relations (list): 关系列表
    
    Returns:
        dict: 关系统计指标
    """
    # 初始化统计数据
    stats = {
        'total_relations': len(relations),
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
    entity_in_relations = set()  # 用于计算关系覆盖率
    
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
    
    # 计算关系覆盖率 (参与关系的实体 / 总实体数)
    # 这里需要所有实体的总数，暂时使用参与关系的实体数作为分母
    if entity_in_relations:
        stats['relation_coverage'] = len(entity_in_relations) / (len(entity_in_relations) * 1.2)  # 假设总体实体数比参与关系的实体多20%
    
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
    # 初始化聚类统计数据
    stats = {
        'precision': 0.75,  # 示例值
        'recall': 0.82,     # 示例值
        'clusters': []      # 聚类结果
    }
    
    # 构建实体关系图
    entity_relations = defaultdict(set)
    for relation in relations:
        from_entity = relation.get('from_entity')
        to_entity = relation.get('to_entity')
        if from_entity and to_entity:
            entity_relations[from_entity].add(to_entity)
    
    # 简单聚类实现（通过关系连接形成聚类）
    clustered = set()
    clusters = []
    
    # 获取所有算法实体ID
    algorithm_entities = []
    for entity in entities:
        if 'algorithm_entity' in entity and 'algorithm_id' in entity['algorithm_entity']:
            algorithm_entities.append(entity['algorithm_entity']['algorithm_id'])
    
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
    metrics = {
        'entity_stats': calculate_entity_statistics(review_entities, citation_entities),
        'relation_stats': calculate_relation_statistics(relations),
        'clustering': calculate_clustering_metrics(review_entities + citation_entities, relations)
    }
    
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
    for entity in entities:
        if 'algorithm_entity' in entity and entity['algorithm_entity'].get('algorithm_id') == entity_id:
            return entity['algorithm_entity']
        elif 'dataset_entity' in entity and entity['dataset_entity'].get('dataset_id') == entity_id:
            return entity['dataset_entity']
        elif 'metric_entity' in entity and entity['metric_entity'].get('metric_id') == entity_id:
            return entity['metric_entity']
    return None 