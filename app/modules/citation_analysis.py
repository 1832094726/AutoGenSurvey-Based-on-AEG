import os
import json
import logging
import shutil
from pathlib import Path
import numpy as np
from collections import defaultdict
from app.config import Config
from app.modules.data_extraction import extract_entities_from_paper
from app.modules.agents import extract_evolution_relations
from app.modules.db_manager import db_manager

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def extract_entities_from_citations(citation_list=None, citation_folder=None, task_id=None):
    """
    从引用的文献中提取实体
    
    Args:
        citation_list (list): 引文列表
        citation_folder (str): 存放引文PDF文件的文件夹
        task_id (str): 任务ID
        
    Returns:
        list: 从引用文献中提取的实体列表
    """
    logging.info(f"从引用文献中提取实体，引文列表长度: {len(citation_list) if citation_list else 0}, 文件夹: {citation_folder}")
    
    # 初始化实体列表和已处理文件集合
    entities = []
    processed_files = set()
    
    # 如果提供了引文文件夹，标准化路径
    if citation_folder:
        citation_folder = os.path.normpath(citation_folder)
    
    # 如果提供了引文文件夹，从文件夹中获取所有PDF
    if citation_folder and os.path.isdir(citation_folder):
        pdf_files = []
        for filename in os.listdir(citation_folder):
            if filename.endswith('.pdf'):
                pdf_files.append(os.path.normpath(os.path.join(citation_folder, filename)))
        logging.info(f"从文件夹 '{citation_folder}' 中找到 {len(pdf_files)} 个PDF文件")
    # 如果提供了引文列表，查找对应的文件
    elif citation_list:
        pdf_files = []
        for citation in citation_list:
            # 根据引文信息在系统中查找对应的PDF文件
            citation_id = citation.get('citation_id', '')
            citation_title = citation.get('title', '')
            
            # 首先在cited_papers目录中查找
            potential_files = []
            for filename in os.listdir(Config.CITED_PAPERS_DIR):
                if filename.endswith('.pdf'):
                    if citation_id and citation_id.lower() in filename.lower():
                        potential_files.append(os.path.normpath(os.path.join(Config.CITED_PAPERS_DIR, filename)))
                    elif citation_title and citation_title.lower() in filename.lower():
                        potential_files.append(os.path.normpath(os.path.join(Config.CITED_PAPERS_DIR, filename)))
            
            # 如果找到了匹配的文件，添加到处理列表
            if potential_files:
                pdf_files.extend(potential_files)
                logging.info(f"找到匹配的PDF文件: {len(potential_files)} 个")
            else:
                logging.warning(f"未找到引文的PDF文件: {citation_id} - {citation_title}")
    else:
        logging.error("未提供引文列表或引文文件夹")
        return []
    
    # 确保不重复处理同一文件
    pdf_files = list(set(pdf_files))
    logging.info(f"总共有 {len(pdf_files)} 个唯一的PDF文件要处理")
    
    # 处理每个PDF文件
    for i, pdf_path in enumerate(pdf_files):
        try:
            filename = os.path.basename(pdf_path)
            if filename in processed_files:
                logging.info(f"文件 '{filename}' 已处理，跳过")
                continue
            
            # 更新处理状态
            if task_id:
                message = f"正在处理文件 {i+1}/{len(pdf_files)}: {filename}"
                db_manager.update_processing_status(
                    task_id=task_id,
                    current_stage='处理引文',
                    progress=i/len(pdf_files),
                    current_file=filename,
                    message=message
                )
            
            logging.info(f"正在处理文件 {i+1}/{len(pdf_files)}: {filename}")
            extracted_entities = extract_entities_from_paper(pdf_path, task_id=task_id)
            
            if extracted_entities:
                entities.extend(extracted_entities)
                processed_files.add(filename)
                logging.info(f"从文件 '{filename}' 中提取了 {len(extracted_entities)} 个实体")
            else:
                logging.warning(f"从文件 '{filename}' 中未提取到实体")
                
        except Exception as e:
            logging.error(f"处理文件 '{pdf_path}' 时出错: {str(e)}")
            import traceback
            logging.error(traceback.format_exc())
    
    # 合并和去除重复实体
    merged_entities = merge_and_deduplicate_entities(entities)
    logging.info(f"原始提取了 {len(entities)} 个实体，合并后剩余 {len(merged_entities)} 个唯一实体")
    
    return merged_entities

def merge_and_deduplicate_entities(entities):
    """
    合并和去除重复实体
    
    Args:
        entities (list): 实体列表
        
    Returns:
        list: 去重后的实体列表
    """
    if not entities:
        return []
    
    # 按实体ID和名称分组
    entity_dict = {}
    
    for entity in entities:
        entity_id = None
        entity_type = None
        
        # 提取实体ID和类型
        if 'entity_id' in entity:
            entity_id = entity.get('entity_id')
            entity_type = entity.get('entity_type', 'Algorithm')
        elif 'algorithm_id' in entity:
            entity_id = entity.get('algorithm_id')
            entity_type = 'Algorithm'
        elif 'dataset_id' in entity:
            entity_id = entity.get('dataset_id')
            entity_type = 'Dataset'
        elif 'metric_id' in entity:
            entity_id = entity.get('metric_id')
            entity_type = 'Metric'
        
        # 如果找不到ID，使用名称作为ID
        if not entity_id:
            entity_name = entity.get('name', '')
            if entity_name:
                entity_id = f"{entity_name.lower().replace(' ', '_')}_{entity_type}"
                # 将生成的ID添加到实体中
                entity['entity_id'] = entity_id
            else:
                # 如果名称为空，跳过该实体
                continue
        
        # 合并同ID实体
        if entity_id in entity_dict:
            # 合并属性
            for key, value in entity.items():
                if key not in entity_dict[entity_id] or not entity_dict[entity_id][key]:
                    entity_dict[entity_id][key] = value
                elif isinstance(value, list) and isinstance(entity_dict[entity_id][key], list):
                    # 合并列表
                    entity_dict[entity_id][key] = list(set(entity_dict[entity_id][key] + value))
                elif isinstance(value, dict) and isinstance(entity_dict[entity_id][key], dict):
                    # 合并字典
                    entity_dict[entity_id][key].update(value)
        else:
            # 添加新实体
            entity_dict[entity_id] = entity
            
            # 确保实体有类型字段
            if 'entity_type' not in entity_dict[entity_id]:
                entity_dict[entity_id]['entity_type'] = entity_type
    
    # 将字典转回列表
    merged_entities = list(entity_dict.values())
    return merged_entities

def extract_relations_from_entities(entities, task_id=None):
    """
    从实体中提取演化关系
    
    Args:
        entities (list): 实体列表
        task_id (str, optional): 处理任务ID
        
    Returns:
        list: 演化关系列表
    """
    if not entities:
        logging.warning("没有实体可用于提取关系")
        return []
    
    # 更新处理状态
    if task_id:
        db_manager.update_processing_status(
            task_id=task_id,
            current_stage='提取演化关系',
            progress=0.7,
            message=f'正在从 {len(entities)} 个实体中提取演化关系'
        )
    
    try:
        # 调用演化关系提取函数
        relations = extract_evolution_relations(entities)
        
        # 合并和去除重复关系
        merged_relations = merge_and_deduplicate_relations(relations)
        
        logging.info(f"提取了 {len(relations)} 个关系，合并后剩余 {len(merged_relations)} 个唯一关系")
        
        return merged_relations
        
    except Exception as e:
        logging.error(f"提取演化关系时出错: {str(e)}")
        import traceback
        logging.error(traceback.format_exc())
        return []

def merge_and_deduplicate_relations(relations):
    """
    合并和去除重复关系
    
    Args:
        relations (list): 关系列表
        
    Returns:
        list: 去重后的关系列表
    """
    if not relations:
        return []
    
    # 按关系特征分组
    relation_dict = {}
    
    for relation in relations:
        # 检查是否是标准格式的关系
        if not isinstance(relation, dict):
            continue
        
        # 处理两种可能的关系格式
        if 'from_entities' in relation and 'to_entities' in relation:
            # 标准格式
            from_entities = relation.get('from_entities', [])
            to_entities = relation.get('to_entities', [])
            relation_type = relation.get('relation_type', '')
            
            # 检查必要字段
            if not from_entities or not to_entities or not relation_type:
                continue
                
            # 生成唯一键
            for from_entity in from_entities:
                for to_entity in to_entities:
                    from_id = from_entity.get('entity_id', '')
                    to_id = to_entity.get('entity_id', '')
                    
                    if not from_id or not to_id:
                        continue
                        
                    key = f"{from_id}_{to_id}_{relation_type}"
                    
                    if key in relation_dict:
                        # 更新现有关系，保留置信度更高的
                        existing_confidence = relation_dict[key].get('confidence', 0)
                        new_confidence = relation.get('confidence', 0)
                        
                        if new_confidence > existing_confidence:
                            relation_dict[key] = relation
                    else:
                        # 添加新关系
                        relation_dict[key] = relation
                        
        elif 'from_entity' in relation and 'to_entity' in relation:
            # 简化格式
            from_id = relation.get('from_entity', '')
            to_id = relation.get('to_entity', '')
            relation_type = relation.get('relation_type', '')
            
            # 检查必要字段
            if not from_id or not to_id or not relation_type:
                continue
                
            # 生成唯一键
            key = f"{from_id}_{to_id}_{relation_type}"
            
            if key in relation_dict:
                # 更新现有关系，保留置信度更高的
                existing_confidence = relation_dict[key].get('confidence', 0)
                new_confidence = relation.get('confidence', 0)
                
                if new_confidence > existing_confidence:
                    # 转换为标准格式
                    standard_relation = {
                        'from_entities': [{'entity_id': from_id, 'entity_type': relation.get('from_entity_type', 'Algorithm')}],
                        'to_entities': [{'entity_id': to_id, 'entity_type': relation.get('to_entity_type', 'Algorithm')}],
                        'relation_type': relation_type,
                        'structure': relation.get('structure', ''),
                        'detail': relation.get('detail', ''),
                        'evidence': relation.get('evidence', ''),
                        'confidence': new_confidence
                    }
                    relation_dict[key] = standard_relation
            else:
                # 转换为标准格式
                standard_relation = {
                    'from_entities': [{'entity_id': from_id, 'entity_type': relation.get('from_entity_type', 'Algorithm')}],
                    'to_entities': [{'entity_id': to_id, 'entity_type': relation.get('to_entity_type', 'Algorithm')}],
                    'relation_type': relation_type,
                    'structure': relation.get('structure', ''),
                    'detail': relation.get('detail', ''),
                    'evidence': relation.get('evidence', ''),
                    'confidence': relation.get('confidence', 0.5)
                }
                relation_dict[key] = standard_relation
    
    # 将字典转回列表
    merged_relations = list(relation_dict.values())
    return merged_relations

def compare_entities(generated_entities, gold_standard_entities):
    """
    比较生成的实体和黄金标准实体
    
    Args:
        generated_entities (list): 生成的实体列表
        gold_standard_entities (list): 黄金标准实体列表
        
    Returns:
        dict: 比较结果
    """
    if not generated_entities or not gold_standard_entities:
        return {
            "precision": 0.0,
            "recall": 0.0,
            "f1_score": 0.0,
            "matched_entities": [],
            "missed_entities": [],
            "extra_entities": []
        }
    
    # 提取实体ID和名称
    generated_ids = set()
    generated_names = set()
    generated_id_map = {}
    
    for entity in generated_entities:
        entity_id = None
        entity_name = entity.get('name', '')
        
        if 'entity_id' in entity:
            entity_id = entity.get('entity_id')
        elif 'algorithm_id' in entity:
            entity_id = entity.get('algorithm_id')
        elif 'dataset_id' in entity:
            entity_id = entity.get('dataset_id')
        elif 'metric_id' in entity:
            entity_id = entity.get('metric_id')
        
        if entity_id:
            generated_ids.add(entity_id)
            generated_id_map[entity_id] = entity
        
        if entity_name:
            generated_names.add(entity_name.lower())
    
    gold_ids = set()
    gold_names = set()
    gold_id_map = {}
    
    for entity in gold_standard_entities:
        entity_id = None
        entity_name = entity.get('name', '')
        
        if 'entity_id' in entity:
            entity_id = entity.get('entity_id')
        elif 'algorithm_id' in entity:
            entity_id = entity.get('algorithm_id')
        elif 'dataset_id' in entity:
            entity_id = entity.get('dataset_id')
        elif 'metric_id' in entity:
            entity_id = entity.get('metric_id')
        
        if entity_id:
            gold_ids.add(entity_id)
            gold_id_map[entity_id] = entity
        
        if entity_name:
            gold_names.add(entity_name.lower())
    
    # 匹配实体
    matched_ids = generated_ids.intersection(gold_ids)
    matched_entities = [{"generated": generated_id_map.get(entity_id), "gold_standard": gold_id_map.get(entity_id)} for entity_id in matched_ids]
    
    # 计算精确度和召回率
    precision = len(matched_ids) / len(generated_ids) if generated_ids else 0.0
    recall = len(matched_ids) / len(gold_ids) if gold_ids else 0.0
    f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    # 找出遗漏和额外的实体
    missed_ids = gold_ids - generated_ids
    extra_ids = generated_ids - gold_ids
    
    missed_entities = [gold_id_map.get(entity_id) for entity_id in missed_ids]
    extra_entities = [generated_id_map.get(entity_id) for entity_id in extra_ids]
    
    return {
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score,
        "matched_entities": matched_entities,
        "missed_entities": missed_entities,
        "extra_entities": extra_entities
    }

def compare_relations(generated_relations, gold_standard_relations):
    """
    比较生成的关系和黄金标准关系
    
    Args:
        generated_relations (list): 生成的关系列表
        gold_standard_relations (list): 黄金标准关系列表
        
    Returns:
        dict: 比较结果
    """
    if not generated_relations or not gold_standard_relations:
        return {
            "precision": 0.0,
            "recall": 0.0,
            "f1_score": 0.0,
            "matched_relations": [],
            "missed_relations": [],
            "extra_relations": []
        }
    
    # 将关系转换为统一的键格式
    def get_relation_keys(relations):
        keys = set()
        relation_map = {}
        
        for relation in relations:
            if 'from_entities' in relation and 'to_entities' in relation:
                for from_entity in relation['from_entities']:
                    for to_entity in relation['to_entities']:
                        from_id = from_entity.get('entity_id', '')
                        to_id = to_entity.get('entity_id', '')
                        relation_type = relation.get('relation_type', '')
                        
                        if from_id and to_id and relation_type:
                            key = f"{from_id}_{to_id}_{relation_type}"
                            keys.add(key)
                            relation_map[key] = relation
            elif 'from_entity' in relation and 'to_entity' in relation:
                from_id = relation.get('from_entity', '')
                to_id = relation.get('to_entity', '')
                relation_type = relation.get('relation_type', '')
                
                if from_id and to_id and relation_type:
                    key = f"{from_id}_{to_id}_{relation_type}"
                    keys.add(key)
                    relation_map[key] = relation
        
        return keys, relation_map
    
    generated_keys, generated_map = get_relation_keys(generated_relations)
    gold_keys, gold_map = get_relation_keys(gold_standard_relations)
    
    # 匹配关系
    matched_keys = generated_keys.intersection(gold_keys)
    matched_relations = [{"generated": generated_map.get(key), "gold_standard": gold_map.get(key)} for key in matched_keys]
    
    # 计算精确度和召回率
    precision = len(matched_keys) / len(generated_keys) if generated_keys else 0.0
    recall = len(matched_keys) / len(gold_keys) if gold_keys else 0.0
    f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    # 找出遗漏和额外的关系
    missed_keys = gold_keys - generated_keys
    extra_keys = generated_keys - gold_keys
    
    missed_relations = [gold_map.get(key) for key in missed_keys]
    extra_relations = [generated_map.get(key) for key in extra_keys]
    
    return {
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score,
        "matched_relations": matched_relations,
        "missed_relations": missed_relations,
        "extra_relations": extra_relations
    }

def compare_method_names(generated_entities, gold_standard_entities):
    """
    比较方法名称
    
    Args:
        generated_entities (list): 生成的实体列表
        gold_standard_entities (list): 黄金标准实体列表
        
    Returns:
        dict: 比较结果
    """
    # 提取方法名称
    generated_names = set()
    for entity in generated_entities:
        if entity.get('entity_type', '') == 'Algorithm':
            name = entity.get('name', '').strip().lower()
            if name:
                generated_names.add(name)
    
    gold_names = set()
    for entity in gold_standard_entities:
        if entity.get('entity_type', '') == 'Algorithm':
            name = entity.get('name', '').strip().lower()
            if name:
                gold_names.add(name)
    
    # 计算准确率
    matched_names = generated_names.intersection(gold_names)
    accuracy = len(matched_names) / len(gold_names) if gold_names else 0.0
    
    return {
        "accuracy": accuracy,
        "matched_names": list(matched_names),
        "missed_names": list(gold_names - generated_names),
        "extra_names": list(generated_names - gold_names)
    }

def compute_cluster_metrics(generated_entities, gold_standard_entities):
    """
    计算算法聚类指标
    
    Args:
        generated_entities (list): 生成的实体列表
        gold_standard_entities (list): 黄金标准实体列表
        
    Returns:
        dict: 聚类指标
    """
    # 按任务聚类
    def cluster_by_task(entities):
        clusters = defaultdict(list)
        for entity in entities:
            if entity.get('entity_type', '') == 'Algorithm':
                tasks = entity.get('task', [])
                if isinstance(tasks, str):
                    tasks = [tasks]
                
                if tasks:
                    for task in tasks:
                        if task:
                            clusters[task.lower()].append(entity)
                else:
                    # 如果没有任务信息，使用通用类别
                    clusters['general'].append(entity)
        
        return clusters
    
    generated_clusters = cluster_by_task(generated_entities)
    gold_clusters = cluster_by_task(gold_standard_entities)
    
    # 计算聚类精确度和召回率
    precision_sum = 0
    recall_sum = 0
    
    for task, gold_entities in gold_clusters.items():
        if task in generated_clusters:
            generated_entities_in_cluster = generated_clusters[task]
            
            # 计算该聚类的精确度和召回率
            gold_entity_names = {entity.get('name', '').lower() for entity in gold_entities if entity.get('name')}
            generated_entity_names = {entity.get('name', '').lower() for entity in generated_entities_in_cluster if entity.get('name')}
            
            intersection = gold_entity_names.intersection(generated_entity_names)
            
            cluster_precision = len(intersection) / len(generated_entity_names) if generated_entity_names else 0.0
            cluster_recall = len(intersection) / len(gold_entity_names) if gold_entity_names else 0.0
            
            precision_sum += cluster_precision
            recall_sum += cluster_recall
    
    # 计算平均精确度和召回率
    precision = precision_sum / len(gold_clusters) if gold_clusters else 0.0
    recall = recall_sum / len(gold_clusters) if gold_clusters else 0.0
    f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return {
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score,
        "generated_clusters": {task: len(entities) for task, entities in generated_clusters.items()},
        "gold_clusters": {task: len(entities) for task, entities in gold_clusters.items()}
    }

def generate_evaluation_metrics(experiment_results, output_path=None):
    """
    生成评估指标表格
    
    Args:
        experiment_results (dict): 实验结果字典
        output_path (str, optional): 输出文件路径
        
    Returns:
        dict: 评估指标表格
    """
    metrics_table = {
        "headers": [
            "模型与实验设置",
            "方法名称准确率（%）",
            "算法聚类精准率（%）",
            "算法聚类召回率（%）",
            "算法关系精准率（%）",
            "算法关系召回率（%）",
            "实体精准率（%）",
            "实体召回率（%）"
        ],
        "rows": []
    }
    
    for setting, results in experiment_results.items():
        row = [
            setting,
            round(results.get('method_name_accuracy', 0) * 100),
            round(results.get('cluster_precision', 0) * 100),
            round(results.get('cluster_recall', 0) * 100),
            round(results.get('relation_precision', 0) * 100),
            round(results.get('relation_recall', 0) * 100),
            round(results.get('entity_precision', 0) * 100),
            round(results.get('entity_recall', 0) * 100)
        ]
        metrics_table['rows'].append(row)
    
    # 保存结果
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(metrics_table, f, ensure_ascii=False, indent=2)
    
    return metrics_table

def process_citations_and_extract_data(review_pdf_path, citation_folder=None, task_id=None, experiment_settings=None):
    """
    处理综述文章和引文，提取实体和关系数据
    
    Args:
        review_pdf_path (str): 综述文章的PDF文件路径
        citation_folder (str, optional): 引文文件夹路径
        task_id (str, optional): 处理任务ID
        experiment_settings (dict, optional): 实验设置参数
        
    Returns:
        tuple: 综述实体、引文实体、综述关系、引文关系
    """
    # 确保目录存在
    os.makedirs(Config.CACHE_DIR, exist_ok=True)
    
    # 更新处理状态
    if task_id:
        db_manager.update_processing_status(
            task_id=task_id,
            current_stage='开始处理',
            progress=0.1,
            message=f'开始处理综述文章和引文'
        )
    
    # 处理综述文章
    logging.info(f"正在处理综述文章: {review_pdf_path}")
    review_entities = extract_entities_from_paper(review_pdf_path, task_id)
    
    # 更新处理状态
    if task_id:
        db_manager.update_processing_status(
            task_id=task_id,
            current_stage='处理引文',
            progress=0.3,
            message=f'从综述中提取了 {len(review_entities)} 个实体，开始处理引文'
        )
    
    # 处理引文
    citation_entities = []
    if citation_folder:
        logging.info(f"正在处理引文文件夹: {citation_folder}")
        citation_entities = extract_entities_from_citations(citation_folder=citation_folder, task_id=task_id)
    
    # 提取关系
    if task_id:
        db_manager.update_processing_status(
            task_id=task_id,
            current_stage='提取关系',
            progress=0.6,
            message=f'处理完成，开始提取演化关系'
        )
    
    # 提取综述文章的关系
    review_relations = extract_relations_from_entities(review_entities, task_id)
    
    # 提取引文的关系
    citation_relations = []
    if citation_entities:
        citation_relations = extract_relations_from_entities(citation_entities, task_id)
    
    # 保存结果
    result_data = {
        "review_entities": review_entities,
        "citation_entities": citation_entities,
        "review_relations": review_relations,
        "citation_relations": citation_relations
    }
    
    # 保存到文件
    result_path = os.path.join(Config.CACHE_DIR, f"citation_analysis_{task_id}.json")
    with open(result_path, 'w', encoding='utf-8') as f:
        json.dump(result_data, f, ensure_ascii=False, indent=2)
    
    # 更新处理状态
    if task_id:
        db_manager.update_processing_status(
            task_id=task_id,
            current_stage='完成',
            progress=1.0,
            message=f'处理完成，提取了 {len(review_entities)} 个综述实体，{len(citation_entities)} 个引文实体，{len(review_relations)} 个综述关系，{len(citation_relations)} 个引文关系'
        )
    
    return review_entities, citation_entities, review_relations, citation_relations

def compare_review_and_citations(review_data, citation_data, gold_standard_data=None, output_path=None):
    """
    比较综述和引文的数据
    
    Args:
        review_data (dict): 综述数据
        citation_data (dict): 引文数据
        gold_standard_data (dict, optional): 黄金标准数据
        output_path (str, optional): 输出文件路径
        
    Returns:
        dict: 比较结果
    """
    # 提取实体和关系
    review_entities = review_data.get('entities', [])
    review_relations = review_data.get('relations', [])
    
    citation_entities = citation_data.get('entities', [])
    citation_relations = citation_data.get('relations', [])
    
    # 准备结果数据
    results = {
        "entity_comparison": compare_entities(review_entities, citation_entities),
        "relation_comparison": compare_relations(review_relations, citation_relations),
        "method_name_comparison": compare_method_names(review_entities, citation_entities),
        "cluster_metrics": compute_cluster_metrics(review_entities, citation_entities)
    }
    
    # 如果提供了黄金标准，与其比较
    if gold_standard_data:
        gold_entities = gold_standard_data.get('entities', [])
        gold_relations = gold_standard_data.get('relations', [])
        
        results["review_vs_gold"] = {
            "entity_comparison": compare_entities(review_entities, gold_entities),
            "relation_comparison": compare_relations(review_relations, gold_relations),
            "method_name_comparison": compare_method_names(review_entities, gold_entities),
            "cluster_metrics": compute_cluster_metrics(review_entities, gold_entities)
        }
        
        results["citation_vs_gold"] = {
            "entity_comparison": compare_entities(citation_entities, gold_entities),
            "relation_comparison": compare_relations(citation_relations, gold_relations),
            "method_name_comparison": compare_method_names(citation_entities, gold_entities),
            "cluster_metrics": compute_cluster_metrics(citation_entities, gold_entities)
        }
    
    # 保存结果
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
    
    return results 