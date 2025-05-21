import logging
import json
import os
import uuid
import time
from datetime import datetime
from tqdm import tqdm
from app.config import Config
from app.modules.data_extraction import extract_entities_from_paper, process_papers_and_extract_data
from app.modules.db_manager import db_manager
from app.modules.knowledge_graph import build_knowledge_graph, export_graph_to_json
import re

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def process_raw_response(response):
    """
    处理千问API的原始响应，支持流式和非流式格式
    
    Args:
        response (list/dict): 千问API的响应
        
    Returns:
        str: 提取的内容
    """
    # 处理Agent3.py中的流式响应处理格式
    if isinstance(response, list) and len(response) > 0:
        if "content" in response[0]:
            return response[0]["content"]
    
    # 处理标准OpenAI格式响应
    if hasattr(response, 'choices') and response.choices:
        return response.choices[0].message.content
    
    # 处理已经是字符串的情况
    if isinstance(response, str):
        return response
    
    # 无法识别的格式
    logging.warning(f"无法处理的响应格式: {type(response)}")
    return ""

def process_review_paper(file_path):
    """
    处理综述文章，提取算法实体和演化关系，并存储到数据库
    
    Args:
        file_path (str): 综述文章的文件路径
    
    Returns:
        tuple: (success, message, task_id)
    """
    try:
        # 生成任务ID
        task_id = str(uuid.uuid4())
        start_time = datetime.now()
        
        # 创建处理任务记录
        db_manager.create_processing_task(task_id, os.path.basename(file_path))
        
        # 更新状态：开始处理
        db_manager.update_processing_status(
            task_id=task_id,
            status='processing',
            current_stage='读取PDF文件',
            progress=0.05,
            current_file=os.path.basename(file_path),
            message='开始处理PDF文件'
        )
        
        # 1. 提取算法实体和要素
        logging.info(f"开始处理文件: {file_path}")
        
        # 更新状态：提取算法实体
        db_manager.update_processing_status(
            task_id=task_id,
            current_stage='提取算法实体',
            progress=0.1,
            message='正在提取算法实体'
        )
        
        # 设置处理超时保护
        max_processing_time = 3600  # 最大处理时间1小时
        stop_timeout_check = False
        
        def check_timeout():
            # 定期检查是否超时
            while not stop_timeout_check:
                time.sleep(60)  # 每60秒检查一次
                current_time = datetime.now()
                elapsed = (current_time - start_time).total_seconds()
                
                if elapsed > max_processing_time:
                    db_manager.update_processing_status(
                        task_id=task_id,
                        status='failed',
                        current_stage='处理超时',
                        message=f'处理时间超过 {max_processing_time/60} 分钟，已自动停止'
                    )
                    logging.error(f"任务 {task_id} 超时")
                    return
                
                # 检查进度文件中的断点信息
                progress_file = os.path.join(Config.CACHE_DIR, f"progress_{task_id}.json")
                if os.path.exists(progress_file):
                    try:
                        with open(progress_file, 'r', encoding='utf-8') as f:
                            progress_data = json.load(f)
                            last_timestamp = progress_data.get('timestamp')
                            if last_timestamp:
                                last_time = datetime.fromisoformat(last_timestamp)
                                idle_time = (datetime.now() - last_time).total_seconds()
                                # 如果超过5分钟没有更新进度，记录日志
                                if idle_time > 300:
                                    logging.warning(f"任务 {task_id} 已超过5分钟未更新进度")
                    except Exception as e:
                        logging.error(f"检查进度文件时出错: {str(e)}")
        
        # 启动超时检查线程
        import threading
        timeout_thread = threading.Thread(target=check_timeout)
        timeout_thread.daemon = True
        timeout_thread.start()
        
        try:
            # 确保处理目录存在
            os.makedirs(Config.CACHE_DIR, exist_ok=True)
            os.makedirs(Config.UPLOAD_DIR, exist_ok=True)
            
            # 使用带有中断恢复功能的函数处理文章
            entities, relations = process_papers_and_extract_data(file_path, task_id)
            
            # 2. 存储数据
            
            # 更新状态：存储数据
            db_manager.update_processing_status(
                task_id=task_id,
                current_stage='存储数据',
                progress=0.9,
                message='正在将提取的数据存储到数据库'
            )
            
            # 保存实体
            for entity in entities:
                try:
                    if 'algorithm_entity' in entity:
                        data_entity = entity['algorithm_entity']
                        # 只有当有实体ID时才存储
                        if 'algorithm_id' in data_entity:
                            algorithm_id = data_entity['algorithm_id']
                            db_manager.store_algorithm_entity(algorithm_id, data_entity)
                except Exception as e:
                    logging.error(f"存储实体时出错: {str(e)}")
            
            # 保存关系
            for relation in relations:
                try:
                    # 遍历所有关系并统一格式
                    for relation in relations:
                        # 确保关系对象的所有必要字段都存在
                        if 'from_entity' not in relation or 'to_entity' not in relation:
                            # 检查是否有旧格式的字段
                            if 'from_paper' in relation:
                                relation['from_entity'] = relation.pop('from_paper')
                            if 'to_paper' in relation:
                                relation['to_entity'] = relation.pop('to_paper')
                            
                            # 如果仍然缺少必要字段，则跳过
                            if 'from_entity' not in relation or 'to_entity' not in relation:
                                logging.warning(f"跳过缺少必要字段的关系: {relation}")
                                continue
                        
                        # 确保其他字段存在
                        relation.setdefault('structure', '')
                        relation.setdefault('detail', '')
                        relation.setdefault('evidence', '')
                        relation.setdefault('confidence', 0.5)
                    
                    db_manager.store_algorithm_relation(relation)
                except Exception as e:
                    logging.error(f"存储关系时出错: {str(e)}")
            
            # 创建和存储图数据
            try:
                graph = build_knowledge_graph(entities, relations)
                graph_file = os.path.join(Config.GRAPH_DATA_DIR, f"{task_id}.json")
                
                export_graph_to_json(graph, graph_file)
                
                logging.info(f"图数据已保存到: {graph_file}")
            except Exception as e:
                logging.error(f"创建或存储图数据时出错: {str(e)}")
            
            # 更新状态：处理完成
            end_time = datetime.now()
            processing_time = (end_time - start_time).total_seconds()
            
            db_manager.update_processing_status(
                task_id=task_id,
                status='completed',
                current_stage='处理完成',
                progress=1.0,
                message=f'处理完成，用时 {processing_time:.2f} 秒'
            )
            
            return True, f"处理完成，共提取 {len(entities)} 个算法实体，{len(relations)} 个演化关系。", task_id
            
        finally:
            # 停止超时检查线程
            stop_timeout_check = True
            
            # 清理临时进度文件
            progress_file = os.path.join(Config.CACHE_DIR, f"progress_{task_id}.json")
            if os.path.exists(progress_file):
                try:
                    os.remove(progress_file)
                    logging.info(f"已清理临时进度文件: {progress_file}")
                except Exception as e:
                    logging.warning(f"清理临时进度文件时出错: {str(e)}")
                    
    except Exception as e:
        logging.error(f"处理文件时出错: {str(e)}")
        
        if task_id:
            db_manager.update_processing_status(
                task_id=task_id,
                status='failed',
                current_stage='处理错误',
                message=f'处理文件时出错: {str(e)}'
            )
        
        return False, f"处理失败: {str(e)}", task_id

def process_multiple_papers(file_paths):
    """
    处理多篇论文，提取算法实体和演化关系，并存储到数据库
    
    Args:
        file_paths (list): 论文文件路径列表
    
    Returns:
        tuple: (success, message, task_id)
    """
    try:
        # 生成任务ID
        task_id = str(uuid.uuid4())
        start_time = datetime.now()
        
        # 创建处理任务记录
        db_manager.create_processing_task(task_id, f"批量处理 {len(file_paths)} 个文件")
        
        # 更新状态：开始处理
        db_manager.update_processing_status(
            task_id=task_id,
            status='processing',
            current_stage='准备处理',
            progress=0.0,
            message=f'开始处理 {len(file_paths)} 个文件'
        )
        
        # 设置处理超时保护
        max_processing_time = 7200  # 最大处理时间2小时
        
        def check_timeout():
            # 定期检查是否超时
            while True:
                time.sleep(60)  # 每60秒检查一次
                current_time = datetime.now()
                elapsed = (current_time - start_time).total_seconds()
                
                if elapsed > max_processing_time:
                    db_manager.update_processing_status(
                        task_id=task_id,
                        status='failed',
                        current_stage='处理超时',
                        message=f'批量处理时间超过 {max_processing_time/60} 分钟，已自动停止'
                    )
                    logging.error(f"批量处理任务 {task_id} 超时")
                    return
                
                # 获取当前状态，检查是否已完成或失败
                status_info = db_manager.get_processing_status(task_id)
                if status_info and status_info.get('status') in ['completed', 'failed']:
                    return
        
        # 启动超时检查线程
        import threading
        timeout_thread = threading.Thread(target=check_timeout)
        timeout_thread.daemon = True
        timeout_thread.start()
        
        all_entities = []
        successful_files = 0
        failed_files = 0
        
        # 遍历处理每个文件
        for i, file_path in enumerate(file_paths):
            filename = os.path.basename(file_path)
            
            # 更新状态：处理当前文件
            progress = (i / len(file_paths)) * 0.8  # 从0%到80%
            db_manager.update_processing_status(
                task_id=task_id,
                current_stage=f'处理文件 {i+1}/{len(file_paths)}',
                progress=progress,
                current_file=filename,
                message=f'正在处理文件: {filename}'
            )
            
            logging.info(f"正在处理文件 {i+1}/{len(file_paths)}: {filename}")
            
            # 单个文件最多尝试两次
            for attempt in range(2):
                try:
                    if attempt > 0:
                        db_manager.update_processing_status(
                            task_id=task_id,
                            message=f'正在重试处理文件: {filename} (尝试 {attempt+1}/2)'
                        )
                    
                    entities = extract_entities_from_paper(file_path, task_id, sub_progress=(i, len(file_paths)))
                    if entities:
                        all_entities.extend(entities)
                        successful_files += 1
                        break  # 成功提取，跳出重试循环
                    elif attempt == 0:  # 第一次失败，继续尝试
                        continue
                except Exception as e:
                    logging.error(f"处理文件 {filename} 时出错 (尝试 {attempt+1}/2): {str(e)}")
                    if attempt == 0:  # 第一次失败，继续尝试
                        time.sleep(5)  # 等待5秒再重试
                        continue
            else:
                # 所有尝试都失败
                failed_files += 1
                db_manager.update_processing_status(
                    task_id=task_id,
                    message=f'无法处理文件: {filename}，跳过并继续'
                )
        
        # 更新处理进度信息
        db_manager.update_processing_status(
            task_id=task_id,
            current_stage='文件处理结果',
            progress=0.8,
            message=f'完成 {successful_files}/{len(file_paths)} 个文件处理，{failed_files} 个失败'
        )
        
        if not all_entities:
            # 更新状态：处理失败
            db_manager.update_processing_status(
                task_id=task_id,
                status='failed',
                current_stage='提取算法实体',
                progress=0.8,
                message='所有文件中都未能提取到算法实体'
            )
            return False, "所有文件中都未能提取到算法实体", task_id
        
        # 更新状态：规范化数据
        db_manager.update_processing_status(
            task_id=task_id,
            current_stage='规范化数据',
            progress=0.85,
            message='正在规范化提取的数据'
        )
        
        # 规范化数据
        normalized_entities = normalize_entities(all_entities)
        
        # 更新状态：存储数据
        db_manager.update_processing_status(
            task_id=task_id,
            current_stage='存储数据',
            progress=0.9,
            message='正在将数据存储到数据库'
        )
        
        # 存储数据到数据库
        db_manager.store_entities(normalized_entities)
        
        # 更新状态：提取演化关系
        db_manager.update_processing_status(
            task_id=task_id,
            current_stage='提取演化关系',
            progress=0.95,
            message='正在提取算法演化关系'
        )
        
        # 提取演化关系
        relations = extract_evolution_relations(normalized_entities)
        
        # 存储演化关系到数据库
        if relations:
            try:
                # 先验证所有关系的目标实体是否都存在
                valid_entities = set()
                
                # 收集数据库中已存在的实体ID
                cursor = db_manager.conn.cursor()
                cursor.execute('SELECT algorithm_id FROM Algorithms')
                existing_ids = set(row[0] for row in cursor.fetchall())
                
                logging.info(f"数据库中有 {len(existing_ids)} 个算法实体")
                
                # 过滤掉引用不存在实体的关系
                valid_relations = []
                skipped_relations = 0
                
                for relation in relations:
                    to_entity = relation.get('to_entity', '')
                    if not to_entity:
                        logging.warning("跳过关系：缺少目标实体ID")
                        skipped_relations += 1
                        continue
                        
                    if to_entity in existing_ids:
                        valid_relations.append(relation)
                    else:
                        logging.warning(f"跳过关系：目标实体 {to_entity} 不存在于数据库中")
                        skipped_relations += 1
                
                if skipped_relations > 0:
                    logging.warning(f"过滤后保留 {len(valid_relations)} 个有效关系，跳过 {skipped_relations} 个无效关系")
                
                # 存储有效关系
                if valid_relations:
                    db_manager.store_relations(valid_relations)
                else:
                    logging.warning("没有有效的演化关系可以存储")
            except Exception as e:
                logging.error(f"验证和存储关系时出错: {str(e)}")
                db_manager.update_processing_status(
                    task_id=task_id,
                    message=f'存储演化关系时出错: {str(e)}'
                )
        
        # 更新状态：完成
        db_manager.update_processing_status(
            task_id=task_id,
            status='completed',
            current_stage='处理完成',
            progress=1.0,
            message=f'批量文件处理完成 (成功: {successful_files}, 失败: {failed_files})',
            completed=True
        )
        
        # 保存处理结果到文件（可选）
        result_path = os.path.join(Config.DATA_DIR, 'batch_processed_results.json')
        with open(result_path, 'w', encoding='utf-8') as f:
            json.dump({
                'entities': [entity for entity in normalized_entities],
                'relations': relations if relations else []
            }, f, ensure_ascii=False, indent=2)
        
        logging.info(f"批量处理完成，结果已保存至 {result_path}")
        
        return True, f"成功处理 {successful_files}/{len(file_paths)} 个文件，{failed_files} 个失败", task_id
        
    except Exception as e:
        logging.error(f"批量处理文件时出错: {str(e)}")
        
        # 如果已创建任务，则更新状态为失败
        if 'task_id' in locals():
            db_manager.update_processing_status(
                task_id=task_id,
                status='failed',
                current_stage='处理错误',
                message=f'批量处理文件时出错: {str(e)}'
            )
        
        return False, f"批量处理文件时出错: {str(e)}", task_id if 'task_id' in locals() else None

def normalize_entities(raw_entities):
    """
    规范化算法实体的各字段，确保一致性
    
    Args:
        raw_entities (List[Dict]): 原始提取的算法实体列表
    
    Returns:
        List[Dict]: 规范化后的算法实体列表
    """
    # 如果输入是API响应而非列表，先提取内容
    if not isinstance(raw_entities, list):
        raw_entities_content = process_raw_response(raw_entities)
        try:
            import json
            # 尝试解析JSON字符串
            raw_entities = json.loads(raw_entities_content)
        except:
            logging.error("无法解析响应内容为实体列表")
            return []
    
    normalized = []
    
    for entity in raw_entities:
        if 'algorithm_entity' in entity:
            algo = entity['algorithm_entity']
            
            # 确保基本字段存在
            if 'algorithm_id' not in algo:
                algo['algorithm_id'] = create_algorithm_id(algo)
            
            # 确保复杂字段是正确的结构
            for field in ['architecture', 'methodology']:
                if field not in algo or not isinstance(algo[field], dict):
                    algo[field] = {}
            
            # 确保列表字段是列表
            list_fields = [
                'authors', 'dataset', 'metrics', 
                'architecture.components', 'architecture.connections', 'architecture.mechanisms',
                'methodology.training_strategy', 'methodology.parameter_tuning',
                'feature_processing', 'evolution_relations'
            ]
            
            for field in list_fields:
                if '.' in field:
                    parent, child = field.split('.')
                    if parent in algo and child not in algo[parent]:
                        algo[parent][child] = []
                    elif parent not in algo:
                        algo[parent] = {child: []}
                elif field not in algo:
                    algo[field] = []
            
            normalized.append(entity)
    
    return normalized

def create_algorithm_id(algo):
    """
    为算法实体创建唯一的ID
    
    Args:
        algo (Dict): 算法实体字典
        
    Returns:
        str: 算法ID
    """
    name = algo.get('name', '')
    year = algo.get('year', '')
    authors = algo.get('authors', [])
    
    if name and year and authors:
        # 使用第一作者姓氏和年份
        first_author = authors[0].split()[-1] if authors[0] else "Unknown"
        return f"{first_author}{year}_{name.replace(' ', '')}"
    elif name and year:
        return f"Unknown{year}_{name.replace(' ', '')}"
    elif name:
        return f"Unknown_{name.replace(' ', '')}"
    else:
        import uuid
        return f"Algo_{str(uuid.uuid4())[:8]}"

def extract_evolution_relations(entities):
    """
    从提取的实体中提取演化关系
    
    Args:
        entities (List[Dict]): 算法实体列表
        
    Returns:
        List[Dict]: 演化关系列表
    """
    relations = []
    logging.info(f"开始从 {len(entities)} 个实体中提取演化关系")
    
    # 如果输入是API响应而非列表，先提取内容
    if not isinstance(entities, list):
        entities_content = process_raw_response(entities)
        try:
            import json
            # 尝试解析JSON字符串
            entities = json.loads(entities_content)
            logging.info(f"已将API响应转换为实体列表，共 {len(entities)} 个实体")
        except:
            logging.error("无法解析响应内容为实体列表")
            return []
    
    # 创建一个ID到标准化ID的映射
    id_mapping = {}
    for entity in entities:
        if 'algorithm_entity' in entity and 'algorithm_id' in entity['algorithm_entity']:
            raw_id = entity['algorithm_entity']['algorithm_id']
            id_mapping[raw_id] = raw_id  # 保持原始ID不变
            logging.info(f"实体ID: {raw_id}")
    
    logging.info(f"创建了 {len(id_mapping)} 个实体ID映射")
    
    # 遍历实体提取关系
    for entity in entities:
        if 'algorithm_entity' in entity and 'evolution_relations' in entity['algorithm_entity']:
            algo = entity['algorithm_entity']
            algo_id = algo.get('algorithm_id', '')  # 保持原始大小写
            relation_count = 0
            
            for relation in algo.get('evolution_relations', []):
                # 优先使用from_entity/to_entity，如果不存在则尝试使用from_paper/to_paper
                from_entity = relation.get('from_entity')
                if from_entity is None and 'from_paper' in relation:
                    from_entity = relation['from_paper']
                    logging.info(f"将from_paper字段转换为from_entity: {from_entity}")
                
                to_entity = relation.get('to_entity')
                if to_entity is None and 'to_paper' in relation:
                    to_entity = relation['to_paper']
                    logging.info(f"将to_paper字段转换为to_entity: {to_entity}")
                
                # 如果to_entity仍然为空，使用当前算法ID
                if not to_entity:
                    to_entity = algo_id
                
                relation_type = relation.get('relation_type')
                if relation_type is None and 'evolution_type' in relation:
                    relation_type = relation['evolution_type']
                    logging.info(f"将evolution_type字段转换为relation_type: {relation_type}")
                
                # 如果缺少必要字段，则跳过
                if not from_entity or not relation_type:
                    logging.warning(f"跳过缺少必要字段的关系: {relation}")
                    continue
                
                relation_obj = {
                    'from_entity': from_entity,
                    'to_entity': to_entity,
                    'relation_type': relation_type,
                    'structure': relation.get('structure', ''),
                    'detail': relation.get('detail', ''),
                    'evidence': relation.get('evidence', ''),
                    'confidence': relation.get('confidence', 0.5)
                }
                relations.append(relation_obj)
                relation_count += 1
                
                logging.info(f"提取关系: {from_entity} -> {to_entity} ({relation_type})")
            
            if relation_count > 0:
                logging.info(f"从实体 {algo_id} 中提取了 {relation_count} 个关系")
    
    logging.info(f"共提取 {len(relations)} 个演化关系")
    return relations

def transform_table_data_to_entities(entity_data, relation_data):
    """
    将从表格中提取的数据转换为标准的算法实体和关系格式。
    
    Args:
        entity_data (List[Dict]): 从表格提取的实体数据
        relation_data (List[Dict]): 从表格提取的关系数据
        
    Returns:
        Tuple[List[Dict], List[Dict]]: 标准化的实体列表和关系列表
    """
    # 转换实体数据
    standard_entities = []
    entity_id_map = {}  # 用于跟踪已添加的实体
    
    for item in entity_data:
        entity_name = item.get("实体名称", "")
        if not entity_name:
            continue
            
        # 简单处理以生成ID（实际使用可能需要更复杂的逻辑）
        entity_id = entity_name.replace(" ", "_").upper()  # 统一使用大写ID
        
        # 跟踪已添加的实体，防止重复
        if entity_id in entity_id_map:
            continue
            
        entity_id_map[entity_id] = True
        
        # 创建简化的实体记录
        entity = {
            "algorithm_entity": {
                "algorithm_id": entity_id,
                "name": entity_name,
                "entity_type": item.get("模式类型", "Unknown"),
                "reference": item.get("引文", "Unknown"),
                "evolution_relations": []
            }
        }
        
        standard_entities.append(entity)
    
    # 转换关系数据
    standard_relations = []
    
    for item in relation_data:
        from_entity = item.get("实体A", "")
        to_entity = item.get("实体B", "")
        
        if not from_entity or not to_entity:
            continue
            
        # 生成实体ID，统一大写
        from_id = from_entity.replace(" ", "_").upper()
        to_id = to_entity.replace(" ", "_").upper()
        
        # 创建关系记录
        relation = {
            "from_entity": from_id,
            "to_entity": to_id,
            "relation_type": item.get("改进类型", "Improve"),
            "structure": item.get("改进模式层级", "Unknown"),
            "detail": item.get("改进内容", ""),
            "evidence": "",  # 表格数据通常不包含证据
            "confidence": 0.9  # 默认置信度
        }
        
        standard_relations.append(relation)
        
        # 将关系添加到对应实体的evolution_relations中
        for entity in standard_entities:
            if entity["algorithm_entity"]["algorithm_id"] == to_id:
                entity["algorithm_entity"]["evolution_relations"].append(relation)
    
    return standard_entities, standard_relations

def save_data_to_json(entities, relations, entities_file='entities.json', relations_file='relations.json'):
    """
    将实体和关系数据保存为JSON文件。
    
    Args:
        entities (List[Dict]): 实体数据列表
        relations (List[Dict]): 关系数据列表
        entities_file (str): 实体数据保存文件名
        relations_file (str): 关系数据保存文件名
    """
    # 确保数据目录存在
    if not os.path.exists(Config.OUTPUT_FOLDER):
        os.makedirs(Config.OUTPUT_FOLDER)
    
    # 保存实体数据
    entities_path = os.path.join(Config.OUTPUT_FOLDER, entities_file)
    with open(entities_path, 'w', encoding='utf-8') as f:
        json.dump(entities, f, ensure_ascii=False, indent=2)
    logging.info(f"实体数据已保存到 {entities_path}")
    
    # 保存关系数据
    relations_path = os.path.join(Config.OUTPUT_FOLDER, relations_file)
    with open(relations_path, 'w', encoding='utf-8') as f:
        json.dump(relations, f, ensure_ascii=False, indent=2)
    logging.info(f"关系数据已保存到 {relations_path}")

def load_data_from_json(entities_file='entities.json', relations_file='relations.json'):
    """
    从JSON文件加载实体和关系数据。
    
    Args:
        entities_file (str): 实体数据文件名
        relations_file (str): 关系数据文件名
        
    Returns:
        Tuple[List[Dict], List[Dict]]: 实体列表和关系列表
    """
    # 加载实体数据
    entities_path = os.path.join(Config.OUTPUT_FOLDER, entities_file)
    try:
        with open(entities_path, 'r', encoding='utf-8') as f:
            entities = json.load(f)
        logging.info(f"从 {entities_path} 加载了实体数据")
    except FileNotFoundError:
        logging.warning(f"实体数据文件 {entities_path} 不存在，返回空列表")
        entities = []
    
    # 加载关系数据
    relations_path = os.path.join(Config.OUTPUT_FOLDER, relations_file)
    try:
        with open(relations_path, 'r', encoding='utf-8') as f:
            relations = json.load(f)
        logging.info(f"从 {relations_path} 加载了关系数据")
    except FileNotFoundError:
        logging.warning(f"关系数据文件 {relations_path} 不存在，返回空列表")
        relations = []
    
    return entities, relations

def table_string_to_dict(content):
    """
    从文本中提取表格数据。
    
    Args:
        content (str): 包含表格的文本内容，或者千问API的直接响应
        
    Returns:
        Tuple[List[Dict], List[Dict]]: 实体提取字典和进化模式字典
    """
    # 如果输入是API响应而非字符串，先提取内容
    if not isinstance(content, str):
        content = process_raw_response(content)
        
    # 实体提取表可能的标识符
    entity_table_patterns = [
        "表格1:", "表格 1:", "表格1：", "表格 1：", 
        "实体提取表", "模式类型，实体名称，引文"
    ]

def process_folder_with_cache(folder_path, cache_key=None):
    """处理文件夹中的所有PDF文件，并缓存结果"""
    # 确保缓存目录存在
    os.makedirs(Config.CACHE_DIR, exist_ok=True)
    
    # 生成缓存键
    if cache_key is None:
        folder_name = os.path.basename(os.path.normpath(folder_path))
        cache_key = f"{folder_name}_processed"
    
    # 缓存文件路径
    cache_path = os.path.join(Config.CACHE_DIR, f"{cache_key}.json")
    
    # 检查缓存是否存在
    if os.path.exists(cache_path):
        try:
            with open(cache_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logging.warning(f"读取缓存文件失败: {str(e)}")
    
    # 如果缓存不存在或读取失败，处理文件夹
    pdf_files = []
    for file in os.listdir(folder_path):
        if file.lower().endswith('.pdf'):
            pdf_files.append(os.path.join(folder_path, file))
    
    # 处理所有PDF文件
    results = []
    for pdf_file in pdf_files:
        try:
            entity = extract_entities_from_paper(pdf_file)
            if entity:
                results.append(entity)
        except Exception as e:
            logging.error(f"处理文件 {pdf_file} 时出错: {str(e)}")
    
    # 缓存结果
    try:
        with open(cache_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
    except Exception as e:
        logging.warning(f"保存缓存文件失败: {str(e)}")
    
    return results

def process_papers_and_extract_data(review_pdf_path, task_id=None, citation_paths=None):
    """
    处理综述文章和引用文献，提取实体和关系数据。
    
    Args:
        review_pdf_path (str): 综述文章的PDF文件路径
        task_id (str, optional): 处理任务ID，用于更新处理状态
        citation_paths (list, optional): 引用文献的PDF文件路径列表
        
    Returns:
        Tuple[List[Dict], List[Dict]]: 提取的实体列表和关系列表
    """
    # 确保所有必要的目录存在
    for directory in [Config.UPLOAD_DIR, Config.CACHE_DIR, Config.CITED_PAPERS_DIR]:
        os.makedirs(directory, exist_ok=True)
    
    # 初始化关系列表
    relations = []
    
    # 检查是否有进度文件，尝试恢复之前的处理
    progress_file = None
    if task_id:
        progress_file = os.path.join(Config.CACHE_DIR, f"progress_{task_id}.json")
        if os.path.exists(progress_file):
            try:
                with open(progress_file, 'r', encoding='utf-8') as f:
                    progress_data = json.load(f)
                    logging.info(f"找到现有进度文件: {progress_file}")
                    
                    # 从缓存中恢复已处理的实体列表
                    entities = progress_data.get('entities', [])
                    processed_files = set(progress_data.get('processed_files', []))
                    
                    # 更新处理状态
                    db_manager.update_processing_status(
                        task_id=task_id,
                        current_stage='恢复处理',
                        progress=progress_data.get('progress', 0.3),
                        message=f'从上次中断处恢复处理，已处理 {len(processed_files)} 个文件'
                    )
                    
                    # 检查主综述是否已处理
                    review_basename = os.path.basename(review_pdf_path)
                    if review_basename in processed_files:
                        logging.info("综述文章已处理，跳过")
                    else:
                        # 需要处理综述文章
                        logging.info("需要处理综述文章")
                        processed_files = set()  # 重置处理列表
                        entities = []  # 重置实体列表
            except Exception as e:
                logging.error(f"读取进度文件时出错: {str(e)}")
                processed_files = set()
                entities = []
        else:
            processed_files = set()
            entities = []
    else:
        processed_files = set()
        entities = []