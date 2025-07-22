from flask import Blueprint, request, jsonify, send_from_directory, current_app
import os
import uuid
import tempfile
import shutil
import json
import logging
import traceback
from datetime import datetime
from werkzeug.utils import secure_filename
from app.config import Config
from app.modules.data_processing import process_review_paper, process_multiple_papers, normalize_entities, transform_table_data_to_entities, save_data_to_json
from app.modules.knowledge_graph import build_knowledge_graph, visualize_graph, export_graph_to_json
from app.modules.db_manager import db_manager
from app.modules.metrics_calculator import calculate_entity_statistics, calculate_relation_statistics, calculate_clustering_metrics
import mysql.connector
import time
import threading

# 创建蓝图
combined_api = Blueprint('combined_api', __name__)

def generate_task_name(original_task_name, task_id, task_data):
    """生成任务名称 - 如果原始名称为空则使用task_id"""
    # 如果原始任务名称为空、None，则使用task_id
    if not original_task_name or original_task_name.strip() == '':
        return task_id if task_id else '未知任务'

    # 否则使用原始名称
    return original_task_name

# =================== 文件上传处理相关API ===================

@combined_api.route('/upload', methods=['POST'])
def upload_file():
    """接收用户上传的文件，处理后生成知识图谱并返回结果"""
    try:
        if 'file' not in request.files:
            return jsonify({
                'success': False,
                'message': '没有发现上传的文件'
            }), 400
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({
                'success': False,
                'message': '未选择文件'
            }), 400
        
        if not os.path.exists(Config.UPLOAD_DIR):
            os.makedirs(Config.UPLOAD_DIR)
        
        # 使用临时文件保存上传的文件
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as temp_file:
            file.save(temp_file.name)
            
            # 创建永久文件路径
            filename = f"{uuid.uuid4()}{os.path.splitext(file.filename)[1]}"
            target_path = os.path.join(Config.UPLOAD_DIR, filename)
            
            # 确保目标目录存在
            os.makedirs(os.path.dirname(target_path), exist_ok=True)
            
            # 将临时文件移动到目标位置
            shutil.copy2(temp_file.name, target_path)
        
        # 处理上传的文件
        success, message, task_id = process_review_paper(target_path)
        
        if success:
            return jsonify({
                'success': True,
                'message': message,
                'task_id': task_id,
                'filename': os.path.basename(file.filename)
            })
        else:
            return jsonify({
                'success': False,
                'message': message,
                'task_id': task_id
            }), 500
        
    except Exception as e:
        logging.error(f"文件上传处理出错: {str(e)}")
        logging.error(traceback.format_exc())
        return jsonify({
            'success': False,
            'message': f'处理失败: {str(e)}'
        }), 500

@combined_api.route('/batch_upload', methods=['POST'])
def batch_upload_files():
    """批量接收用户上传的文件，处理后生成知识图谱并返回结果"""
    try:
        if 'files[]' not in request.files:
            return jsonify({
                'success': False,
                'message': '没有发现上传的文件'
            }), 400
        
        files = request.files.getlist('files[]')
        
        if not files or all(file.filename == '' for file in files):
            return jsonify({
                'success': False,
                'message': '未选择文件'
            }), 400
        
        if not os.path.exists(Config.UPLOAD_DIR):
            os.makedirs(Config.UPLOAD_DIR)
        
        file_paths = []
        file_names = []
        
        for file in files:
            if file.filename == '':
                continue
                
            # 使用临时文件保存上传的文件
            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as temp_file:
                file.save(temp_file.name)
                
                # 创建永久文件路径
                filename = f"{uuid.uuid4()}{os.path.splitext(file.filename)[1]}"
                target_path = os.path.join(Config.UPLOAD_DIR, filename)
                
                # 确保目标目录存在
                os.makedirs(os.path.dirname(target_path), exist_ok=True)
                
                # 将临时文件移动到目标位置
                shutil.copy2(temp_file.name, target_path)
                
                file_paths.append(target_path)
                file_names.append(os.path.basename(file.filename))
        
        if not file_paths:
            return jsonify({
                'success': False,
                'message': '没有有效的文件上传'
            }), 400
        
        # 批量处理上传的文件
        success, message, task_id = process_multiple_papers(file_paths)
        
        if success:
            return jsonify({
                'success': True,
                'message': message,
                'task_id': task_id,
                'filenames': file_names
            })
        else:
            return jsonify({
                'success': False,
                'message': message,
                'task_id': task_id
            }), 500
        
    except Exception as e:
        logging.error(f"批量文件上传处理出错: {str(e)}")
        logging.error(traceback.format_exc())
        return jsonify({
            'success': False,
            'message': f'批量处理失败: {str(e)}'
        }), 500

# =================== 知识图谱相关API ===================

@combined_api.route('/graph/data', methods=['GET'])
def get_graph_data():
    """获取知识图谱数据"""
    try:
        force_refresh = request.args.get('refresh', '0') == '1'
        
        # 从数据库构建图谱
        entities = db_manager.get_all_entities()
        relations = db_manager.get_all_relations()
        
        if not entities:
            return jsonify({
                'success': False,
                'message': '没有可用的实体数据'
            }), 404
        
        # 构建图
        graph = build_knowledge_graph(entities, relations)
        
        # 获取图的networkx部分用于后续操作
        nx_graph = graph.get("networkx_graph", None)
        
        # 如果需要从graph获取json格式数据
        nodes = graph.get("nodes", [])
        edges = graph.get("edges", [])
        
        # 导出为JSON格式
        output_path = os.path.join(Config.GRAPH_DATA_DIR, 'graph.json')
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        graph_data = export_graph_to_json(graph, output_path)
        
        return jsonify({
            'success': True,
            'nodes': graph_data['nodes'],
            'edges': graph_data['edges']
        })
        
    except Exception as e:
        logging.error(f"获取图谱数据出错: {str(e)}")
        logging.error(traceback.format_exc())
        return jsonify({
            'success': False,
            'message': f'获取图谱数据失败: {str(e)}'
        }), 500

@combined_api.route('/graph/image', methods=['GET'])
def get_graph_image():
    """获取知识图谱图片"""
    try:
        # 确保图数据目录存在
        os.makedirs(Config.GRAPH_DATA_DIR, exist_ok=True)
        
        graph_image_path = os.path.join(Config.GRAPH_DATA_DIR, 'graph.png')
        
        if not os.path.exists(graph_image_path):
            # 如果图片不存在，尝试从数据库创建
            entities = db_manager.get_all_entities()
            relations = db_manager.get_all_relations()
            
            if not entities:
                return jsonify({'success': False, 'message': '没有可用的数据'}), 404
            
            graph = build_knowledge_graph(entities, relations)
            visualize_graph(graph, output_path=graph_image_path)
        
        return send_from_directory(os.path.dirname(graph_image_path), os.path.basename(graph_image_path))
        
    except Exception as e:
        logging.error(f"获取图片时出错: {str(e)}")
        return jsonify({'success': False, 'message': f'获取图片时出错: {str(e)}'}), 500

# =================== 实体相关API ===================

@combined_api.route('/entities', methods=['GET'])
def get_entities():
    """获取所有实体"""
    try:
        entity_type = request.args.get('type', None)
        
        if entity_type:
            entities = db_manager.get_entities_by_type(entity_type)
        else:
            entities = db_manager.get_all_entities()
        
        # 规范化实体格式，处理可能的嵌套结构
        processed_entities = []
        for entity in entities:
            if 'algorithm_entity' in entity:
                algorithm = entity['algorithm_entity']
                algorithm['entity_type'] = 'Algorithm'
                # 确保source字段存在
                if 'source' not in algorithm:
                    algorithm['source'] = entity.get('source', '未知')
                processed_entities.append(algorithm)
            elif 'dataset_entity' in entity:
                dataset = entity['dataset_entity']
                dataset['entity_type'] = 'Dataset'
                # 确保source字段存在
                if 'source' not in dataset:
                    dataset['source'] = entity.get('source', '未知')
                processed_entities.append(dataset)
            elif 'metric_entity' in entity:
                metric = entity['metric_entity']
                metric['entity_type'] = 'Metric'
                # 确保source字段存在
                if 'source' not in metric:
                    metric['source'] = entity.get('source', '未知')
                processed_entities.append(metric)
            elif 'entity_type' in entity:
                # 确保source字段存在
                if 'source' not in entity:
                    entity['source'] = '未知'
                processed_entities.append(entity)
            else:
                # 如果没有特定结构，则尝试根据字段推断类型
                if any(key.startswith('algorithm_') for key in entity.keys()):
                    entity['entity_type'] = 'Algorithm'
                elif any(key.startswith('dataset_') for key in entity.keys()):
                    entity['entity_type'] = 'Dataset'
                elif any(key.startswith('metric_') for key in entity.keys()):
                    entity['entity_type'] = 'Metric'
                # 确保source字段存在
                if 'source' not in entity:
                    entity['source'] = '未知'
                processed_entities.append(entity)
        
        logging.info(f"获取到实体总数: {len(processed_entities)}")
        
        # 记录各类型实体数量
        algo_count = sum(1 for e in processed_entities if e.get('entity_type') == 'Algorithm')
        dataset_count = sum(1 for e in processed_entities if e.get('entity_type') == 'Dataset')
        metric_count = sum(1 for e in processed_entities if e.get('entity_type') == 'Metric')
        
        # 按来源分类统计
        review_count = 0
        citation_count = 0
        unknown_count = 0
        
        # 按来源分组
        entities_by_source = {
            '综述': [],
            '引文': [],
            '未知': []
        }
        
        # 正确处理嵌套结构中的source字段
        for e in processed_entities:
            # 从嵌套结构中获取source字段
            source = '未知'
            if 'algorithm_entity' in e:
                source = e['algorithm_entity'].get('source', '未知')
            elif 'dataset_entity' in e:
                source = e['dataset_entity'].get('source', '未知')
            elif 'metric_entity' in e:
                source = e['metric_entity'].get('source', '未知')
            else:
                source = e.get('source', '未知')
                
            # 更新计数
            if source == '综述':
                review_count += 1
                entities_by_source['综述'].append(e)
            elif source == '引文':
                citation_count += 1
                entities_by_source['引文'].append(e)
            else:
                unknown_count += 1
                entities_by_source['未知'].append(e)
        
        logging.info(f"算法实体: {algo_count}, 数据集实体: {dataset_count}, 评价指标实体: {metric_count}")
        logging.info(f"综述来源: {review_count}, 引文来源: {citation_count}, 未知来源: {unknown_count}")
        
        return jsonify({
            'success': True,
            'count': len(processed_entities),
            'entities': processed_entities,
            'by_type': {
                'algorithm': algo_count,
                'dataset': dataset_count,
                'metric': metric_count
            },
            'by_source': {
                'review': review_count,
                'citation': citation_count,
                'unknown': unknown_count
            },
            'groups': entities_by_source
        })
        
    except Exception as e:
        logging.error(f"获取实体列表出错: {str(e)}")
        logging.error(traceback.format_exc())
        return jsonify({
            'success': False,
            'message': f'获取实体列表失败: {str(e)}'
        }), 500

@combined_api.route('/entities/<entity_id>', methods=['GET'])
def get_entity_by_id(entity_id):
    """通过ID获取特定实体信息（与/entity/<entity_id>功能相同，为兼容前端路径）"""
    try:
        logging.info(f"请求获取实体详情: {entity_id}")
        
        # 从数据库获取实体
        entity = db_manager.get_entity_by_id(entity_id)
        
        if entity:
            # 直接返回完整数据，不做格式处理，保留所有字段
            return jsonify({
                'success': True,
                'data': entity
            })
        else:
            logging.warning(f"未找到实体: {entity_id}")
            return jsonify({
                'success': False,
                'message': f'未找到实体: {entity_id}'
            }), 404
    
    except Exception as e:
        logging.error(f"获取实体详情时出错: {str(e)}")
        logging.error(traceback.format_exc())
        return jsonify({
            'success': False,
            'message': f'获取实体详情时出错: {str(e)}'
        }), 500

@combined_api.route('/entity/<entity_id>', methods=['GET'])
def get_entity(entity_id):
    """获取指定ID的实体详情"""
    try:
        logging.info(f"请求获取实体详情: {entity_id}")
        
        # 从数据库获取实体
        entity = db_manager.get_entity_by_id(entity_id)
        
        if entity:
            # 直接返回完整数据，不做格式处理，保留所有字段
            return jsonify({
                'success': True,
                'data': entity
            })
        else:
            logging.warning(f"未找到实体: {entity_id}")
            return jsonify({
                'success': False,
                'message': f'未找到实体: {entity_id}'
            }), 404
    
    except Exception as e:
        logging.error(f"获取实体详情时出错: {str(e)}")
        import traceback
        logging.error(traceback.format_exc())
        return jsonify({
            'success': False,
            'message': f'获取实体详情时出错: {str(e)}'
        }), 500

@combined_api.route('/entity/search', methods=['GET'])
def search_entities():
    """搜索实体，支持按名称、类型和其他属性搜索"""
    try:
        # 获取查询参数
        name = request.args.get('name', '')
        entity_type = request.args.get('type', '')
        
        logging.info(f"搜索实体，名称关键词: {name}, 类型: {entity_type}")
        
        # 根据条件搜索实体
        entities = db_manager.search_entities(name, entity_type)
        
        # 规范化实体格式
        processed_entities = []
        for entity in entities:
            if 'algorithm_entity' in entity:
                algorithm = entity['algorithm_entity']
                algorithm['entity_type'] = 'Algorithm'
                processed_entities.append(algorithm)
            elif 'dataset_entity' in entity:
                dataset = entity['dataset_entity']
                dataset['entity_type'] = 'Dataset'
                processed_entities.append(dataset)
            elif 'metric_entity' in entity:
                metric = entity['metric_entity']
                metric['entity_type'] = 'Metric'
                processed_entities.append(metric)
            elif 'entity_type' in entity:
                processed_entities.append(entity)
        
        return jsonify({
            'success': True,
            'entities': processed_entities,
            'count': len(processed_entities)
        })
        
    except Exception as e:
        logging.error(f"搜索实体时出错: {str(e)}")
        import traceback
        logging.error(traceback.format_exc())
        return jsonify({
            'success': False,
            'message': f'搜索实体时出错: {str(e)}'
        }), 500

# =================== 关系相关API ===================

@combined_api.route('/relations', methods=['GET'])
def get_relations():
    """获取所有关系"""
    try:
        relation_type = request.args.get('type', None)
        
        if relation_type:
            relations = db_manager.get_relations_by_type(relation_type)
        else:
            relations = db_manager.get_all_relations()
        
        # 确保每个关系都有source字段
        for relation in relations:
            if 'source' not in relation or not relation['source']:
                relation['source'] = '未知'
        
        # 按关系类型统计
        relation_types = {}
        for relation in relations:
            relation_type = relation.get('relation_type', 'unknown')
            if relation_type not in relation_types:
                relation_types[relation_type] = 0
            relation_types[relation_type] += 1
            
        # 按来源统计
        review_count = sum(1 for r in relations if r.get('source') == '综述')
        citation_count = sum(1 for r in relations if r.get('source') == '引文')
        unknown_count = sum(1 for r in relations if r.get('source') == '未知')
        
        logging.info(f"获取到关系总数: {len(relations)}")
        logging.info(f"综述来源: {review_count}, 引文来源: {citation_count}, 未知来源: {unknown_count}")
        
        # 按来源分组
        relations_by_source = {
            '综述': [r for r in relations if r.get('source') == '综述'],
            '引文': [r for r in relations if r.get('source') == '引文'],
            '未知': [r for r in relations if r.get('source') == '未知']
        }
        
        return jsonify({
            'success': True,
            'count': len(relations),
            'relations': relations,
            'by_type': relation_types,
            'by_source': {
                'review': review_count,
                'citation': citation_count,
                'unknown': unknown_count
            },
            'groups': relations_by_source
        })
        
    except Exception as e:
        logging.error(f"获取关系列表出错: {str(e)}")
        logging.error(traceback.format_exc())
        return jsonify({
            'success': False,
            'message': f'获取关系列表失败: {str(e)}'
        }), 500

@combined_api.route('/relation/<relation_id>', methods=['GET'])
def get_relation(relation_id):
    """获取指定ID的关系详情"""
    try:
        relation = db_manager.get_relation_by_id(relation_id)
        
        if relation:
            # 直接返回完整数据，不做格式处理，保留所有字段
            return jsonify({
                'success': True,
                'data': relation
            })
        else:
            return jsonify({
                'success': False,
                'message': f'未找到关系: {relation_id}'
            }), 404
    
    except Exception as e:
        logging.error(f"获取关系详情时出错: {str(e)}")
        logging.error(traceback.format_exc())
        return jsonify({
            'success': False,
            'message': f'获取关系详情时出错: {str(e)}'
        }), 500


# =================== 数据导入相关API ===================

@combined_api.route('/import/table', methods=['POST'])
def import_table_data():
    """从表格数据导入"""
    try:
        data = request.json
        
        if not data or 'entities' not in data or 'relations' not in data:
            return jsonify({'success': False, 'message': '无效的数据格式'}), 400
        
        entity_data = data['entities']
        relation_data = data['relations']
        
        # 转换表格数据为标准格式
        entities, relations = transform_table_data_to_entities(entity_data, relation_data)
        
        # 保存到数据库
        db_manager.store_entities(entities)
        db_manager.store_relations(relations)
        
        # 保存为JSON文件
        save_data_to_json(entities, relations)
        
        # 确保图数据目录存在
        os.makedirs(Config.GRAPH_DATA_DIR, exist_ok=True)
        
        # 构建图并可视化
        graph = build_knowledge_graph(entities, relations)
        graph_image_path = os.path.join(Config.GRAPH_DATA_DIR, 'graph.png')
        visualize_graph(graph, output_path=graph_image_path)
        
        # 导出图数据供前端使用
        graph_json_path = os.path.join(Config.GRAPH_DATA_DIR, 'graph.json')
        export_graph_to_json(graph, graph_json_path)
        
        return jsonify({
            'success': True, 
            'message': '表格数据导入成功',
            'entities_count': len(entities),
            'relations_count': len(relations)
        })
        
    except Exception as e:
        logging.error(f"导入表格数据时出错: {str(e)}")
        logging.error(traceback.format_exc())
        return jsonify({
            'success': False,
            'message': f'导入表格数据时出错: {str(e)}'
        }), 500

# =================== 文档相关API ===================

@combined_api.route('/document/analyze', methods=['POST'])
def analyze_document():
    """分析文档并提取实体和关系"""
    try:
        if 'file' not in request.files:
            return jsonify({'success': False, 'message': '没有上传文件'}), 400
            
        file = request.files['file']
        if file.filename == '':
            return jsonify({'success': False, 'message': '没有选择文件'}), 400
            
        # 保存文件并处理
        # 创建唯一任务ID
        task_id = str(uuid.uuid4())
        
        # 保存文件
        if not os.path.exists(Config.UPLOAD_DIR):
            os.makedirs(Config.UPLOAD_DIR)
            
        file_path = os.path.join(Config.UPLOAD_DIR, f"{task_id}_{file.filename}")
        file.save(file_path)
        
        # 启动异步处理任务
        from app.modules.data_extraction import process_papers_and_extract_data
        import threading
        
        def process_in_background():
            try:
                process_papers_and_extract_data(file_path, task_id)
            except Exception as e:
                logging.error(f"后台处理文档时出错: {str(e)}")
                logging.error(traceback.format_exc())
        
        thread = threading.Thread(target=process_in_background)
        thread.daemon = True
        thread.start()
        
        return jsonify({
            'success': True,
            'message': '文档分析任务已启动',
            'task_id': task_id
        })
        
    except Exception as e:
        logging.error(f"文档分析出错: {str(e)}")
        logging.error(traceback.format_exc())
        return jsonify({
            'success': False,
            'message': f'文档分析出错: {str(e)}'
        }), 500

# =================== 辅助功能API ===================

@combined_api.route('/clear/all', methods=['POST'])
def clear_all_data():
    """清除所有缓存和数据库数据"""
    try:
        logging.warning("收到清除所有数据的请求")
        
        # 清除数据库中的数据
        db_manager.clear_all_data()
        logging.warning("已清除数据库中的所有数据")
        
        # 清除缓存文件
        import shutil
        
        # 清除上传文件夹
        if os.path.exists(Config.UPLOAD_FOLDER):
            shutil.rmtree(Config.UPLOAD_FOLDER)
            os.makedirs(Config.UPLOAD_FOLDER, exist_ok=True)
            logging.warning(f"已清除上传文件夹: {Config.UPLOAD_FOLDER}")
            
        # 清除图数据文件夹
        if os.path.exists(Config.GRAPH_DATA_DIR):
            shutil.rmtree(Config.GRAPH_DATA_DIR)
            os.makedirs(Config.GRAPH_DATA_DIR, exist_ok=True)
            logging.warning(f"已清除图数据文件夹: {Config.GRAPH_DATA_DIR}")
            
        # 清除缓存目录
        if os.path.exists(Config.CACHE_DIR):
            shutil.rmtree(Config.CACHE_DIR)
            os.makedirs(Config.CACHE_DIR, exist_ok=True)
            logging.warning(f"已清除缓存文件夹: {Config.CACHE_DIR}")
        
        return jsonify({
            "success": True,
            "message": "已成功清除所有缓存和数据库数据"
        })
    except Exception as e:
        logging.error(f"清除数据时出错: {str(e)}")
        logging.error(traceback.format_exc())
        return jsonify({
            "success": False,
            "message": f"清除数据时出错: {str(e)}",
            "traceback": traceback.format_exc()
        }), 500

@combined_api.route('/clear/cache', methods=['POST'])
def clear_cache_keep_file_ids():
    """清除除file_ids外的所有缓存和数据库数据"""
    try:
        logging.warning("收到清除缓存的请求（保留file_ids）")
        
        # 清除数据库中的数据
        db_manager.clear_all_data()
        logging.warning("已清除数据库中的所有数据")
        
        # 清除缓存文件
        import shutil
        
        # 清除上传文件夹
        if os.path.exists(Config.UPLOAD_FOLDER):
            shutil.rmtree(Config.UPLOAD_FOLDER)
            os.makedirs(Config.UPLOAD_FOLDER, exist_ok=True)
            logging.warning(f"已清除上传文件夹: {Config.UPLOAD_FOLDER}")
            
        # 清除图数据文件夹
        if os.path.exists(Config.GRAPH_DATA_DIR):
            shutil.rmtree(Config.GRAPH_DATA_DIR)
            os.makedirs(Config.GRAPH_DATA_DIR, exist_ok=True)
            logging.warning(f"已清除图数据文件夹: {Config.GRAPH_DATA_DIR}")
            
        # 清除缓存目录中除file_ids外的所有内容
        if os.path.exists(Config.CACHE_DIR):
            file_ids_dir = os.path.join(Config.CACHE_DIR, "file_ids")
            
            # 备份file_ids目录（如果存在）
            temp_file_ids_backup = None
            if os.path.exists(file_ids_dir):
                import tempfile
                temp_file_ids_backup = tempfile.mkdtemp()
                shutil.copytree(file_ids_dir, os.path.join(temp_file_ids_backup, "file_ids"))
                logging.info(f"已备份file_ids目录到临时位置: {temp_file_ids_backup}")
            
            # 清除整个缓存目录
            shutil.rmtree(Config.CACHE_DIR)
            os.makedirs(Config.CACHE_DIR, exist_ok=True)
            logging.warning(f"已清除缓存文件夹: {Config.CACHE_DIR}")
            
            # 恢复file_ids目录
            if temp_file_ids_backup:
                if not os.path.exists(file_ids_dir):
                    os.makedirs(file_ids_dir, exist_ok=True)
                shutil.copytree(os.path.join(temp_file_ids_backup, "file_ids"), file_ids_dir, dirs_exist_ok=True)
                # 清除临时备份
                shutil.rmtree(temp_file_ids_backup)
                logging.info(f"已恢复file_ids目录: {file_ids_dir}")
        
        return jsonify({
            "success": True,
            "message": "已成功清除缓存和数据库数据，保留file_ids"
        })
    except Exception as e:
        logging.error(f"清除缓存时出错: {str(e)}")
        logging.error(traceback.format_exc())
        return jsonify({
            "success": False,
            "message": f"清除缓存时出错: {str(e)}",
            "traceback": traceback.format_exc()
        }), 500

# =================== 比较分析相关API ===================

@combined_api.route('/comparison/reextract_relations', methods=['POST'])
def reextract_relations():
    """基于已有任务的实体重新提取演化关系"""
    try:
        data = request.get_json()

        # 验证必需的参数
        required_fields = ['source_task_id', 'model_name', 'review_path', 'citation_paths']
        for field in required_fields:
            if field not in data:
                return jsonify({'error': f'缺少必需的参数: {field}'}), 400

        source_task_id = data['source_task_id']
        model_name = data['model_name']
        review_path = data['review_path']  # 新增：综述PDF路径
        citation_paths = data['citation_paths']  # 新增：引文PDF路径列表

        # 验证综述PDF文件是否存在
        if not review_path or not os.path.exists(review_path):
            return jsonify({'error': '综述PDF文件不存在或路径无效'}), 400

        # 验证引文PDF文件是否存在
        if not citation_paths or not isinstance(citation_paths, list):
            return jsonify({'error': '必须提供引文PDF文件路径列表'}), 400

        # 检查PDF文件是否存在
        missing_files = []
        for pdf_path in citation_paths:
            if not os.path.exists(pdf_path):
                missing_files.append(pdf_path)

        if missing_files:
            return jsonify({'error': f'以下PDF文件不存在: {missing_files}'}), 400

        # 验证源任务是否存在
        source_task = db_manager.get_processing_status(source_task_id)
        if not source_task:
            return jsonify({'error': f'源任务 {source_task_id} 不存在'}), 404

        # 检查源任务是否有实体数据
        entities = db_manager.get_entities_by_task(source_task_id)
        if not entities:
            return jsonify({'error': f'源任务 {source_task_id} 中没有实体数据'}), 400

        # 生成新任务ID
        new_task_id = str(uuid.uuid4())

        # 创建新任务记录
        task_name = f"重新提取关系 - 基于任务 {source_task_id[:8]}"
        db_manager.create_processing_task(new_task_id, task_name)

        # 更新任务状态
        db_manager.update_processing_status(
            task_id=new_task_id,
            status='处理中',
            current_stage='初始化',
            progress=0.0,
            message='正在初始化重新提取关系任务'
        )

        # 启动后台任务
        thread = threading.Thread(
            target=reextract_relations_from_existing_task,
            args=(source_task_id, new_task_id, model_name, review_path, citation_paths)
        )
        thread.daemon = True
        thread.start()

        return jsonify({
            'task_id': new_task_id,
            'source_task_id': source_task_id,
            'review_file': os.path.basename(review_path),
            'citation_files_count': len(citation_paths),
            'message': '重新提取关系任务已启动'
        })

    except Exception as e:
        logging.error(f"启动重新提取关系任务时出错: {str(e)}")
        return jsonify({'error': str(e)}), 500

@combined_api.route('/comparison/upload_and_reextract', methods=['POST'])
def upload_and_reextract():
    """上传PDF文件并启动重新提取关系任务"""
    try:
        # 获取表单数据
        source_task_id = request.form.get('source_task_id')
        model_name = request.form.get('model_name')

        if not source_task_id or not model_name:
            return jsonify({'error': '缺少必需的参数: source_task_id 或 model_name'}), 400

        # 获取上传的文件
        review_file = request.files.get('review_file')
        citation_files = request.files.getlist('citation_files')

        if not review_file:
            return jsonify({'error': '没有上传综述PDF文件'}), 400

        if not citation_files:
            return jsonify({'error': '没有上传引文PDF文件'}), 400

        # 验证源任务是否存在
        source_task = db_manager.get_processing_status(source_task_id)
        if not source_task:
            return jsonify({'error': f'源任务 {source_task_id} 不存在'}), 404

        # 检查源任务是否有实体数据
        entities = db_manager.get_entities_by_task(source_task_id)
        if not entities:
            return jsonify({'error': f'源任务 {source_task_id} 中没有实体数据'}), 400

        # 保存上传的文件
        import tempfile
        import shutil

        temp_dir = tempfile.mkdtemp(prefix='reextract_')
        review_path = None
        citation_paths = []

        try:
            # 保存综述文件
            if review_file.filename and review_file.filename.endswith('.pdf'):
                review_filename = f"review_{review_file.filename}"
                review_path = os.path.join(temp_dir, review_filename)
                review_file.save(review_path)
                logging.info(f"保存综述文件: {review_path}")
            else:
                shutil.rmtree(temp_dir)
                return jsonify({'error': '综述文件必须是PDF格式'}), 400

            # 保存引文文件
            for i, file in enumerate(citation_files):
                if file.filename and file.filename.endswith('.pdf'):
                    # 生成安全的文件名
                    safe_filename = f"citation_{i}_{file.filename}"
                    file_path = os.path.join(temp_dir, safe_filename)
                    file.save(file_path)
                    citation_paths.append(file_path)
                    logging.info(f"保存引文文件: {file_path}")

            if not citation_paths:
                shutil.rmtree(temp_dir)
                return jsonify({'error': '没有有效的引文PDF文件'}), 400

            # 生成新任务ID
            new_task_id = str(uuid.uuid4())

            # 创建新任务记录
            task_name = f"重新提取关系 - 基于任务 {source_task_id[:8]} (1个综述+{len(citation_paths)}个引文)"
            db_manager.create_processing_task(new_task_id, task_name)

            # 更新任务状态
            db_manager.update_processing_status(
                task_id=new_task_id,
                status='处理中',
                current_stage='初始化',
                progress=0.0,
                message=f'正在初始化重新提取关系任务，已上传1个综述PDF和{len(citation_paths)}个引文PDF'
            )

            # 启动后台任务
            thread = threading.Thread(
                target=reextract_relations_from_existing_task,
                args=(source_task_id, new_task_id, model_name, review_path, citation_paths)
            )
            thread.daemon = True
            thread.start()

            return jsonify({
                'task_id': new_task_id,
                'source_task_id': source_task_id,
                'review_file': os.path.basename(review_path),
                'citation_files_count': len(citation_paths),
                'temp_dir': temp_dir,
                'message': '文件上传成功，重新提取关系任务已启动'
            })

        except Exception as e:
            # 清理临时文件
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
            raise e

    except Exception as e:
        logging.error(f"上传文件并启动重新提取关系任务时出错: {str(e)}")
        return jsonify({'error': str(e)}), 500

@combined_api.route('/comparison/start', methods=['POST'])
def start_comparison():
    """启动比较分析任务"""
    try:
        # 检查是否有文件上传
        if 'review_paper' not in request.files:
            return jsonify({'status': 'error', 'message': '未提供综述论文'}), 400
        
        review_file = request.files['review_paper']
        if review_file.filename == '':
            return jsonify({'status': 'error', 'message': '未选择综述论文文件'}), 400
        
        # 检查文件类型
        if not review_file.filename.lower().endswith('.pdf'):
            return jsonify({'status': 'error', 'message': '综述论文必须是PDF格式'}), 400
        
        # 获取模型名称
        model_name = request.form.get('model', 'qwen-long')
        logging.info(f"使用模型: {model_name}")
        
        # 创建临时文件列表，用于后续清理
        temp_files = []
        
        # 保存综述论文
        review_filename = secure_filename(review_file.filename)
        review_path = os.path.join(Config.UPLOAD_DIR, review_filename)
        review_file.save(review_path)
        temp_files.append(review_path)
        
        # 处理引用文献
        citation_paths = []
        if 'citation_papers' in request.files:
            citation_files = request.files.getlist('citation_papers')
            for citation_file in citation_files:
                if citation_file and citation_file.filename != '' and citation_file.filename.lower().endswith('.pdf'):
                    citation_filename = secure_filename(citation_file.filename)
                    citation_path = os.path.join(Config.UPLOAD_DIR, citation_filename)
                    citation_file.save(citation_path)
                    citation_paths.append(citation_path)
                    temp_files.append(citation_path)
        
        # 生成任务ID
        task_id = str(uuid.uuid4())
                # 创建处理任务
        task_name = f"比较分析任务 - review_filename"
        db_manager.create_processing_task(
            task_name=task_name,
            task_id=task_id
        )
        

        
        # 启动后台处理线程
        thread = threading.Thread(
            target=run_comparison_task,
            args=(task_id, review_path, citation_paths, model_name, temp_files)
        )
        thread.daemon = True
        thread.start()
        
        # 返回任务ID和初始状态
        return jsonify({
            'status': 'success',
            'message': '比较分析任务已启动',
            'task_id': task_id
        })
        
    except Exception as e:
        logging.error(f"启动比较分析任务时出错: {str(e)}")
        logging.error(traceback.format_exc())
        
        # 清理临时文件
        for file_path in temp_files if 'temp_files' in locals() else []:
            try:
                if os.path.exists(file_path):
                    os.remove(file_path)
                    logging.info(f"已删除临时文件: {file_path}")
            except Exception as file_err:
                logging.error(f"删除临时文件出错: {str(file_err)}")
                
        return jsonify({'status': 'error', 'message': f'处理请求时出错: {str(e)}'}), 500

def run_comparison_task(task_id, review_path, citation_paths, model_name, temp_files):
    """在后台运行比较分析任务"""
    try:
        logging.info(f"启动比较分析任务 {task_id}，使用模型: {model_name}")
        
        # 设置环境变量指定模型
        os.environ['AI_MODEL'] = model_name
        
        # 更新处理状态
        db_manager.update_processing_status(
            task_id=task_id,
            current_stage='开始处理',
            progress=0.1,
            message=f'开始处理综述文章和 {len(citation_paths)} 篇引用文献'
        )
        
        # 导入所需模块
        from app.modules.data_extraction import calculate_comparison_metrics
        
        # 更新状态：提取实体
        db_manager.update_processing_status(
            task_id=task_id,
            current_stage='提取实体',
            progress=0.2,
            message='正在从综述论文中提取实体'
        )
        
        # 处理综述论文，提取实体，使用相同的任务ID
        review_entities, _ = extract_entities_from_review(review_path, task_id, model_name=model_name)
        
        # 确保所有综述实体都有正确的来源标记
        for entity in review_entities:
            # 标记外层source字段
            if 'source' not in entity or not entity['source'] or entity['source'] == '未知':
                entity['source'] = '综述'
            
            # 标记内层source字段
            if 'algorithm_entity' in entity and isinstance(entity['algorithm_entity'], dict):
                entity['algorithm_entity']['source'] = '综述'
            elif 'dataset_entity' in entity and isinstance(entity['dataset_entity'], dict):
                entity['dataset_entity']['source'] = '综述'
            elif 'metric_entity' in entity and isinstance(entity['metric_entity'], dict):
                entity['metric_entity']['source'] = '综述'
        
        # 更新状态：处理引用文献
        db_manager.update_processing_status(
            task_id=task_id,
            current_stage='处理引用文献',
            progress=0.4,
            message='正在从引用文献中提取实体'
        )
        
        # 处理引用文献，提取实体，使用相同的任务ID
        citation_entities, _ = extract_entities_from_citations(
            review_entities=review_entities,
            citation_paths=citation_paths,
            batch_size=15, 
            task_id=task_id,
            model_name=model_name)
        
        # 确保所有引文实体都有正确的来源标记
        for entity in citation_entities:
            # 标记外层source字段
            if 'source' not in entity or not entity['source'] or entity['source'] == '未知':
                entity['source'] = '引文'
            
            # 标记内层source字段
            if 'algorithm_entity' in entity and isinstance(entity['algorithm_entity'], dict):
                entity['algorithm_entity']['source'] = '引文'
            elif 'dataset_entity' in entity and isinstance(entity['dataset_entity'], dict):
                entity['dataset_entity']['source'] = '引文'
            elif 'metric_entity' in entity and isinstance(entity['metric_entity'], dict):
                entity['metric_entity']['source'] = '引文'
        
        # 更新状态：提取关系
        db_manager.update_processing_status(
            task_id=task_id,
            current_stage='提取关系',
            progress=0.6,
            message='正在分析实体之间的演化关系'
        )
        
        # 导入关系提取函数
        from app.modules.agents import extract_evolution_relations
        
        # 分别提取综述关系和引文关系
        
        # 1. 提取综述关系（只使用综述PDF和综述实体）
        db_manager.update_processing_status(
            task_id=task_id,
            current_stage='提取关系',
            progress=0.65,
            message='正在从综述中提取算法演化关系'
        )
        review_evolution_relations = extract_evolution_relations(
            entities=review_entities, 
            pdf_paths=[review_path],  # 只传入综述PDF
            task_id=task_id,
            previous_relations=None,
            batch_size=15,
            model_name=model_name
        )
        
        # 确保所有综述关系都有正确的来源标记
        for relation in review_evolution_relations:
            relation['source'] = '综述'
        
        # 2. 提取引文关系（只使用引文PDF和引文实体）
        if citation_entities and citation_paths:
            db_manager.update_processing_status(
                task_id=task_id,
                current_stage='提取关系',
                progress=0.75,
                message='正在从引文中提取算法演化关系'
            )
            citation_evolution_relations = extract_evolution_relations(
                entities=citation_entities, 
                pdf_paths=citation_paths,  # 只传入引文PDF
                review_relations=review_evolution_relations,
                task_id=task_id,
                max_attempts=5,
                previous_relations=None,
                batch_size=15, 
                model_name=model_name
            )
            
            # 确保所有引文关系都有正确的来源标记
            for relation in citation_evolution_relations:
                relation['source'] = '引文'
        else:
            citation_evolution_relations = []
        
        # 合并两种来源的演化关系
        evolution_relations = review_evolution_relations + citation_evolution_relations
        
        # 合并所有实体（保留原始代码）
        all_entities = review_entities + citation_entities
        
        # 更新状态：计算指标
        db_manager.update_processing_status(
            task_id=task_id,
            current_stage='计算指标',
            progress=0.8,
             message='正在计算比较指标'
        )
        
        # 使用calculate_comparison_metrics函数计算比较指标
        metrics = calculate_comparison_metrics(review_entities, citation_entities, evolution_relations)
        
        # 保存实体和关系到数据库，使用相同的任务ID
        save_entities_and_relations(all_entities, evolution_relations, task_id)
        
        # 构建结果
        result_data = {
            'review_entities_count': len(review_entities),
            'citation_entities_count': len(citation_entities),
            'relations_count': len(evolution_relations),
            'metrics': metrics
        }
        
        # 更新任务状态为完成
        db_manager.update_processing_status(
            task_id=task_id,
            status='已完成',
            current_stage='任务完成',
            progress=1.0,
            message=json.dumps(result_data, ensure_ascii=False),
            completed=True
        )
        
        logging.info(f"比较分析任务 {task_id} 已完成")
        
    except Exception as e:
        logging.error(f"运行比较分析任务 {task_id} 时出错: {str(e)}")
        import traceback
        logging.error(traceback.format_exc())
        
        # 更新处理状态为错误
        db_manager.update_processing_status(
            task_id=task_id,
            status='错误',
            current_stage='处理出错',
            progress=0,
            message=f'处理任务时出错: {str(e)}'
        )
    
    finally:
        # 无论成功还是失败，都清理临时文件
        logging.info(f"正在清理任务 {task_id} 的临时文件")
        for file_path in temp_files:
            try:
                if os.path.exists(file_path):
                    os.remove(file_path)
                    logging.info(f"已删除临时文件: {file_path}")
            except Exception as file_err:
                logging.error(f"删除临时文件出错: {str(file_err)}")

def reextract_relations_from_existing_task(source_task_id, new_task_id, model_name, review_path, citation_paths):
    """基于已有任务的实体重新提取演化关系"""
    try:
        logging.info(f"开始基于任务 {source_task_id} 的实体重新提取关系，新任务ID: {new_task_id}")
        logging.info(f"综述PDF文件: {os.path.basename(review_path)}")
        logging.info(f"引文PDF文件: {len(citation_paths)} 个文件")

        # 设置环境变量指定模型
        os.environ['AI_MODEL'] = model_name

        # 更新处理状态
        db_manager.update_processing_status(
            task_id=new_task_id,
            current_stage='获取已有实体',
            progress=0.1,
            message=f'正在从任务 {source_task_id} 获取已提取的实体'
        )

        # 从数据库获取已有任务的实体
        all_entities = db_manager.get_entities_by_task(source_task_id)
        if not all_entities:
            raise ValueError(f"任务 {source_task_id} 中没有找到实体数据")

        # 分离综述实体和引文实体
        review_entities = []
        citation_entities = []

        for entity in all_entities:
            # 检查实体的来源，支持多种实体类型
            source = None

            # 检查外层source字段
            if 'source' in entity and entity['source']:
                source = entity['source']
            # 检查algorithm_entity中的source
            elif 'algorithm_entity' in entity and entity['algorithm_entity'] and 'source' in entity['algorithm_entity']:
                source = entity['algorithm_entity']['source']
            # 检查dataset_entity中的source
            elif 'dataset_entity' in entity and entity['dataset_entity'] and 'source' in entity['dataset_entity']:
                source = entity['dataset_entity']['source']
            # 检查metric_entity中的source
            elif 'metric_entity' in entity and entity['metric_entity'] and 'source' in entity['metric_entity']:
                source = entity['metric_entity']['source']

            # 根据来源分类
            if source == '综述':
                review_entities.append(entity)
            elif source == '引文':
                citation_entities.append(entity)

        logging.info(f"获取到 {len(review_entities)} 个综述实体，{len(citation_entities)} 个引文实体")

        # 更新状态：重新提取关系
        db_manager.update_processing_status(
            task_id=new_task_id,
            current_stage='重新提取关系',
            progress=0.3,
            message='正在重新分析实体之间的演化关系'
        )

        # 导入关系提取函数
        from app.modules.agents import extract_evolution_relations

        # 1. 重新提取综述关系
        db_manager.update_processing_status(
            task_id=new_task_id,
            current_stage='提取综述关系',
            progress=0.4,
            message='正在重新从综述实体中提取算法演化关系'
        )

        review_evolution_relations = extract_evolution_relations(
            entities=review_entities,
            pdf_paths=[review_path],  # 使用提供的综述PDF路径
            task_id=new_task_id,
            previous_relations=None,
            batch_size=15,
            model_name=model_name
        )

        # 确保所有综述关系都有正确的来源标记
        for relation in review_evolution_relations:
            relation['source'] = '综述'

        # 2. 重新提取引文关系
        if citation_entities:
            db_manager.update_processing_status(
                task_id=new_task_id,
                current_stage='提取引文关系',
                progress=0.6,
                message='正在重新从引文PDF和实体中提取算法演化关系'
            )

            citation_evolution_relations = extract_evolution_relations(
                entities=citation_entities,
                pdf_paths=citation_paths,  # 使用提供的引文PDF路径
                review_relations=review_evolution_relations,
                task_id=new_task_id,
                max_attempts=6,
                previous_relations=None,
                batch_size=20,
                model_name=model_name
            )

            # 确保所有引文关系都有正确的来源标记
            for relation in citation_evolution_relations:
                relation['source'] = '引文'
        else:
            citation_evolution_relations = []

        # 合并两种来源的演化关系
        evolution_relations = review_evolution_relations + citation_evolution_relations

        # 更新状态：计算指标
        db_manager.update_processing_status(
            task_id=new_task_id,
            current_stage='计算指标',
            progress=0.8,
            message='正在计算比较指标'
        )

        # 导入所需模块
        from app.modules.data_extraction import calculate_comparison_metrics

        # 使用calculate_comparison_metrics函数计算比较指标
        metrics = calculate_comparison_metrics(review_entities, citation_entities, evolution_relations)

        # 保存实体和关系到数据库，使用新任务ID
        save_entities_and_relations(all_entities, evolution_relations, new_task_id)

        # 构建结果
        result_data = {
            'source_task_id': source_task_id,
            'review_entities_count': len(review_entities),
            'citation_entities_count': len(citation_entities),
            'relations_count': len(evolution_relations),
            'metrics': metrics
        }

        # 更新任务状态为完成
        db_manager.update_processing_status(
            task_id=new_task_id,
            status='已完成',
            current_stage='任务完成',
            progress=1.0,
            message=json.dumps(result_data, ensure_ascii=False),
            completed=True
        )

        logging.info(f"基于任务 {source_task_id} 的关系重新提取任务 {new_task_id} 已完成")
        return result_data

    except Exception as e:
        logging.error(f"重新提取关系任务 {new_task_id} 时出错: {str(e)}")
        import traceback
        logging.error(traceback.format_exc())

        # 更新处理状态为错误
        db_manager.update_processing_status(
            task_id=new_task_id,
            status='错误',
            current_stage='处理出错',
            progress=0,
            message=f'重新提取关系时出错: {str(e)}'
        )
        raise

def extract_entities_from_review(review_path, task_id, model_name):
    """从综述论文中提取实体和关系"""
    from app.modules.agents import extract_paper_entities

    # 检查文件是否存在
    if not os.path.exists(review_path):
        logging.error(f"综述论文文件不存在: {review_path}")
        return [], []

    logging.info(f"开始处理综述论文: {os.path.basename(review_path)}")

    # 从综述论文中提取实体，使用主任务ID（采用GPT/DeepSeek的PDF提取方法）
    entities, is_complete = extract_paper_entities(
        review_entities=None,
        pdf_paths=review_path,
        model_name=model_name if model_name else "gpt-4.1-mini",  # 默认使用GPT
        task_id=task_id  # 使用主任务ID
    )

    # 确保所有实体都有正确的来源标记
    for entity in entities:
        # 标记外层source字段
        if 'source' not in entity or not entity['source'] or entity['source'] == '未知':
            entity['source'] = '综述'

        # 标记内层source字段
        if 'algorithm_entity' in entity and isinstance(entity['algorithm_entity'], dict):
            entity['algorithm_entity']['source'] = '综述'
        elif 'dataset_entity' in entity and isinstance(entity['dataset_entity'], dict):
            entity['dataset_entity']['source'] = '综述'
        elif 'metric_entity' in entity and isinstance(entity['metric_entity'], dict):
            entity['metric_entity']['source'] = '综述'

    logging.info(f"从综述论文中提取到 {len(entities)} 个实体")

    # 暂时返回空的关系列表，关系将在后续步骤中提取
    return entities, []

def extract_entities_from_citations(review_entities, citation_paths, batch_size, task_id, model_name):
    """从引用文献中提取实体和关系"""
    from app.modules.agents import extract_paper_entities
    
    all_entities = []
    
    if not citation_paths:
        logging.warning("没有提供引用文献")
        return all_entities, []
        
    valid_paths = [p for p in citation_paths if os.path.exists(p)]
    if len(valid_paths) < len(citation_paths):
        logging.warning(f"有 {len(citation_paths) - len(valid_paths)} 个引用文献文件不存在")
    
    if not valid_paths:
        logging.error("没有有效的引用文献文件")
        return all_entities, []
    
    # 提取实体，使用主任务ID（采用GPT/DeepSeek的PDF提取方法）
    all_entities, is_complete = extract_paper_entities(
        review_entities=review_entities,
        pdf_paths=valid_paths,
        max_attempts=6,
        batch_size=batch_size,
        model_name=model_name if model_name else "gpt-4.1-mini",  # 默认使用GPT
        task_id=task_id  # 使用主任务ID
    )
    
    # 确保所有实体都有正确的来源标记
    for entity in all_entities:
        # 标记外层source字段
        if 'source' not in entity or not entity['source'] or entity['source'] == '未知':
            entity['source'] = '引文'
        
        # 标记内层source字段
        if 'algorithm_entity' in entity and isinstance(entity['algorithm_entity'], dict):
            entity['algorithm_entity']['source'] = '引文'
        elif 'dataset_entity' in entity and isinstance(entity['dataset_entity'], dict):
            entity['dataset_entity']['source'] = '引文'
        elif 'metric_entity' in entity and isinstance(entity['metric_entity'], dict):
            entity['metric_entity']['source'] = '引文'

    # 暂时返回空的关系列表，关系将在后续步骤中提取
    return all_entities, []

def save_entities_and_relations(entities, relations, task_id=None):
    """保存实体和关系到数据库"""
    if not entities and not relations:
        logging.warning("没有实体和关系需要保存")
        return
        
    # 保存实体
    entity_count = 0
    for entity in entities:
        if entity:
            db_manager.store_algorithm_entity(entity, task_id)
            entity_count += 1
    
    # 保存关系
    relation_count = 0
    for relation in relations:
        if relation:
            db_manager.store_algorithm_relation(relation, task_id)
            relation_count += 1
    
    logging.info(f"已保存 {entity_count} 个实体和 {relation_count} 个关系到数据库, 任务ID: {task_id}")


@combined_api.route('/comparison/history', methods=['GET'])
def get_comparison_history():
    """获取比较分析的历史任务记录，支持论文分析任务选择"""
    try:
        # 使用db_manager的get_comparison_history方法获取任务记录
        tasks = db_manager.get_comparison_history(limit=200)

        # 为论文分析页面格式化任务数据
        formatted_tasks = []
        for task in tasks:
            # 获取原始任务名称
            original_task_name = task.get('task_name', '')
            task_id = task.get('task_id', '')

            # 生成任务名称
            task_name = generate_task_name(original_task_name, task_id, task)

            # 确保每个任务都有必要的字段
            formatted_task = {
                "task_id": task_id,
                "id": task_id,  # 兼容不同的ID字段名
                "name": task_name,
                "description": task.get('description', f"创建于 {task.get('start_time', 'Unknown')}"),
                "status": task.get('status', 'completed'),
                "created_at": task.get('created_at', ''),
                "entity_count": task.get('entity_count', 0),
                "relation_count": task.get('relation_count', 0),
                # 添加时间字段
                "start_time": task.get('start_time'),
                "update_time": task.get('update_time'),
                "end_time": task.get('end_time'),
                "task_name": task_name  # 使用任务名称
            }

            formatted_tasks.append(formatted_task)

        # 按创建时间倒序排列，最新的在前面
        formatted_tasks.sort(key=lambda x: x.get('start_time', ''), reverse=True)

        return jsonify({
            "success": True,
            "tasks": formatted_tasks,
            "total_count": len(formatted_tasks),
            "message": f"找到 {len(formatted_tasks)} 个可用任务"
        })

    except Exception as e:
        logging.error(f"获取比较分析历史记录时出错: {str(e)}")
        logging.error(traceback.format_exc())
        return jsonify({
            "success": False,
            "message": f"获取历史记录出错: {str(e)}",
            "tasks": []  # 返回空任务列表而不是完全失败
        }), 500

@combined_api.route('/comparison/<task_id>/entities', methods=['GET'])
def get_comparison_entities(task_id):
    """获取比较分析任务的实体数据"""
    try:
        # 检查任务ID格式
        if not task_id or len(task_id) < 10:
            return jsonify({
                "success": False,
                "message": f"无效的任务ID: {task_id}"
            }), 400
            
        # 获取任务关联的实体
        try:
            entities_data = db_manager.get_entities_by_task(task_id)
        except Exception as db_error:
            # 处理数据库查询错误
            logging.error(f"数据库查询任务 {task_id} 实体时出错: {str(db_error)}")
            logging.error(traceback.format_exc())
            
            # 返回空的实体列表，避免前端失败
            return jsonify({
                "success": True,
                "message": f"查询实体时出错: {str(db_error)}",
                "entities": [],
                "by_source": {},
                "count": {"total": 0}
            })
        
        if not entities_data or len(entities_data) == 0:
            return jsonify({
                "success": True,
                "message": "未找到实体数据",
                "entities": [],
                "by_source": {},
                "count": {"total": 0}
            })
        
        # 按来源组织实体
        entities_by_source = {}
        for entity in entities_data:
            # 从嵌套结构中获取source字段
            source = None
            if 'algorithm_entity' in entity:
                source = entity['algorithm_entity'].get('source', '未知')
            elif 'dataset_entity' in entity:
                source = entity['dataset_entity'].get('source', '未知')
            elif 'metric_entity' in entity:
                source = entity['metric_entity'].get('source', '未知')
            else:
                source = entity.get('source', '未知')
                
            if source not in entities_by_source:
                entities_by_source[source] = []
            entities_by_source[source].append(entity)
        
        # 统计各类型实体数量
        entity_counts = {
            "total": len(entities_data),
            "by_type": {}
        }
        
        for entity in entities_data:
            # 从嵌套结构中获取entity_type字段
            if 'algorithm_entity' in entity:
                entity_type = entity['algorithm_entity'].get('entity_type', 'Algorithm')
            elif 'dataset_entity' in entity:
                entity_type = entity['dataset_entity'].get('entity_type', 'Dataset')
            elif 'metric_entity' in entity:
                entity_type = entity['metric_entity'].get('entity_type', 'Metric')
            else:
                entity_type = entity.get('entity_type', '未知')
            
            # 统计不同类型的实体数量
            if entity_type not in entity_counts["by_type"]:
                entity_counts["by_type"][entity_type] = 0
            entity_counts["by_type"][entity_type] += 1
        
        return jsonify({
            "success": True,
            "entities": entities_data,
            "by_source": entities_by_source,
            "count": entity_counts
        })
        
    except Exception as e:
        logging.error(f"获取任务 {task_id} 的实体数据时出错: {str(e)}")
        logging.error(traceback.format_exc())
        return jsonify({
            "success": False,
            "message": f"获取实体数据出错: {str(e)}",
            "entities": []
        }), 500

@combined_api.route('/comparison/<task_id>/relations', methods=['GET'])
def get_comparison_relations(task_id):
    """获取比较分析任务的关系数据"""
    try:
        # 检查任务ID格式
        if not task_id or len(task_id) < 10:
            return jsonify({
                "success": False,
                "message": f"无效的任务ID: {task_id}"
            }), 400
            
        # 获取任务关联的关系
        try:
            relations_data = db_manager.get_relations_by_task(task_id)
        except Exception as db_error:
            # 处理数据库查询错误
            logging.error(f"数据库查询任务 {task_id} 关系时出错: {str(db_error)}")
            logging.error(traceback.format_exc())
            
            # 返回空的关系列表，避免前端失败
            return jsonify({
                "success": True,
                "message": f"查询关系时出错: {str(db_error)}",
                "relations": [],
                "by_source": {},
                "count": {"total": 0}
            })
        
        if not relations_data or len(relations_data) == 0:
            return jsonify({
                "success": True,
                "message": "未找到关系数据",
                "relations": [],
                "by_source": {},
                "count": {"total": 0}
            })
        
        # 按来源组织关系
        relations_by_source = {}
        for relation in relations_data:
            source = relation.get('source', '未知')
            if source not in relations_by_source:
                relations_by_source[source] = []
            relations_by_source[source].append(relation)
        
        # 统计各类型关系数量
        relation_counts = {
            "total": len(relations_data),
            "by_type": {}
        }
        
        for relation in relations_data:
            relation_type = relation.get('relation_type', '未知')
            if relation_type not in relation_counts["by_type"]:
                relation_counts["by_type"][relation_type] = 0
            relation_counts["by_type"][relation_type] += 1
        
        return jsonify({
            "success": True,
            "relations": relations_data,
            "by_source": relations_by_source,
            "count": relation_counts
        })
        
    except Exception as e:
        logging.error(f"获取任务 {task_id} 的关系数据时出错: {str(e)}")
        logging.error(traceback.format_exc())
        return jsonify({
            "success": False,
            "message": f"获取关系数据出错: {str(e)}",
            "relations": []
        }), 500

@combined_api.route('/tasks/<task_id>/status', methods=['GET'])
@combined_api.route('/comparison/<task_id>/status', methods=['GET'])
def get_comparison_status(task_id):
    """获取比较分析任务的状态信息"""
    try:
        # 检查任务ID格式
        if not task_id or len(task_id) < 10:
            return jsonify({
                "success": False,
                "message": f"无效的任务ID: {task_id}"
            }), 400
            
        # 获取任务状态
        try:
            task_status = db_manager.get_processing_status(task_id)
        except Exception as db_error:
            # 处理数据库查询错误
            logging.error(f"数据库查询任务 {task_id} 状态时出错: {str(db_error)}")
            logging.error(traceback.format_exc())
            
            # 返回有限的状态信息，避免完全失败
            return jsonify({
                "success": True,
                "task": {
                    "task_id": task_id,
                    "status": "ERROR",
                    "message": f"查询状态时出错: {str(db_error)}",
                    "progress": 0
                }
            })
        
        if not task_status:
            # 如果任务不存在，返回一个通用的空状态
            return jsonify({
                "success": True,
                "task": {
                    "task_id": task_id,
                    "status": "NOT_FOUND",
                    "message": "任务不存在或已被删除",
                    "progress": 0
                }
            })
            
        return jsonify({
            "success": True,
            "task": task_status
        })
        
    except Exception as e:
        logging.error(f"获取任务 {task_id} 的状态时出错: {str(e)}")
        logging.error(traceback.format_exc())
        return jsonify({
            "success": False,
            "message": f"获取任务状态出错: {str(e)}"
        }), 500

# =================== 来源统计API ===================

@combined_api.route('/stats/source', methods=['GET'])
def get_source_stats():
    """获取按来源分类的实体和关系统计信息"""
    try:
        task_id = request.args.get('task_id', None)
        
        # 获取实体和关系
        if task_id:
            # 如果提供了任务ID，只获取该任务的实体和关系
            entities = db_manager.get_entities_by_task(task_id)
            relations = db_manager.get_relations_by_task(task_id)
        else:
            # 否则获取所有实体和关系
            entities = db_manager.get_all_entities()
            relations = db_manager.get_all_relations()
        
        # 处理实体，确保获取正确的source字段
        processed_entities = []
        for entity in entities:
            if 'algorithm_entity' in entity:
                algorithm = entity['algorithm_entity']
                algorithm['entity_type'] = 'Algorithm'
                algorithm['source'] = algorithm.get('source', entity.get('source', '未知'))
                processed_entities.append(algorithm)
            elif 'dataset_entity' in entity:
                dataset = entity['dataset_entity']
                dataset['entity_type'] = 'Dataset'
                dataset['source'] = dataset.get('source', entity.get('source', '未知'))
                processed_entities.append(dataset)
            elif 'metric_entity' in entity:
                metric = entity['metric_entity']
                metric['entity_type'] = 'Metric'
                metric['source'] = metric.get('source', entity.get('source', '未知'))
                processed_entities.append(metric)
            elif 'entity_type' in entity:
                entity['source'] = entity.get('source', '未知')
                processed_entities.append(entity)
        
        # 确保关系的source字段
        for relation in relations:
            if 'source' not in relation or not relation['source']:
                relation['source'] = '未知'
        
        # 统计实体，直接根据数据库中的source字段统计
        entities_by_source = {
            '综述': [],
            '引文': [],
            '未知': []
        }
        
        for entity in processed_entities:
            source = entity.get('source', '未知')
            if source == '综述':
                entities_by_source['综述'].append(entity)
            elif source == '引文':
                entities_by_source['引文'].append(entity)
            else:
                entities_by_source['未知'].append(entity)
        
        # 按来源和类型统计实体
        entity_stats = {
            '综述': {
                'total': len(entities_by_source['综述']),
                'algorithm': sum(1 for e in entities_by_source['综述'] if e.get('entity_type') == 'Algorithm'),
                'dataset': sum(1 for e in entities_by_source['综述'] if e.get('entity_type') == 'Dataset'),
                'metric': sum(1 for e in entities_by_source['综述'] if e.get('entity_type') == 'Metric')
            },
            '引文': {
                'total': len(entities_by_source['引文']),
                'algorithm': sum(1 for e in entities_by_source['引文'] if e.get('entity_type') == 'Algorithm'),
                'dataset': sum(1 for e in entities_by_source['引文'] if e.get('entity_type') == 'Dataset'),
                'metric': sum(1 for e in entities_by_source['引文'] if e.get('entity_type') == 'Metric')
            },
            '未知': {
                'total': len(entities_by_source['未知']),
                'algorithm': sum(1 for e in entities_by_source['未知'] if e.get('entity_type') == 'Algorithm'),
                'dataset': sum(1 for e in entities_by_source['未知'] if e.get('entity_type') == 'Dataset'),
                'metric': sum(1 for e in entities_by_source['未知'] if e.get('entity_type') == 'Metric')
            }
        }
        
        # 统计关系，直接根据数据库中的source字段统计
        relations_by_source = {
            '综述': [],
            '引文': [],
            '未知': []
        }
        
        for relation in relations:
            source = relation.get('source', '未知')
            if source == '综述':
                relations_by_source['综述'].append(relation)
            elif source == '引文':
                relations_by_source['引文'].append(relation)
            else:
                relations_by_source['未知'].append(relation)
        
        # 按来源和类型统计关系
        relation_stats = {
            '综述': {
                'total': len(relations_by_source['综述']),
                'types': {}
            },
            '引文': {
                'total': len(relations_by_source['引文']),
                'types': {}
            },
            '未知': {
                'total': len(relations_by_source['未知']),
                'types': {}
            }
        }
        
        # 统计关系类型
        for source, rels in relations_by_source.items():
            for rel in rels:
                rel_type = rel.get('relation_type', 'unknown')
                if rel_type not in relation_stats[source]['types']:
                    relation_stats[source]['types'][rel_type] = 0
                relation_stats[source]['types'][rel_type] += 1
        
        return jsonify({
            'success': True,
            'entities': {
                'total': len(processed_entities),
                'by_source': entity_stats
            },
            'relations': {
                'total': len(relations),
                'by_source': relation_stats
            },
            'task_id': task_id
        })
        
    except Exception as e:
        logging.error(f"获取来源统计信息时出错: {str(e)}")
        logging.error(traceback.format_exc())
        return jsonify({
            'success': False,
            'message': f'获取统计信息失败: {str(e)}'
        }), 500

@combined_api.route('/comparison/<task_id>/recalculate', methods=['POST'])
def recalculate_metrics(task_id):
    """重新计算比较分析指标"""
    try:
        # 检查任务ID格式
        if not task_id or len(task_id) < 10:
            return jsonify({
                "success": False,
                "message": f"无效的任务ID: {task_id}"
            }), 400
            
        # 获取请求参数
        data = request.get_json(silent=True) or {}
        metric_type = data.get('metric_type', 'all')
        
        # 根据指标类型计算不同的指标
        metrics = {}
        
        try:
            # 获取实体和关系数据
            entities = db_manager.get_entities_by_task(task_id)
            relations = db_manager.get_relations_by_task(task_id)
            
            if not entities and not relations:
                return jsonify({
                    "success": True,
                    "message": "没有找到可计算的数据",
                    "metrics": {
                        "entity_stats": {
                            "total": 0,
                            "message": "无数据"
                        },
                        "relation_stats": {
                            "total": 0,
                            "message": "无数据"
                        },
                        "clustering": {
                            "clusters": [],
                            "message": "无数据"
                        }
                    }
                })
                
            # 根据请求计算指标
            if metric_type == 'all' or metric_type == 'entity_stats':
                # 按source字段分离综述和引文实体
                # 从嵌套结构中获取source字段
                review_entities = []
                citation_entities = []
                for e in entities:
                    # 检查实体类型并从对应的内层字典获取source
                    if 'algorithm_entity' in e:
                        source = e['algorithm_entity'].get('source', '')
                        if source == '综述':
                            review_entities.append(e)
                        elif source == '引文':
                            citation_entities.append(e)
                    elif 'dataset_entity' in e:
                        source = e['dataset_entity'].get('source', '')
                        if source == '综述':
                            review_entities.append(e)
                        elif source == '引文':
                            citation_entities.append(e)
                    elif 'metric_entity' in e:
                        source = e['metric_entity'].get('source', '')
                        if source == '综述':
                            review_entities.append(e)
                        elif source == '引文':
                            citation_entities.append(e)
                
                metrics['entity_stats'] = calculate_entity_statistics(review_entities, citation_entities)
            
            if metric_type == 'all' or metric_type == 'relation_stats':
                metrics['relation_stats'] = calculate_relation_statistics(relations)
                
            if metric_type == 'all' or metric_type == 'clustering':
                metrics['clustering'] = calculate_clustering_metrics(entities, relations)
                
        except Exception as calc_error:
            logging.error(f"计算指标时出错: {str(calc_error)}")
            logging.error(traceback.format_exc())
            
            # 返回尽可能多的有效指标，对错误的部分返回错误消息
            if metric_type == 'all' or metric_type == 'entity_stats':
                if 'entity_stats' not in metrics:
                    metrics['entity_stats'] = {"error": str(calc_error)}
                    
            if metric_type == 'all' or metric_type == 'relation_stats':
                if 'relation_stats' not in metrics:
                    metrics['relation_stats'] = {"error": str(calc_error)}
                    
            if metric_type == 'all' or metric_type == 'clustering':
                if 'clustering' not in metrics:
                    metrics['clustering'] = {"error": str(calc_error)}
        
        return jsonify({
            "success": True,
            "metrics": metrics
        })
        
    except Exception as e:
        logging.error(f"重新计算任务 {task_id} 的指标时出错: {str(e)}")
        logging.error(traceback.format_exc())
        return jsonify({
            "success": False,
            "message": f"计算指标出错: {str(e)}"
        }), 500 