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

# 创建蓝图
combined_api = Blueprint('combined_api', __name__)

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
            else:
                # 如果没有特定结构，则尝试根据字段推断类型
                if any(key.startswith('algorithm_') for key in entity.keys()):
                    entity['entity_type'] = 'Algorithm'
                elif any(key.startswith('dataset_') for key in entity.keys()):
                    entity['entity_type'] = 'Dataset'
                elif any(key.startswith('metric_') for key in entity.keys()):
                    entity['entity_type'] = 'Metric'
                processed_entities.append(entity)
        
        logging.info(f"获取到实体总数: {len(processed_entities)}")
        # 记录各类型实体数量
        algo_count = sum(1 for e in processed_entities if e.get('entity_type') == 'Algorithm')
        dataset_count = sum(1 for e in processed_entities if e.get('entity_type') == 'Dataset')
        metric_count = sum(1 for e in processed_entities if e.get('entity_type') == 'Metric')
        
        logging.info(f"算法实体: {algo_count}, 数据集实体: {dataset_count}, 评价指标实体: {metric_count}")
        
        return jsonify({
            'success': True,
            'count': len(processed_entities),
            'entities': processed_entities
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
            # 规范化实体格式
            if 'algorithm_entity' in entity:
                result = entity['algorithm_entity']
                result['entity_type'] = 'Algorithm'
            elif 'dataset_entity' in entity:
                result = entity['dataset_entity']
                result['entity_type'] = 'Dataset'
            elif 'metric_entity' in entity:
                result = entity['metric_entity']
                result['entity_type'] = 'Metric'
            else:
                result = entity
                
            return jsonify(result)
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
            return jsonify(entity)
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
        
        return jsonify({
            'success': True,
            'count': len(relations),
            'relations': relations
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
            return jsonify(relation)
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

# =================== 任务状态相关API ===================

@combined_api.route('/task/<task_id>/status', methods=['GET'])
def get_task_status(task_id):
    """获取任务处理状态"""
    try:
        # 查询任务状态
        task_status = db_manager.get_processing_status(task_id)
        
        if not task_status:
            return jsonify({
                "success": False,
                "message": f"找不到任务ID: {task_id}"
            }), 404
            
        return jsonify({
            "success": True,
            "data": task_status
        })
        
    except Exception as e:
        logging.error(f"获取任务状态时出错: {str(e)}")
        logging.error(traceback.format_exc())
        return jsonify({
            "success": False,
            "message": f"获取任务状态时出错: {str(e)}",
            "traceback": traceback.format_exc()
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
def clear_cache_keep_pdf_text():
    """清除除PDF提取文本外的所有缓存和数据库数据"""
    try:
        logging.warning("收到清除缓存的请求（保留PDF文本提取）")
        
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
            
        # 清除缓存目录中除PDF文本外的所有内容
        if os.path.exists(Config.CACHE_DIR):
            pdf_text_dir = os.path.join(Config.CACHE_DIR, "pdf_text")
            
            # 备份PDF文本目录（如果存在）
            temp_pdf_text_backup = None
            if os.path.exists(pdf_text_dir):
                import tempfile
                temp_pdf_text_backup = tempfile.mkdtemp()
                shutil.copytree(pdf_text_dir, os.path.join(temp_pdf_text_backup, "pdf_text"))
                logging.info(f"已备份PDF文本目录到临时位置: {temp_pdf_text_backup}")
            
            # 清除整个缓存目录
            shutil.rmtree(Config.CACHE_DIR)
            os.makedirs(Config.CACHE_DIR, exist_ok=True)
            logging.warning(f"已清除缓存文件夹: {Config.CACHE_DIR}")
            
            # 恢复PDF文本目录
            if temp_pdf_text_backup:
                if not os.path.exists(pdf_text_dir):
                    os.makedirs(pdf_text_dir, exist_ok=True)
                shutil.copytree(os.path.join(temp_pdf_text_backup, "pdf_text"), pdf_text_dir, dirs_exist_ok=True)
                # 清除临时备份
                shutil.rmtree(temp_pdf_text_backup)
                logging.info(f"已恢复PDF文本目录: {pdf_text_dir}")
        
        return jsonify({
            "success": True,
            "message": "已成功清除缓存和数据库数据，保留PDF文本提取"
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

@combined_api.route('/comparison/start', methods=['POST'])
def start_comparison():
    """启动比较分析任务"""
    temp_files = []  # 用于跟踪所有临时文件
    try:
        # 获取表单数据
        review_paper = request.files.get('review_paper')
        citation_papers = request.files.getlist('citation_papers')
        model_name = request.form.get('model', 'chatgpt')
        from pathlib import Path
        # 验证输入
        if not review_paper:
            return jsonify({'status': 'error', 'message': '未提供综述论文'}), 400
        from datetime import datetime
        # 生成任务ID
        task_id = str(uuid.uuid4())
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # 创建上传目录
        os.makedirs(Config.UPLOAD_DIR, exist_ok=True)
        os.makedirs(Config.CITED_PAPERS_DIR, exist_ok=True)
        
        # 保存综述论文，使用任务ID避免冲突
        review_filename = review_paper.filename
        # 不要生成安全的文件名
        safe_filename = f"{Path(review_filename).name}"  
        # 直接保存文件到目标位置
        review_path = os.path.join(Config.UPLOAD_FOLDER, safe_filename)
        review_paper.save(review_path)
        temp_files.append(review_path)
        
        # 保存引用论文
        citation_paths = []
        for i, paper in enumerate(citation_papers):
            if paper.filename:
                citation_filename = secure_filename(f"{task_id}_{i}_{paper.filename}")
                citation_path = os.path.join(Config.CITED_PAPERS_DIR, citation_filename)
                paper.save(citation_path)
                citation_paths.append(citation_path)
                temp_files.append(citation_path)
        
        # 将任务信息保存到数据库
        task_info = {
            'task_id': task_id,
            'review_paper': review_path,
            'citation_papers': citation_paths,
            'model': model_name,
            'timestamp': timestamp,
            'status': 'started',
            'temp_files': temp_files  # 记录所有临时文件以便稍后清理
        }
        
        # 创建处理任务 - 只使用支持的参数
        task_name = f"比较分析任务 - 模型: {model_name} - 文件: {review_paper.filename}"
        db_manager.create_processing_task(
            task_id=task_id,
            task_name=task_name
        )
        
        # 更新任务状态以包含更多信息
        db_manager.update_processing_status(
            task_id=task_id,
            status='处理中',
            current_stage='初始化',
            progress=0.0,
            message='任务已创建，准备开始处理'
        )
        
        # 在后台启动处理任务
        import threading
        process_thread = threading.Thread(
            target=run_comparison_task,
            args=(task_id, review_path, citation_paths, model_name, temp_files)
        )
        process_thread.daemon = True
        process_thread.start()
        
        return jsonify({
            'status': 'success',
            'message': '比较分析任务已启动',
            'task_id': task_id
        })
    
    except Exception as e:
        logging.error(f"启动比较分析任务时出错: {str(e)}")
        import traceback
        logging.error(traceback.format_exc())
        
        # 发生错误时清理所有临时文件
        for file_path in temp_files:
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
        from app.modules.data_extraction import process_papers_and_extract_data
        
        # 运行处理任务
        entities, relations, metrics = process_papers_and_extract_data(
            review_pdf_path=review_path,
            task_id=task_id,
            citation_paths=citation_paths
        )
        
        # 保存结果到数据库
        from datetime import datetime
        result_data = {
            'entities_count': len(entities),
            'relations_count': len(relations),
            'metrics': metrics,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        db_manager.update_processing_status(
            task_id=task_id,
            status='已完成',
            current_stage='任务完成',
            progress=1.0,
            message='比较分析任务已完成',
            result=json.dumps(result_data)
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

@combined_api.route('/comparison/status/<task_id>', methods=['GET'])
def get_comparison_status(task_id):
    """获取比较分析任务处理状态"""
    try:
        # 从数据库获取任务状态
        task_status = db_manager.get_processing_status(task_id)
        
        if not task_status:
            return jsonify({'status': 'error', 'message': '找不到指定的任务'}), 404
        
        # 转换为JSON格式
        status_data = {
            'task_id': task_id,
            'status': task_status.get('status', '未知'),
            'current_stage': task_status.get('current_stage', ''),
            'progress': task_status.get('progress', 0),
            'message': task_status.get('message', ''),
            'result': json.loads(task_status.get('result', '{}')) if task_status.get('result') else {}
        }
        
        return jsonify({
            'status': 'success',
            'data': status_data
        })
    
    except Exception as e:
        logging.error(f"获取任务 {task_id} 状态时出错: {str(e)}")
        return jsonify({'status': 'error', 'message': f'获取任务状态时出错: {str(e)}'}), 500 