from flask import Blueprint, render_template, request, jsonify, send_from_directory
import os
import json
import logging
import shutil
import uuid
from pathlib import Path
from app.modules.data_extraction import process_papers_and_extract_data
from app.modules.data_processing import normalize_entities, transform_table_data_to_entities, save_data_to_json, load_data_from_json
from app.modules.knowledge_graph import build_knowledge_graph, visualize_graph, export_graph_to_json
from app.modules.db_manager import db_manager
from app.config import Config
import datetime
from app.modules.relation_generator import generate_relations, update_entities_with_relations
from threading import Thread
from app.modules.agents import extract_paper_entities

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# 创建蓝图
main = Blueprint('main', __name__)

@main.route('/')
def index():
    """渲染主页"""
    return render_template('index.html')

@main.route('/upload')
def upload():
    """渲染上传页面"""
    return render_template('upload.html')

@main.route('/graph')
def graph():
    """渲染关系图页面"""
    return render_template('graph.html')

@main.route('/comparison')
def comparison():
    """渲染对比分析页面"""
    return render_template('comparison.html')

@main.route('/table')
def table():
    """渲染表格页面"""
    return render_template('table.html')

@main.route('/entity/<entity_id>')
@main.route('/entities/<entity_id>')
def entity_detail(entity_id):
    """实体详情页面"""
    entity = db_manager.get_entity_by_id(entity_id)
    
    if not entity:
        return render_template('error.html', message=f"找不到ID为 {entity_id} 的实体")
    
    # 根据嵌套结构提取实体数据和类型
    if 'algorithm_entity' in entity:
        entity_data = entity['algorithm_entity']
        entity_type = entity_data.get('entity_type', 'Algorithm')
    elif 'dataset_entity' in entity:
        entity_data = entity['dataset_entity']
        entity_type = entity_data.get('entity_type', 'Dataset')
    elif 'metric_entity' in entity:
        entity_data = entity['metric_entity']
        entity_type = entity_data.get('entity_type', 'Metric')
    else:
        entity_data = entity
        entity_type = entity.get('entity_type', 'Unknown')
    
    # 确保entity_data中有entity_id
    if 'entity_id' not in entity_data and entity_id:
        entity_data['entity_id'] = entity_id
    
    # 获取与该实体相关的关系
    relations = db_manager.get_relations_by_entity(entity_id)
    
    return render_template('entity_detail.html', 
                          entity=entity_data, 
                          entity_id=entity_id,
                          entity_type=entity_type,
                          relations=relations)

@main.route('/process', methods=['POST'])
def process_papers():
    """处理上传的论文"""
    temp_file_path = None
    try:
        file = request.files.get('paper')
        if not file:
            return jsonify({'success': False, 'message': '未上传文件'}), 400
        
        # 生成唯一的task_id
        import uuid
        import time
        task_id = f"task_{int(time.time())}_{str(uuid.uuid4())[:8]}"
        
        # 创建处理任务记录
        original_filename = file.filename or "未命名文件"
        db_manager.create_processing_task(
            task_id=task_id,
            task_name=f"处理文件: {original_filename}"
        )
        logging.info(f"创建处理任务: {task_id} 用于文件 {original_filename}")
        
        # 确保上传目录存在
        os.makedirs(Config.UPLOAD_FOLDER, exist_ok=True)
        
        # 生成安全的文件名，使用task_id避免冲突
        safe_filename = f"{task_id}_{Path(original_filename).name}"  # 使用任务ID前缀避免冲突
        
        # 直接保存文件到目标位置
        filename = os.path.join(Config.UPLOAD_FOLDER, safe_filename)
        file.save(filename)
        temp_file_path = filename  # 记录临时文件路径以便后续清理
        
        logging.info(f"成功保存文件: {filename}")
        
        # 更新任务状态
        db_manager.update_processing_status(
            task_id=task_id,
            status='processing',
            current_stage='开始处理',
            progress=0.05,
            current_file=os.path.basename(filename),
            message=f'开始处理文件: {os.path.basename(filename)}'
        )
        
        # 在后台线程中异步处理文件
        def process_in_background():
            try:
                # 处理文件
                entities, relations = process_papers_and_extract_data(filename, task_id)
                
                # 规范化实体
                normalized_entities = normalize_entities(entities)
                
                # 确保图数据目录存在
                os.makedirs(Config.GRAPH_DATA_DIR, exist_ok=True)
                
                # 构建图并可视化
                graph = build_knowledge_graph(normalized_entities, relations)
                graph_image_path = os.path.join(Config.GRAPH_DATA_DIR, 'graph.png')
                visualize_graph(graph, output_path=graph_image_path)
                
                # 导出图数据供前端使用
                graph_json_path = os.path.join(Config.GRAPH_DATA_DIR, 'graph.json')
                export_graph_to_json(graph, graph_json_path)
                
                # 更新任务状态为完成
                db_manager.update_processing_status(
                    task_id=task_id,
                    status='completed',
                    current_stage='完成处理',
                    progress=1.0,
                    message=f'成功处理文件: {os.path.basename(filename)}，提取了{len(normalized_entities)}个实体和{len(relations)}个关系',
                    completed=True
                )
                
                logging.info(f"任务 {task_id} 处理完成")
                
            except Exception as e:
                logging.error(f"后台处理文件时出错: {str(e)}")
                import traceback
                error_details = traceback.format_exc()
                logging.error(error_details)
                
                # 更新任务状态为失败
                db_manager.update_processing_status(
                    task_id=task_id,
                    status='failed',
                    current_stage='处理失败',
                    message=f'处理失败: {str(e)}',
                    completed=True
                )
            finally:
                # 清理临时文件
                try:
                    if os.path.exists(filename):
                        os.remove(filename)
                        logging.info(f"已删除临时文件: {filename}")
                except Exception as file_err:
                    logging.error(f"删除临时文件出错: {str(file_err)}")
        
        # 启动后台处理线程
        import threading
        thread = threading.Thread(target=process_in_background)
        thread.daemon = True
        thread.start()
        
        return jsonify({
            'success': True, 
            'message': '文件上传成功，开始处理中',
            'task_id': task_id
        })
        
    except Exception as e:
        logging.error(f"处理文件时出错: {str(e)}")
        # 发生异常时也清理临时文件
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.remove(temp_file_path)
                logging.info(f"已删除临时文件: {temp_file_path}")
            except Exception as file_err:
                logging.error(f"删除临时文件出错: {str(file_err)}")
        return jsonify({'success': False, 'message': f'处理文件时出错: {str(e)}'}), 500

def load_or_build_graph_data(force_refresh=False):
    """
    加载或构建图数据，可被不同的API路由复用
    
    Args:
        force_refresh (bool): 是否强制刷新缓存

    Returns:
        dict: 图数据字典，包含nodes和edges
    """
    try:
        logging.warning("============ 开始加载或构建图谱数据 ============")
        # 确保图数据目录存在
        os.makedirs(Config.GRAPH_DATA_DIR, exist_ok=True)
        
        graph_json_path = os.path.join(Config.GRAPH_DATA_DIR, 'graph.json')
        debug_log_path = os.path.join(Config.GRAPH_DATA_DIR, 'debug_log.json')
        
        debug_info = {
            "timestamp": str(datetime.datetime.now()),
            "force_refresh": force_refresh,
            "action_log": []
        }
        
        # 如果请求刷新或JSON文件不存在，从数据库重建图谱
        if force_refresh or not os.path.exists(graph_json_path):
            # 从数据库获取最新数据
            logging.warning("从数据库刷新图谱数据 - force_refresh=%s, file_exists=%s", 
                          force_refresh, os.path.exists(graph_json_path))
            debug_info["action_log"].append("从数据库加载数据")
            
            entities = db_manager.get_all_entities()
            relations = db_manager.get_all_relations()
            
            debug_info["entities_count"] = len(entities)
            debug_info["relations_count"] = len(relations)
            if len(entities) > 0:
                debug_info["first_entity"] = entities[0]
            
            logging.warning("从数据库获取的实体数量: %d, 关系数量: %d", 
                          len(entities), len(relations))
            
            # 构建图
            debug_info["action_log"].append("构建图谱")
            graph = build_knowledge_graph(entities, relations)
            
            # 确保导出目录存在
            os.makedirs(os.path.dirname(graph_json_path), exist_ok=True)
            
            # 导出图数据
            debug_info["action_log"].append("导出图谱数据")
            graph_data = export_graph_to_json(graph, graph_json_path)
            debug_info["nodes_count"] = len(graph_data.get("nodes", []))
            debug_info["edges_count"] = len(graph_data.get("edges", []))
            
            logging.warning("导出的节点数量: %d, 边数量: %d", 
                         len(graph_data.get("nodes", [])), 
                         len(graph_data.get("edges", [])))
        else:
            debug_info["action_log"].append("使用缓存的图谱数据")
            logging.warning("使用缓存的图谱数据: %s", graph_json_path)
        
        # 读取JSON文件
        if os.path.exists(graph_json_path):
            with open(graph_json_path, 'r', encoding='utf-8') as f:
                graph_data = json.load(f)
                
                debug_info["from_cache"] = not force_refresh
                debug_info["cache_path"] = graph_json_path
                debug_info["cached_nodes"] = len(graph_data.get("nodes", []))
                debug_info["cached_edges"] = len(graph_data.get("edges", []))
                
                # 检查是否返回的是空数据
                if not graph_data.get('nodes'):
                    logging.warning("图谱数据为空，尝试强制刷新")
                    debug_info["action_log"].append("空数据，强制刷新")
                    
                    # 强制刷新一次
                    entities = db_manager.get_all_entities()
                    relations = db_manager.get_all_relations()
                    
                    debug_info["forced_entities_count"] = len(entities)
                    debug_info["forced_relations_count"] = len(relations)
                    if len(entities) > 0:
                        debug_info["forced_first_entity"] = entities[0]
                    
                    logging.warning("强制刷新 - 从数据库获取的实体数量: %d, 关系数量: %d", 
                                  len(entities), len(relations))
                    
                    # 为调试保存原始数据
                    raw_data_path = os.path.join(Config.GRAPH_DATA_DIR, 'raw_data.json')
                    with open(raw_data_path, 'w', encoding='utf-8') as raw_f:
                        json.dump({"entities": entities, "relations": relations}, raw_f, 
                                 ensure_ascii=False, indent=2)
                        debug_info["action_log"].append(f"保存原始数据到 {raw_data_path}")
                    
                    # 构建图
                    graph = build_knowledge_graph(entities, relations)
                    
                    # 导出图数据
                    graph_data = export_graph_to_json(graph, graph_json_path)
                    debug_info["forced_nodes_count"] = len(graph_data.get("nodes", []))
                    debug_info["forced_edges_count"] = len(graph_data.get("edges", []))
                    
                    # 重新读取
                    with open(graph_json_path, 'r', encoding='utf-8') as f:
                        graph_data = json.load(f)
                
                logging.warning("返回图谱数据: 节点数 %d, 边数 %d", 
                              len(graph_data.get('nodes', [])), 
                              len(graph_data.get('edges', [])))
                
                # 保存调试信息
                with open(debug_log_path, 'w', encoding='utf-8') as debug_f:
                    json.dump(debug_info, debug_f, ensure_ascii=False, indent=2)
                
                # 复制一份数据到graph_api的缓存路径
                try:
                    api_cache_path = os.path.join(Config.GRAPH_DATA_DIR, 'graph_data.json')
                    if graph_data and (len(graph_data.get('nodes', [])) > 0 or len(graph_data.get('edges', [])) > 0):
                        with open(api_cache_path, 'w', encoding='utf-8') as f:
                            json.dump(graph_data, f, ensure_ascii=False, indent=2)
                        logging.warning(f"已将图数据复制到API缓存: {api_cache_path}")
                except Exception as e:
                    logging.error(f"复制到API缓存失败: {str(e)}")
                
                return graph_data
        else:
            logging.error("JSON文件不存在: %s", graph_json_path)
            debug_info["error"] = f"JSON文件不存在: {graph_json_path}"
            with open(debug_log_path, 'w', encoding='utf-8') as debug_f:
                json.dump(debug_info, debug_f, ensure_ascii=False, indent=2)
                
            return {'nodes': [], 'edges': [], 'error': 'JSON文件不存在'}
        
    except Exception as e:
        logging.error("获取图数据时出错: %s", str(e))
        import traceback
        tb = traceback.format_exc()
        logging.error(tb)
        
        # 保存错误日志
        error_log = {
            "timestamp": str(datetime.datetime.now()),
            "error": str(e),
            "traceback": tb
        }
        error_log_path = os.path.join(Config.GRAPH_DATA_DIR, 'error_log.json')
        with open(error_log_path, 'w', encoding='utf-8') as f:
            json.dump(error_log, f, ensure_ascii=False, indent=2)
            
        return {'nodes': [], 'edges': [], 'error': str(e), 'traceback': tb}

# @main.route('/api/graph/data')
# def get_graph_data():
#     """获取图数据API接口"""
#     force_refresh = request.args.get('refresh', '0') == '1'
#     graph_data = load_or_build_graph_data(force_refresh)
#     return jsonify(graph_data)

# @main.route('/api/graph/image')
# def get_graph_image():
#     """获取图片"""
#     try:
#         # 确保图数据目录存在
#         os.makedirs(Config.GRAPH_DATA_DIR, exist_ok=True)
#         
#         graph_image_path = os.path.join(Config.GRAPH_DATA_DIR, 'graph.png')
#         
#         if not os.path.exists(graph_image_path):
#             # 如果图片不存在，尝试从数据库创建
#             entities = db_manager.get_all_entities()
#             relations = db_manager.get_all_relations()
#             
#             if not entities:
#                 return jsonify({'success': False, 'message': '没有可用的数据'}), 404
#             
#             graph = build_knowledge_graph(entities, relations)
#             visualize_graph(graph, output_path=graph_image_path)
#         
#         return send_from_directory(os.path.dirname(graph_image_path), os.path.basename(graph_image_path))
#         
#     except Exception as e:
#         logging.error(f"获取图片时出错: {str(e)}")
#         return jsonify({'success': False, 'message': f'获取图片时出错: {str(e)}'}), 500

@main.route('/api/import/table', methods=['POST'])
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

@main.route('/api/debug/entities')
def debug_entities():
    """调试端点：获取所有算法实体数据"""
    try:
        logging.warning("调试请求：获取所有算法实体数据")
        entities = db_manager.get_all_entities()
        return jsonify({
            "count": len(entities),
            "entities": entities
        })
    except Exception as e:
        logging.error(f"获取算法实体时出错: {str(e)}")
        import traceback
        tb = traceback.format_exc()
        logging.error(tb)
        return jsonify({"error": str(e), "traceback": tb}), 500

@main.route('/api/debug/relations')
def debug_relations():
    """调试端点：获取所有演化关系数据"""
    try:
        logging.warning("调试请求：获取所有演化关系数据")
        relations = db_manager.get_all_relations()
        return jsonify({
            "count": len(relations),
            "relations": relations
        })
    except Exception as e:
        logging.error(f"获取演化关系时出错: {str(e)}")
        import traceback
        tb = traceback.format_exc()
        logging.error(tb)
        return jsonify({"error": str(e), "traceback": tb}), 500

@main.route('/api/clear/all', methods=['POST'])
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
            
        # 清除缓存文件夹
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
        import traceback
        tb = traceback.format_exc()
        logging.error(tb)
        return jsonify({
            "success": False,
            "message": f"清除数据时出错: {str(e)}",
            "traceback": tb
        }), 500

@main.route('/api/tasks/<task_id>/status', methods=['GET'])
def get_task_status(task_id):
    """获取任务处理状态API"""
    try:
        # 查询任务状态
        query = """
        SELECT id, task_id, status, current_stage, progress, current_file, message, 
               start_time, update_time, end_time
        FROM ProcessingStatus
        WHERE task_id = %s
        """
        
        db_manager.cursor.execute(query, (task_id,))
        row = db_manager.cursor.fetchone()
        
        if not row:
            return jsonify({
                "success": False,
                "message": f"找不到任务ID: {task_id}"
            }), 404
            
        # 解析行数据
        status_data = {
            "id": row[0],
            "task_id": row[1],
            "status": row[2],
            "current_stage": row[3],
            "progress": float(row[4]) if row[4] is not None else 0,
            "current_file": row[5],
            "message": row[6],
            "start_time": row[7].strftime('%Y-%m-%d %H:%M:%S') if row[7] else None,
            "update_time": row[8].strftime('%Y-%m-%d %H:%M:%S') if row[8] else None,
            "end_time": row[9].strftime('%Y-%m-%d %H:%M:%S') if row[9] else None,
            "is_completed": row[2] in ['completed', 'failed']
        }
        
        return jsonify({
            "success": True,
            "data": status_data
        })
        
    except Exception as e:
        logging.error(f"获取任务状态时出错: {str(e)}")
        import traceback
        tb = traceback.format_exc()
        logging.error(tb)
        return jsonify({
            "success": False,
            "message": f"获取任务状态时出错: {str(e)}",
            "traceback": tb
        }), 500

