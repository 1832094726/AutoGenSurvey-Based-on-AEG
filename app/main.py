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
from threading import Thread
from app.modules.agents import extract_paper_entities
import platform
from flask import current_app
from app.modules.db_pool import db_utils

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# 创建蓝图
main = Blueprint('main', __name__)

# 数据库连接辅助函数
def get_db_connection():
    """获取数据库连接，使用db_pool连接池
    
    Returns:
        connection: 数据库连接对象，实际上无需返回连接，应直接使用db_utils
    """
    logging.info("使用连接池获取数据库连接")
    # 不再需要手动reconnect，连接池会自动管理
    return db_utils  # 返回db_utils实例，而不是直接的连接

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

@main.route('/paper-analysis')
def paper_analysis():
    """渲染论文分析页面"""
    return render_template('paper_analysis.html')

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
        
        # 不要生成安全的文件名
        safe_filename = f"{Path(original_filename).name}"  # 使用任务ID前缀避免冲突
        
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

@main.route('/api/debug/routes', methods=['GET'])
def get_all_routes():
    """返回应用中所有注册的路由"""
    routes = []
    for rule in current_app.url_map.iter_rules():
        methods = ','.join(rule.methods)
        routes.append({
            'endpoint': rule.endpoint,
            'methods': methods,
            'rule': str(rule)
        })
    
    logging.info(f"已请求路由列表，找到 {len(routes)} 个路由")
    return jsonify({
        'success': True,
        'message': f'找到 {len(routes)} 个路由',
        'data': sorted(routes, key=lambda r: r['rule'])
    })

@main.route('/api/tasks/test', methods=['GET'])
def test_tasks_api():
    """API测试端点"""
    logging.info(f"API测试端点被调用")
    return jsonify({
        'success': True,
        'message': 'API端点可用',
        'test_data': {
            'timestamp': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'request_method': request.method,
            'request_path': request.path,
            'request_args': dict(request.args)
        }
    })

@main.route('/api/task_status/<task_id>', methods=['GET'])
def get_task_status_alt(task_id):
    """获取任务状态（备用路由）"""
    logging.info(f"备用路由：获取任务状态 - ID: {task_id}")
    logging.info(f"请求URL: {request.url}")
    logging.info(f"请求方法: {request.method}")
    logging.info(f"请求头: {dict(request.headers)}")
    
    try:
        # 记录请求信息
        logging.info(f"正在查询任务 {task_id} 的状态")
        
        # 从数据库获取任务状态
        with get_db_connection() as conn:
            task_status = get_task_status_from_db(conn, task_id)
        
        if task_status:
            logging.info(f"任务状态获取成功: {task_status}")
            return jsonify({
                'success': True,
                'message': '任务状态获取成功',
                'data': task_status
            })
        else:
            logging.warning(f"任务 {task_id} 不存在")
            return jsonify({
                'success': False,
                'message': f'找不到任务 {task_id}'
            }), 404
    except Exception as e:
        logging.error(f"获取任务状态时出错: {str(e)}")
        logging.exception(e)
        return jsonify({
            'success': False,
            'message': f'获取任务状态时出错: {str(e)}'
        }), 500

@main.route('/api/tasks/<task_id>/status', methods=['GET'])
def get_task_status(task_id):
    """获取任务状态"""
    # 记录详细的请求信息，帮助调试路由问题
    logging.info(f"主路由：获取任务状态 - ID: {task_id}")
    logging.info(f"请求URL: {request.url}")
    logging.info(f"请求方法: {request.method}")
    logging.info(f"请求头: {dict(request.headers)}")
    
    try:
        # 记录请求信息
        logging.info(f"正在查询任务 {task_id} 的状态")
        
        # 从数据库获取任务状态
        with get_db_connection() as conn:
            task_status = get_task_status_from_db(conn, task_id)
        
        if task_status:
            logging.info(f"任务状态获取成功: {task_status}")
            return jsonify({
                'success': True,
                'message': '任务状态获取成功',
                'data': task_status
            })
        else:
            logging.warning(f"任务 {task_id} 不存在")
            return jsonify({
                'success': False,
                'message': f'找不到任务 {task_id}'
            }), 404
    except Exception as e:
        logging.error(f"获取任务状态时出错: {str(e)}")
        logging.exception(e)
        return jsonify({
            'success': False,
            'message': f'获取任务状态时出错: {str(e)}'
        }), 500

# 辅助函数：从数据库获取任务状态
def get_task_status_from_db(conn, task_id):
    """从数据库获取任务状态信息"""
    # 首先检查ProcessingStatus表
    query = """
    SELECT task_id, status, current_stage, progress, message, start_time, current_file
        FROM ProcessingStatus
        WHERE task_id = %s
    ORDER BY start_time DESC
    LIMIT 1
    """
    try:
        cursor = conn.cursor(dictionary=True)
        cursor.execute(query, (task_id,))
        task_data = cursor.fetchone()
        cursor.close()
        
        if task_data:
            logging.info(f"在ProcessingStatus表中找到任务: {task_data}")
            # 格式化日期时间
            if 'start_time' in task_data and task_data['start_time']:
                task_data['start_time'] = task_data['start_time'].strftime('%Y-%m-%d %H:%M:%S')
                
            # 确保键名一致，使用current_file作为任务名称
            task_data['task_name'] = task_data.get('current_file', '未命名任务')
            return task_data
            
        # 如果在ProcessingStatus表中找不到，尝试查找相关实体
        logging.info(f"在ProcessingStatus表中未找到任务 {task_id}，尝试检查相关实体")
        entity_data = check_task_entities(conn, task_id)
        if entity_data:
            logging.info(f"找到任务相关实体: {entity_data}")
            return entity_data
            
        logging.warning(f"在所有表中均未找到任务 {task_id}")
        return None
    except Exception as e:
        logging.error(f"查询任务状态时出错: {str(e)}")
        logging.exception(e)
        raise

# 辅助函数：检查是否存在与任务相关的实体
def check_task_entities(conn, task_id):
    """检查是否存在与任务相关的实体（算法、数据集、指标）"""
    try:
        # 检查算法表
        cursor = conn.cursor(dictionary=True)
        tables = ['Algorithms', 'Datasets', 'Metrics']
        id_fields = {
            'Algorithms': 'algorithm_id',
            'Datasets': 'dataset_id',
            'Metrics': 'metric_id'
        }
        
        for table in tables:
            id_field = id_fields[table]
            query = f"""
            SELECT {id_field} as entity_id, name, task_id 
            FROM {table}
            WHERE task_id = %s
            LIMIT 1
            """
            cursor.execute(query, (task_id,))
            result = cursor.fetchone()
            
            if result:
                logging.info(f"在{table}表中找到任务相关记录: {result}")
                entity_type = table[:-1]  # 移除表名末尾的's'
                
                # 创建任务状态记录
                now = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                task_data = {
                    'task_id': task_id,
                    'task_name': f"{result.get('name', '未命名')}（{entity_type}）",
                    'status': 'COMPLETED',  # 如果实体存在，说明任务已完成
                    'current_stage': 'ENTITY_EXTRACTION',
                    'progress': 100,
                    'message': f'已完成{entity_type}实体提取',
                    'start_time': now
                }
                return task_data
        
        cursor.close()
        return None
    except Exception as e:
        logging.error(f"检查任务实体时出错: {str(e)}")
        logging.exception(e)
        return None

@main.route('/comparison/results/<task_id>')
@main.route('/comparison_results.html/<task_id>')
def comparison_results(task_id):
    """显示比较分析结果页面"""
    try:
        # 简化函数，只传递任务ID，其他数据通过API动态获取
        return render_template('comparison_results.html', task_id=task_id)
    except Exception as e:
        logging.error(f"加载比较结果页面时出错: {str(e)}")
        logging.error(traceback.format_exc())
        flash(f"加载比较结果页面时出错: {str(e)}", 'danger')
        return redirect(url_for('main.comparison'))

@main.route('/api/graph/task/<task_id>')
def get_graph_data_by_task(task_id):
    """获取指定任务的图谱数据API接口，支持新的多关系边格式"""
    try:
        logging.info(f"请求任务 {task_id} 的图谱数据")
        
        # 从数据库获取与任务相关的实体和关系
        entities = db_manager.get_entities_by_task(task_id)
        relations = db_manager.get_relations_by_task(task_id)
        
        logging.info(f"从数据库获取的实体数量: {len(entities)}, 关系数量: {len(relations)}")
        
        # 构建图
        graph_data = build_knowledge_graph(entities, relations, task_id)
        
        # 在返回前移除networkx_graph字段，因为它不可JSON序列化
        if 'networkx_graph' in graph_data:
            del graph_data['networkx_graph']
        
        logging.info(f"返回图谱数据: 节点数 {len(graph_data['nodes'])}, 边数 {len(graph_data['edges'])}")
        return jsonify(graph_data)
        
    except Exception as e:
        logging.error(f"获取任务 {task_id} 的图谱数据时出错: {str(e)}")
        import traceback
        tb = traceback.format_exc()
        logging.error(tb)
        return jsonify({'nodes': [], 'edges': [], 'error': str(e), 'traceback': tb})

@main.route('/api/tasks')
def get_tasks():
    """获取所有任务的列表"""
    logging.warning("===== 开始处理任务列表请求 =====")
    
    try:
        # 记录请求详情
        logging.warning("请求头: %s", str(request.headers))
        logging.warning("请求方法: %s, 路径: %s", request.method, request.path)
        logging.warning("请求参数: %s", str(request.args))
        
        logging.warning("正在获取任务列表...")
        
        # 使用连接池，不再需要手动检查数据库连接
        start_time = datetime.datetime.now()
        logging.warning(f"开始从数据库获取任务列表: {start_time}")
        
        try:
            tasks = db_manager.get_comparison_history(limit=50)
        
            end_time = datetime.datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            logging.warning(f"数据库查询完成时间: {end_time}, 耗时: {duration}秒")
            logging.warning(f"成功获取 {len(tasks)} 条任务记录")
            
            # 打印部分任务数据用于调试
            if tasks:
                logging.warning(f"第一条任务数据示例: {tasks[0]}")
            
            # 构建响应
            response = jsonify(tasks)
            
            # 添加响应头，解决可能的CORS问题
            response.headers.add('Access-Control-Allow-Origin', '*')
            response.headers.add('Access-Control-Allow-Headers', 'Content-Type')
            response.headers.add('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
            
            logging.warning("===== 任务列表请求处理完成，准备返回响应 =====")
            return response
            
        except Exception as db_err:
            # 数据库操作错误
            logging.error(f"调用get_comparison_history时出错: {str(db_err)}")
            import traceback
            db_tb = traceback.format_exc()
            logging.error(db_tb)
            
            # 返回错误响应
            error_response = jsonify({
                'error': f"数据库操作错误: {str(db_err)}",
                'traceback': db_tb
            })
            error_response.headers.add('Access-Control-Allow-Origin', '*')
            logging.warning("===== 任务列表请求处理失败，返回数据库错误 =====")
            return error_response
            
    except Exception as e:
        # 其他未知错误
        logging.error(f"获取任务列表过程中出现未知错误: {str(e)}")
        import traceback
        tb = traceback.format_exc()
        logging.error(tb)
        
        error_response = jsonify({
            'error': str(e), 
            'traceback': tb
        })
        error_response.headers.add('Access-Control-Allow-Origin', '*')
        logging.warning("===== 任务列表请求处理失败，返回未知错误 =====")
        return error_response

@main.route('/api/system/db-status', methods=['GET'])
def check_db_status():
    """检查数据库连接状态"""
    try:
        # 使用连接池执行简单查询测试连接
        result = db_utils.select_one("SELECT 1 as test")
        
        return jsonify({
            'success': True,
            'status': 'connected',
            'test_query': result and result.get('test') == 1,
            'pool_info': {
                'message': '使用连接池连接数据库正常'
            }
        })
    except Exception as e:
        logging.error(f"检查数据库状态时出错: {str(e)}")
        import traceback
        tb = traceback.format_exc()
        logging.error(tb)
        return jsonify({
            'success': False,
            'status': 'disconnected',
            'error': str(e),
            'traceback': tb
        })

@main.route('/api/system/ping', methods=['GET'])
def ping_test():
    """简单的连接测试接口"""
    try:
        logging.warning("收到ping测试请求")
        
        # 记录请求头和参数
        logging.warning("请求头: %s", str(request.headers))
        logging.warning("请求路径: %s", request.path)
        
        # 构造响应
        response_data = {
            'status': 'success',
            'message': 'pong',
            'timestamp': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'server_info': {
                'python_version': platform.python_version(),
                'system': platform.system(),
                'machine': platform.machine()
            }
        }
        
        response = jsonify(response_data)
        response.headers.add('Access-Control-Allow-Origin', '*')
        logging.warning("ping测试请求成功返回")
        
        return response
    except Exception as e:
        logging.error("ping测试处理出错: %s", str(e))
        error_response = jsonify({'status': 'error', 'message': str(e)})
        error_response.headers.add('Access-Control-Allow-Origin', '*')
        return error_response

