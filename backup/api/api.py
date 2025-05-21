from flask import Blueprint, request, jsonify, send_from_directory, current_app
import os
import uuid
import tempfile
import shutil
import json
import logging
import traceback
from app.config import Config
from app.modules.data_processing import process_review_paper, process_multiple_papers
from app.modules.knowledge_graph import generate_knowledge_graph
from app.modules.db_manager import db_manager

api = Blueprint('api', __name__)

@api.route('/upload', methods=['POST'])
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

@api.route('/batch_upload', methods=['POST'])
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

@api.route('/knowledge_graph', methods=['GET'])
def get_knowledge_graph():
    """获取知识图谱数据"""
    try:
        graph_data = generate_knowledge_graph()
        
        return jsonify({
            'success': True,
            'nodes': graph_data['nodes'],
            'links': graph_data['links']
        })
        
    except Exception as e:
        logging.error(f"获取知识图谱数据出错: {str(e)}")
        logging.error(traceback.format_exc())
        return jsonify({
            'success': False,
            'message': f'获取知识图谱数据失败: {str(e)}'
        }), 500

@api.route('/status/<task_id>', methods=['GET'])
def get_processing_status(task_id):
    """获取处理任务的状态"""
    try:
        logging.info(f"获取任务状态请求: {task_id}")
        status = db_manager.get_processing_status(task_id)
        
        if status:
            logging.info(f"获取到任务状态: {status}")
            return jsonify({
                'success': True,
                'status': status
            })
        else:
            logging.warning(f"未找到任务ID: {task_id}")
            return jsonify({
                'success': False,
                'message': f'未找到任务ID: {task_id}'
            }), 404
            
    except Exception as e:
        logging.error(f"获取处理状态出错: {str(e)}")
        logging.error(traceback.format_exc())
        return jsonify({
            'success': False,
            'message': f'获取处理状态失败: {str(e)}'
        }), 500

@api.route('/status', methods=['GET'])
def get_all_processing_status():
    """获取所有处理任务的状态"""
    try:
        statuses = db_manager.get_processing_status()
        
        return jsonify({
            'success': True,
            'statuses': statuses
        })
            
    except Exception as e:
        logging.error(f"获取所有处理状态出错: {str(e)}")
        logging.error(traceback.format_exc())
        return jsonify({
            'success': False,
            'message': f'获取所有处理状态失败: {str(e)}'
        }), 500

@api.route('/algorithms', methods=['GET'])
def get_algorithms():
    """获取所有算法实体"""
    try:
        algorithms = db_manager.get_all_entities()
        
        return jsonify({
            'success': True,
            'algorithms': algorithms
        })
        
    except Exception as e:
        logging.error(f"获取算法实体出错: {str(e)}")
        logging.error(traceback.format_exc())
        return jsonify({
            'success': False,
            'message': f'获取算法实体失败: {str(e)}'
        }), 500

@api.route('/algorithm/<algorithm_id>', methods=['GET'])
def get_algorithm(algorithm_id):
    """获取指定算法实体的详细信息"""
    try:
        algorithm = db_manager.get_entity_by_id(algorithm_id)
        
        if algorithm:
            return jsonify({
                'success': True,
                'algorithm': algorithm
            })
        else:
            return jsonify({
                'success': False,
                'message': f'未找到算法: {algorithm_id}'
            }), 404
            
    except Exception as e:
        logging.error(f"获取算法详情出错: {str(e)}")
        logging.error(traceback.format_exc())
        return jsonify({
            'success': False,
            'message': f'获取算法详情失败: {str(e)}'
        }), 500

@api.route('/entities/<entity_id>', methods=['GET'])
def get_entity_details(entity_id):
    """获取指定实体的详细信息，包括算法、数据集和评价指标"""
    try:
        logging.info(f"请求获取实体详情: {entity_id}")
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
        logging.error(f"获取实体详情出错: {str(e)}")
        logging.error(traceback.format_exc())
        return jsonify({
            'success': False,
            'message': f'获取实体详情失败: {str(e)}'
        }), 500 