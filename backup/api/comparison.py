import os
import logging
import json
import uuid
import shutil
from datetime import datetime
from flask import Blueprint, request, jsonify
from werkzeug.utils import secure_filename

from app.config import Config
from app.modules.db_manager import db_manager
from app.modules.data_extraction import process_papers_and_extract_data

# 创建蓝图
comparison_bp = Blueprint('comparison_api', __name__)

@comparison_bp.route('/start', methods=['POST'])
def start_comparison():
    """启动比较分析任务"""
    temp_files = []  # 用于跟踪所有临时文件
    try:
        # 获取表单数据
        review_paper = request.files.get('review_paper')
        citation_papers = request.files.getlist('citation_papers')
        model_name = request.form.get('model', 'chatgpt')
        
        # 验证输入
        if not review_paper:
            return jsonify({'status': 'error', 'message': '未提供综述论文'}), 400
        
        # 生成任务ID
        task_id = str(uuid.uuid4())
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # 创建上传目录
        os.makedirs(Config.UPLOAD_DIR, exist_ok=True)
        os.makedirs(Config.CITED_PAPERS_DIR, exist_ok=True)
        
        # 保存综述论文，使用任务ID避免冲突
        review_filename = secure_filename(f"{task_id}_{review_paper.filename}")
        review_path = os.path.join(Config.UPLOAD_DIR, review_filename)
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
        
        # 运行处理任务
        entities, relations, metrics = process_papers_and_extract_data(
            review_pdf_path=review_path,
            task_id=task_id,
            citation_paths=citation_paths
        )
        
        # 保存结果到数据库
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

@comparison_bp.route('/status/<task_id>', methods=['GET'])
def get_task_status(task_id):
    """获取任务处理状态"""
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