#!/usr/bin/env python3
"""
智能综述生成API路由
基于算法脉络分析的自定义综述生成功能
"""

from flask import Blueprint, request, jsonify
import logging
import uuid
import threading
import time
from datetime import datetime
from typing import Dict, Any, List

from app.modules.survey_generator import survey_generator
from app.modules.db_manager import db_manager

# 创建蓝图
smart_survey_bp = Blueprint('smart_survey', __name__, url_prefix='/api/smart_survey')

# 日志配置
logger = logging.getLogger(__name__)

# 任务状态存储
survey_tasks = {}

class SurveyTask:
    """综述生成任务类"""
    
    def __init__(self, task_id: str, task_ids: List[str], topic: str, 
                 section_num: int = 7, subsection_len: int = 700):
        self.task_id = task_id
        self.task_ids = task_ids
        self.topic = topic
        self.section_num = section_num
        self.subsection_len = subsection_len
        self.status = 'initializing'
        self.progress = 0.0
        self.current_stage = '初始化任务'
        self.message = '正在准备综述生成...'
        self.result = None
        self.error = None
        self.start_time = datetime.now()
        self.end_time = None
    
    def update_progress(self, progress: float, stage: str, message: str):
        """更新任务进度"""
        self.progress = progress
        self.current_stage = stage
        self.message = message
        logger.info(f"任务 {self.task_id}: {stage} - {message} ({progress:.1%})")
    
    def complete_success(self, result: Dict[str, Any]):
        """标记任务成功完成"""
        self.status = 'completed'
        self.progress = 1.0
        self.current_stage = '已完成'
        self.message = '综述生成完成'
        self.result = result
        self.end_time = datetime.now()
    
    def complete_error(self, error: str):
        """标记任务失败"""
        self.status = 'error'
        self.current_stage = '错误'
        self.message = f'生成失败: {error}'
        self.error = error
        self.end_time = datetime.now()

@smart_survey_bp.route('/generate', methods=['POST'])
def generate_smart_survey():
    """启动智能综述生成"""
    try:
        data = request.get_json()
        
        # 验证请求参数
        if not data:
            return jsonify({'success': False, 'message': '请求数据为空'}), 400
        
        task_ids = data.get('task_ids', [])
        topic = data.get('topic', '').strip()
        section_num = data.get('section_num', 7)
        subsection_len = data.get('subsection_len', 700)
        
        if not task_ids:
            return jsonify({'success': False, 'message': '请选择至少一个任务'}), 400
        
        if not topic:
            return jsonify({'success': False, 'message': '请输入综述主题'}), 400
        
        # 创建任务
        survey_task_id = str(uuid.uuid4())
        task = SurveyTask(survey_task_id, task_ids, topic, section_num, subsection_len)
        survey_tasks[survey_task_id] = task
        
        # 启动后台生成线程
        thread = threading.Thread(
            target=_generate_survey_background,
            args=(task,),
            daemon=True
        )
        thread.start()
        
        return jsonify({
            'success': True,
            'task_id': survey_task_id,
            'message': '综述生成任务已启动',
            'task': {
                'task_id': survey_task_id,
                'status': task.status,
                'progress': task.progress,
                'current_stage': task.current_stage,
                'message': task.message
            }
        })
        
    except Exception as e:
        logger.error(f"启动综述生成失败: {str(e)}")
        return jsonify({'success': False, 'message': f'启动失败: {str(e)}'}), 500

@smart_survey_bp.route('/progress/<task_id>', methods=['GET'])
def get_survey_progress(task_id: str):
    """获取综述生成进度"""
    try:
        task = survey_tasks.get(task_id)
        if not task:
            return jsonify({'success': False, 'message': '任务不存在'}), 404
        
        return jsonify({
            'success': True,
            'task': {
                'task_id': task.task_id,
                'status': task.status,
                'progress': task.progress,
                'current_stage': task.current_stage,
                'message': task.message,
                'start_time': task.start_time.isoformat(),
                'end_time': task.end_time.isoformat() if task.end_time else None
            }
        })
        
    except Exception as e:
        logger.error(f"获取任务进度失败: {str(e)}")
        return jsonify({'success': False, 'message': f'获取进度失败: {str(e)}'}), 500

@smart_survey_bp.route('/result/<task_id>', methods=['GET'])
def get_survey_result(task_id: str):
    """获取综述生成结果"""
    try:
        task = survey_tasks.get(task_id)
        if not task:
            return jsonify({'success': False, 'message': '任务不存在'}), 404
        
        if task.status == 'completed' and task.result:
            return jsonify({
                'success': True,
                'result': task.result
            })
        elif task.status == 'error':
            return jsonify({
                'success': False,
                'message': task.error or '生成失败'
            }), 400
        else:
            return jsonify({
                'success': False,
                'message': '任务未完成'
            }), 202
        
    except Exception as e:
        logger.error(f"获取任务结果失败: {str(e)}")
        return jsonify({'success': False, 'message': f'获取结果失败: {str(e)}'}), 500

@smart_survey_bp.route('/tasks', methods=['GET'])
def list_survey_tasks():
    """获取综述生成任务列表"""
    try:
        tasks_list = []
        for task_id, task in survey_tasks.items():
            tasks_list.append({
                'task_id': task_id,
                'topic': task.topic,
                'status': task.status,
                'progress': task.progress,
                'current_stage': task.current_stage,
                'start_time': task.start_time.isoformat(),
                'end_time': task.end_time.isoformat() if task.end_time else None,
                'source_tasks_count': len(task.task_ids)
            })
        
        # 按开始时间排序
        tasks_list.sort(key=lambda x: x['start_time'], reverse=True)
        
        return jsonify({
            'success': True,
            'tasks': tasks_list
        })
        
    except Exception as e:
        logger.error(f"获取任务列表失败: {str(e)}")
        return jsonify({'success': False, 'message': f'获取任务列表失败: {str(e)}'}), 500

@smart_survey_bp.route('/available_tasks', methods=['GET'])
def get_available_tasks():
    """获取可用的任务列表"""
    try:
        # 使用现有的比较分析历史记录作为任务列表
        tasks = db_manager.get_comparison_history(limit=200)
        
        available_tasks = []
        for task in tasks:
            task_id = task.get('task_id', '')
            if not task_id:
                continue
            
            try:
                # 获取任务的实体和关系数量
                entities = db_manager.get_entities_by_task(task_id)
                relations = db_manager.get_relations_by_task(task_id)
                
                if entities and len(entities) > 0:
                    entity_count = len(entities)
                    relation_count = len(relations) if relations else 0
                    
                    # 生成任务名称
                    original_task_name = task.get('task_name', '')
                    if not original_task_name or original_task_name.startswith('比较分析任务'):
                        task_name = f"算法演进分析 {task_id[:8]}"
                    else:
                        task_name = original_task_name
                    
                    # 只返回有足够数据的任务（至少2个实体）
                    if entity_count >= 2:
                        available_tasks.append({
                            'task_id': task_id,
                            'task_name': task_name,
                            'entity_count': entity_count,      # 前端期望的字段
                            'relation_count': relation_count,  # 前端期望的字段
                            'algorithms_count': entity_count,  # 兼容字段
                            'relations_count': relation_count, # 兼容字段
                            'datasets_count': 0,  # 暂时设为0
                            'metrics_count': 0,   # 暂时设为0
                            'created_time': task.get('start_time', ''),
                            'status': task.get('status', 'completed')
                        })
            except Exception as task_error:
                logger.warning(f"检查任务 {task_id} 时出错: {str(task_error)}")
                continue
        
        # 按创建时间倒序排列
        available_tasks.sort(key=lambda x: x.get('created_time', ''), reverse=True)
        
        return jsonify({
            'success': True,
            'tasks': available_tasks
        })
        
    except Exception as e:
        logger.error(f"获取可用任务失败: {str(e)}")
        return jsonify({'success': False, 'message': f'获取可用任务失败: {str(e)}'}), 500

def _generate_survey_background(task: SurveyTask):
    """后台生成综述"""
    try:
        logger.info(f"开始生成综述，任务ID: {task.task_id}, 主题: {task.topic}")
        
        # 阶段1: 数据验证
        task.update_progress(0.1, '数据验证', '验证任务数据完整性...')
        
        # 验证任务存在并有实体数据
        for task_id in task.task_ids:
            entities = db_manager.get_entities_by_task(task_id)
            if not entities or len(entities) == 0:
                raise ValueError(f"任务 {task_id} 没有实体数据")
        
        # 阶段2: 构建算法图
        task.update_progress(0.3, '图结构构建', '正在构建算法演进图...')
        
        # 阶段3: 脉络分析  
        task.update_progress(0.5, '脉络分析', '正在分析算法发展脉络...')
        
        # 阶段4: 内容生成
        task.update_progress(0.7, '内容生成', '正在生成综述内容...')
        
        # 调用综述生成器
        result = survey_generator.generate_survey(
            task.task_ids, 
            task.topic, 
            task.section_num, 
            task.subsection_len
        )
        
        # 阶段5: 格式化和完成
        task.update_progress(0.9, '格式化', '正在格式化输出结果...')
        
        # 添加任务元数据
        result['metadata']['task_id'] = task.task_id
        result['metadata']['source_tasks'] = task.task_ids
        result['metadata']['generation_time'] = datetime.now().isoformat()
        
        task.complete_success(result)
        logger.info(f"综述生成完成，任务ID: {task.task_id}")
        
    except Exception as e:
        error_msg = str(e)
        logger.error(f"综述生成失败，任务ID: {task.task_id}, 错误: {error_msg}")
        task.complete_error(error_msg)

# 清理旧任务的定时器
def _cleanup_old_tasks():
    """清理超过24小时的旧任务"""
    current_time = datetime.now()
    tasks_to_remove = []
    
    for task_id, task in survey_tasks.items():
        time_diff = current_time - task.start_time
        if time_diff.total_seconds() > 24 * 3600:  # 24小时
            tasks_to_remove.append(task_id)
    
    for task_id in tasks_to_remove:
        del survey_tasks[task_id]
        logger.info(f"清理旧任务: {task_id}")

# 启动清理定时器
def start_cleanup_timer():
    """启动清理定时器"""
    def cleanup_loop():
        while True:
            time.sleep(3600)  # 每小时执行一次
            _cleanup_old_tasks()
    
    cleanup_thread = threading.Thread(target=cleanup_loop, daemon=True)
    cleanup_thread.start()

# 自动启动清理定时器
start_cleanup_timer()
