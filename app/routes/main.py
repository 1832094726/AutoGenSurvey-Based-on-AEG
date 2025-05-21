from flask import Blueprint, render_template, request, jsonify, flash, redirect, url_for
import json
from app.modules.db_manager import db_manager

# 创建蓝图 - 使用不同名称避免冲突
routes_main = Blueprint('routes_main', __name__, url_prefix='/routes')

@routes_main.route('/comparison')
def comparison():
    """比较分析页面"""
    return render_template('comparison.html')

@routes_main.route('/comparison/results/<task_id>')
def comparison_results(task_id):
    """比较分析结果页面"""
    from app.modules.db_manager import db_manager
    import json
    
    task_status = db_manager.get_processing_status(task_id)
    
    if not task_status:
        flash('找不到指定的任务ID', 'danger')
        return redirect(url_for('routes_main.comparison'))
    
    if task_status.get('status') != '已完成':
        flash('任务尚未完成，无法显示结果', 'warning')
        return redirect(url_for('routes_main.comparison'))
    
    # 解析结果数据
    result_data = {}
    if task_status.get('result'):
        try:
            result_data = json.loads(task_status.get('result'))
        except:
            flash('无法解析任务结果数据', 'danger')
    
    # 获取指标数据
    entities_count = result_data.get('entities_count', 0)
    relations_count = result_data.get('relations_count', 0)
    metrics = result_data.get('metrics', {})
    
    return render_template('comparison_results.html', 
                           task_id=task_id,
                           entities_count=entities_count,
                           relations_count=relations_count,
                           metrics=metrics) 