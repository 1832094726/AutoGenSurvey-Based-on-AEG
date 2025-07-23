"""
AutoSurvey集成功能的路由定义
"""

from flask import Blueprint, request, jsonify, render_template, send_file
import asyncio
import logging
import uuid
from datetime import datetime
import json
import os
import tempfile
import io
from typing import Dict, Any

from app.modules.autosurvey_integration import (
    TaskSelector, EntityRelationExtractor, DataFormatConverter,
    AutoSurveyConnector, AutoSurveyConfig, AlgorithmLineageAnalyzer,
    ProcessingStatus, task_manager, get_progress_tracker, cleanup_progress_tracker,
    RetryManager, ResourceMonitor
)
from app.modules.survey_storage_manager import SurveyStorageManager
from app.modules.db_manager import DatabaseManager

# 创建蓝图
autosurvey_bp = Blueprint('autosurvey', __name__, url_prefix='/autosurvey')

# 全局变量存储处理状态
processing_tasks = {}

# 初始化存储管理器
storage_manager = SurveyStorageManager()

@autosurvey_bp.route('/')
def index():
    """AutoSurvey集成主页"""
    return render_template('autosurvey.html')

@autosurvey_bp.route('/api/tasks', methods=['GET'])
def get_tasks():
    """获取可用任务列表"""
    try:
        db_manager = DatabaseManager()
        task_selector = TaskSelector(db_manager)
        
        tasks = task_selector.get_available_tasks()
        
        return jsonify({
            "success": True,
            "tasks": tasks,
            "total": len(tasks)
        })
        
    except Exception as e:
        logging.error(f"获取任务列表失败: {str(e)}")
        return jsonify({
            "success": False,
            "message": f"获取任务列表失败: {str(e)}"
        }), 500

@autosurvey_bp.route('/api/tasks/validate', methods=['POST'])
def validate_task_selection():
    """验证任务选择"""
    try:
        data = request.get_json()
        task_ids = data.get('task_ids', [])
        
        if not task_ids:
            return jsonify({
                "success": False,
                "message": "请选择至少一个任务"
            }), 400
        
        db_manager = DatabaseManager()
        task_selector = TaskSelector(db_manager)
        
        is_valid, issues = task_selector.validate_task_selection(task_ids)
        
        return jsonify({
            "success": True,
            "valid": is_valid,
            "issues": issues
        })
        
    except Exception as e:
        logging.error(f"验证任务选择失败: {str(e)}")
        return jsonify({
            "success": False,
            "message": f"验证失败: {str(e)}"
        }), 500

@autosurvey_bp.route('/api/generate', methods=['POST'])
def start_generation():
    """开始综述生成"""
    try:
        data = request.get_json()
        task_ids = data.get('task_ids', [])
        topic = data.get('topic', '').strip()
        parameters = data.get('parameters', {})
        
        # 验证输入
        if not task_ids:
            return jsonify({
                "success": False,
                "message": "请选择任务"
            }), 400
        
        if not topic:
            return jsonify({
                "success": False,
                "message": "请输入综述主题"
            }), 400
        
        # 生成任务ID
        generation_id = str(uuid.uuid4())
        
        # 创建处理状态
        status = ProcessingStatus(generation_id)
        processing_tasks[generation_id] = status
        
        # 使用异步任务管理器启动生成任务
        await task_manager.submit_task(
            generation_id,
            run_generation_task,
            generation_id, task_ids, topic, parameters
        )
        
        return jsonify({
            "success": True,
            "generation_id": generation_id,
            "message": "综述生成任务已启动"
        })
        
    except Exception as e:
        logging.error(f"启动生成任务失败: {str(e)}")
        return jsonify({
            "success": False,
            "message": f"启动失败: {str(e)}"
        }), 500

@autosurvey_bp.route('/api/progress/<generation_id>', methods=['GET'])
def get_progress(generation_id):
    """获取生成进度"""
    try:
        if generation_id not in processing_tasks:
            return jsonify({
                "success": False,
                "message": "未找到生成任务"
            }), 404
        
        status = processing_tasks[generation_id]
        
        progress_data = {
            "status": status.status,
            "percentage": status.progress * 100,
            "stage": status.current_stage,
            "message": status.message,
            "start_time": status.start_time.isoformat(),
            "errors": status.errors
        }
        
        # 如果任务完成，添加结果
        if hasattr(status, 'result') and status.result:
            progress_data["result"] = status.result
        
        return jsonify({
            "success": True,
            "progress": progress_data
        })
        
    except Exception as e:
        logging.error(f"获取进度失败: {str(e)}")
        return jsonify({
            "success": False,
            "message": f"获取进度失败: {str(e)}"
        }), 500

@autosurvey_bp.route('/api/cancel/<generation_id>', methods=['POST'])
def cancel_generation(generation_id):
    """取消生成任务"""
    try:
        if generation_id not in processing_tasks:
            return jsonify({
                "success": False,
                "message": "未找到生成任务"
            }), 404
        
        status = processing_tasks[generation_id]
        status.status = "cancelled"
        status.message = "任务已被用户取消"
        
        return jsonify({
            "success": True,
            "message": "任务已取消"
        })
        
    except Exception as e:
        logging.error(f"取消任务失败: {str(e)}")
        return jsonify({
            "success": False,
            "message": f"取消失败: {str(e)}"
        }), 500

@autosurvey_bp.route('/api/download/<generation_id>')
def download_result(generation_id):
    """下载生成结果"""
    try:
        format_type = request.args.get('format', 'markdown').lower()
        
        if generation_id not in processing_tasks:
            return jsonify({
                "success": False,
                "message": "未找到生成任务"
            }), 404
        
        status = processing_tasks[generation_id]
        
        if not hasattr(status, 'result') or not status.result:
            return jsonify({
                "success": False,
                "message": "生成结果不可用"
            }), 404
        
        # 创建临时文件
        temp_dir = tempfile.mkdtemp()
        
        if format_type == 'markdown':
            filename = f"survey_{generation_id}.md"
            filepath = os.path.join(temp_dir, filename)
            
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(status.result.get('content', ''))
            
            return send_file(
                filepath,
                as_attachment=True,
                download_name=filename,
                mimetype='text/markdown'
            )
        
        elif format_type == 'json':
            filename = f"survey_{generation_id}.json"
            filepath = os.path.join(temp_dir, filename)
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(status.result, f, ensure_ascii=False, indent=2)
            
            return send_file(
                filepath,
                as_attachment=True,
                download_name=filename,
                mimetype='application/json'
            )
        
        else:
            return jsonify({
                "success": False,
                "message": f"不支持的格式: {format_type}"
            }), 400
        
    except Exception as e:
        logging.error(f"下载结果失败: {str(e)}")
        return jsonify({
            "success": False,
            "message": f"下载失败: {str(e)}"
        }), 500

async def run_generation_task(generation_id: str, task_ids: list, topic: str, parameters: dict):
    """运行生成任务（增强版异步处理）"""
    # 创建进度跟踪器
    progress = get_progress_tracker(generation_id, 100)
    resource_monitor = ResourceMonitor()
    retry_manager = RetryManager(max_retries=2)

    try:
        progress.update(5, "初始化", "正在初始化组件...")

        # 初始化组件
        db_manager = DatabaseManager()
        extractor = EntityRelationExtractor(db_manager)
        converter = DataFormatConverter()
        lineage_analyzer = AlgorithmLineageAnalyzer(db_manager)
        config = AutoSurveyConfig()

        # 步骤1: 提取数据（带重试）
        progress.update(10, "提取任务数据", "正在从数据库提取实体和关系...")

        async def extract_data():
            if len(task_ids) == 1:
                return extractor.extract_task_data(task_ids[0])
            else:
                return extractor.merge_multiple_tasks(task_ids)

        task_data = await retry_manager.retry_async(extract_data)

        progress.update(25, "数据提取完成", f"成功提取 {len(task_data.entities)} 个实体和 {len(task_data.relations)} 个关系")

        # 检查资源使用情况
        memory_info = resource_monitor.check_memory_usage()
        progress.add_log(f"内存使用: {memory_info.get('current_memory_mb', 0):.1f} MB")

        # 步骤2: 数据转换
        progress.update(30, "转换数据格式", "正在转换为AutoSurvey输入格式...")

        autosurvey_input = converter.convert_to_autosurvey_format(task_data, topic, parameters)

        progress.update(40, "数据转换完成", "数据格式转换成功")

        # 步骤3: 算法脉络分析
        progress.update(45, "分析算法脉络", "正在构建算法演进图谱...")

        lineage_analysis = lineage_analyzer.analyze_algorithm_lineage(task_data)

        progress.update(60, "脉络分析完成", f"识别出 {len(lineage_analysis.get('key_nodes', []))} 个关键节点")

        # 步骤4: 调用AutoSurvey（带重试）
        progress.update(65, "生成综述", "正在调用AutoSurvey生成综述...")

        async def generate_survey():
            async with AutoSurveyConnector(config) as connector:
                from app.modules.autosurvey_integration import SurveyGenerationRequest

                request = SurveyGenerationRequest(
                    topic=topic,
                    task_data=task_data,
                    parameters=parameters
                )

                return await connector.generate_survey(request)

        survey_result = await retry_manager.retry_async(generate_survey)

        progress.update(85, "综述生成完成", "AutoSurvey调用成功")

        # 步骤5: 增强结果
        progress.update(90, "增强综述内容", "正在整合算法脉络分析...")

        # 将脉络分析结果整合到综述中
        survey_result.algorithm_lineage = lineage_analysis

        # 步骤6: 完成
        progress.update(100, "生成完成", "综述生成成功完成")

        # 获取最终资源统计
        final_memory = resource_monitor.check_memory_usage()
        runtime_stats = resource_monitor.get_runtime_stats()

        # 保存结果
        result_data = {
            "survey_id": survey_result.survey_id,
            "topic": survey_result.topic,
            "content": survey_result.content,
            "outline": survey_result.outline,
            "references": survey_result.references,
            "algorithm_lineage": survey_result.algorithm_lineage,
            "quality_metrics": survey_result.quality_metrics,
            "generation_time": survey_result.generation_time.isoformat(),
            "word_count": len(survey_result.content.split()),
            "section_count": len(survey_result.outline.get("sections", [])),
            "reference_count": len(survey_result.references),
            "quality_score": survey_result.quality_metrics.get("overall_score", 0.0),
            "processing_stats": {
                "runtime": runtime_stats,
                "memory_usage": final_memory,
                "input_entities": len(task_data.entities),
                "input_relations": len(task_data.relations)
            }
        }

        # 存储综述结果
        try:
            storage_record = storage_manager.store_survey(
                survey_id=survey_result.survey_id,
                topic=survey_result.topic,
                content=survey_result.content,
                task_ids=task_ids,
                metadata=result_data,
                formats=["markdown", "html", "pdf"]
            )
            result_data["storage_record"] = {
                "survey_id": storage_record.survey_id,
                "file_paths": storage_record.file_paths,
                "version": storage_record.version
            }
        except Exception as e:
            progress.add_log(f"存储结果时出错: {str(e)}", "warning")

        # 将结果保存到任务管理器
        if generation_id in processing_tasks:
            processing_tasks[generation_id].result = result_data
            processing_tasks[generation_id].status = "completed"

        progress.add_log("任务成功完成", "success")

        return result_data

    except Exception as e:
        logging.error(f"生成任务失败 {generation_id}: {str(e)}")
        progress.add_log(f"任务失败: {str(e)}", "error")

        if generation_id in processing_tasks:
            processing_tasks[generation_id].status = "failed"
            processing_tasks[generation_id].add_error(str(e))

        raise
    finally:
        # 清理资源
        cleanup_progress_tracker(generation_id)

@autosurvey_bp.route('/api/surveys', methods=['GET'])
def list_surveys():
    """获取综述列表"""
    try:
        topic_filter = request.args.get('topic')
        status_filter = request.args.get('status')
        limit = int(request.args.get('limit', 20))
        offset = int(request.args.get('offset', 0))

        surveys = storage_manager.list_surveys(
            topic_filter=topic_filter,
            status_filter=status_filter,
            limit=limit,
            offset=offset
        )

        survey_list = []
        for survey in surveys:
            survey_list.append({
                "survey_id": survey.survey_id,
                "topic": survey.topic,
                "generation_time": survey.generation_time.isoformat(),
                "status": survey.status,
                "version": survey.version,
                "tags": survey.tags,
                "available_formats": list(survey.file_paths.keys()),
                "metadata": {
                    "word_count": survey.metadata.get("word_count", 0),
                    "section_count": survey.metadata.get("section_count", 0),
                    "reference_count": survey.metadata.get("reference_count", 0)
                }
            })

        return jsonify({
            "success": True,
            "surveys": survey_list,
            "total": len(survey_list)
        })

    except Exception as e:
        logging.error(f"获取综述列表失败: {str(e)}")
        return jsonify({
            "success": False,
            "message": f"获取列表失败: {str(e)}"
        }), 500

@autosurvey_bp.route('/api/surveys/<survey_id>', methods=['GET'])
def get_survey_detail(survey_id):
    """获取综述详情"""
    try:
        survey = storage_manager.get_survey(survey_id)

        if not survey:
            return jsonify({
                "success": False,
                "message": "综述不存在"
            }), 404

        return jsonify({
            "success": True,
            "survey": {
                "survey_id": survey.survey_id,
                "topic": survey.topic,
                "task_ids": survey.task_ids,
                "generation_time": survey.generation_time.isoformat(),
                "status": survey.status,
                "version": survey.version,
                "tags": survey.tags,
                "file_paths": survey.file_paths,
                "metadata": survey.metadata
            }
        })

    except Exception as e:
        logging.error(f"获取综述详情失败: {str(e)}")
        return jsonify({
            "success": False,
            "message": f"获取详情失败: {str(e)}"
        }), 500

@autosurvey_bp.route('/api/surveys/<survey_id>/download/<format_type>')
def download_survey_file(survey_id, format_type):
    """下载综述文件"""
    try:
        content = storage_manager.get_file_content(survey_id, format_type)

        if not content:
            return jsonify({
                "success": False,
                "message": "文件不存在"
            }), 404

        # 确定MIME类型和文件名
        mime_types = {
            "markdown": "text/markdown",
            "html": "text/html",
            "latex": "application/x-latex",
            "pdf": "application/pdf",
            "word": "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        }

        extensions = {
            "markdown": ".md",
            "html": ".html",
            "latex": ".tex",
            "pdf": ".pdf",
            "word": ".docx"
        }

        mime_type = mime_types.get(format_type, "application/octet-stream")
        extension = extensions.get(format_type, ".txt")
        filename = f"survey_{survey_id}{extension}"

        return send_file(
            io.BytesIO(content),
            mimetype=mime_type,
            as_attachment=True,
            download_name=filename
        )

    except Exception as e:
        logging.error(f"下载文件失败: {str(e)}")
        return jsonify({
            "success": False,
            "message": f"下载失败: {str(e)}"
        }), 500

@autosurvey_bp.route('/api/surveys/<survey_id>/status', methods=['PUT'])
def update_survey_status(survey_id):
    """更新综述状态"""
    try:
        data = request.get_json()
        new_status = data.get('status')

        if not new_status:
            return jsonify({
                "success": False,
                "message": "缺少状态参数"
            }), 400

        success = storage_manager.update_survey_status(survey_id, new_status)

        if success:
            return jsonify({
                "success": True,
                "message": "状态更新成功"
            })
        else:
            return jsonify({
                "success": False,
                "message": "状态更新失败"
            }), 500

    except Exception as e:
        logging.error(f"更新状态失败: {str(e)}")
        return jsonify({
            "success": False,
            "message": f"更新失败: {str(e)}"
        }), 500

@autosurvey_bp.route('/api/surveys/<survey_id>/tags', methods=['POST'])
def add_survey_tags(survey_id):
    """添加综述标签"""
    try:
        data = request.get_json()
        tags = data.get('tags', [])

        if not tags:
            return jsonify({
                "success": False,
                "message": "缺少标签参数"
            }), 400

        success = storage_manager.add_tags(survey_id, tags)

        if success:
            return jsonify({
                "success": True,
                "message": "标签添加成功"
            })
        else:
            return jsonify({
                "success": False,
                "message": "标签添加失败"
            }), 500

    except Exception as e:
        logging.error(f"添加标签失败: {str(e)}")
        return jsonify({
            "success": False,
            "message": f"添加失败: {str(e)}"
        }), 500

@autosurvey_bp.route('/api/surveys/search', methods=['GET'])
def search_surveys():
    """搜索综述"""
    try:
        query = request.args.get('q', '').strip()

        if not query:
            return jsonify({
                "success": False,
                "message": "缺少搜索关键词"
            }), 400

        surveys = storage_manager.search_surveys(query)

        survey_list = []
        for survey in surveys:
            survey_list.append({
                "survey_id": survey.survey_id,
                "topic": survey.topic,
                "generation_time": survey.generation_time.isoformat(),
                "status": survey.status,
                "tags": survey.tags
            })

        return jsonify({
            "success": True,
            "surveys": survey_list,
            "total": len(survey_list),
            "query": query
        })

    except Exception as e:
        logging.error(f"搜索综述失败: {str(e)}")
        return jsonify({
            "success": False,
            "message": f"搜索失败: {str(e)}"
        }), 500

@autosurvey_bp.route('/api/storage/stats', methods=['GET'])
def get_storage_stats():
    """获取存储统计信息"""
    try:
        stats = storage_manager.get_storage_stats()

        return jsonify({
            "success": True,
            "stats": stats
        })

    except Exception as e:
        logging.error(f"获取存储统计失败: {str(e)}")
        return jsonify({
            "success": False,
            "message": f"获取统计失败: {str(e)}"
        }), 500

# 错误处理
@autosurvey_bp.errorhandler(404)
def not_found(error):
    return jsonify({
        "success": False,
        "message": "请求的资源不存在"
    }), 404

@autosurvey_bp.errorhandler(500)
def internal_error(error):
    return jsonify({
        "success": False,
        "message": "服务器内部错误"
    }), 500
