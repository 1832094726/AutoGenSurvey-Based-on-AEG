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
import re
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
                 section_num: int = 7, subsection_len: int = 700, reference_num: int = 100):
        self.task_id = task_id
        self.task_ids = task_ids
        self.topic = topic
        self.section_num = section_num
        self.subsection_len = subsection_len
        self.reference_num = reference_num  # 添加参考文献数量参数
        self.status = 'initializing'
        self.progress = 0.0
        self.current_stage = 'Initializing Task'
        self.message = 'Preparing survey generation...'
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
        self.current_stage = 'Completed'
        self.message = 'Survey generation completed'
        self.result = result
        self.end_time = datetime.now()
    
    def complete_error(self, error: str):
        """标记任务失败"""
        self.status = 'error'
        self.current_stage = 'Error'
        self.message = f'Generation failed: {error}'
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
        reference_num = data.get('reference_num', 100)
        
        if not task_ids:
            return jsonify({'success': False, 'message': '请选择至少一个任务'}), 400
        
        if not topic:
            return jsonify({'success': False, 'message': '请输入综述主题'}), 400
        
        # 创建任务
        survey_task_id = str(uuid.uuid4())
        task = SurveyTask(survey_task_id, task_ids, topic, section_num, subsection_len, reference_num)
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
            'message': 'Survey generation task started',
            'task': {
                'task_id': survey_task_id,
                'status': task.status,
                'progress': task.progress,
                'current_stage': task.current_stage,
                'message': task.message
            }
        })
        
    except Exception as e:
        logger.error(f"Failed to start survey generation: {str(e)}")
        return jsonify({'success': False, 'message': f'Startup failed: {str(e)}'}), 500

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
        
        # 阶段1: 数据收集
        task.update_progress(0.1, 'Data Collection', 'Collecting entity and relation data...')
        
        # 收集所有任务的实体和关系数据
        all_entities = []
        all_relations = []
        
        for task_id in task.task_ids:
            entities = db_manager.get_entities_by_task(task_id)
            relations = db_manager.get_relations_by_task(task_id)
            
            if not entities or len(entities) == 0:
                raise ValueError(f"任务 {task_id} 没有实体数据")
            
            all_entities.extend(entities)
            all_relations.extend(relations if relations else [])
        
        logger.info(f"收集到 {len(all_entities)} 个实体和 {len(all_relations)} 个关系")
        
        # 阶段2: 数据处理
        task.update_progress(0.3, 'Data Processing', 'Processing and formatting data...')
        
        # 处理实体数据，提取算法信息
        processed_entities = []
        for entity in all_entities:
            if isinstance(entity, dict) and 'algorithm_entity' in entity:
                alg_data = entity['algorithm_entity']
                processed_entities.append({
                    'name': alg_data.get('name', ''),
                    'year': alg_data.get('year', ''),
                    'authors': alg_data.get('authors', []),
                    'task': alg_data.get('task', ''),
                    'dataset': alg_data.get('dataset', []),
                    'metrics': alg_data.get('metrics', []),
                    'architecture': alg_data.get('architecture', {}),
                    'methodology': alg_data.get('methodology', {}),
                    'source': alg_data.get('source', '')
                })
        
        # 处理关系数据
        processed_relations = []
        for relation in all_relations:
            if isinstance(relation, dict):
                processed_relations.append({
                    'from_entity': relation.get('from_entity_id', ''),
                    'to_entity': relation.get('to_entity_id', ''),
                    'relation_type': relation.get('relation_type', ''),
                    'structure': relation.get('structure', ''),
                    'detail': relation.get('detail', ''),
                    'evidence': relation.get('evidence', ''),
                    'confidence': relation.get('confidence', 0.0),
                    'source': relation.get('source', '')
                })
        
        # Phase 3: Generate survey by sections
        task.update_progress(0.5, 'Generating Outline', 'Generating survey chapter outline...')
        
        # 使用agents模块的现有方法调用qwen-long
        from openai import OpenAI
        from app.config import Config
        
        # 创建qwen-long客户端
        client = OpenAI(
            api_key=Config.QWEN_API_KEY,
            base_url=Config.QWEN_BASE_URL
        )
        
        # 第一步：生成章节大纲
        outline_prompt = _build_outline_prompt(task.topic, processed_entities, processed_relations, task.section_num)
        
        try:
            outline_response = client.chat.completions.create(
                model=Config.QWEN_MODEL,
                messages=[
                    {"role": "system", "content": "You are a senior academic researcher specialized in designing high-quality technical survey chapter structures and outlines."},
                    {"role": "user", "content": outline_prompt}
                ],
                max_tokens=None,  # 设置为None
                temperature=0.3,
                top_p=0.8,
                stream=False
            )
            
            outline_content = outline_response.choices[0].message.content
            logger.info(f"生成的大纲长度: {len(outline_content)} 字符")
            
        except Exception as e:
            logger.error(f"生成大纲失败: {str(e)}")
            raise ValueError(f"生成大纲失败: {str(e)}")
        
        # 第二步：解析章节并逐章生成内容
        task.update_progress(0.6, 'Parsing Sections', 'Parsing chapter structure...')
        
        # 简单解析章节（假设按## 标题分割）
        sections = _parse_sections_from_outline(outline_content, task.section_num)
        logger.info(f"解析到 {len(sections)} 个章节")
        
        # 第三步：逐章节生成详细内容
        full_survey_content = ""
        total_sections = len(sections)
        
        for i, section in enumerate(sections):
            progress = 0.6 + (i / total_sections) * 0.25  # 0.6-0.85的进度
            task.update_progress(progress, f'Generating Chapter {i+1}', f'Generating Chapter {i+1}: {section["title"]}...')
            
            # 为每个章节生成详细内容
            section_prompt = _build_section_prompt(
                task.topic, 
                section, 
                processed_entities, 
                processed_relations, 
                i+1, 
                total_sections
            )
            
            try:
                section_response = client.chat.completions.create(
                    model=Config.QWEN_MODEL,
                    messages=[
                        {"role": "system", "content": f"You are a senior academic researcher writing Chapter {i+1} of a comprehensive survey paper on {task.topic}. Please generate detailed, professional academic content in English."},
                        {"role": "user", "content": section_prompt}
                    ],
                    max_tokens=None,  # 设置为None
                    temperature=0.3,
                    top_p=0.8,
                    stream=False
                )
                
                section_content = section_response.choices[0].message.content
                # 清理可能的元信息
                section_content = _clean_generated_content(section_content)
                full_survey_content += section_content + "\n\n"
                logger.info(f"第{i+1}章生成完成，长度: {len(section_content)} 字符")
                
            except Exception as e:
                logger.error(f"生成第{i+1}章失败: {str(e)}")
                # 如果某章节失败，添加错误信息但继续其他章节
                full_survey_content += f"\n\n## {section['title']}\n\n[本章节生成失败: {str(e)}]\n\n"
        
        survey_content = full_survey_content.strip()
        logger.info(f"完整综述生成完成，总长度: {len(survey_content)} 字符")
        
        # 阶段4: 结果处理
        task.update_progress(0.9, 'Result Processing', 'Processing generation results...')
        
        # 构建返回结果
        result = {
            'content': {
                'markdown': survey_content,  # 前端期望的格式
                'formats': ['markdown']
            },
            'algorithm_lineage': {
                'entities_count': len(processed_entities),
                'relations_count': len(processed_relations),
                'summary': f'基于{len(task.task_ids)}个任务分析算法演进脉络，发现{len(processed_entities)}个关键算法实体和{len(processed_relations)}个演化关系。'
            },
            'task_ids': task.task_ids,
            'parameters': {
                'section_num': task.section_num,
                'subsection_len': task.subsection_len,
                'rag_num': len(processed_entities)  # 用实体数量作为参考数量
            },
            'metadata': {
                'task_id': task.task_id,
                'source_tasks': task.task_ids,
                'topic': task.topic,
                'entities_count': len(processed_entities),
                'relations_count': len(processed_relations),
                'generation_time': datetime.now().isoformat(),
                'model': 'qwen-long',
                'content_length': len(survey_content),
                'word_count': len(survey_content.split()) if survey_content else 0
            }
        }
        
        task.complete_success(result)
        logger.info(f"综述生成完成，任务ID: {task.task_id}")
        
    except Exception as e:
        error_msg = str(e)
        logger.error(f"综述生成失败，任务ID: {task.task_id}, 错误: {error_msg}")
        task.complete_error(error_msg)

def _build_outline_prompt(topic: str, entities: list, relations: list, section_num: int) -> str:
    """构建大纲生成提示词"""
    
    # 简化的实体概要 - 使用所有实体，不限制数量
    entities_summary = f"Total {len(entities)} algorithm entities, covering years from 2000 to 2024, including:"
    entity_names = [entity['name'] for entity in entities if entity.get('name')]
    entities_summary += f"{', '.join(entity_names[:20])} and other algorithms."  # 只在显示时限制，实际使用所有实体
    
    # 简化的关系概要 - 使用所有关系，不限制数量
    relations_summary = f"Total {len(relations)} evolutionary relationships, main relationship types include:"
    relation_types = list(set([rel['relation_type'] for rel in relations if rel.get('relation_type')]))
    relations_summary += f"{', '.join(relation_types)} etc."
    
    prompt = f"""As an academic research expert, please design a detailed chapter outline for the survey on "{topic}".

## Data Overview:
- {entities_summary}
- {relations_summary}

## Outline Requirements:
Please design a detailed outline for {section_num} main chapters, each chapter should include:
1. Chapter title
2. Brief chapter description
3. Main subsections (3-5 subsections)

## Output Format:
Please generate {section_num} chapters according to requirements, strictly follow the format below for each chapter:

Format Example:
## 1. Chapter Title
**Summary**: Brief description of the chapter
**Subsections**: 
- Subsection 1
- Subsection 2
- Subsection 3

Please ensure:
1. The first chapter is usually an abstract or introduction
2. The last chapter is usually a conclusion
3. Middle chapters are reasonably arranged according to the {topic} theme
4. Each chapter must have complete summary and subsections information
5. Chapter numbers from 1 to {section_num}
6. Focus on academic content structure, not formatting details

Please strictly follow the above format to generate the outline, ensuring all {section_num} chapters are included."""

    return prompt

def _parse_sections_from_outline(outline_content: str, expected_section_num: int) -> list:
    """从大纲中解析章节信息"""
    sections = []
    
    try:
        # 按## 分割章节
        parts = outline_content.split('## ')
        
        for part in parts[1:]:  # 跳过第一个空白部分
            lines = part.strip().split('\n')
            if not lines:
                continue
                
            # 解析标题
            title_line = lines[0].strip()
            
            # 解析摘要/总结
            summary = ""
            subsections = []
            
            for line in lines[1:]:
                line = line.strip()
                if line.startswith('**Summary**') or line.startswith('**Abstract**') or line.startswith('**摘要**'):
                    summary = line.replace('**Summary**:', '').replace('**Summary**：', '').replace('**Abstract**:', '').replace('**Abstract**：', '').replace('**摘要**:', '').replace('**摘要**：', '').strip()
                elif line.startswith('**Subsections**') or line.startswith('**子章节**'):
                    continue
                elif line.startswith('- '):
                    subsections.append(line[2:].strip())
            
            # 确保标题格式正确（去掉多余的数字）
            if title_line and not title_line.startswith(str(len(sections) + 1)):
                # 添加正确的章节编号
                title_line = f"{len(sections) + 1}. {title_line.lstrip('0123456789. ')}"
            
            section = {
                'title': title_line,
                'summary': summary,
                'subsections': subsections,
                'word_count': '2000-3000 words'  # 提供默认字数范围
            }
            sections.append(section)
        
        logger.info(f"从大纲解析出 {len(sections)} 个章节")
        
    except Exception as e:
        logger.warning(f"解析大纲失败: {str(e)}，使用默认章节结构")
        sections = []
    
    # 如果解析失败或章节数不够，创建默认的章节结构
    if len(sections) < expected_section_num:
        logger.info(f"章节数量不足({len(sections)})，创建标准{expected_section_num}章节结构")
        
        # 根据期望的章节数创建默认结构
        default_sections = []
        for i in range(expected_section_num):
            if i == 0:
                # 第一章：摘要/引言
                default_sections.append({
                    'title': f'1. Abstract', 
                    'summary': 'Survey abstract and overview', 
                    'subsections': ['Research Background', 'Main Findings', 'Article Contributions'], 
                    'word_count': '500-800 words'
                })
            elif i == expected_section_num - 1:
                # 最后一章：结论
                default_sections.append({
                    'title': f'{i+1}. Conclusion', 
                    'summary': 'Summary and future prospects', 
                    'subsections': ['Main Contributions', 'Limitations', 'Future Work'], 
                    'word_count': '1000-1500 words'
                })
            else:
                # 中间章节：根据位置分配内容
                chapter_titles = [
                    'Introduction',
                    'Related Work', 
                    'Algorithm Evolution',
                    'Key Technologies',
                    'Applications and Evaluation',
                    'Future Trends',
                    'Technical Challenges',
                    'Method Comparison'
                ]
                
                # 选择合适的标题
                title_idx = min(i-1, len(chapter_titles)-1)
                chapter_title = chapter_titles[title_idx]
                
                default_sections.append({
                    'title': f'{i+1}. {chapter_title}', 
                    'summary': f'Chapter {i+1} content', 
                    'subsections': ['Technical Background', 'Main Methods', 'Current Development'], 
                    'word_count': '2000-3000 words'
                })
        
        sections = default_sections
    
    # 确保每个章节都有正确的编号
    for i, section in enumerate(sections):
        if not section['title'].startswith(str(i + 1)):
            # 重新编号
            title_without_number = section['title'].lstrip('0123456789. ')
            section['title'] = f"{i + 1}. {title_without_number}"
    
    logger.info(f"最终确定 {len(sections)} 个章节")
    return sections

def _build_section_prompt(topic: str, section: dict, entities: list, relations: list, section_num: int, total_sections: int) -> str:
    """构建单个章节的生成提示词"""
    
    # 使用所有实体和关系，不再限制数量（qwen-long有大的context window）
    relevant_entities = entities  # 使用所有实体
    relevant_relations = relations # 使用所有关系
    
    # 构建实体信息（为了避免prompt过长，只显示前50个的详细信息）
    entities_text = ""
    display_entities = relevant_entities[:50]  # 只显示前50个详细信息
    for i, entity in enumerate(display_entities):
        entities_text += f"- {entity['name']} ({entity['year']}): {entity.get('task', 'N/A')}\n"
        if entity.get('architecture') and isinstance(entity['architecture'], dict):
            components = entity['architecture'].get('components', [])
            if components and isinstance(components, list):
                entities_text += f"  Architecture: {', '.join(components[:3])}\n"
    
    if len(relevant_entities) > 50:
        entities_text += f"\n... and {len(relevant_entities) - 50} other related entities\n"
    
    # 构建关系信息（为了避免prompt过长，只显示前30个的详细信息）
    relations_text = ""
    display_relations = relevant_relations[:30]  # 只显示前30个详细信息
    for i, relation in enumerate(display_relations):
        relations_text += f"- {relation['from_entity']} → {relation['to_entity']} ({relation['relation_type']})\n"
        if relation.get('detail'):
            relations_text += f"  Details: {relation['detail'][:100]}...\n"
    
    if len(relevant_relations) > 30:
        relations_text += f"\n... and {len(relevant_relations) - 30} other related relationships\n"
    
    prompt = f"""Please write Chapter {section_num} of the survey on "{topic}": {section['title']}

## Chapter Requirements:
- Chapter should cover: {section['summary']}
- Include these main topics: {', '.join(section['subsections'])}
- Target approximately {section['word_count']} words

## Available Data:
### Related Algorithm Entities:
{entities_text}

### Related Evolutionary Relationships:
{relations_text}

## Writing Requirements:
1. This is Chapter {section_num} of {total_sections} chapters
2. Use academic language, professional and rigorous
3. Make full use of the provided algorithm entity and evolutionary relationship information
4. Each subsection must have detailed content
5. Include specific technical analysis and case studies
6. Maintain logical coherence and academic standards
7. Appropriately cite algorithm names and technical details
8. Write in fluent English with proper academic writing style
9. **IMPORTANT**: Only generate the actual chapter content - do not include meta-information like word counts, formatting instructions, or structural notes
10. Start directly with the chapter title and content

Please generate the complete chapter content with proper academic structure:"""

    return prompt

def _clean_generated_content(content: str) -> str:
    """清理生成内容中的元信息和格式化说明"""
    import re
    
    # 移除常见的元信息模式
    patterns_to_remove = [
        r'Word Count:.*?\n',  # 移除字数统计
        r'Estimated Word Count:.*?\n',  # 移除预计字数
        r'Target Word Count:.*?\n',  # 移除目标字数
        r'---\n.*?---\n',  # 移除分隔线包围的内容
        r'\*\*Word Count\*\*:.*?\n',  # 移除加粗的字数统计
        r'\*\*Target.*?\*\*:.*?\n',  # 移除目标相关的元信息
        r'Abstract\s*$',  # 移除单独的Abstract行
        r'乱入其他内容',  # 移除任何中文乱入内容
        r'\n\s*\n\s*\n',  # 合并多个连续空行
    ]
    
    cleaned_content = content
    for pattern in patterns_to_remove:
        cleaned_content = re.sub(pattern, '\n', cleaned_content, flags=re.MULTILINE | re.IGNORECASE)
    
    # 移除开头和结尾的多余空白
    cleaned_content = cleaned_content.strip()
    
    # 确保章节之间有适当的间距
    cleaned_content = re.sub(r'\n{3,}', '\n\n', cleaned_content)
    
    return cleaned_content

def _build_survey_prompt(topic: str, entities: list, relations: list, section_num: int) -> str:
    """构建综述生成提示词"""
    
    # 使用所有实体，不再限制数量（qwen-long有大的context window）
    # 但为了控制prompt长度，只显示前200个实体的详细信息
    display_entities = entities[:200]
    entities_text = ""
    for i, entity in enumerate(display_entities):
        entities_text += f"{i+1}. **{entity['name']}** ({entity['year']})\n"
        if entity.get('authors'):
            authors_list = entity['authors'][:5] if isinstance(entity['authors'], list) else []
            if authors_list:
                entities_text += f"   - Authors: {', '.join(authors_list)}{'...' if len(entity.get('authors', [])) > 5 else ''}\n"
        if entity.get('task'):
            entities_text += f"   - Application Domain: {entity['task']}\n"
        if entity.get('dataset') and isinstance(entity['dataset'], list):
            datasets = entity['dataset'][:3]
            if datasets:
                entities_text += f"   - Datasets: {', '.join(datasets)}{'...' if len(entity.get('dataset', [])) > 3 else ''}\n"
        if entity.get('metrics') and isinstance(entity['metrics'], list):
            metrics = entity['metrics'][:3]
            if metrics:
                entities_text += f"   - Evaluation Metrics: {', '.join(metrics)}{'...' if len(entity.get('metrics', [])) > 3 else ''}\n"
        if entity.get('architecture') and isinstance(entity['architecture'], dict):
            components = entity['architecture'].get('components', [])
            if components and isinstance(components, list):
                comp_list = components[:4]
                entities_text += f"   - Architecture Components: {', '.join(comp_list)}{'...' if len(components) > 4 else ''}\n"
            connections = entity['architecture'].get('connections', [])
            if connections and isinstance(connections, list):
                conn_list = connections[:3]
                entities_text += f"   - Connection Methods: {', '.join(conn_list)}{'...' if len(connections) > 3 else ''}\n"
        if entity.get('methodology') and isinstance(entity['methodology'], dict):
            training = entity['methodology'].get('training_strategy', [])
            if training and isinstance(training, list):
                train_list = training[:3]
                entities_text += f"   - Training Strategy: {', '.join(train_list)}{'...' if len(training) > 3 else ''}\n"
        entities_text += f"   - Source: {entity.get('source', 'Unknown')}\n\n"
    
    if len(entities) > 200:
        entities_text += f"\n... and {len(entities) - 200} other related entities\n"
    
    # 使用所有关系，不再限制数量
    # 但为了控制prompt长度，只显示前100个关系的详细信息
    display_relations = relations[:100]
    relations_text = ""
    for i, relation in enumerate(display_relations):
        relations_text += f"{i+1}. **{relation['from_entity']}** → **{relation['to_entity']}**\n"
        relations_text += f"   - Relationship Type: {relation['relation_type']}\n"
        if relation.get('structure'):
            relations_text += f"   - Impact Structure: {relation['structure']}\n"
        if relation.get('detail'):
            relations_text += f"   - Detailed Description: {relation['detail']}\n"
        if relation.get('evidence'):
            relations_text += f"   - Supporting Evidence: {relation['evidence']}\n"
        if relation.get('confidence'):
            relations_text += f"   - Confidence Level: {relation['confidence']}\n"
        relations_text += f"   - Source: {relation.get('source', 'Unknown')}\n\n"
    
    if len(relations) > 100:
        relations_text += f"\n... and {len(relations) - 100} other related relationships\n"
    
    prompt = f"""As a senior academic researcher, please generate a detailed academic survey on "{topic}" based on the following algorithm entity and evolutionary relationship data.

## Important Requirements:
1. This is an **academic survey paper**, requiring detailed, in-depth, and professional content
2. Must make full use of the provided {len(entities)} algorithm entities and {len(relations)} evolutionary relationships
3. Each chapter must contain rich technical details and in-depth analysis
4. The generated survey should reach a length of **15000-20000 words**
5. Focus on analyzing evolutionary relationships between algorithms, technical innovations, and development trends
6. Reference AutoSurvey's survey generation methodology to ensure academic rigor and completeness

## Algorithm Entity Data ({len(entities)} algorithms):
{entities_text}

## Evolutionary Relationship Data ({len(relations)} relationships):
{relations_text}

## Survey Structure Requirements:
Please generate a detailed survey with {section_num} main chapters. Based on the set number of chapters, reasonably allocate content:

- If 7 chapters: Recommend including Abstract, Introduction, Related Work, Algorithm Development History, Key Technical Analysis, Development Trends, Conclusion
- If 5 chapters: Recommend including Abstract, Introduction, Core Technical Analysis, Development Trends, Conclusion  
- If other numbers: Please reasonably plan chapter content allocation

## Writing Requirements:
- **Every algorithm must be mentioned and analyzed**, cannot omit important entities
- **Every evolutionary relationship must be reflected in corresponding chapters**
- Use academic language, ensuring content is professional and accurate
- Provide detailed technical analysis and in-depth discussion
- Include specific technical details and experimental result analysis
- Maintain logical coherence between chapters
- Appropriately cite relevant algorithms and research work
- Use standard academic paper format
- Reasonably allocate content length based on chapter numbers, overall target 15000-20000 words
- **Write entirely in English with proper academic writing style**

Please be sure to generate a **complete, detailed, high-quality** academic survey that fully demonstrates the technological development landscape of the {topic} field."""

    return prompt

@smart_survey_bp.route('/download/<task_id>/<format>', methods=['GET'])
def download_survey_result(task_id: str, format: str):
    """下载智能综述生成结果文件"""
    try:
        task = survey_tasks.get(task_id)
        if not task:
            return jsonify({'success': False, 'message': '任务不存在'}), 404
        
        if task.status != 'completed' or not task.result:
            return jsonify({'success': False, 'message': '任务未完成或无结果'}), 404
        
        # 获取生成的内容
        content = task.result.get('content', {})
        markdown_content = content.get('markdown', '')
        
        if not markdown_content:
            return jsonify({'success': False, 'message': '无可下载内容'}), 404
        
        # 根据格式返回相应内容
        if format.lower() == 'markdown':
            from flask import Response
            
            # 生成文件名
            filename = f"{task.topic}_survey_{task.task_id[:8]}.md"
            
            return Response(
                markdown_content,
                mimetype='text/markdown',
                headers={
                    'Content-Disposition': f'attachment; filename="{filename}"',
                    'Content-Type': 'text/markdown; charset=utf-8'
                }
            )
        
        elif format.lower() == 'txt':
            from flask import Response
            
            # 生成纯文本版本（去除markdown格式）
            import re
            text_content = re.sub(r'[#*`_~\[\]()>-]', '', markdown_content)
            text_content = re.sub(r'\n{3,}', '\n\n', text_content)
            
            filename = f"{task.topic}_survey_{task.task_id[:8]}.txt"
            
            return Response(
                text_content,
                mimetype='text/plain',
                headers={
                    'Content-Disposition': f'attachment; filename="{filename}"',
                    'Content-Type': 'text/plain; charset=utf-8'
                }
            )
        
        elif format.lower() == 'html':
            from flask import Response
            
            # 简单的markdown到HTML转换
            html_content = _markdown_to_html(markdown_content)
            filename = f"{task.topic}_survey_{task.task_id[:8]}.html"
            
            return Response(
                html_content,
                mimetype='text/html',
                headers={
                    'Content-Disposition': f'attachment; filename="{filename}"',
                    'Content-Type': 'text/html; charset=utf-8'
                }
            )
        
        else:
            return jsonify({'success': False, 'message': f'不支持的文件格式: {format}'}), 400
        
    except Exception as e:
        logger.error(f"下载文件失败: {str(e)}")
        return jsonify({'success': False, 'message': f'下载失败: {str(e)}'}), 500

def _markdown_to_html(markdown_content: str) -> str:
    """简单的Markdown到HTML转换"""
    html = markdown_content
    
    # 替换标题
    html = re.sub(r'^# (.+)$', r'<h1>\1</h1>', html, flags=re.MULTILINE)
    html = re.sub(r'^## (.+)$', r'<h2>\1</h2>', html, flags=re.MULTILINE)
    html = re.sub(r'^### (.+)$', r'<h3>\1</h3>', html, flags=re.MULTILINE)
    html = re.sub(r'^#### (.+)$', r'<h4>\1</h4>', html, flags=re.MULTILINE)
    
    # 替换粗体和斜体
    html = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', html)
    html = re.sub(r'\*(.+?)\*', r'<em>\1</em>', html)
    
    # 替换换行
    html = html.replace('\n\n', '</p><p>')
    html = html.replace('\n', '<br>')
    
    # 包装在HTML文档中
    html_doc = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>综述文档</title>
    <style>
        body {{ font-family: 'Times New Roman', serif; line-height: 1.6; margin: 40px; }}
        h1 {{ color: #2c3e50; border-bottom: 2px solid #3498db; }}
        h2 {{ color: #34495e; margin-top: 30px; }}
        h3 {{ color: #7f8c8d; }}
        p {{ text-align: justify; margin-bottom: 15px; }}
        strong {{ color: #2c3e50; }}
    </style>
</head>
<body>
    <p>{html}</p>
</body>
</html>"""
    
    return html_doc

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
