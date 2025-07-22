# 论文分析后台任务实现
# 文件：app/modules/paper_analysis_tasks.py

import json
import logging
import threading
from typing import Dict, List, Optional
from app.config import Config
from app.modules.db_manager import db_manager
from app.modules.paper_analysis_utils import call_llm_with_prompt, extract_json_from_response
from app.modules.paper_analysis_prompts import (
    get_method_extraction_prompt,
    get_coverage_analysis_prompt,
    get_task_relation_coverage_prompt,
    get_entity_classification_prompt,
    get_recommended_model_for_task
)

def start_paper_task_analysis_with_config(analysis_id: str, paper_path: str, task_id: str, model_name: str):
    """启动论文与任务比较分析的后台任务（基于现有配置）"""
    
    def analysis_worker():
        try:
            # 第1步：从论文提取方法（第1次大模型调用）
            db_manager.update_processing_status(
                task_id=analysis_id,
                current_stage='提取论文方法',
                progress=20,
                message='正在从论文中提取研究方法和算法'
            )
            
            # 使用推荐的模型进行PDF分析
            pdf_model = get_recommended_model_for_task('pdf_analysis') if model_name == Config.DEFAULT_MODEL else model_name
            
            paper_response = call_llm_with_prompt(
                prompt=get_method_extraction_prompt(),
                model_name=pdf_model,
                pdf_path=paper_path
            )
            paper_methods = extract_json_from_response(paper_response)
            
            # 第2步：从任务引文数据提取方法（第2次大模型调用）
            db_manager.update_processing_status(
                task_id=analysis_id,
                current_stage='提取引文方法',
                progress=50,
                message='正在从任务引文数据中提取研究方法和算法'
            )
            
            citation_data = get_task_citation_data_with_config(task_id)
            citation_methods = convert_entities_to_methods_with_config(citation_data["entities"], model_name)
            
            # 第3步：比较计算覆盖率（第3次大模型调用）
            db_manager.update_processing_status(
                task_id=analysis_id,
                current_stage='计算覆盖率',
                progress=80,
                message='正在比较分析并计算覆盖率'
            )
            
            # 使用推荐的模型进行文本比较
            comparison_model = get_recommended_model_for_task('text_comparison') if model_name == Config.DEFAULT_MODEL else model_name
            
            coverage_prompt = get_coverage_analysis_prompt().format(
                reference_data=json.dumps(citation_methods, ensure_ascii=False, indent=2),
                comparison_data=json.dumps(paper_methods, ensure_ascii=False, indent=2)
            )
            
            coverage_response = call_llm_with_prompt(
                prompt=coverage_prompt,
                model_name=comparison_model
            )
            coverage_result = extract_json_from_response(coverage_response)
            
            # 保存最终结果
            final_results = {
                "paper_methods": paper_methods,
                "citation_methods": citation_methods,
                "coverage_analysis": coverage_result,
                "models_used": {
                    "pdf_analysis": pdf_model,
                    "text_comparison": comparison_model
                }
            }
            
            # 缓存结果到配置目录
            cache_analysis_result_to_config(analysis_id, final_results)
            
            db_manager.update_processing_status(
                task_id=analysis_id,
                current_stage='分析完成',
                progress=100,
                message='论文与任务比较分析完成',
                results=final_results
            )
            
        except Exception as e:
            logging.error(f"论文与任务分析失败: {str(e)}")
            db_manager.update_processing_status(
                task_id=analysis_id,
                current_stage='分析失败',
                progress=-1,
                message=f'分析失败: {str(e)}'
            )
    
    # 使用现有配置的线程池
    thread = threading.Thread(target=analysis_worker)
    thread.daemon = True
    thread.start()

def start_paper_paper_analysis_with_config(analysis_id: str, paper1_path: str, paper2_path: str, model_name: str):
    """启动论文与论文比较分析的后台任务（基于现有配置）"""
    
    def analysis_worker():
        try:
            # 使用推荐的模型进行PDF分析
            pdf_model = get_recommended_model_for_task('pdf_analysis') if model_name == Config.DEFAULT_MODEL else model_name
            
            # 第1步：从论文1提取方法
            db_manager.update_processing_status(
                task_id=analysis_id,
                current_stage='提取论文1方法',
                progress=25,
                message='正在从第一篇论文中提取研究方法和算法'
            )
            
            paper1_response = call_llm_with_prompt(
                prompt=get_method_extraction_prompt(),
                model_name=pdf_model,
                pdf_path=paper1_path
            )
            paper1_methods = extract_json_from_response(paper1_response)
            
            # 第2步：从论文2提取方法
            db_manager.update_processing_status(
                task_id=analysis_id,
                current_stage='提取论文2方法',
                progress=50,
                message='正在从第二篇论文中提取研究方法和算法'
            )
            
            paper2_response = call_llm_with_prompt(
                prompt=get_method_extraction_prompt(),
                model_name=pdf_model,
                pdf_path=paper2_path
            )
            paper2_methods = extract_json_from_response(paper2_response)
            
            # 第3步：比较计算覆盖率
            db_manager.update_processing_status(
                task_id=analysis_id,
                current_stage='计算覆盖率',
                progress=80,
                message='正在比较两篇论文并计算覆盖率'
            )
            
            # 使用推荐的模型进行文本比较
            comparison_model = get_recommended_model_for_task('text_comparison') if model_name == Config.DEFAULT_MODEL else model_name
            
            coverage_prompt = get_coverage_analysis_prompt().format(
                reference_data=json.dumps(paper1_methods, ensure_ascii=False, indent=2),
                comparison_data=json.dumps(paper2_methods, ensure_ascii=False, indent=2)
            )
            
            coverage_response = call_llm_with_prompt(
                prompt=coverage_prompt,
                model_name=comparison_model
            )
            coverage_result = extract_json_from_response(coverage_response)
            
            # 保存最终结果
            final_results = {
                "paper1_methods": paper1_methods,
                "paper2_methods": paper2_methods,
                "coverage_analysis": coverage_result,
                "models_used": {
                    "pdf_analysis": pdf_model,
                    "text_comparison": comparison_model
                }
            }
            
            # 缓存结果
            cache_analysis_result_to_config(analysis_id, final_results)
            
            db_manager.update_processing_status(
                task_id=analysis_id,
                current_stage='分析完成',
                progress=100,
                message='论文与论文比较分析完成',
                results=final_results
            )
            
        except Exception as e:
            logging.error(f"论文对比分析失败: {str(e)}")
            db_manager.update_processing_status(
                task_id=analysis_id,
                current_stage='分析失败',
                progress=-1,
                message=f'分析失败: {str(e)}'
            )
    
    # 启动后台线程
    thread = threading.Thread(target=analysis_worker)
    thread.daemon = True
    thread.start()

def start_relation_coverage_analysis_with_config(analysis_id: str, task_id: str, model_name: str):
    """启动任务关系覆盖率分析的后台任务（基于现有配置）"""
    
    def analysis_worker():
        try:
            # 获取关系数据（复用现有功能）
            db_manager.update_processing_status(
                task_id=analysis_id,
                current_stage='获取关系数据',
                progress=30,
                message='正在获取任务的综述和引文关系数据'
            )
            
            all_relations = db_manager.get_relations_by_task(task_id)
            review_relations = [r for r in all_relations if r.get('source') == '综述']
            citation_relations = [r for r in all_relations if r.get('source') == '引文']
            
            logging.info(f"获取到 {len(review_relations)} 个综述关系，{len(citation_relations)} 个引文关系")
            
            # 分析关系覆盖率（1次大模型调用）
            db_manager.update_processing_status(
                task_id=analysis_id,
                current_stage='分析关系覆盖率',
                progress=70,
                message='正在分析综述和引文关系的覆盖率'
            )
            
            # 使用推荐的模型进行关系分析
            relation_model = get_recommended_model_for_task('relation_analysis') if model_name == Config.DEFAULT_MODEL else model_name
            
            relation_prompt = get_task_relation_coverage_prompt().format(
                review_relations=json.dumps(review_relations, ensure_ascii=False, indent=2),
                citation_relations=json.dumps(citation_relations, ensure_ascii=False, indent=2)
            )
            
            relation_response = call_llm_with_prompt(
                prompt=relation_prompt,
                model_name=relation_model
            )
            relation_result = extract_json_from_response(relation_response)
            
            # 保存最终结果
            final_results = {
                "review_relations_count": len(review_relations),
                "citation_relations_count": len(citation_relations),
                "relation_coverage": relation_result,
                "model_used": relation_model
            }
            
            # 缓存结果
            cache_analysis_result_to_config(analysis_id, final_results)
            
            db_manager.update_processing_status(
                task_id=analysis_id,
                current_stage='分析完成',
                progress=100,
                message='任务关系覆盖率分析完成',
                results=final_results
            )
            
        except Exception as e:
            logging.error(f"关系覆盖率分析失败: {str(e)}")
            db_manager.update_processing_status(
                task_id=analysis_id,
                current_stage='分析失败',
                progress=-1,
                message=f'分析失败: {str(e)}'
            )
    
    # 启动后台线程
    thread = threading.Thread(target=analysis_worker)
    thread.daemon = True
    thread.start()

# 辅助函数（基于现有配置）

def get_task_citation_data_with_config(task_id: str) -> dict:
    """获取任务的引文数据（复用现有数据库功能）"""
    all_entities = db_manager.get_entities_by_task(task_id)
    all_relations = db_manager.get_relations_by_task(task_id)
    
    citation_entities = [e for e in all_entities if e.get('source') == '引文']
    citation_relations = [r for r in all_relations if r.get('source') == '引文']
    
    logging.info(f"获取到 {len(citation_entities)} 个引文实体，{len(citation_relations)} 个引文关系")
    
    return {
        "entities": citation_entities,
        "relations": citation_relations
    }

def convert_entities_to_methods_with_config(entities: list, model_name: str) -> dict:
    """将实体数据转换为方法-算法结构（使用现有配置）"""
    
    if not entities:
        logging.warning("没有找到引文实体数据")
        return {"methods": []}
    
    # 构建实体信息文本
    entities_text = ""
    for entity in entities:
        if 'algorithm_entity' in entity:
            algo = entity['algorithm_entity']
            entities_text += f"Algorithm: {algo.get('name', '')}\n"
            entities_text += f"Description: {algo.get('description', '')}\n"
            entities_text += f"Application: {algo.get('application_domain', '')}\n\n"
    
    if not entities_text.strip():
        logging.warning("没有找到有效的算法实体数据")
        return {"methods": []}
    
    # 使用大模型进行方法分类
    classification_prompt = get_entity_classification_prompt().format(
        entities_data=entities_text
    )
    
    try:
        response = call_llm_with_prompt(classification_prompt, model_name)
        methods_data = extract_json_from_response(response)
        
        logging.info(f"成功分类了 {len(methods_data.get('methods', []))} 个方法类别")
        return methods_data
    except Exception as e:
        logging.error(f"实体分类失败: {str(e)}")
        return {"methods": []}

def cache_analysis_result_to_config(analysis_id: str, result_data: dict) -> str:
    """将分析结果缓存到配置指定的目录"""
    import os
    
    # 使用现有配置的缓存目录
    cache_dir = Config.CACHE_DIR
    
    # 确保目录存在
    os.makedirs(cache_dir, exist_ok=True)
    
    # 生成缓存文件路径
    cache_file = os.path.join(cache_dir, f"paper_analysis_{analysis_id}.json")
    
    # 保存结果
    with open(cache_file, 'w', encoding='utf-8') as f:
        json.dump(result_data, f, ensure_ascii=False, indent=2)
    
    logging.info(f"分析结果已缓存到: {cache_file}")
    return cache_file
