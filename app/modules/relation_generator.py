import os
import json
import logging
import time
from typing import List, Dict, Optional, Tuple
from openai import OpenAI
from app.config import Config
from app.modules.db_manager import db_manager

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_entities(file_path: str = None) -> List[Dict]:
    """
    从文件加载实体数据
    
    Args:
        file_path: 实体文件路径，默认使用Config中的配置
        
    Returns:
        实体列表
    """
    if not file_path:
        file_path = os.path.join(Config.DATA_DIR, 'entities.json')
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            entities = json.load(f)
            logging.info(f"成功加载 {len(entities)} 个实体")
            return entities
    except Exception as e:
        logging.error(f"加载实体数据时出错: {str(e)}")
        return []

def load_relations(file_path: str = None) -> List[Dict]:
    """
    从文件加载关系数据
    
    Args:
        file_path: 关系文件路径，默认使用Config中的配置
        
    Returns:
        关系列表
    """
    if not file_path:
        file_path = os.path.join(Config.DATA_DIR, 'relations.json')
    
    try:
        if os.path.exists(file_path):
            with open(file_path, 'r', encoding='utf-8') as f:
                relations = json.load(f)
                logging.info(f"成功加载 {len(relations)} 个关系")
                return relations
        else:
            logging.warning(f"关系文件不存在: {file_path}，将创建新文件")
            return []
    except Exception as e:
        logging.error(f"加载关系数据时出错: {str(e)}")
        return []

def save_relations(relations: List[Dict], file_path: str = None) -> bool:
    """
    保存关系数据到文件
    
    Args:
        relations: 关系列表
        file_path: 保存路径，默认使用Config中的配置
        
    Returns:
        是否保存成功
    """
    if not file_path:
        file_path = os.path.join(Config.DATA_DIR, 'relations.json')
    
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(relations, f, ensure_ascii=False, indent=2)
            logging.info(f"成功保存 {len(relations)} 个关系到文件")
        return True
    except Exception as e:
        logging.error(f"保存关系数据时出错: {str(e)}")
        return False

def extract_entity_info(entities: List[Dict]) -> List[Dict]:
    """
    提取实体的关键信息，用于关系生成
    
    Args:
        entities: 完整实体列表
        
    Returns:
        简化的实体信息列表
    """
    simplified_entities = []
    
    for entity in entities:
        if 'algorithm_entity' in entity:
            algo = entity['algorithm_entity']
            entity_type = algo.get('entity_type', 'Algorithm')
            
            simplified = {
                'id': algo.get('algorithm_id', '').upper(),
                'type': entity_type,
                'name': algo.get('name', ''),
                'year': algo.get('year', 0),
                'task': algo.get('task', ''),
                'title': algo.get('title', '')
            }
            
            # 根据不同实体类型提取特定字段
            if entity_type == 'Algorithm':
                # 算法实体特有字段
                simplified['methodology'] = algo.get('methodology', {}).get('training_strategy', [])
                simplified['architecture'] = algo.get('architecture', {}).get('components', [])
            elif entity_type == 'Dataset':
                # 数据集实体特有字段
                simplified['domain'] = algo.get('domain', '')
                simplified['size'] = algo.get('size', '')
            elif entity_type == 'Metric':
                # 指标实体特有字段
                simplified['focus'] = algo.get('focus', '')
                simplified['range'] = algo.get('range', '')
            
            simplified_entities.append(simplified)
    
    return simplified_entities

def generate_relation_prompt(entities: List[Dict], papers: List[Dict]) -> str:
    """
    生成用于关系识别的提示词
    
    Args:
        entities: 算法实体列表
        papers: 论文列表
        
    Returns:
        提示词字符串
    """
    # 生成实体描述
    entity_descriptions = []
    for i, entity in enumerate(entities):
        entity_descriptions.append(f"""
Entity {i+1}:
- ID: {entity['id']}
- Name: {entity['name']}
- Type: {entity['type']}
- Year: {entity['year']}
- Description: {entity['description']}
""")
    
    # 生成论文描述
    paper_descriptions = []
    for i, paper in enumerate(papers):
        paper_descriptions.append(f"""
Paper {i+1}:
- Title: {paper['title']}
- Authors: {paper['authors']}
- Year: {paper['year']}
- Abstract: {paper['abstract']}
""")
    
    # 生成提示词
    prompt = f"""分析以下算法实体和论文，判断它们之间是否存在演化关系。实体类型包括算法(Algorithm)、数据集(Dataset)和指标(Metric)。
    
可能的关系类型包括:
1. Improve（改进）：论文改进了算法的某些方面
2. Extend（扩展）：论文扩展了算法的功能或应用范围
3. Adapt（改编）：论文将算法应用到新领域或环境
4. Replace（替代）：论文替代了算法的功能
5. Inspire（启发）：算法启发了论文的创建
6. Use（使用）：论文使用了算法作为组件、数据集或指标
7. Evaluate（评估）：论文用于评估算法的性能
8. Compare（比较）：论文与算法进行了直接比较

请分析以下{str(len(entities))}个实体和{str(len(papers))}篇论文，对于每个实体和论文，如果存在关系，请以JSON格式给出:
{{
  "entity_id": "实体ID",
  "paper_id": "论文ID",
  "relation_type": "关系类型",
  "structure": "结构变化类别(如Architecture.Component, Methodology.Training, Feature.Processing等)",
  "detail": "详细说明关系内容",
  "evidence": "推断依据",
  "confidence": "0.0-1.0之间的置信度分数"
}}

如果不存在明确关系，请返回:
{{
  "entity_id": "实体ID",
  "paper_id": "论文ID",
  "relation_type": null,
  "confidence": 0.0
}}

请确保所有ID使用大写。只回复JSON格式，不要添加其他解释。以数组形式返回所有结果:
[
  {{result for entity 1 and paper 1}},
  {{result for entity 1 and paper 2}},
  ...
]

实体信息如下:
{"".join(entity_descriptions)}

论文信息如下:
{"".join(paper_descriptions)}
"""
    return prompt

def call_llm_api(prompt: str, temperature: float = 0.1) -> Dict:
    """
    调用大模型API获取关系识别结果
    
    Args:
        prompt: 提示词
        temperature: 采样温度，控制输出随机性
        
    Returns:
        API响应结果
    """
    client = OpenAI(
        api_key=Config.QWEN_API_KEY,
        base_url=Config.QWEN_BASE_URL
    )
    
    try:
        logging.info("正在调用大模型API...")
        start_time = time.time()
        
        # 尝试流式输出
        response_content = ""
        chunk_count = 0
        try:
            completion = client.chat.completions.create(
                model="qwen-long",
                messages=[
                    {"role": "system", "content": "你是一个专注于分析算法演化关系的AI助手。"},
                    {"role": "user", "content": prompt}
                ],
                temperature=temperature,
                max_tokens=None,  # 不限制token数量
                stream=True
            )
            for chunk in completion:
                chunk_count += 1
                if hasattr(chunk, 'choices') and chunk.choices and hasattr(chunk.choices[0], 'delta') and chunk.choices[0].delta.content:
                    content_piece = chunk.choices[0].delta.content
                    response_content += content_piece
                    if chunk_count % 10 == 0:
                        logging.info(f"响应块 #{chunk_count}，当前响应长度: {len(response_content)}")
            logging.info(f"响应接收完成，共 {chunk_count} 个响应块，总长度: {len(response_content)}")
            
            # 解析JSON结果
            # 尝试从响应文本中提取关系数据
            try:
                import re
                from app.modules.agents import extract_json_from_text
                
                # 使用改进的JSON提取函数
                json_str = extract_json_from_text(response_content)
                if json_str:
                    relations = json.loads(json_str)
                    if isinstance(relations, list):
                        logging.info(f"成功提取 {len(relations)} 个关系信息")
                        return {
                            "success": True,
                            "data": relations
                        }
                
                logging.warning("未能从响应中提取有效的关系信息")
                return {
                    "success": False,
                    "message": "未能提取有效的关系信息",
                    "raw_response": response_content[:1000] + "..." if len(response_content) > 1000 else response_content
                }
            except json.JSONDecodeError as e:
                logging.error(f"解析JSON响应出错: {str(e)}")
                return {
                    "success": False,
                    "message": f"解析响应失败: {str(e)}",
                    "raw_response": response_content[:1000] + "..." if len(response_content) > 1000 else response_content
                }
                
        except Exception as api_err:
            logging.error(f"API调用出错: {str(api_err)}")
            import traceback
            logging.error(traceback.format_exc())
            return {
                "success": False,
                "message": f"API调用失败: {str(api_err)}"
            }
            
        elapsed_time = time.time() - start_time
        logging.info(f"API调用完成，耗时: {elapsed_time:.2f}秒")
        
    except Exception as e:
        logging.error(f"调用关系识别API出错: {str(e)}")
        import traceback
        logging.error(traceback.format_exc())
        return {
            "success": False,
            "message": f"处理出错: {str(e)}"
        }

def parse_api_response(response: Dict) -> List[Dict]:
    """
    解析API返回的关系数据
    
    Args:
        response: API调用结果
        
    Returns:
        解析后的关系列表
    """
    if not response.get("success", False):
        logging.error("API调用失败，无法解析响应")
        return []
    
    # 兼容新的响应格式
    if "data" in response:
        # 新格式：直接返回data字段中的关系列表
        relation_data = response.get("data", [])
        
        # 确保结果是列表
        if not isinstance(relation_data, list):
            relation_data = [relation_data]
        
        # 过滤和规范化关系
        valid_relations = []
        for relation in relation_data:
            if relation.get("relation_type") not in [None, "", "null"]:
                # 确保所有ID都是大写
                if "from_entity" in relation:
                    relation["from_entity"] = str(relation["from_entity"]).upper()
                if "to_entity" in relation:
                    relation["to_entity"] = str(relation["to_entity"]).upper()
                
                # 移除pair_id字段
                if "pair_id" in relation:
                    del relation["pair_id"]
                
                valid_relations.append(relation)
                logging.info("找到有效关系: " + str(relation.get('from_entity', '')) + " -> " + str(relation.get('to_entity', '')) + " (" + str(relation.get('relation_type', '')) + ")")
        
        return valid_relations
    
    # 兼容旧的响应格式
    content = response.get("content", "")
    if not content:
        logging.error("API响应内容为空")
        return []
    
    try:
        # 尝试从文本中提取JSON部分
        content = content.strip()
        
        # 如果内容被反引号包围，提取JSON部分
        if "```json" in content and "```" in content:
            start_idx = content.find("```json") + 7
            end_idx = content.rfind("```")
            if start_idx > 7 and end_idx > start_idx:
                content = content[start_idx:end_idx].strip()
        elif "```" in content:
            start_idx = content.find("```") + 3
            end_idx = content.rfind("```")
            if start_idx > 3 and end_idx > start_idx:
                content = content[start_idx:end_idx].strip()
        
        # 解析JSON
        relation_data = json.loads(content)
        
        # 确保结果是列表
        if not isinstance(relation_data, list):
            relation_data = [relation_data]
        
        # 过滤掉没有关系的条目
        valid_relations = []
        for relation in relation_data:
            if relation.get("relation_type") not in [None, "", "null"]:
                # 确保所有ID都是大写
                if "from_entity" in relation:
                    relation["from_entity"] = str(relation["from_entity"]).upper()
                if "to_entity" in relation:
                    relation["to_entity"] = str(relation["to_entity"]).upper()
                
                # 移除pair_id字段
                if "pair_id" in relation:
                    del relation["pair_id"]
                
                valid_relations.append(relation)
                logging.info("找到有效关系: " + str(relation.get('from_entity', '')) + " -> " + str(relation.get('to_entity', '')) + " (" + str(relation.get('relation_type', '')) + ")")
        
        return valid_relations
    except Exception as e:
        logging.error("解析API响应时出错: " + str(e))
        logging.debug("原始响应内容: " + content)
        return []

def generate_potential_entity_pairs(entities: List[Dict]) -> List[Dict]:
    """
    生成可能存在关系的实体对
    
    Args:
        entities: 实体信息列表
        
    Returns:
        实体对列表
    """
    pairs = []
    
    # 按类型分组实体
    algorithms = [e for e in entities if e['type'] == 'Algorithm']
    datasets = [e for e in entities if e['type'] == 'Dataset']
    metrics = [e for e in entities if e['type'] == 'Metric']
    
    # 1. 算法之间的关系 - 按时间排序，假设新算法可能基于旧算法
    sorted_algos = sorted(algorithms, key=lambda x: x.get('year', 0))
    
    # 生成算法间时间上可能存在的关系对
    for i in range(len(sorted_algos)):
        for j in range(i+1, len(sorted_algos)):
            from_algo = sorted_algos[i]
            to_algo = sorted_algos[j]
            
            # 如果两个算法的发布时间相差不超过8年，且任务相同，可能存在关系
            year_diff = to_algo.get('year', 0) - from_algo.get('year', 0)
            same_task = to_algo.get('task', '') and to_algo.get('task', '') == from_algo.get('task', '')
            
            if 0 < year_diff <= 8 or same_task:
                pairs.append({
                    'from_entity': from_algo,
                    'to_entity': to_algo,
                    'potential_relation': 'Algorithm-Algorithm'
                })
    
    # 2. 算法与数据集的关系
    for algo in algorithms:
        for dataset in datasets:
            # 如果算法晚于数据集出现，可能使用了该数据集
            if algo.get('year', 0) >= dataset.get('year', 0):
                pairs.append({
                    'from_entity': dataset,
                    'to_entity': algo,
                    'potential_relation': 'Dataset-Algorithm'
                })
    
    # 3. 算法与指标的关系
    for algo in algorithms:
        for metric in metrics:
            # 算法可能使用或关注特定指标
            pairs.append({
                'from_entity': metric,
                'to_entity': algo,
                'potential_relation': 'Metric-Algorithm'
            })
    
    logging.info("生成了 " + str(len(pairs)) + " 个潜在关系对")
    return pairs

def generate_relations(max_pairs: int = 100, batch_size: int = 5):
    """
    主函数: 生成并保存算法实体间的演化关系
    
    Args:
        max_pairs: 最大处理的实体对数量，避免API调用过多
        batch_size: 每次API调用处理的实体对数量
    """
    logging.info("==== 开始生成算法演化关系 ====")
    
    # 1. 加载实体和已有关系
    entities = load_entities()
    existing_relations = load_relations()
    
    if not entities:
        logging.error("没有找到实体数据，无法生成关系")
        return
    
    # 2. 提取实体关键信息
    simplified_entities = extract_entity_info(entities)
    logging.info("提取了 " + str(len(simplified_entities)) + " 个实体的关键信息")
    
    # 3. 准备算法实体（带有描述字段）
    algorithm_entities = []
    for entity in simplified_entities:
        # 从原始实体中提取或生成描述字段
        if 'description' not in entity:
            # 生成简单描述
            description = f"{entity['name']} is a {entity['type'].lower()} "
            if entity['type'] == 'Algorithm':
                description += f"for {entity.get('task', 'various tasks')}."
            elif entity['type'] == 'Dataset':
                description += f"in the domain of {entity.get('domain', 'unknown')}."
            elif entity['type'] == 'Metric':
                description += f"focusing on {entity.get('focus', 'performance evaluation')}."
            
            entity['description'] = description
        
        algorithm_entities.append(entity)
    
    logging.info("准备了 " + str(len(algorithm_entities)) + " 个实体用于关系生成")
    
    # 4. 准备论文列表（从entities中提取）
    papers = []
    for entity in entities:
        if 'algorithm_entity' in entity:
            algo = entity['algorithm_entity']
            if 'title' in algo and 'authors' in algo:
                paper = {
                    'id': algo.get('algorithm_id', '').upper(),
                    'title': algo.get('title', ''),
                    'authors': algo.get('authors', []),
                    'year': algo.get('year', 0),
                    'abstract': algo.get('abstract', '')
                }
                papers.append(paper)
    
    logging.info("准备了 " + str(len(papers)) + " 篇论文用于关系生成")
    
    # 5. 生成提示词
    prompt = generate_relation_prompt(algorithm_entities[:max_pairs], papers[:max_pairs])
    
    # 6. 调用API
    logging.info("开始调用大模型API生成关系")
    api_response = call_llm_api(prompt)
    
    # 7. 解析结果
    new_relations = parse_api_response(api_response)
    logging.info("解析出 " + str(len(new_relations)) + " 个新关系")
    
    # 8. 转换为标准关系格式（如果需要）
    standard_relations = []
    for relation in new_relations:
        # 转换格式
        if 'entity_id' in relation and 'paper_id' in relation:
            standard_relation = {
                'from_entity': relation.get('entity_id', '').upper(),
                'to_entity': relation.get('paper_id', '').upper(),
                'relation_type': relation.get('relation_type', 'Unknown'),
                'structure': relation.get('structure', ''),
                'detail': relation.get('detail', ''),
                'evidence': relation.get('evidence', ''),
                'confidence': relation.get('confidence', 0.5)
            }
            standard_relations.append(standard_relation)
    
    # 9. 合并新旧关系并保存
    merged_relations = existing_relations + (standard_relations or new_relations)
    logging.info("合并后共有 " + str(len(merged_relations)) + " 个关系")
    
    # 10. 保存到文件
    save_success = save_relations(merged_relations)
    
    # 11. 更新数据库(如果需要)
    try:
        if save_success and new_relations:
            relations_to_store = standard_relations or new_relations
            db_manager.store_relations(relations_to_store)
            logging.info("成功将 " + str(len(relations_to_store)) + " 个新关系存入数据库")
    except Exception as e:
        logging.error("存储关系到数据库时出错: " + str(e))
    
    logging.info("==== 关系生成完成 ====")
    
    # 返回生成的关系
    return standard_relations or new_relations

def update_entities_with_relations():
    """
    将生成的关系更新到实体的evolution_relations字段
    """
    logging.info("开始更新实体的演化关系字段")
    
    # 1. 加载实体和关系
    entities = load_entities()
    relations = load_relations()
    
    if not entities or not relations:
        logging.error("没有找到足够的数据来更新实体关系")
        return
    
    # 2. 创建实体ID到索引的映射
    entity_map = {}
    for i, entity in enumerate(entities):
        if 'algorithm_entity' in entity:
            algo_id = entity['algorithm_entity'].get('algorithm_id', '').upper()
            if algo_id:
                entity_map[algo_id] = i
    
    # 3. 按目标实体对关系进行分组
    relation_groups = {}
    for relation in relations:
        to_entity = relation.get('to_entity', '')
        if to_entity:
            if to_entity not in relation_groups:
                relation_groups[to_entity] = []
            relation_groups[to_entity].append(relation)
    
    # 4. 更新实体的evolution_relations字段
    updated_count = 0
    for entity_id, relations in relation_groups.items():
        if entity_id in entity_map:
            entity_idx = entity_map[entity_id]
            # 清空原有的evolution_relations
            entities[entity_idx]['algorithm_entity']['evolution_relations'] = []
            
            # 添加新的关系
            for relation in relations:
                entities[entity_idx]['algorithm_entity']['evolution_relations'].append({
                    'from_entity': relation.get('from_entity', ''),
                    'to_entity': relation.get('to_entity', ''),
                    'relation_type': relation.get('relation_type', ''),
                    'structure': relation.get('structure', ''),
                    'detail': relation.get('detail', ''),
                    'evidence': relation.get('evidence', ''),
                    'confidence': relation.get('confidence', 0.5)
                })
            updated_count += 1
    
    logging.info(f"更新了 {updated_count} 个实体的演化关系")
    
    # 5. 保存更新后的实体
    try:
        entities_file = os.path.join(Config.DATA_DIR, 'entities.json')
        with open(entities_file, 'w', encoding='utf-8') as f:
            json.dump(entities, f, ensure_ascii=False, indent=2)
        logging.info(f"成功保存更新后的实体数据到 {entities_file}")
    except Exception as e:
        logging.error(f"保存更新后的实体数据时出错: {str(e)}")

if __name__ == "__main__":
    # 生成新的关系
    generate_relations(max_pairs=200, batch_size=5)
    
    # 更新实体的evolution_relations字段
    update_entities_with_relations() 