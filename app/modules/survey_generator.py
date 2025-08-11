#!/usr/bin/env python3
"""
自定义综述生成引擎
基于算法脉络分析生成高质量综述内容
"""

import os
import json
import logging
from datetime import datetime
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass
from collections import defaultdict, Counter
import networkx as nx
from app.config import Config
from app.modules.db_manager import db_manager

@dataclass
class AlgorithmEntity:
    """算法实体数据类"""
    entity_id: str
    name: str
    year: int
    authors: List[str]
    task: str
    dataset: List[str]
    metrics: List[str]
    architecture: Dict[str, Any]
    methodology: Dict[str, Any]
    feature_processing: List[str]
    source: str
    description: str = ""

@dataclass
class EvolutionRelation:
    """演化关系数据类"""
    from_entity_id: str
    to_entity_id: str
    relation_type: str
    structure: str
    detail: str
    evidence: str
    confidence: float
    source: str

@dataclass
class SurveySection:
    """综述章节数据类"""
    title: str
    content: str
    level: int
    subsections: List['SurveySection']
    
class AlgorithmLineageAnalyzer:
    """算法脉络分析器"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.graph = nx.DiGraph()
        
    def build_evolution_graph(self, entities: List[AlgorithmEntity], 
                            relations: List[EvolutionRelation]) -> nx.DiGraph:
        """构建算法演进图"""
        self.graph.clear()
        
        # 添加节点
        for entity in entities:
            self.graph.add_node(entity.entity_id, **{
                'name': entity.name,
                'year': entity.year,
                'authors': entity.authors,
                'task': entity.task,
                'architecture': entity.architecture,
                'methodology': entity.methodology,
                'entity': entity
            })
        
        # 添加边
        for relation in relations:
            if (relation.from_entity_id in self.graph.nodes and 
                relation.to_entity_id in self.graph.nodes):
                self.graph.add_edge(
                    relation.from_entity_id, 
                    relation.to_entity_id,
                    relation_type=relation.relation_type,
                    structure=relation.structure,
                    detail=relation.detail,
                    confidence=relation.confidence,
                    relation=relation
                )
        
        return self.graph
    
    def identify_key_algorithms(self) -> List[Tuple[str, Dict[str, Any]]]:
        """识别关键算法节点"""
        key_algorithms = []
        
        # 计算各种中心性指标
        try:
            betweenness = nx.betweenness_centrality(self.graph)
            in_degree = dict(self.graph.in_degree())
            out_degree = dict(self.graph.out_degree())
            
            # 综合评分
            for node_id in self.graph.nodes():
                node_data = self.graph.nodes[node_id]
                score = {
                    'betweenness': betweenness.get(node_id, 0),
                    'in_degree': in_degree.get(node_id, 0),
                    'out_degree': out_degree.get(node_id, 0),
                    'total_degree': in_degree.get(node_id, 0) + out_degree.get(node_id, 0)
                }
                
                # 计算综合重要性分数
                importance_score = (
                    score['betweenness'] * 0.4 +
                    score['in_degree'] * 0.3 +
                    score['out_degree'] * 0.2 +
                    score['total_degree'] * 0.1
                )
                
                key_algorithms.append((node_id, {
                    'name': node_data.get('name', ''),
                    'year': node_data.get('year', 0),
                    'importance_score': importance_score,
                    'metrics': score,
                    'entity': node_data.get('entity')
                }))
        
        except Exception as e:
            self.logger.error(f"计算关键算法时出错: {str(e)}")
        
        # 按重要性排序
        key_algorithms.sort(key=lambda x: x[1]['importance_score'], reverse=True)
        return key_algorithms[:10]  # 返回前10个关键算法
    
    def find_evolution_paths(self) -> List[List[str]]:
        """找到主要的演进路径"""
        paths = []
        
        try:
            # 找到起始节点（入度为0的节点）
            start_nodes = [node for node in self.graph.nodes() 
                          if self.graph.in_degree(node) == 0]
            
            # 找到终止节点（出度为0的节点）
            end_nodes = [node for node in self.graph.nodes() 
                        if self.graph.out_degree(node) == 0]
            
            # 寻找从起始到终止的所有路径
            for start in start_nodes:
                for end in end_nodes:
                    try:
                        for path in nx.all_simple_paths(self.graph, start, end, cutoff=10):
                            if len(path) >= 3:  # 至少包含3个节点的路径才有意义
                                paths.append(path)
                    except nx.NetworkXNoPath:
                        continue
        
        except Exception as e:
            self.logger.error(f"寻找演进路径时出错: {str(e)}")
        
        # 按路径长度排序，优先返回较长的路径
        paths.sort(key=len, reverse=True)
        return paths[:5]  # 返回前5条主要路径
    
    def analyze_technical_trends(self) -> Dict[str, Any]:
        """分析技术发展趋势"""
        trends = {
            'architecture_evolution': defaultdict(list),
            'methodology_evolution': defaultdict(list),
            'yearly_distribution': defaultdict(int),
            'task_distribution': defaultdict(int)
        }
        
        # 分析架构演进
        for node_id in self.graph.nodes():
            node_data = self.graph.nodes[node_id]
            entity = node_data.get('entity')
            if entity and hasattr(entity, 'year'):
                year = entity.year
                trends['yearly_distribution'][year] += 1
                
                # 安全访问task字段
                task = getattr(entity, 'task', '') or ''
                trends['task_distribution'][task] += 1
                
                # 安全分析架构组件
                try:
                    architecture = getattr(entity, 'architecture', {})
                    if isinstance(architecture, dict) and 'components' in architecture:
                        components = architecture['components']
                        if isinstance(components, list):
                            for component in components:
                                if component:  # 确保组件名称不为空
                                    trends['architecture_evolution'][component].append(year)
                except Exception as e:
                    self.logger.warning(f"处理实体 {entity.entity_id} 的架构数据时出错: {e}")
                
                # 安全分析方法论
                try:
                    methodology = getattr(entity, 'methodology', {})
                    if isinstance(methodology, dict) and 'training_strategy' in methodology:
                        strategies = methodology['training_strategy']
                        if isinstance(strategies, list):
                            for strategy in strategies:
                                if strategy:  # 确保策略名称不为空
                                    trends['methodology_evolution'][strategy].append(year)
                except Exception as e:
                    self.logger.warning(f"处理实体 {entity.entity_id} 的方法论数据时出错: {e}")
        
        # 计算趋势统计
        for component, years in trends['architecture_evolution'].items():
            trends['architecture_evolution'][component] = {
                'first_appearance': min(years) if years else 0,
                'last_appearance': max(years) if years else 0,
                'frequency': len(years),
                'years': sorted(years)
            }
        
        for method, years in trends['methodology_evolution'].items():
            trends['methodology_evolution'][method] = {
                'first_appearance': min(years) if years else 0,
                'last_appearance': max(years) if years else 0,
                'frequency': len(years),
                'years': sorted(years)
            }
        
        return trends

class SurveyGenerator:
    """综述生成器"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.lineage_analyzer = AlgorithmLineageAnalyzer()
        
    def generate_survey(self, task_ids: List[str], topic: str, 
                       section_num: int = 7, subsection_len: int = 700) -> Dict[str, Any]:
        """生成综述"""
        try:
            # 1. 获取数据
            entities, relations = self._load_data(task_ids)
            
            if not entities:
                raise ValueError("未找到有效的算法实体数据")
            
            # 2. 构建演进图和分析
            self.lineage_analyzer.build_evolution_graph(entities, relations)
            key_algorithms = self.lineage_analyzer.identify_key_algorithms()
            evolution_paths = self.lineage_analyzer.find_evolution_paths()
            technical_trends = self.lineage_analyzer.analyze_technical_trends()
            
            # 3. 生成综述内容
            survey_content = self._generate_survey_content(
                topic, entities, relations, key_algorithms, 
                evolution_paths, technical_trends, section_num, subsection_len
            )
            
            # 4. 生成结果
            result = {
                'content': {
                    'formats': ['markdown', 'json'],
                    'markdown': survey_content,
                    'json': {
                        'topic': topic,
                        'sections': self._parse_sections(survey_content),
                        'statistics': {
                            'total_algorithms': len(entities),
                            'total_relations': len(relations),
                            'key_algorithms_count': len(key_algorithms),
                            'evolution_paths_count': len(evolution_paths)
                        },
                        'analysis': {
                            'key_algorithms': key_algorithms,
                            'evolution_paths': evolution_paths,
                            'technical_trends': technical_trends
                        }
                    },
                    'files': {}
                },
                'status': 'completed',
                'topic': topic,
                'timestamp': datetime.now().isoformat(),
                'metadata': {
                    'generation_method': 'custom_survey_generator',
                    'data_sources': task_ids,
                    'entities_analyzed': len(entities),
                    'relations_analyzed': len(relations)
                }
            }
            
            return result
            
        except Exception as e:
            self.logger.error(f"生成综述失败: {str(e)}")
            return {
                'content': {
                    'formats': ['markdown'],
                    'markdown': f"# 综述生成失败\n\n错误信息: {str(e)}",
                    'files': {}
                },
                'status': 'error',
                'error': str(e),
                'topic': topic,
                'timestamp': datetime.now().isoformat()
            }
    
    def _load_data(self, task_ids: List[str]) -> Tuple[List[AlgorithmEntity], List[EvolutionRelation]]:
        """加载数据"""
        entities = []
        relations = []
        
        for task_id in task_ids:
            try:
                # 获取实体（返回实体列表，不是字典）
                entities_data = db_manager.get_entities_by_task(task_id)
                self.logger.info(f"任务 {task_id} 获取到 {len(entities_data)} 个实体")
                
                # 处理实体数据
                for i, entity_wrapper in enumerate(entities_data):
                    self.logger.debug(f"处理第 {i} 个实体包装器: {type(entity_wrapper)}")
                    
                    # 检查实体类型并提取算法实体
                    if isinstance(entity_wrapper, dict) and 'algorithm_entity' in entity_wrapper:
                        alg_data = entity_wrapper['algorithm_entity']
                        
                        # 安全处理年份字段
                        year_value = alg_data.get('year', 0)
                        if isinstance(year_value, str):
                            try:
                                year = int(year_value) if year_value.isdigit() else 0
                            except:
                                year = 0
                        elif isinstance(year_value, (int, float)):
                            year = int(year_value)
                        else:
                            year = 0
                        
                        # 安全处理列表字段
                        authors = alg_data.get('authors', [])
                        if not isinstance(authors, list):
                            authors = []
                        
                        dataset = alg_data.get('dataset', [])
                        if not isinstance(dataset, list):
                            dataset = []
                        
                        metrics = alg_data.get('metrics', [])
                        if not isinstance(metrics, list):
                            metrics = []
                        
                        feature_processing = alg_data.get('feature_processing', [])
                        if not isinstance(feature_processing, list):
                            feature_processing = []
                        
                        # 安全处理字典字段
                        architecture = alg_data.get('architecture', {})
                        if not isinstance(architecture, dict):
                            architecture = {}
                        
                        methodology = alg_data.get('methodology', {})
                        if not isinstance(methodology, dict):
                            methodology = {}
                        
                        # 创建AlgorithmEntity对象
                        entity = AlgorithmEntity(
                            entity_id=alg_data.get('algorithm_id', alg_data.get('entity_id', '')),
                            name=alg_data.get('name', ''),
                            year=year,
                            authors=authors,
                            task=alg_data.get('task', ''),
                            dataset=dataset,
                            metrics=metrics,
                            architecture=architecture,
                            methodology=methodology,
                            feature_processing=feature_processing,
                            source=alg_data.get('source', ''),
                            description=alg_data.get('description', '')
                        )
                        entities.append(entity)
                        self.logger.debug(f"成功添加算法实体: {entity.name}")
                    
                    # 可以扩展处理数据集和指标实体，但目前专注于算法
                    elif isinstance(entity_wrapper, dict) and 'dataset_entity' in entity_wrapper:
                        # 暂时跳过数据集实体，可以后续扩展
                        pass
                    elif isinstance(entity_wrapper, dict) and 'metric_entity' in entity_wrapper:
                        # 暂时跳过指标实体，可以后续扩展
                        pass
                    else:
                        self.logger.warning(f"未知的实体包装器格式: {entity_wrapper}")
                
                # 获取演化关系
                evolution_data = db_manager.get_relations_by_task(task_id)
                self.logger.info(f"任务 {task_id} 获取到 {len(evolution_data)} 个关系")
                
                for rel_data in evolution_data:
                    if isinstance(rel_data, dict):
                        relation = EvolutionRelation(
                            from_entity_id=rel_data.get('from_entity_id', ''),
                            to_entity_id=rel_data.get('to_entity_id', ''),
                            relation_type=rel_data.get('relation_type', ''),
                            structure=rel_data.get('structure', ''),
                            detail=rel_data.get('detail', ''),
                            evidence=rel_data.get('evidence', ''),
                            confidence=float(rel_data.get('confidence', 0.0)),
                            source=rel_data.get('source', '')
                        )
                        relations.append(relation)
                    
            except Exception as e:
                self.logger.error(f"处理任务 {task_id} 时出错: {str(e)}")
                continue
        
        self.logger.info(f"总共加载了 {len(entities)} 个算法实体和 {len(relations)} 个演化关系")
        return entities, relations
        
        return entities, relations
    
    def _generate_survey_content(self, topic: str, entities: List[AlgorithmEntity],
                               relations: List[EvolutionRelation], key_algorithms: List[Tuple[str, Dict]],
                               evolution_paths: List[List[str]], technical_trends: Dict[str, Any],
                               section_num: int, subsection_len: int) -> str:
        """生成综述内容"""
        
        content_parts = []
        
        # 标题和摘要
        content_parts.append(f"# {topic}综述\n")
        content_parts.append(self._generate_abstract(entities, relations, key_algorithms))
        
        # 1. 引言
        content_parts.append("## 1. 引言\n")
        content_parts.append(self._generate_introduction(topic, entities, technical_trends))
        
        # 2. 相关工作和背景
        content_parts.append("## 2. 相关工作和背景\n")
        content_parts.append(self._generate_background(entities, technical_trends))
        
        # 3. 算法发展历程
        content_parts.append("## 3. 算法发展历程\n")
        content_parts.append(self._generate_evolution_history(entities, evolution_paths, technical_trends))
        
        # 4. 关键算法分析
        content_parts.append("## 4. 关键算法分析\n")
        content_parts.append(self._generate_key_algorithms_analysis(key_algorithms, entities, relations))
        
        # 5. 技术演进路径
        content_parts.append("## 5. 技术演进路径\n")
        content_parts.append(self._generate_evolution_paths_analysis(evolution_paths, entities, relations))
        
        # 6. 发展趋势分析
        content_parts.append("## 6. 发展趋势分析\n")
        content_parts.append(self._generate_trends_analysis(technical_trends, entities))
        
        # 7. 总结与展望
        content_parts.append("## 7. 总结与展望\n")
        content_parts.append(self._generate_conclusion(topic, key_algorithms, technical_trends))
        
        return "\n\n".join(content_parts)
    
    def _generate_abstract(self, entities: List[AlgorithmEntity], 
                          relations: List[EvolutionRelation], 
                          key_algorithms: List[Tuple[str, Dict]]) -> str:
        """生成摘要"""
        total_algorithms = len(entities)
        total_relations = len(relations)
        key_count = len(key_algorithms)
        
        # 年份范围
        years = [e.year for e in entities if e.year > 0]
        year_range = f"{min(years)}年至{max(years)}年" if years else "近年来"
        
        # 主要任务类型
        tasks = Counter([e.task for e in entities if e.task])
        main_tasks = [task for task, _ in tasks.most_common(3)]
        
        abstract = f"""## 摘要

本综述对{year_range}的算法发展进行了系统性分析，涵盖了{total_algorithms}个算法实体和{total_relations}个演化关系。
通过算法脉络分析，我们识别出{key_count}个关键算法节点，主要涉及{', '.join(main_tasks)}等任务领域。

研究发现，算法发展呈现出明显的演进模式，从早期的基础方法逐步发展到现代的复杂架构。
关键技术创新点集中在网络架构设计、训练策略优化和特征处理方法等方面。
本综述为理解该领域的技术发展脉络提供了系统性的分析框架。

**关键词**: 算法演进、技术脉络、发展趋势、{', '.join(main_tasks[:2])}"""
        
        return abstract
    
    def _generate_introduction(self, topic: str, entities: List[AlgorithmEntity], 
                              technical_trends: Dict[str, Any]) -> str:
        """生成引言"""
        # 统计信息
        total_algorithms = len(entities)
        years = [e.year for e in entities if e.year > 0]
        year_span = max(years) - min(years) if years else 0
        
        # 主要任务和数据集
        tasks = Counter([e.task for e in entities if e.task])
        datasets = Counter([ds for e in entities for ds in e.dataset if ds])
        
        main_task = tasks.most_common(1)[0][0] if tasks else topic
        popular_datasets = [ds for ds, _ in datasets.most_common(3)]
        
        intro = f"""{topic}作为人工智能领域的重要分支，在过去{year_span}年中经历了快速发展。
本领域的研究主要集中在{main_task}等任务上，涉及{total_algorithms}个主要算法的演进和发展。

从技术发展的角度来看，该领域的研究呈现出以下特点：

1. **技术演进的连续性**: 新算法往往基于已有方法进行改进和创新
2. **数据集驱动的发展**: {', '.join(popular_datasets[:2])}等标准数据集推动了算法性能的持续提升
3. **架构创新的重要性**: 网络架构的设计成为性能突破的关键因素

本综述通过分析算法实体间的演化关系，揭示了该领域技术发展的内在规律和趋势，
为研究者理解技术脉络和预测未来发展方向提供了重要参考。"""
        
        return intro
    
    def _generate_background(self, entities: List[AlgorithmEntity], 
                           technical_trends: Dict[str, Any]) -> str:
        """生成背景介绍"""
        # 早期算法
        early_algorithms = sorted([e for e in entities if e.year > 0], key=lambda x: x.year)[:5]
        
        # 架构演进
        arch_components = technical_trends.get('architecture_evolution', {})
        early_components = []
        modern_components = []
        
        for component, info in arch_components.items():
            if isinstance(info, dict) and 'first_appearance' in info:
                if info['first_appearance'] < 2015:
                    early_components.append(component)
                else:
                    modern_components.append(component)
        
        background = f"""### 2.1 早期发展阶段

该领域的早期研究可以追溯到{early_algorithms[0].year if early_algorithms else '20世纪'}年。
早期的代表性算法包括：

"""
        
        for alg in early_algorithms[:3]:
            background += f"- **{alg.name}** ({alg.year}年): "
            if alg.description:
                background += alg.description
            else:
                background += f"在{alg.task}任务上的重要贡献"
            background += "\n"
        
        background += f"""
### 2.2 技术基础

早期算法主要采用{', '.join(early_components[:3])}等基础组件，
奠定了该领域的技术基础。这一阶段的特点是：

1. **算法结构相对简单**: 主要使用传统的机器学习方法
2. **数据规模较小**: 受限于计算资源和数据获取能力
3. **性能评估标准**: 建立了基本的评估指标体系

### 2.3 现代发展

随着计算能力的提升和数据规模的扩大，现代算法开始采用
{', '.join(modern_components[:3])}等先进技术，
推动了整个领域的快速发展。"""
        
        return background
    
    def _generate_evolution_history(self, entities: List[AlgorithmEntity], 
                                  evolution_paths: List[List[str]], 
                                  technical_trends: Dict[str, Any]) -> str:
        """生成发展历程"""
        # 按年份分组
        yearly_algorithms = defaultdict(list)
        for entity in entities:
            if entity.year > 0:
                yearly_algorithms[entity.year].append(entity)
        
        # 按年份排序
        sorted_years = sorted(yearly_algorithms.keys())
        
        history = "### 3.1 时间线发展\n\n"
        
        # 分阶段描述
        if sorted_years:
            early_period = [y for y in sorted_years if y < 2010]
            middle_period = [y for y in sorted_years if 2010 <= y < 2017]
            recent_period = [y for y in sorted_years if y >= 2017]
            
            if early_period:
                history += f"**早期阶段 ({min(early_period)}-2009年)**:\n"
                for year in early_period[-3:]:  # 最后3年
                    algs = yearly_algorithms[year]
                    history += f"- {year}年: {', '.join([a.name for a in algs])}\n"
                history += "\n"
            
            if middle_period:
                history += f"**发展阶段 (2010-2016年)**:\n"
                for year in middle_period[-3:]:
                    algs = yearly_algorithms[year]
                    history += f"- {year}年: {', '.join([a.name for a in algs])}\n"
                history += "\n"
            
            if recent_period:
                history += f"**现代阶段 (2017年至今)**:\n"
                for year in recent_period[-3:]:
                    algs = yearly_algorithms[year]
                    history += f"- {year}年: {', '.join([a.name for a in algs])}\n"
                history += "\n"
        
        # 主要演进路径
        history += "### 3.2 主要演进路径\n\n"
        
        for i, path in enumerate(evolution_paths[:3], 1):
            history += f"**路径 {i}**: "
            path_names = []
            for entity_id in path:
                entity = next((e for e in entities if e.entity_id == entity_id), None)
                if entity:
                    path_names.append(f"{entity.name}({entity.year})")
            
            history += " → ".join(path_names)
            history += "\n\n"
        
        return history
    
    def _generate_key_algorithms_analysis(self, key_algorithms: List[Tuple[str, Dict]], 
                                        entities: List[AlgorithmEntity], 
                                        relations: List[EvolutionRelation]) -> str:
        """生成关键算法分析"""
        analysis = ""
        
        for i, (entity_id, info) in enumerate(key_algorithms[:5], 1):
            entity = info.get('entity')
            if not entity:
                continue
            
            analysis += f"### 4.{i} {entity.name}\n\n"
            
            # 基本信息
            analysis += f"**发布年份**: {entity.year}年\n"
            analysis += f"**主要任务**: {entity.task}\n"
            analysis += f"**重要性评分**: {info['importance_score']:.3f}\n\n"
            
            # 技术特点
            if entity.architecture:
                analysis += "**技术特点**:\n"
                if 'components' in entity.architecture:
                    analysis += f"- 架构组件: {', '.join(entity.architecture['components'])}\n"
                if 'mechanisms' in entity.architecture:
                    analysis += f"- 核心机制: {', '.join(entity.architecture['mechanisms'])}\n"
            
            # 创新点
            incoming_relations = [r for r in relations if r.to_entity_id == entity_id]
            if incoming_relations:
                analysis += "\n**主要创新点**:\n"
                for rel in incoming_relations[:3]:
                    analysis += f"- {rel.detail}\n"
            
            # 影响力
            outgoing_relations = [r for r in relations if r.from_entity_id == entity_id]
            if outgoing_relations:
                analysis += f"\n**技术影响**: 直接影响了{len(outgoing_relations)}个后续算法的发展\n"
            
            analysis += "\n"
        
        return analysis
    
    def _generate_evolution_paths_analysis(self, evolution_paths: List[List[str]], 
                                         entities: List[AlgorithmEntity], 
                                         relations: List[EvolutionRelation]) -> str:
        """生成演进路径分析"""
        analysis = "技术演进路径反映了算法发展的逻辑脉络和创新规律。通过分析主要演进路径，我们可以识别出技术发展的关键转折点和驱动因素。\n\n"
        
        for i, path in enumerate(evolution_paths[:3], 1):
            analysis += f"### 5.{i} 演进路径 {i}\n\n"
            
            # 路径描述
            path_entities = []
            for entity_id in path:
                entity = next((e for e in entities if e.entity_id == entity_id), None)
                if entity:
                    path_entities.append(entity)
            
            if len(path_entities) >= 2:
                analysis += f"**路径概述**: 从{path_entities[0].name}({path_entities[0].year})到{path_entities[-1].name}({path_entities[-1].year})的技术演进\n\n"
                
                # 分析演进关系
                analysis += "**演进分析**:\n"
                for j in range(len(path) - 1):
                    from_id, to_id = path[j], path[j + 1]
                    relation = next((r for r in relations 
                                   if r.from_entity_id == from_id and r.to_entity_id == to_id), None)
                    
                    from_entity = next((e for e in entities if e.entity_id == from_id), None)
                    to_entity = next((e for e in entities if e.entity_id == to_id), None)
                    
                    if relation and from_entity and to_entity:
                        analysis += f"- {from_entity.name} → {to_entity.name}: {relation.detail}\n"
                
                analysis += "\n"
        
        return analysis
    
    def _generate_trends_analysis(self, technical_trends: Dict[str, Any], 
                                entities: List[AlgorithmEntity]) -> str:
        """生成趋势分析"""
        analysis = ""
        
        # 年度分布趋势
        yearly_dist = technical_trends.get('yearly_distribution', {})
        if yearly_dist:
            analysis += "### 6.1 发展活跃度趋势\n\n"
            sorted_years = sorted(yearly_dist.keys())
            
            # 找出高峰年份
            peak_years = sorted(yearly_dist.items(), key=lambda x: x[1], reverse=True)[:3]
            analysis += f"算法发展的高峰期出现在{', '.join([str(year) for year, _ in peak_years])}年，"
            analysis += f"其中{peak_years[0][0]}年发布了{peak_years[0][1]}个重要算法。\n\n"
        
        # 架构演进趋势
        arch_evolution = technical_trends.get('architecture_evolution', {})
        if arch_evolution:
            analysis += "### 6.2 架构技术趋势\n\n"
            
            # 按首次出现时间排序
            sorted_components = sorted(
                [(comp, info) for comp, info in arch_evolution.items() 
                 if isinstance(info, dict) and 'first_appearance' in info],
                key=lambda x: x[1]['first_appearance']
            )
            
            # 早期技术
            early_tech = [comp for comp, info in sorted_components[:5]]
            analysis += f"**早期主流技术**: {', '.join(early_tech)}\n"
            
            # 现代技术
            modern_tech = [comp for comp, info in sorted_components[-5:]]
            analysis += f"**现代前沿技术**: {', '.join(modern_tech)}\n\n"
        
        # 方法论趋势
        method_evolution = technical_trends.get('methodology_evolution', {})
        if method_evolution:
            analysis += "### 6.3 方法论发展趋势\n\n"
            
            popular_methods = sorted(
                [(method, info) for method, info in method_evolution.items()
                 if isinstance(info, dict) and 'frequency' in info],
                key=lambda x: x[1]['frequency'], reverse=True
            )[:5]
            
            analysis += "**主流训练策略**:\n"
            for method, info in popular_methods:
                analysis += f"- {method}: 在{info['frequency']}个算法中使用\n"
            
            analysis += "\n"
        
        return analysis
    
    def _generate_conclusion(self, topic: str, key_algorithms: List[Tuple[str, Dict]], 
                           technical_trends: Dict[str, Any]) -> str:
        """生成总结与展望"""
        conclusion = f"""### 7.1 主要贡献总结

通过对{topic}领域的系统性分析，本综述得出以下主要结论：

1. **技术演进规律**: 算法发展呈现出明显的继承性和创新性，新技术往往在前代基础上进行改进
2. **关键技术节点**: """
        
        if key_algorithms:
            top_algorithms = [info['entity'].name for _, info in key_algorithms[:3] 
                            if info.get('entity')]
            conclusion += f"{', '.join(top_algorithms)}等算法在技术发展中起到了关键作用\n"
        
        conclusion += """3. **发展驱动因素**: 计算能力提升、数据规模扩大和理论创新共同推动了技术进步

### 7.2 未来发展趋势

基于当前技术发展脉络，预测未来发展将呈现以下趋势：

1. **技术深度融合**: 不同技术路径将进一步融合，形成更加复杂的混合架构
2. **效率优化**: 在保持性能的同时，算法效率和可解释性将成为重要考量
3. **应用场景扩展**: 技术将向更多实际应用场景扩展，推动产业化发展

### 7.3 研究挑战与机遇

当前该领域面临的主要挑战包括：
- 算法复杂度与效率的平衡
- 大规模数据处理的可扩展性
- 跨领域知识的有效整合

同时，新兴技术的发展也为该领域带来了新的机遇，
为研究者提供了广阔的创新空间。"""
        
        return conclusion
    
    def _parse_sections(self, content: str) -> List[Dict[str, Any]]:
        """解析章节结构"""
        sections = []
        lines = content.split('\n')
        current_section = None
        current_content = []
        
        for line in lines:
            if line.startswith('## '):
                # 保存之前的章节
                if current_section:
                    sections.append({
                        'title': current_section,
                        'content': '\n'.join(current_content).strip(),
                        'level': 2
                    })
                
                # 开始新章节
                current_section = line[3:].strip()
                current_content = []
            elif line.startswith('### '):
                current_content.append(line)
            else:
                current_content.append(line)
        
        # 添加最后一个章节
        if current_section:
            sections.append({
                'title': current_section,
                'content': '\n'.join(current_content).strip(),
                'level': 2
            })
        
        return sections

# 全局实例
survey_generator = SurveyGenerator()
