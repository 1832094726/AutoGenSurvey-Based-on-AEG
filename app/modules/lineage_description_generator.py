"""
算法脉络描述生成器
基于算法演进图谱生成详细的文字描述，包括时间线、技术关联、影响分析
"""

import logging
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass

@dataclass
class LineageDescription:
    """脉络描述数据结构"""
    timeline_description: str
    technical_relationships: str
    influence_analysis: str
    development_summary: str
    key_insights: List[str]

class AlgorithmLineageDescriptionGenerator:
    """算法脉络描述生成器"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def generate_lineage_description(self, algorithm_lineage: Dict[str, Any], topic: str) -> LineageDescription:
        """生成完整的算法脉络描述"""
        try:
            self.logger.info(f"开始生成算法脉络描述: {topic}")
            
            # 生成时间线描述
            timeline_desc = self._generate_timeline_description(algorithm_lineage, topic)
            
            # 生成技术关联描述
            tech_relations_desc = self._generate_technical_relationships_description(algorithm_lineage, topic)
            
            # 生成影响分析描述
            influence_desc = self._generate_influence_analysis(algorithm_lineage, topic)
            
            # 生成发展总结
            development_summary = self._generate_development_summary(algorithm_lineage, topic)
            
            # 提取关键洞察
            key_insights = self._extract_key_insights(algorithm_lineage, topic)
            
            return LineageDescription(
                timeline_description=timeline_desc,
                technical_relationships=tech_relations_desc,
                influence_analysis=influence_desc,
                development_summary=development_summary,
                key_insights=key_insights
            )
            
        except Exception as e:
            self.logger.error(f"生成算法脉络描述失败: {str(e)}")
            raise
    
    def _generate_timeline_description(self, algorithm_lineage: Dict[str, Any], topic: str) -> str:
        """生成时间线描述"""
        key_nodes = algorithm_lineage.get("key_nodes", [])
        if not key_nodes:
            return f"{topic}领域的算法发展时间线信息不足。"
        
        # 按年份排序节点
        nodes_with_years = [node for node in key_nodes if node.get("year")]
        nodes_with_years.sort(key=lambda x: x.get("year", 0))
        
        if not nodes_with_years:
            return f"{topic}领域包含{len(key_nodes)}个重要算法，但缺乏详细的时间信息。"
        
        timeline_parts = []
        timeline_parts.append(f"## {topic}算法发展时间线\n")
        
        # 按时期分组
        periods = self._group_by_periods(nodes_with_years)
        
        for period, nodes in periods.items():
            timeline_parts.append(f"### {period}\n")
            
            if len(nodes) == 1:
                node = nodes[0]
                timeline_parts.append(f"**{node['year']}年**: {node['name']} 的提出标志着{topic}领域的重要进展。")
            else:
                years = [str(node['year']) for node in nodes]
                names = [node['name'] for node in nodes]
                timeline_parts.append(f"**{'-'.join(years)}年**: 这一时期见证了多个重要算法的诞生，包括{', '.join(names)}，形成了{topic}领域的技术基础。")
            
            # 添加时期特征分析
            period_analysis = self._analyze_period_characteristics(nodes, period)
            if period_analysis:
                timeline_parts.append(f"\n{period_analysis}")
        
        # 添加发展趋势分析
        trend_analysis = self._analyze_development_trends(nodes_with_years)
        if trend_analysis:
            timeline_parts.append(f"\n### 发展趋势\n\n{trend_analysis}")
        
        return "\n".join(timeline_parts)
    
    def _group_by_periods(self, nodes_with_years: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        """按时期分组节点"""
        periods = {}
        
        for node in nodes_with_years:
            year = node.get("year", 0)
            
            # 按十年分组
            if year < 2000:
                period = "早期发展阶段 (2000年前)"
            elif year < 2010:
                period = "基础建立阶段 (2000-2009年)"
            elif year < 2015:
                period = "快速发展阶段 (2010-2014年)"
            elif year < 2020:
                period = "深度学习时代 (2015-2019年)"
            else:
                period = "现代发展阶段 (2020年至今)"
            
            if period not in periods:
                periods[period] = []
            periods[period].append(node)
        
        return periods
    
    def _analyze_period_characteristics(self, nodes: List[Dict[str, Any]], period: str) -> str:
        """分析时期特征"""
        if not nodes:
            return ""
        
        characteristics = []
        
        # 分析算法类型
        algorithm_types = self._classify_algorithm_types(nodes)
        if algorithm_types:
            characteristics.append(f"主要技术特征包括{', '.join(algorithm_types)}。")
        
        # 分析重要性分布
        importance_scores = [node.get("importance_score", 0) for node in nodes]
        avg_importance = sum(importance_scores) / len(importance_scores) if importance_scores else 0
        
        if avg_importance > 0.7:
            characteristics.append("这一时期的算法普遍具有较高的影响力。")
        elif avg_importance > 0.4:
            characteristics.append("这一时期见证了多个有影响力的技术突破。")
        
        return " ".join(characteristics)
    
    def _classify_algorithm_types(self, nodes: List[Dict[str, Any]]) -> List[str]:
        """分类算法类型"""
        types = []
        names = [node.get("name", "").lower() for node in nodes]
        
        # 基于算法名称的简单分类
        if any("neural" in name or "network" in name for name in names):
            types.append("神经网络方法")
        if any("deep" in name or "cnn" in name or "rnn" in name for name in names):
            types.append("深度学习技术")
        if any("svm" in name or "support" in name for name in names):
            types.append("支持向量机")
        if any("tree" in name or "forest" in name for name in names):
            types.append("树模型")
        if any("cluster" in name or "kmeans" in name for name in names):
            types.append("聚类算法")
        
        return types[:3]  # 返回前3个类型
    
    def _analyze_development_trends(self, nodes_with_years: List[Dict[str, Any]]) -> str:
        """分析发展趋势"""
        if len(nodes_with_years) < 3:
            return ""
        
        trends = []
        
        # 分析时间分布
        years = [node.get("year", 0) for node in nodes_with_years]
        year_span = max(years) - min(years)
        
        if year_span > 20:
            trends.append("该领域经历了长期的持续发展")
        elif year_span > 10:
            trends.append("该领域在过去十多年中快速发展")
        else:
            trends.append("该领域的主要算法集中在较短时间内涌现")
        
        # 分析发展速度
        recent_count = sum(1 for year in years if year >= 2015)
        total_count = len(years)
        
        if recent_count / total_count > 0.6:
            trends.append("近年来发展尤为迅速")
        elif recent_count / total_count > 0.3:
            trends.append("保持稳定的发展势头")
        
        return "，".join(trends) + "。"
    
    def _generate_technical_relationships_description(self, algorithm_lineage: Dict[str, Any], topic: str) -> str:
        """生成技术关联描述"""
        relationships_parts = []
        relationships_parts.append(f"## {topic}技术关联分析\n")
        
        # 分析算法图谱
        graph_data = algorithm_lineage.get("algorithm_graph", {})
        if not graph_data:
            return f"{topic}领域的技术关联信息不足。"
        
        nodes = graph_data.get("nodes", {})
        edges = graph_data.get("edges", [])
        
        # 分析关系类型分布
        relation_analysis = self._analyze_relation_types(edges)
        if relation_analysis:
            relationships_parts.append(f"### 关系类型分析\n\n{relation_analysis}")
        
        # 分析技术集群
        clusters = algorithm_lineage.get("algorithm_clusters", {})
        cluster_analysis = self._analyze_technical_clusters(clusters)
        if cluster_analysis:
            relationships_parts.append(f"\n### 技术集群分析\n\n{cluster_analysis}")
        
        # 分析核心技术
        core_tech_analysis = self._analyze_core_technologies(nodes, edges)
        if core_tech_analysis:
            relationships_parts.append(f"\n### 核心技术分析\n\n{core_tech_analysis}")
        
        return "\n".join(relationships_parts)
    
    def _analyze_relation_types(self, edges: List[Dict[str, Any]]) -> str:
        """分析关系类型"""
        if not edges:
            return ""
        
        # 统计关系类型
        relation_counts = {}
        for edge in edges:
            rel_type = edge.get("type", "unknown")
            relation_counts[rel_type] = relation_counts.get(rel_type, 0) + 1
        
        # 生成描述
        analysis_parts = []
        total_relations = len(edges)
        
        analysis_parts.append(f"算法间共存在{total_relations}个关联关系，主要包括：")
        
        # 按频率排序
        sorted_relations = sorted(relation_counts.items(), key=lambda x: x[1], reverse=True)
        
        for rel_type, count in sorted_relations[:5]:
            percentage = (count / total_relations) * 100
            rel_desc = self._get_relation_description(rel_type)
            analysis_parts.append(f"- **{rel_desc}**: {count}个关系 ({percentage:.1f}%)")
        
        return "\n".join(analysis_parts)
    
    def _get_relation_description(self, rel_type: str) -> str:
        """获取关系类型的中文描述"""
        descriptions = {
            "improve": "改进关系",
            "extend": "扩展关系", 
            "based_on": "基于关系",
            "compare": "对比关系",
            "similar": "相似关系",
            "combine": "结合关系",
            "inspire": "启发关系"
        }
        
        for key, desc in descriptions.items():
            if key in rel_type.lower():
                return desc
        
        return rel_type
    
    def _analyze_technical_clusters(self, clusters: Dict[str, Any]) -> str:
        """分析技术集群"""
        if not clusters:
            return ""
        
        analysis_parts = []
        
        # 时间集群分析
        temporal_clusters = clusters.get("temporal_clusters", [])
        if temporal_clusters:
            analysis_parts.append(f"从时间维度看，识别出{len(temporal_clusters)}个主要发展阶段：")
            for cluster in temporal_clusters[:3]:
                period = cluster.get("period", "未知时期")
                size = cluster.get("size", 0)
                analysis_parts.append(f"- {period}: {size}个相关算法")
        
        # 作者集群分析
        author_clusters = clusters.get("author_clusters", [])
        if author_clusters:
            analysis_parts.append(f"\n从研究团队角度，发现{len(author_clusters)}个主要研究集群：")
            for cluster in author_clusters[:3]:
                author = cluster.get("author", "未知作者")
                size = cluster.get("size", 0)
                analysis_parts.append(f"- {author}团队: 贡献{size}个相关算法")
        
        # 连通性集群分析
        connectivity_clusters = clusters.get("connectivity_clusters", [])
        if connectivity_clusters:
            analysis_parts.append(f"\n从技术关联角度，形成{len(connectivity_clusters)}个技术集群，体现了不同技术路线的聚合特征。")
        
        return "\n".join(analysis_parts)
    
    def _analyze_core_technologies(self, nodes: Dict[str, Any], edges: List[Dict[str, Any]]) -> str:
        """分析核心技术"""
        if not nodes or not edges:
            return ""
        
        # 计算节点度数
        node_degrees = {}
        for node_id in nodes:
            degree = sum(1 for edge in edges if edge["source"] == node_id or edge["target"] == node_id)
            node_degrees[node_id] = degree
        
        # 找到度数最高的节点
        core_nodes = sorted(node_degrees.items(), key=lambda x: x[1], reverse=True)[:3]
        
        if not core_nodes:
            return ""
        
        analysis_parts = []
        analysis_parts.append("核心技术节点分析：")
        
        for node_id, degree in core_nodes:
            if node_id in nodes:
                node_data = nodes[node_id]
                name = node_data.get("name", "未知算法")
                year = node_data.get("year", "未知年份")
                analysis_parts.append(f"- **{name}** ({year}): 连接度为{degree}，在技术网络中起到关键枢纽作用")
        
        return "\n".join(analysis_parts)
    
    def _generate_influence_analysis(self, algorithm_lineage: Dict[str, Any], topic: str) -> str:
        """生成影响分析描述"""
        influence_parts = []
        influence_parts.append(f"## {topic}影响力分析\n")
        
        key_nodes = algorithm_lineage.get("key_nodes", [])
        if not key_nodes:
            return f"{topic}领域的影响力分析数据不足。"
        
        # 分析高影响力算法
        high_impact_analysis = self._analyze_high_impact_algorithms(key_nodes)
        if high_impact_analysis:
            influence_parts.append(f"### 高影响力算法\n\n{high_impact_analysis}")
        
        # 分析影响力传播
        propagation_analysis = self._analyze_influence_propagation(algorithm_lineage)
        if propagation_analysis:
            influence_parts.append(f"\n### 影响力传播\n\n{propagation_analysis}")
        
        # 分析长期影响
        long_term_analysis = self._analyze_long_term_influence(key_nodes)
        if long_term_analysis:
            influence_parts.append(f"\n### 长期影响\n\n{long_term_analysis}")
        
        return "\n".join(influence_parts)
    
    def _analyze_high_impact_algorithms(self, key_nodes: List[Dict[str, Any]]) -> str:
        """分析高影响力算法"""
        # 按重要性评分排序
        sorted_nodes = sorted(key_nodes, key=lambda x: x.get("importance_score", 0), reverse=True)
        top_nodes = sorted_nodes[:5]
        
        analysis_parts = []
        analysis_parts.append("以下算法在该领域具有突出的影响力：")
        
        for i, node in enumerate(top_nodes, 1):
            name = node.get("name", "未知算法")
            year = node.get("year", "未知年份")
            score = node.get("importance_score", 0)
            
            # 分析影响力类型
            influence_types = []
            if node.get("out_degree", 0) >= 3:
                influence_types.append("技术启发")
            if node.get("in_degree", 0) >= 2:
                influence_types.append("技术整合")
            if score >= 0.8:
                influence_types.append("基础性贡献")
            
            influence_desc = "、".join(influence_types) if influence_types else "重要贡献"
            
            analysis_parts.append(f"{i}. **{name}** ({year}): 重要性评分{score:.2f}，主要体现在{influence_desc}方面")
        
        return "\n".join(analysis_parts)
    
    def _analyze_influence_propagation(self, algorithm_lineage: Dict[str, Any]) -> str:
        """分析影响力传播"""
        dev_paths = algorithm_lineage.get("development_paths", {})
        if not isinstance(dev_paths, dict):
            return ""
        
        main_paths = dev_paths.get("main_paths", [])
        if not main_paths:
            return ""
        
        analysis_parts = []
        analysis_parts.append("影响力传播路径分析：")
        
        for i, path in enumerate(main_paths[:3], 1):
            length = path.get("length", 0)
            time_span = path.get("time_span", {})
            start_year = time_span.get("start_year", "未知")
            end_year = time_span.get("end_year", "未知")
            
            analysis_parts.append(f"- 传播路径{i}: 跨越{length}个算法节点，时间跨度从{start_year}年到{end_year}年，展现了技术影响的延续性")
        
        return "\n".join(analysis_parts)
    
    def _analyze_long_term_influence(self, key_nodes: List[Dict[str, Any]]) -> str:
        """分析长期影响"""
        # 分析早期算法的持续影响
        early_nodes = [node for node in key_nodes if node.get("year", 2030) < 2010]
        recent_nodes = [node for node in key_nodes if node.get("year", 0) >= 2015]
        
        if not early_nodes:
            return ""
        
        analysis_parts = []
        
        if early_nodes:
            early_names = [node.get("name", "") for node in early_nodes[:3]]
            analysis_parts.append(f"早期算法如{', '.join(early_names)}等，至今仍在该领域发挥重要作用，体现了基础性技术的持久影响力。")
        
        if recent_nodes and early_nodes:
            analysis_parts.append(f"从{len(early_nodes)}个早期重要算法到{len(recent_nodes)}个近期重要算法的发展，展现了该领域的持续创新能力。")
        
        return "\n".join(analysis_parts)
    
    def _generate_development_summary(self, algorithm_lineage: Dict[str, Any], topic: str) -> str:
        """生成发展总结"""
        summary_parts = []
        
        # 基本统计
        key_nodes = algorithm_lineage.get("key_nodes", [])
        graph_data = algorithm_lineage.get("algorithm_graph", {})
        
        total_algorithms = len(graph_data.get("nodes", {}))
        key_algorithms = len(key_nodes)
        total_relations = len(graph_data.get("edges", []))
        
        summary_parts.append(f"{topic}领域的算法发展呈现以下特征：")
        summary_parts.append(f"- 技术规模: 涵盖{total_algorithms}个算法，其中{key_algorithms}个为关键节点")
        summary_parts.append(f"- 关联密度: 算法间存在{total_relations}个关联关系，形成了紧密的技术网络")
        
        # 发展阶段总结
        if key_nodes:
            years = [node.get("year") for node in key_nodes if node.get("year")]
            if years:
                span = max(years) - min(years)
                summary_parts.append(f"- 发展历程: 从{min(years)}年到{max(years)}年，经历了{span}年的发展历程")
        
        # 技术特征总结
        clusters = algorithm_lineage.get("algorithm_clusters", {})
        if clusters:
            temporal_clusters = clusters.get("temporal_clusters", [])
            if temporal_clusters:
                summary_parts.append(f"- 发展阶段: 可分为{len(temporal_clusters)}个主要发展阶段，每个阶段都有其独特的技术特征")
        
        return "\n".join(summary_parts)
    
    def _extract_key_insights(self, algorithm_lineage: Dict[str, Any], topic: str) -> List[str]:
        """提取关键洞察"""
        insights = []
        
        key_nodes = algorithm_lineage.get("key_nodes", [])
        graph_data = algorithm_lineage.get("algorithm_graph", {})
        
        # 洞察1: 技术集中度
        if key_nodes:
            high_impact_count = sum(1 for node in key_nodes if node.get("importance_score", 0) > 0.7)
            if high_impact_count >= 3:
                insights.append(f"{topic}领域存在多个高影响力的核心技术，形成了相对均衡的技术生态")
            elif high_impact_count >= 1:
                insights.append(f"{topic}领域的技术发展相对集中，少数核心算法起到主导作用")
        
        # 洞察2: 发展模式
        dev_paths = algorithm_lineage.get("development_paths", {})
        if isinstance(dev_paths, dict):
            main_paths = dev_paths.get("main_paths", [])
            if len(main_paths) >= 3:
                insights.append("该领域呈现多元化发展模式，存在多条并行的技术发展路径")
            elif len(main_paths) >= 1:
                insights.append("该领域的技术发展相对线性，主要沿着少数几条路径演进")
        
        # 洞察3: 创新活跃度
        if key_nodes:
            recent_nodes = [node for node in key_nodes if node.get("year", 0) >= 2015]
            if len(recent_nodes) / len(key_nodes) > 0.5:
                insights.append("近年来该领域创新活跃，新技术不断涌现")
            else:
                insights.append("该领域的基础技术相对成熟，近期主要以改进和应用为主")
        
        # 洞察4: 技术关联性
        edges = graph_data.get("edges", [])
        nodes = graph_data.get("nodes", {})
        if edges and nodes:
            density = len(edges) / max(1, len(nodes) * (len(nodes) - 1))
            if density > 0.1:
                insights.append("算法间关联密切，技术融合程度较高")
            else:
                insights.append("算法相对独立，技术路线分化明显")
        
        return insights[:5]  # 返回最多5个关键洞察
