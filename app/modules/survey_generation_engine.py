"""
综述内容生成引擎
基于模板和算法脉络分析生成高质量的学术综述
"""

import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
import json
import re
from dataclasses import dataclass

@dataclass
class SurveyTemplate:
    """综述模板"""
    name: str
    description: str
    sections: List[Dict[str, Any]]
    style: str  # academic, technical, overview
    target_length: int  # 目标字数
    
class SurveyContentGenerator:
    """综述内容生成器"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.templates = self._load_survey_templates()
        
    def generate_survey_content(self, 
                              topic: str,
                              autosurvey_result: Dict[str, Any],
                              algorithm_lineage: Dict[str, Any],
                              template_name: str = "academic_standard") -> Dict[str, Any]:
        """生成综述内容"""
        try:
            self.logger.info(f"开始生成综述内容: {topic}")
            
            # 选择模板
            template = self._get_template(template_name)
            
            # 分析输入数据
            content_analysis = self._analyze_input_data(autosurvey_result, algorithm_lineage)
            
            # 生成各个章节
            sections = []
            for section_config in template.sections:
                section_content = self._generate_section(
                    section_config, topic, autosurvey_result, algorithm_lineage, content_analysis
                )
                sections.append(section_content)
            
            # 整合内容
            full_content = self._integrate_sections(sections, template, topic)
            
            # 后处理和优化
            optimized_content = self._optimize_content(full_content, template)
            
            # 生成元数据
            metadata = self._generate_metadata(optimized_content, algorithm_lineage)
            
            return {
                "content": optimized_content,
                "sections": sections,
                "template_used": template_name,
                "metadata": metadata,
                "generation_time": datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"生成综述内容失败: {str(e)}")
            raise
    
    def _load_survey_templates(self) -> Dict[str, SurveyTemplate]:
        """加载综述模板"""
        templates = {}
        
        # 学术标准模板
        templates["academic_standard"] = SurveyTemplate(
            name="学术标准模板",
            description="适用于学术期刊的标准综述格式",
            style="academic",
            target_length=8000,
            sections=[
                {
                    "name": "abstract",
                    "title": "摘要",
                    "type": "summary",
                    "target_length": 300,
                    "required": True
                },
                {
                    "name": "introduction",
                    "title": "引言",
                    "type": "introduction",
                    "target_length": 1000,
                    "required": True,
                    "subsections": ["背景与动机", "研究意义", "文章结构"]
                },
                {
                    "name": "related_work",
                    "title": "相关工作",
                    "type": "literature_review",
                    "target_length": 1500,
                    "required": True,
                    "subsections": ["传统方法", "现代方法", "最新进展"]
                },
                {
                    "name": "algorithm_evolution",
                    "title": "算法演进分析",
                    "type": "lineage_analysis",
                    "target_length": 2000,
                    "required": True,
                    "subsections": ["关键节点识别", "发展路径分析", "技术分支与融合"]
                },
                {
                    "name": "methodology",
                    "title": "方法论",
                    "type": "methodology",
                    "target_length": 1200,
                    "required": True,
                    "subsections": ["研究方法", "评估标准", "数据来源"]
                },
                {
                    "name": "analysis",
                    "title": "分析与讨论",
                    "type": "analysis",
                    "target_length": 1500,
                    "required": True,
                    "subsections": ["技术对比", "性能分析", "应用场景"]
                },
                {
                    "name": "future_directions",
                    "title": "未来方向",
                    "type": "future_work",
                    "target_length": 800,
                    "required": True,
                    "subsections": ["技术挑战", "发展趋势", "研究机会"]
                },
                {
                    "name": "conclusion",
                    "title": "结论",
                    "type": "conclusion",
                    "target_length": 500,
                    "required": True
                }
            ]
        )
        
        # 技术概览模板
        templates["technical_overview"] = SurveyTemplate(
            name="技术概览模板",
            description="适用于技术报告和概览文档",
            style="technical",
            target_length=5000,
            sections=[
                {
                    "name": "executive_summary",
                    "title": "执行摘要",
                    "type": "summary",
                    "target_length": 200,
                    "required": True
                },
                {
                    "name": "technology_landscape",
                    "title": "技术全景",
                    "type": "overview",
                    "target_length": 1200,
                    "required": True
                },
                {
                    "name": "key_algorithms",
                    "title": "关键算法",
                    "type": "algorithm_focus",
                    "target_length": 1500,
                    "required": True
                },
                {
                    "name": "evolution_timeline",
                    "title": "演进时间线",
                    "type": "timeline",
                    "target_length": 1000,
                    "required": True
                },
                {
                    "name": "practical_applications",
                    "title": "实际应用",
                    "type": "applications",
                    "target_length": 800,
                    "required": True
                },
                {
                    "name": "recommendations",
                    "title": "建议",
                    "type": "recommendations",
                    "target_length": 300,
                    "required": True
                }
            ]
        )
        
        return templates
    
    def _get_template(self, template_name: str) -> SurveyTemplate:
        """获取指定模板"""
        if template_name not in self.templates:
            self.logger.warning(f"模板 {template_name} 不存在，使用默认模板")
            template_name = "academic_standard"
        
        return self.templates[template_name]
    
    def _analyze_input_data(self, autosurvey_result: Dict[str, Any], algorithm_lineage: Dict[str, Any]) -> Dict[str, Any]:
        """分析输入数据"""
        analysis = {
            "autosurvey_quality": self._assess_autosurvey_quality(autosurvey_result),
            "lineage_richness": self._assess_lineage_richness(algorithm_lineage),
            "content_themes": self._extract_content_themes(autosurvey_result),
            "key_algorithms": self._extract_key_algorithms(algorithm_lineage),
            "temporal_coverage": self._analyze_temporal_coverage(algorithm_lineage),
            "reference_quality": self._assess_reference_quality(autosurvey_result)
        }
        
        return analysis
    
    def _generate_section(self, 
                         section_config: Dict[str, Any],
                         topic: str,
                         autosurvey_result: Dict[str, Any],
                         algorithm_lineage: Dict[str, Any],
                         content_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """生成单个章节"""
        section_type = section_config["type"]
        
        if section_type == "summary":
            content = self._generate_abstract(topic, autosurvey_result, algorithm_lineage)
        elif section_type == "introduction":
            content = self._generate_introduction(topic, autosurvey_result, algorithm_lineage, content_analysis)
        elif section_type == "literature_review":
            content = self._generate_literature_review(topic, autosurvey_result, content_analysis)
        elif section_type == "lineage_analysis":
            content = self._generate_lineage_analysis(topic, algorithm_lineage)
        elif section_type == "methodology":
            content = self._generate_methodology(topic, content_analysis)
        elif section_type == "analysis":
            content = self._generate_analysis(topic, autosurvey_result, algorithm_lineage)
        elif section_type == "future_work":
            content = self._generate_future_directions(topic, algorithm_lineage, content_analysis)
        elif section_type == "conclusion":
            content = self._generate_conclusion(topic, autosurvey_result, algorithm_lineage)
        else:
            content = self._generate_generic_section(section_config, topic, autosurvey_result, algorithm_lineage)
        
        return {
            "name": section_config["name"],
            "title": section_config["title"],
            "type": section_type,
            "content": content,
            "word_count": len(content.split()),
            "subsections": section_config.get("subsections", [])
        }
    
    def _generate_abstract(self, topic: str, autosurvey_result: Dict[str, Any], algorithm_lineage: Dict[str, Any]) -> str:
        """生成摘要"""
        # 提取关键信息
        key_algorithms = algorithm_lineage.get("key_nodes", [])[:3]
        main_paths = algorithm_lineage.get("development_paths", [])[:2]
        
        algorithm_names = [node.get("name", "") for node in key_algorithms]
        
        abstract = f"""本文对{topic}领域进行了全面的综述分析。通过构建算法演进图谱，我们识别出了{len(key_algorithms)}个关键算法节点，包括{', '.join(algorithm_names[:2])}等重要技术。"""
        
        if main_paths:
            abstract += f"分析发现该领域存在{len(main_paths)}条主要发展路径，展现了从传统方法到现代技术的清晰演进脉络。"
        
        abstract += f"本综述基于{len(autosurvey_result.get('references', []))}篇相关文献，为研究者和从业者提供了{topic}领域的全景视图和未来发展方向。"
        
        return abstract
    
    def _generate_introduction(self, topic: str, autosurvey_result: Dict[str, Any], 
                             algorithm_lineage: Dict[str, Any], content_analysis: Dict[str, Any]) -> str:
        """生成引言"""
        intro_parts = []
        
        # 背景与动机
        intro_parts.append(f"## 背景与动机\n\n{topic}作为人工智能领域的重要分支，近年来得到了广泛关注和快速发展。")
        
        # 基于时间覆盖分析添加发展历程
        temporal_info = content_analysis.get("temporal_coverage", {})
        if temporal_info.get("start_year") and temporal_info.get("end_year"):
            intro_parts.append(f"从{temporal_info['start_year']}年到{temporal_info['end_year']}年，该领域经历了{temporal_info['span_years']}年的发展历程。")
        
        # 研究意义
        intro_parts.append(f"\n## 研究意义\n\n深入理解{topic}的发展脉络对于把握技术趋势、指导未来研究具有重要意义。")
        
        # 文章结构
        intro_parts.append("\n## 文章结构\n\n本文首先回顾相关工作，然后分析算法演进关系，接着讨论技术发展趋势，最后提出未来研究方向。")
        
        return "\n".join(intro_parts)
    
    def _generate_literature_review(self, topic: str, autosurvey_result: Dict[str, Any], content_analysis: Dict[str, Any]) -> str:
        """生成文献综述"""
        # 从AutoSurvey结果中提取相关工作部分
        autosurvey_content = autosurvey_result.get("content", "")
        
        # 尝试提取相关工作章节
        related_work_pattern = r"##?\s*(?:相关工作|Related Work|Literature Review).*?(?=##|\Z)"
        match = re.search(related_work_pattern, autosurvey_content, re.DOTALL | re.IGNORECASE)
        
        if match:
            base_content = match.group(0)
        else:
            # 如果没有找到，使用通用模板
            base_content = f"## 相关工作\n\n{topic}领域的研究可以分为几个主要阶段和方向。"
        
        # 增强内容
        themes = content_analysis.get("content_themes", [])
        if themes:
            theme_text = f"\n\n当前研究主要集中在以下几个主题：{', '.join(themes[:5])}。"
            base_content += theme_text
        
        return base_content
    
    def _generate_lineage_analysis(self, topic: str, algorithm_lineage: Dict[str, Any]) -> str:
        """生成算法脉络分析"""
        analysis_parts = []
        
        # 关键节点识别
        key_nodes = algorithm_lineage.get("key_nodes", [])
        if key_nodes:
            analysis_parts.append("## 关键节点识别\n")
            analysis_parts.append(f"通过算法演进图谱分析，我们识别出{len(key_nodes)}个关键算法节点：\n")
            
            for i, node in enumerate(key_nodes[:5], 1):
                node_name = node.get("name", "未知算法")
                node_year = node.get("year", "未知年份")
                importance = node.get("importance_score", 0)
                analysis_parts.append(f"{i}. **{node_name}** ({node_year}) - 重要性评分: {importance:.2f}")
        
        # 发展路径分析
        dev_paths = algorithm_lineage.get("development_paths", {})
        if isinstance(dev_paths, dict) and dev_paths.get("main_paths"):
            analysis_parts.append("\n## 发展路径分析\n")
            main_paths = dev_paths["main_paths"]
            analysis_parts.append(f"识别出{len(main_paths)}条主要发展路径：\n")
            
            for i, path in enumerate(main_paths[:3], 1):
                path_length = path.get("length", 0)
                time_span = path.get("time_span", {})
                start_year = time_span.get("start_year", "未知")
                end_year = time_span.get("end_year", "未知")
                analysis_parts.append(f"{i}. 路径{i}: {path_length}个节点，时间跨度 {start_year}-{end_year}")
        
        # 技术分支与融合
        branches_merges = algorithm_lineage.get("development_paths", {}).get("branches_and_merges", {})
        if branches_merges:
            branches = branches_merges.get("branches", [])
            merges = branches_merges.get("merges", [])
            
            analysis_parts.append("\n## 技术分支与融合\n")
            if branches:
                analysis_parts.append(f"发现{len(branches)}个技术分支点，表明算法发展的多样化趋势。")
            if merges:
                analysis_parts.append(f"识别出{len(merges)}个技术融合点，体现了不同技术路线的交汇整合。")
        
        return "\n".join(analysis_parts)

class SurveyQualityAssessor:
    """综述质量评估器"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def assess_survey_quality(self, survey_content: str, metadata: Dict[str, Any]) -> Dict[str, float]:
        """评估综述质量"""
        scores = {}
        
        # 内容完整性
        scores["completeness"] = self._assess_completeness(survey_content, metadata)
        
        # 结构合理性
        scores["structure"] = self._assess_structure(survey_content)
        
        # 内容连贯性
        scores["coherence"] = self._assess_coherence(survey_content)
        
        # 技术深度
        scores["technical_depth"] = self._assess_technical_depth(survey_content, metadata)
        
        # 创新性
        scores["novelty"] = self._assess_novelty(survey_content, metadata)
        
        # 综合评分
        scores["overall"] = sum(scores.values()) / len(scores)
        
        return scores
    
    def _assess_completeness(self, content: str, metadata: Dict[str, Any]) -> float:
        """评估内容完整性"""
        required_sections = ["引言", "相关工作", "算法演进", "结论"]
        found_sections = 0
        
        for section in required_sections:
            if section in content:
                found_sections += 1
        
        return found_sections / len(required_sections)
    
    def _assess_structure(self, content: str) -> float:
        """评估结构合理性"""
        # 检查标题层次结构
        h1_count = len(re.findall(r'^#\s+', content, re.MULTILINE))
        h2_count = len(re.findall(r'^##\s+', content, re.MULTILINE))
        h3_count = len(re.findall(r'^###\s+', content, re.MULTILINE))
        
        # 合理的层次结构应该是递减的
        if h1_count > 0 and h2_count >= h1_count and h3_count >= 0:
            return min(1.0, (h1_count + h2_count * 0.5 + h3_count * 0.3) / 10)
        else:
            return 0.5
    
    def _assess_coherence(self, content: str) -> float:
        """评估内容连贯性"""
        # 简化的连贯性评估：检查段落长度和连接词
        paragraphs = content.split('\n\n')
        
        # 段落长度合理性
        avg_paragraph_length = sum(len(p.split()) for p in paragraphs) / max(1, len(paragraphs))
        length_score = min(1.0, avg_paragraph_length / 100)  # 理想段落长度约100词
        
        # 连接词使用
        transition_words = ["然而", "因此", "此外", "同时", "另外", "相比之下", "总之"]
        transition_count = sum(content.count(word) for word in transition_words)
        transition_score = min(1.0, transition_count / 20)
        
        return (length_score + transition_score) / 2
    
    def _assess_technical_depth(self, content: str, metadata: Dict[str, Any]) -> float:
        """评估技术深度"""
        # 基于算法名称、技术术语的出现频率
        technical_terms = ["算法", "模型", "方法", "技术", "框架", "架构"]
        term_count = sum(content.count(term) for term in technical_terms)
        
        # 基于内容长度标准化
        content_length = len(content.split())
        term_density = term_count / max(1, content_length / 100)
        
        return min(1.0, term_density / 10)
    
    def _assess_novelty(self, content: str, metadata: Dict[str, Any]) -> float:
        """评估创新性"""
        # 基于算法脉络分析的存在和质量
        lineage_keywords = ["演进", "发展", "脉络", "关系", "路径"]
        lineage_mentions = sum(content.count(keyword) for keyword in lineage_keywords)
        
        return min(1.0, lineage_mentions / 10)

    def _assess_autosurvey_quality(self, autosurvey_result: Dict[str, Any]) -> Dict[str, Any]:
        """评估AutoSurvey结果质量"""
        content = autosurvey_result.get("content", "")
        references = autosurvey_result.get("references", [])

        return {
            "content_length": len(content.split()),
            "reference_count": len(references),
            "has_structure": "##" in content,
            "quality_score": len(content.split()) / 5000 + len(references) / 100  # 简化评分
        }

    def _assess_lineage_richness(self, algorithm_lineage: Dict[str, Any]) -> Dict[str, Any]:
        """评估算法脉络数据丰富度"""
        key_nodes = algorithm_lineage.get("key_nodes", [])
        dev_paths = algorithm_lineage.get("development_paths", {})

        return {
            "key_nodes_count": len(key_nodes),
            "has_development_paths": bool(dev_paths),
            "richness_score": len(key_nodes) / 10 + (1 if dev_paths else 0)
        }

    def _extract_content_themes(self, autosurvey_result: Dict[str, Any]) -> List[str]:
        """提取内容主题"""
        content = autosurvey_result.get("content", "")

        # 简化的主题提取：基于关键词频率
        import re
        from collections import Counter

        # 提取可能的技术术语
        words = re.findall(r'\b[A-Za-z]{4,}\b', content)
        word_counts = Counter(words)

        # 过滤常见词汇，保留技术术语
        common_words = {"this", "that", "with", "from", "they", "have", "been", "were", "will", "more", "such", "also", "than", "only", "very", "well", "much", "most", "many", "some", "time", "work", "data", "used", "using", "based", "method", "methods", "approach", "approaches", "model", "models", "algorithm", "algorithms"}

        themes = []
        for word, count in word_counts.most_common(20):
            if word.lower() not in common_words and count >= 3:
                themes.append(word)

        return themes[:10]

    def _extract_key_algorithms(self, algorithm_lineage: Dict[str, Any]) -> List[Dict[str, Any]]:
        """提取关键算法"""
        key_nodes = algorithm_lineage.get("key_nodes", [])
        return key_nodes[:10]  # 返回前10个关键算法

    def _analyze_temporal_coverage(self, algorithm_lineage: Dict[str, Any]) -> Dict[str, Any]:
        """分析时间覆盖范围"""
        key_nodes = algorithm_lineage.get("key_nodes", [])

        years = []
        for node in key_nodes:
            year = node.get("year")
            if year and isinstance(year, int):
                years.append(year)

        if not years:
            return {"start_year": None, "end_year": None, "span_years": 0}

        return {
            "start_year": min(years),
            "end_year": max(years),
            "span_years": max(years) - min(years),
            "year_distribution": self._calculate_year_distribution(years)
        }

    def _calculate_year_distribution(self, years: List[int]) -> Dict[str, int]:
        """计算年份分布"""
        from collections import Counter

        # 按十年分组
        decades = [(year // 10) * 10 for year in years]
        decade_counts = Counter(decades)

        return {f"{decade}s": count for decade, count in decade_counts.items()}

    def _assess_reference_quality(self, autosurvey_result: Dict[str, Any]) -> Dict[str, Any]:
        """评估参考文献质量"""
        references = autosurvey_result.get("references", [])

        if not references:
            return {"count": 0, "has_recent": False, "quality_score": 0.0}

        # 检查是否有近期文献
        current_year = datetime.now().year
        recent_refs = 0

        for ref in references:
            ref_year = ref.get("year")
            if ref_year and isinstance(ref_year, int) and ref_year >= current_year - 5:
                recent_refs += 1

        return {
            "count": len(references),
            "recent_count": recent_refs,
            "has_recent": recent_refs > 0,
            "quality_score": min(1.0, len(references) / 50 + recent_refs / 20)
        }

    def _generate_methodology(self, topic: str, content_analysis: Dict[str, Any]) -> str:
        """生成方法论章节"""
        methodology_parts = []

        methodology_parts.append("## 研究方法\n")
        methodology_parts.append("本研究采用系统性文献调研方法，结合算法演进图谱分析，对{topic}领域进行全面梳理。")

        methodology_parts.append("\n## 评估标准\n")
        methodology_parts.append("评估算法的重要性主要基于以下标准：技术创新性、影响力、实用性和发展潜力。")

        methodology_parts.append("\n## 数据来源\n")
        ref_quality = content_analysis.get("reference_quality", {})
        ref_count = ref_quality.get("count", 0)
        methodology_parts.append(f"本研究基于{ref_count}篇相关文献，涵盖了{topic}领域的主要研究成果。")

        return "\n".join(methodology_parts)

    def _generate_analysis(self, topic: str, autosurvey_result: Dict[str, Any], algorithm_lineage: Dict[str, Any]) -> str:
        """生成分析与讨论章节"""
        analysis_parts = []

        analysis_parts.append("## 技术对比\n")
        key_algorithms = algorithm_lineage.get("key_nodes", [])[:5]
        if key_algorithms:
            analysis_parts.append("主要算法的技术特点对比如下：\n")
            for i, alg in enumerate(key_algorithms, 1):
                name = alg.get("name", "未知算法")
                year = alg.get("year", "未知年份")
                analysis_parts.append(f"{i}. **{name}** ({year}): 在{topic}领域具有重要影响")

        analysis_parts.append("\n## 性能分析\n")
        analysis_parts.append(f"通过对比分析，我们发现{topic}领域的算法性能呈现持续提升的趋势。")

        analysis_parts.append("\n## 应用场景\n")
        analysis_parts.append(f"{topic}技术在多个应用场景中展现出良好的效果，包括理论研究和实际应用。")

        return "\n".join(analysis_parts)

    def _generate_future_directions(self, topic: str, algorithm_lineage: Dict[str, Any], content_analysis: Dict[str, Any]) -> str:
        """生成未来方向章节"""
        future_parts = []

        future_parts.append("## 技术挑战\n")
        future_parts.append(f"当前{topic}领域面临的主要技术挑战包括算法效率、可解释性和泛化能力等方面。")

        future_parts.append("\n## 发展趋势\n")
        temporal_info = content_analysis.get("temporal_coverage", {})
        if temporal_info.get("end_year"):
            future_parts.append(f"基于{temporal_info['end_year']}年以来的发展趋势，预计{topic}领域将朝着更加智能化和自动化的方向发展。")

        future_parts.append("\n## 研究机会\n")
        future_parts.append("未来的研究机会主要集中在跨领域融合、新兴应用场景和理论突破等方面。")

        return "\n".join(future_parts)

    def _generate_conclusion(self, topic: str, autosurvey_result: Dict[str, Any], algorithm_lineage: Dict[str, Any]) -> str:
        """生成结论章节"""
        key_nodes_count = len(algorithm_lineage.get("key_nodes", []))
        ref_count = len(autosurvey_result.get("references", []))

        conclusion = f"""本文对{topic}领域进行了全面的综述分析。通过构建算法演进图谱，识别出{key_nodes_count}个关键算法节点，分析了技术发展的主要路径和趋势。

基于{ref_count}篇相关文献的分析，我们总结了{topic}领域的主要成就和发展脉络。研究发现，该领域呈现出持续创新和快速发展的特点。

本综述为{topic}领域的研究者和从业者提供了全面的技术全景图，有助于把握发展趋势和识别研究机会。未来的研究应该关注技术融合、应用拓展和理论创新等方向。"""

        return conclusion

    def _generate_generic_section(self, section_config: Dict[str, Any], topic: str,
                                autosurvey_result: Dict[str, Any], algorithm_lineage: Dict[str, Any]) -> str:
        """生成通用章节"""
        section_name = section_config.get("title", "未命名章节")
        return f"## {section_name}\n\n本章节讨论{topic}领域的相关内容。"

    def _integrate_sections(self, sections: List[Dict[str, Any]], template: SurveyTemplate, topic: str) -> str:
        """整合各章节内容"""
        content_parts = [f"# {topic} 综述\n"]

        for section in sections:
            if section["content"].strip():
                content_parts.append(f"\n# {section['title']}\n")
                content_parts.append(section["content"])

        return "\n".join(content_parts)

    def _optimize_content(self, content: str, template: SurveyTemplate) -> str:
        """优化内容"""
        # 基本的内容优化
        optimized = content

        # 移除多余的空行
        optimized = re.sub(r'\n{3,}', '\n\n', optimized)

        # 确保标题格式一致
        optimized = re.sub(r'^#{1,6}\s*', lambda m: '#' * min(len(m.group(0).strip()), 3) + ' ', optimized, flags=re.MULTILINE)

        return optimized.strip()

    def _generate_metadata(self, content: str, algorithm_lineage: Dict[str, Any]) -> Dict[str, Any]:
        """生成元数据"""
        words = content.split()

        return {
            "word_count": len(words),
            "character_count": len(content),
            "section_count": len(re.findall(r'^#\s+', content, re.MULTILINE)),
            "subsection_count": len(re.findall(r'^##\s+', content, re.MULTILINE)),
            "algorithm_mentions": len(algorithm_lineage.get("key_nodes", [])),
            "estimated_reading_time": len(words) // 200,  # 假设每分钟200词
            "content_density": len(words) / max(1, len(content.split('\n\n')))  # 每段平均词数
        }
