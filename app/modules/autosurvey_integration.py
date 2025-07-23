"""
AutoSurvey集成模块 - 数据流设计和接口定义
"""

import json
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
import asyncio
import aiohttp
from app.config import Config
from app.modules.survey_generation_engine import SurveyContentGenerator, SurveyQualityAssessor

@dataclass
class EntityData:
    """实体数据结构"""
    entity_id: str
    entity_type: str  # Algorithm, Dataset, Metric
    name: str
    title: Optional[str] = None
    year: Optional[int] = None
    authors: Optional[List[str]] = None
    task: Optional[str] = None
    description: Optional[str] = None
    source: str = "未知"
    
    def to_autosurvey_format(self) -> Dict[str, Any]:
        """转换为AutoSurvey输入格式"""
        return {
            "id": self.entity_id,
            "type": self.entity_type.lower(),
            "name": self.name,
            "title": self.title or self.name,
            "year": self.year,
            "authors": self.authors or [],
            "description": self.description or "",
            "metadata": {
                "task": self.task,
                "source": self.source
            }
        }

@dataclass
class RelationData:
    """关系数据结构"""
    from_entity: str
    to_entity: str
    relation_type: str
    from_entity_type: str
    to_entity_type: str
    detail: Optional[str] = None
    evidence: Optional[str] = None
    confidence: float = 0.0
    
    def to_autosurvey_format(self) -> Dict[str, Any]:
        """转换为AutoSurvey输入格式"""
        return {
            "source": self.from_entity,
            "target": self.to_entity,
            "type": self.relation_type,
            "source_type": self.from_entity_type.lower(),
            "target_type": self.to_entity_type.lower(),
            "description": self.detail or "",
            "evidence": self.evidence or "",
            "confidence": self.confidence
        }

@dataclass
class TaskData:
    """任务数据结构"""
    task_id: str
    task_name: str
    entities: List[EntityData]
    relations: List[RelationData]
    created_at: datetime
    
    def to_autosurvey_input(self) -> Dict[str, Any]:
        """转换为AutoSurvey完整输入格式"""
        return {
            "task_info": {
                "task_id": self.task_id,
                "task_name": self.task_name,
                "created_at": self.created_at.isoformat()
            },
            "entities": [entity.to_autosurvey_format() for entity in self.entities],
            "relations": [relation.to_autosurvey_format() for relation in self.relations],
            "metadata": {
                "entity_count": len(self.entities),
                "relation_count": len(self.relations),
                "entity_types": list(set(e.entity_type for e in self.entities)),
                "relation_types": list(set(r.relation_type for r in self.relations))
            }
        }

@dataclass
class SurveyGenerationRequest:
    """综述生成请求结构"""
    topic: str
    task_data: TaskData
    parameters: Dict[str, Any]
    
    def to_autosurvey_request(self) -> Dict[str, Any]:
        """转换为AutoSurvey API请求格式"""
        return {
            "topic": self.topic,
            "input_data": self.task_data.to_autosurvey_input(),
            "generation_params": {
                "section_num": self.parameters.get("section_num", 7),
                "subsection_len": self.parameters.get("subsection_len", 700),
                "rag_num": self.parameters.get("rag_num", 60),
                "outline_reference_num": self.parameters.get("outline_reference_num", 1500),
                "model": self.parameters.get("model", "gpt-4o-2024-05-13"),
                **self.parameters
            }
        }

@dataclass
class SurveyResult:
    """综述生成结果结构"""
    survey_id: str
    topic: str
    content: str
    outline: Dict[str, Any]
    references: List[Dict[str, Any]]
    algorithm_lineage: Dict[str, Any]
    generation_time: datetime
    quality_metrics: Dict[str, float]
    
    def to_storage_format(self) -> Dict[str, Any]:
        """转换为存储格式"""
        return {
            "survey_id": self.survey_id,
            "topic": self.topic,
            "content": self.content,
            "outline": json.dumps(self.outline, ensure_ascii=False),
            "references": json.dumps(self.references, ensure_ascii=False),
            "algorithm_lineage": json.dumps(self.algorithm_lineage, ensure_ascii=False),
            "generation_time": self.generation_time.isoformat(),
            "quality_metrics": json.dumps(self.quality_metrics, ensure_ascii=False)
        }

class DataFlowManager:
    """数据流管理器"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
    def extract_task_data(self, task_id: str) -> TaskData:
        """从数据库提取任务数据"""
        # 这里将实现从数据库提取实体和关系数据的逻辑
        pass
    
    def transform_to_autosurvey_format(self, task_data: TaskData, topic: str, params: Dict[str, Any]) -> SurveyGenerationRequest:
        """转换数据为AutoSurvey格式"""
        return SurveyGenerationRequest(
            topic=topic,
            task_data=task_data,
            parameters=params
        )
    
    def validate_data_quality(self, task_data: TaskData) -> Tuple[bool, List[str]]:
        """验证数据质量"""
        issues = []
        
        if not task_data.entities:
            issues.append("没有找到实体数据")
        
        if not task_data.relations:
            issues.append("没有找到关系数据")
        
        # 检查实体完整性
        entity_ids = {e.entity_id for e in task_data.entities}
        for relation in task_data.relations:
            if relation.from_entity not in entity_ids:
                issues.append(f"关系中的源实体 {relation.from_entity} 不存在")
            if relation.to_entity not in entity_ids:
                issues.append(f"关系中的目标实体 {relation.to_entity} 不存在")
        
        return len(issues) == 0, issues
    
    def enhance_with_algorithm_lineage(self, task_data: TaskData) -> Dict[str, Any]:
        """增强算法脉络分析"""
        # 这里将实现算法脉络分析逻辑
        pass

# 数据流接口规范
class DataFlowInterface:
    """数据流接口规范"""
    
    @staticmethod
    def task_selection_to_extraction(task_ids: List[str]) -> List[TaskData]:
        """任务选择到数据提取的接口"""
        pass
    
    @staticmethod
    def extraction_to_transformation(task_data: TaskData) -> Dict[str, Any]:
        """数据提取到格式转换的接口"""
        pass
    
    @staticmethod
    def transformation_to_autosurvey(request_data: Dict[str, Any]) -> Dict[str, Any]:
        """格式转换到AutoSurvey调用的接口"""
        pass
    
    @staticmethod
    def autosurvey_to_enhancement(survey_result: Dict[str, Any], lineage_data: Dict[str, Any]) -> Dict[str, Any]:
        """AutoSurvey结果到脉络增强的接口"""
        pass
    
    @staticmethod
    def enhancement_to_storage(enhanced_result: Dict[str, Any]) -> bool:
        """增强结果到存储的接口"""
        pass

# 错误处理和状态管理
class ProcessingStatus:
    """处理状态管理"""
    
    def __init__(self, task_id: str):
        self.task_id = task_id
        self.status = "initialized"
        self.progress = 0.0
        self.current_stage = "准备中"
        self.message = ""
        self.start_time = datetime.now()
        self.errors = []
    
    def update_status(self, status: str, progress: float, stage: str, message: str = ""):
        """更新处理状态"""
        self.status = status
        self.progress = progress
        self.current_stage = stage
        self.message = message
        self.logger.info(f"任务 {self.task_id} 状态更新: {stage} ({progress:.1%})")
    
    def add_error(self, error: str):
        """添加错误信息"""
        self.errors.append({
            "timestamp": datetime.now().isoformat(),
            "error": error
        })
        self.logger.error(f"任务 {self.task_id} 错误: {error}")

# 模块架构设计

class TaskSelector:
    """任务选择器模块"""

    def __init__(self, db_manager):
        self.db_manager = db_manager
        self.logger = logging.getLogger(__name__)

    def get_available_tasks(self) -> List[Dict[str, Any]]:
        """获取可用任务列表"""
        try:
            tasks = self.db_manager.get_all_tasks()
            return [
                {
                    "task_id": task["task_id"],
                    "task_name": task.get("task_name", "未命名任务"),
                    "entity_count": self._count_entities(task["task_id"]),
                    "relation_count": self._count_relations(task["task_id"]),
                    "created_at": task.get("start_time", "未知"),
                    "status": task.get("status", "未知")
                }
                for task in tasks
            ]
        except Exception as e:
            self.logger.error(f"获取任务列表失败: {str(e)}")
            return []

    def _count_entities(self, task_id: str) -> int:
        """统计任务中的实体数量"""
        try:
            algorithms = self.db_manager.get_algorithms_by_task(task_id)
            datasets = self.db_manager.get_datasets_by_task(task_id)
            metrics = self.db_manager.get_metrics_by_task(task_id)
            return len(algorithms) + len(datasets) + len(metrics)
        except:
            return 0

    def _count_relations(self, task_id: str) -> int:
        """统计任务中的关系数量"""
        try:
            relations = self.db_manager.get_relations_by_task(task_id)
            return len(relations)
        except:
            return 0

    def validate_task_selection(self, task_ids: List[str]) -> Tuple[bool, List[str]]:
        """验证任务选择的有效性"""
        issues = []
        valid_tasks = []

        for task_id in task_ids:
            entity_count = self._count_entities(task_id)
            relation_count = self._count_relations(task_id)

            if entity_count == 0:
                issues.append(f"任务 {task_id} 没有实体数据")
            elif entity_count < 3:
                issues.append(f"任务 {task_id} 实体数据过少 ({entity_count}个)")
            else:
                valid_tasks.append(task_id)

        return len(valid_tasks) > 0, issues

class EntityRelationExtractor:
    """实体关系提取器模块"""

    def __init__(self, db_manager):
        self.db_manager = db_manager
        self.logger = logging.getLogger(__name__)

    def extract_task_data(self, task_id: str) -> TaskData:
        """提取单个任务的数据"""
        try:
            self.logger.info(f"开始提取任务数据: {task_id}")

            # 提取实体数据
            entities = self._extract_entities(task_id)
            self.logger.info(f"提取到 {len(entities)} 个实体")

            # 提取关系数据
            relations = self._extract_relations(task_id)
            self.logger.info(f"提取到 {len(relations)} 个关系")

            # 获取任务信息
            task_info = self._get_task_info(task_id)
            task_name = task_info.get("task_name", f"任务_{task_id}") if task_info else f"任务_{task_id}"

            # 验证数据完整性
            self._validate_extracted_data(entities, relations)

            return TaskData(
                task_id=task_id,
                task_name=task_name,
                entities=entities,
                relations=relations,
                created_at=datetime.now()
            )
        except Exception as e:
            self.logger.error(f"提取任务数据失败 {task_id}: {str(e)}")
            raise

    def _get_task_info(self, task_id: str) -> Dict[str, Any]:
        """获取任务信息"""
        try:
            # 从ProcessingStatus表获取任务信息
            sql = "SELECT task_name, status, start_time FROM ProcessingStatus WHERE task_id = %s"
            result = self.db_manager.db_utils.fetch_one(sql, (task_id,))

            if result:
                return {
                    "task_name": result.get("task_name", ""),
                    "status": result.get("status", ""),
                    "start_time": result.get("start_time", "")
                }
            else:
                # 如果ProcessingStatus中没有，尝试从其他表推断
                return {"task_name": f"任务_{task_id}"}
        except Exception as e:
            self.logger.warning(f"获取任务信息失败 {task_id}: {str(e)}")
            return {"task_name": f"任务_{task_id}"}

    def _validate_extracted_data(self, entities: List[EntityData], relations: List[RelationData]):
        """验证提取的数据"""
        if not entities:
            raise ValueError("没有提取到任何实体数据")

        # 检查实体ID唯一性
        entity_ids = [e.entity_id for e in entities]
        if len(entity_ids) != len(set(entity_ids)):
            self.logger.warning("发现重复的实体ID")

        # 检查关系的实体引用
        entity_id_set = set(entity_ids)
        invalid_relations = []

        for relation in relations:
            if relation.from_entity not in entity_id_set:
                invalid_relations.append(f"关系源实体不存在: {relation.from_entity}")
            if relation.to_entity not in entity_id_set:
                invalid_relations.append(f"关系目标实体不存在: {relation.to_entity}")

        if invalid_relations:
            self.logger.warning(f"发现 {len(invalid_relations)} 个无效关系: {invalid_relations[:5]}")  # 只显示前5个

    def _extract_entities(self, task_id: str) -> List[EntityData]:
        """提取实体数据"""
        entities = []

        try:
            # 提取算法实体
            algorithms = self._get_algorithms_by_task(task_id)
            for alg in algorithms:
                try:
                    # 处理年份数据
                    year = None
                    if alg.get("year"):
                        year_str = str(alg.get("year", "")).strip()
                        if year_str.isdigit():
                            year = int(year_str)
                        else:
                            # 尝试从年份字符串中提取数字
                            import re
                            year_match = re.search(r'\d{4}', year_str)
                            if year_match:
                                year = int(year_match.group())

                    # 处理作者数据
                    authors = []
                    if alg.get("authors"):
                        try:
                            if isinstance(alg["authors"], str):
                                authors = json.loads(alg["authors"])
                            elif isinstance(alg["authors"], list):
                                authors = alg["authors"]
                        except (json.JSONDecodeError, TypeError):
                            # 如果JSON解析失败，尝试按逗号分割
                            authors = [a.strip() for a in str(alg["authors"]).split(",") if a.strip()]

                    entities.append(EntityData(
                        entity_id=alg.get("algorithm_id", ""),
                        entity_type="Algorithm",
                        name=alg.get("name", "未命名算法"),
                        title=alg.get("title", ""),
                        year=year,
                        authors=authors,
                        task=alg.get("task", ""),
                        description=alg.get("title", alg.get("name", "")),
                        source=alg.get("source", "未知")
                    ))
                except Exception as e:
                    self.logger.warning(f"处理算法实体时出错 {alg.get('algorithm_id', 'unknown')}: {str(e)}")
                    continue

            # 提取数据集实体
            datasets = self._get_datasets_by_task(task_id)
            for ds in datasets:
                try:
                    # 处理年份
                    year = None
                    if ds.get("year"):
                        year_str = str(ds.get("year", "")).strip()
                        if year_str.isdigit():
                            year = int(year_str)

                    # 处理创建者
                    creators = []
                    if ds.get("creators"):
                        try:
                            if isinstance(ds["creators"], str):
                                creators = json.loads(ds["creators"])
                            elif isinstance(ds["creators"], list):
                                creators = ds["creators"]
                        except (json.JSONDecodeError, TypeError):
                            creators = [c.strip() for c in str(ds["creators"]).split(",") if c.strip()]

                    entities.append(EntityData(
                        entity_id=ds.get("dataset_id", ""),
                        entity_type="Dataset",
                        name=ds.get("name", "未命名数据集"),
                        description=ds.get("description", ""),
                        year=year,
                        authors=creators,
                        source=ds.get("source", "未知")
                    ))
                except Exception as e:
                    self.logger.warning(f"处理数据集实体时出错 {ds.get('dataset_id', 'unknown')}: {str(e)}")
                    continue

            # 提取指标实体
            metrics = self._get_metrics_by_task(task_id)
            for metric in metrics:
                try:
                    entities.append(EntityData(
                        entity_id=metric.get("metric_id", ""),
                        entity_type="Metric",
                        name=metric.get("name", "未命名指标"),
                        description=metric.get("description", ""),
                        source=metric.get("source", "未知")
                    ))
                except Exception as e:
                    self.logger.warning(f"处理指标实体时出错 {metric.get('metric_id', 'unknown')}: {str(e)}")
                    continue

        except Exception as e:
            self.logger.error(f"提取实体数据时出错: {str(e)}")
            raise

        return entities

    def _get_algorithms_by_task(self, task_id: str) -> List[Dict[str, Any]]:
        """从数据库获取算法实体"""
        try:
            sql = """
            SELECT algorithm_id, name, title, year, authors, task, source
            FROM Algorithms
            WHERE task_id = %s
            """
            return self.db_manager.db_utils.fetch_all(sql, (task_id,))
        except Exception as e:
            self.logger.error(f"获取算法实体失败: {str(e)}")
            return []

    def _get_datasets_by_task(self, task_id: str) -> List[Dict[str, Any]]:
        """从数据库获取数据集实体"""
        try:
            sql = """
            SELECT dataset_id, name, description, year, creators, source
            FROM Datasets
            WHERE task_id = %s
            """
            return self.db_manager.db_utils.fetch_all(sql, (task_id,))
        except Exception as e:
            self.logger.error(f"获取数据集实体失败: {str(e)}")
            return []

    def _get_metrics_by_task(self, task_id: str) -> List[Dict[str, Any]]:
        """从数据库获取指标实体"""
        try:
            sql = """
            SELECT metric_id, name, description, source
            FROM Metrics
            WHERE task_id = %s
            """
            return self.db_manager.db_utils.fetch_all(sql, (task_id,))
        except Exception as e:
            self.logger.error(f"获取指标实体失败: {str(e)}")
            return []

    def _extract_relations(self, task_id: str) -> List[RelationData]:
        """提取关系数据"""
        relations = []

        try:
            relation_records = self._get_relations_by_task(task_id)

            for rel in relation_records:
                try:
                    # 处理置信度
                    confidence = 0.0
                    if rel.get("confidence"):
                        try:
                            confidence = float(rel["confidence"])
                        except (ValueError, TypeError):
                            confidence = 0.0

                    # 验证关系类型
                    relation_type = rel.get("relation_type", "").strip()
                    if not relation_type:
                        self.logger.warning(f"关系缺少类型: {rel.get('from_entity')} -> {rel.get('to_entity')}")
                        relation_type = "未知关系"

                    relations.append(RelationData(
                        from_entity=rel.get("from_entity", ""),
                        to_entity=rel.get("to_entity", ""),
                        relation_type=relation_type,
                        from_entity_type=rel.get("from_entity_type", "Algorithm"),
                        to_entity_type=rel.get("to_entity_type", "Algorithm"),
                        detail=rel.get("detail", ""),
                        evidence=rel.get("evidence", ""),
                        confidence=confidence
                    ))
                except Exception as e:
                    self.logger.warning(f"处理关系时出错: {str(e)}")
                    continue

        except Exception as e:
            self.logger.error(f"提取关系数据时出错: {str(e)}")
            raise

        return relations

    def _get_relations_by_task(self, task_id: str) -> List[Dict[str, Any]]:
        """从数据库获取关系数据"""
        try:
            sql = """
            SELECT from_entity, to_entity, relation_type, from_entity_type, to_entity_type,
                   detail, evidence, confidence
            FROM EvolutionRelations
            WHERE task_id = %s
            """
            return self.db_manager.db_utils.fetch_all(sql, (task_id,))
        except Exception as e:
            self.logger.error(f"获取关系数据失败: {str(e)}")
            return []

    def merge_multiple_tasks(self, task_ids: List[str]) -> TaskData:
        """合并多个任务的数据"""
        all_entities = []
        all_relations = []
        task_names = []

        for task_id in task_ids:
            task_data = self.extract_task_data(task_id)
            all_entities.extend(task_data.entities)
            all_relations.extend(task_data.relations)
            task_names.append(task_data.task_name)

        # 去重实体
        unique_entities = {}
        for entity in all_entities:
            key = f"{entity.entity_type}_{entity.entity_id}"
            if key not in unique_entities:
                unique_entities[key] = entity

        # 去重关系
        unique_relations = {}
        for relation in all_relations:
            key = f"{relation.from_entity}_{relation.to_entity}_{relation.relation_type}"
            if key not in unique_relations:
                unique_relations[key] = relation

        return TaskData(
            task_id="_".join(task_ids),
            task_name=" + ".join(task_names),
            entities=list(unique_entities.values()),
            relations=list(unique_relations.values()),
            created_at=datetime.now()
        )

# 配置管理
class AutoSurveyConfig:
    """AutoSurvey配置管理"""

    def __init__(self):
        self.api_url = Config.OPENAI_BASE_URL or "https://api.openai.com/v1/chat/completions"
        self.api_key = Config.OPENAI_API_KEY
        self.model = Config.OPENAI_MODEL or "gpt-4o-2024-05-13"
        self.timeout = 300  # 5分钟超时
        self.max_retries = 3

    def get_generation_params(self) -> Dict[str, Any]:
        """获取生成参数"""
        return {
            "section_num": 7,
            "subsection_len": 700,
            "rag_num": 60,
            "outline_reference_num": 1500,
            "model": self.model,
            "api_url": self.api_url,
            "api_key": self.api_key
        }

class AutoSurveyConnector:
    """AutoSurvey连接器模块"""

    def __init__(self, config: AutoSurveyConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.session = None

    async def __aenter__(self):
        """异步上下文管理器入口"""
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=self.config.timeout)
        )
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """异步上下文管理器出口"""
        if self.session:
            await self.session.close()

    async def generate_survey(self, request: SurveyGenerationRequest) -> SurveyResult:
        """生成综述"""
        try:
            # 模拟AutoSurvey API调用
            # 实际实现中需要根据AutoSurvey的真实API进行调用

            self.logger.info(f"开始生成综述: {request.topic}")

            # 准备请求数据
            request_data = request.to_autosurvey_request()

            # 这里应该是真实的API调用
            # response = await self._call_autosurvey_api(request_data)

            # 模拟响应
            mock_response = await self._mock_autosurvey_response(request_data)

            return self._parse_survey_result(mock_response, request.topic)

        except Exception as e:
            self.logger.error(f"生成综述失败: {str(e)}")
            raise

    async def _call_autosurvey_api(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """调用AutoSurvey API"""
        headers = {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json"
        }

        for attempt in range(self.config.max_retries):
            try:
                async with self.session.post(
                    self.config.api_url,
                    json=request_data,
                    headers=headers
                ) as response:
                    if response.status == 200:
                        return await response.json()
                    else:
                        error_text = await response.text()
                        raise Exception(f"API调用失败: {response.status} - {error_text}")

            except Exception as e:
                if attempt == self.config.max_retries - 1:
                    raise
                self.logger.warning(f"API调用失败，重试 {attempt + 1}/{self.config.max_retries}: {str(e)}")
                await asyncio.sleep(2 ** attempt)  # 指数退避

    async def _call_autosurvey_command_line(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """通过命令行调用AutoSurvey"""
        import subprocess
        import tempfile
        import uuid
        import os

        # 创建临时文件
        temp_dir = tempfile.mkdtemp()
        input_file = os.path.join(temp_dir, f"input_{uuid.uuid4().hex}.json")
        output_dir = os.path.join(temp_dir, "output")
        os.makedirs(output_dir, exist_ok=True)

        try:
            # 保存输入数据
            with open(input_file, 'w', encoding='utf-8') as f:
                json.dump(request_data, f, ensure_ascii=False, indent=2)

            # 构建命令
            cmd = [
                'python', 'main.py',
                '--topic', request_data.get('topic', ''),
                '--input_file', input_file,
                '--output_dir', output_dir,
                '--model', request_data.get('generation_params', {}).get('model', 'gpt-4o-2024-05-13'),
                '--section_num', str(request_data.get('generation_params', {}).get('section_num', 7)),
                '--subsection_len', str(request_data.get('generation_params', {}).get('subsection_len', 700)),
                '--rag_num', str(request_data.get('generation_params', {}).get('rag_num', 60)),
                '--outline_reference_num', str(request_data.get('generation_params', {}).get('outline_reference_num', 1500)),
                '--api_key', self.config.api_key,
                '--api_url', self.config.api_url
            ]

            # 执行命令
            self.logger.info(f"执行AutoSurvey命令: {' '.join(cmd)}")

            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd='./AutoSurvey'  # 假设AutoSurvey在此目录
            )

            stdout, stderr = await process.communicate()

            if process.returncode == 0:
                # 读取输出结果
                result_file = os.path.join(output_dir, 'survey_result.json')
                if os.path.exists(result_file):
                    with open(result_file, 'r', encoding='utf-8') as f:
                        result = json.load(f)
                    return result
                else:
                    # 如果没有JSON结果文件，尝试读取Markdown文件
                    md_files = [f for f in os.listdir(output_dir) if f.endswith('.md')]
                    if md_files:
                        with open(os.path.join(output_dir, md_files[0]), 'r', encoding='utf-8') as f:
                            content = f.read()
                        return self._create_result_from_content(content, request_data)
                    else:
                        raise Exception("未找到输出文件")
            else:
                error_msg = stderr.decode('utf-8') if stderr else "未知错误"
                raise Exception(f"AutoSurvey执行失败: {error_msg}")

        finally:
            # 清理临时文件
            import shutil
            shutil.rmtree(temp_dir, ignore_errors=True)

    def _create_result_from_content(self, content: str, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """从内容创建结果对象"""
        topic = request_data.get('topic', '未知主题')

        # 简单解析Markdown内容
        lines = content.split('\n')
        sections = []
        current_section = None

        for line in lines:
            if line.startswith('# '):
                if current_section:
                    sections.append(current_section)
                current_section = {"title": line[2:].strip(), "subsections": []}
            elif line.startswith('## ') and current_section:
                current_section["subsections"].append(line[3:].strip())

        if current_section:
            sections.append(current_section)

        return {
            "survey_id": f"survey_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "topic": topic,
            "content": content,
            "outline": {"sections": sections},
            "references": [],  # 需要从内容中解析
            "generation_metadata": {
                "model": request_data.get("generation_params", {}).get("model", "unknown"),
                "generation_time": datetime.now().isoformat(),
                "input_entity_count": len(request_data.get("entities", [])),
                "input_relation_count": len(request_data.get("relations", []))
            }
        }

    async def _mock_autosurvey_response(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """模拟AutoSurvey响应（用于测试）"""
        await asyncio.sleep(2)  # 模拟处理时间

        topic = request_data.get("topic", "未知主题")
        entity_count = len(request_data.get("entities", []))
        relation_count = len(request_data.get("relations", []))

        # 生成更真实的内容
        content = self._generate_mock_content(topic, entity_count, relation_count, request_data)

        return {
            "survey_id": f"survey_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "topic": topic,
            "content": content,
            "outline": {
                "sections": [
                    {"title": "引言", "subsections": ["背景与动机", "研究意义", "文章结构"]},
                    {"title": "相关工作", "subsections": ["传统方法", "深度学习方法", "最新进展"]},
                    {"title": "方法论", "subsections": ["研究方法", "数据收集", "评估标准"]},
                    {"title": "算法演进分析", "subsections": ["关键节点识别", "发展路径分析", "技术趋势"]},
                    {"title": "实验与分析", "subsections": ["实验设置", "结果分析", "性能对比"]},
                    {"title": "讨论", "subsections": ["主要发现", "局限性分析", "未来挑战"]},
                    {"title": "结论与展望", "subsections": ["主要贡献", "实际应用", "未来方向"]}
                ]
            },
            "references": self._generate_mock_references(entity_count),
            "generation_metadata": {
                "model": request_data.get("generation_params", {}).get("model", "gpt-4o-2024-05-13"),
                "generation_time": datetime.now().isoformat(),
                "input_entity_count": entity_count,
                "input_relation_count": relation_count,
                "processing_time": 2.0,
                "word_count": len(content.split()),
                "section_count": 7
            }
        }

    def _generate_mock_content(self, topic: str, entity_count: int, relation_count: int, request_data: Dict[str, Any]) -> str:
        """生成模拟的综述内容"""
        entities = request_data.get("entities", [])
        relations = request_data.get("relations", [])

        # 提取一些实体名称用于内容生成
        algorithm_names = [e.get("name", "") for e in entities if e.get("type") == "algorithm"][:5]
        dataset_names = [e.get("name", "") for e in entities if e.get("type") == "dataset"][:3]

        content = f"""# {topic} 综述

## 1. 引言

### 1.1 背景与动机

{topic}是当前人工智能领域的重要研究方向。本综述基于 {entity_count} 个核心实体和 {relation_count} 个关系，对该领域进行了系统性的分析和总结。

### 1.2 研究意义

通过深入分析算法演进关系，我们能够更好地理解技术发展脉络，为未来研究提供指导。

### 1.3 文章结构

本文首先回顾相关工作，然后介绍研究方法，接着分析算法演进，最后讨论结果并展望未来。

## 2. 相关工作

### 2.1 传统方法

早期的研究主要集中在传统机器学习方法上。这些方法虽然在某些场景下表现良好，但在处理复杂数据时存在局限性。

### 2.2 深度学习方法

随着深度学习的兴起，{', '.join(algorithm_names[:3]) if algorithm_names else '各种神经网络方法'}等算法被广泛应用。

### 2.3 最新进展

近年来，研究者们在{topic}领域取得了显著进展，特别是在{', '.join(algorithm_names[3:]) if len(algorithm_names) > 3 else '新兴技术'}方面。

## 3. 方法论

### 3.1 研究方法

本研究采用系统性文献调研方法，结合定量和定性分析，构建了完整的算法演进图谱。

### 3.2 数据收集

我们收集了{entity_count}个相关实体的详细信息，包括算法、数据集和评价指标。主要数据集包括{', '.join(dataset_names) if dataset_names else '多个标准数据集'}。

### 3.3 评估标准

采用多维度评估框架，包括技术创新性、实用性和影响力等指标。

## 4. 算法演进分析

### 4.1 关键节点识别

通过分析{relation_count}个关系，我们识别出了多个关键的算法节点，这些节点在技术发展中起到了重要作用。

### 4.2 发展路径分析

算法发展呈现出明显的演进路径，从传统方法逐步发展到现代深度学习方法。

### 4.3 技术趋势

当前的技术趋势显示，{topic}领域正朝着更加智能化和自动化的方向发展。

## 5. 实验与分析

### 5.1 实验设置

我们设计了全面的实验来验证算法演进分析的有效性。

### 5.2 结果分析

实验结果表明，基于实体关系的分析方法能够有效识别技术发展趋势。

### 5.3 性能对比

与传统分析方法相比，我们的方法在准确性和完整性方面都有显著提升。

## 6. 讨论

### 6.1 主要发现

本研究的主要发现包括：技术发展具有明显的阶段性特征，关键算法节点对后续发展有重要影响。

### 6.2 局限性分析

当前研究还存在一些局限性，如数据覆盖范围有限、某些新兴技术的分析深度不够等。

### 6.3 未来挑战

{topic}领域面临的主要挑战包括技术标准化、跨领域融合和实际应用等方面。

## 7. 结论与展望

### 7.1 主要贡献

本文的主要贡献包括：构建了完整的算法演进图谱，识别了关键技术节点，分析了发展趋势。

### 7.2 实际应用

研究结果可以为研究者选择技术路线、企业制定技术战略提供参考。

### 7.3 未来方向

未来的研究方向包括：扩大数据覆盖范围、深化跨领域分析、加强实际应用验证等。

## 参考文献

[1] 示例论文1. 作者1等. {topic}的早期研究. 会议/期刊名, 2020.
[2] 示例论文2. 作者2等. 深度学习在{topic}中的应用. 会议/期刊名, 2021.
[3] 示例论文3. 作者3等. {topic}的最新进展. 会议/期刊名, 2022.
"""

        return content

    def _generate_mock_references(self, entity_count: int) -> List[Dict[str, Any]]:
        """生成模拟的参考文献"""
        references = []
        ref_count = min(50, max(10, entity_count // 2))

        for i in range(ref_count):
            references.append({
                "title": f"示例论文{i+1}: 相关技术研究",
                "authors": [f"作者{i+1}", f"作者{i+2}"],
                "year": 2020 + (i % 4),
                "venue": "顶级会议/期刊",
                "url": f"https://example.com/paper{i+1}"
            })

        return references

    def _parse_survey_result(self, response: Dict[str, Any], topic: str) -> SurveyResult:
        """解析综述生成结果"""
        return SurveyResult(
            survey_id=response.get("survey_id", ""),
            topic=topic,
            content=response.get("content", ""),
            outline=response.get("outline", {}),
            references=response.get("references", []),
            algorithm_lineage={},  # 将在后续步骤中填充
            generation_time=datetime.now(),
            quality_metrics=self._calculate_quality_metrics(response)
        )

    async def generate_enhanced_survey(self, request: SurveyGenerationRequest,
                                     algorithm_lineage: Dict[str, Any]) -> SurveyResult:
        """生成增强版综述（集成算法脉络分析）"""
        try:
            self.logger.info(f"开始生成增强版综述: {request.topic}")

            # 首先调用AutoSurvey生成基础综述
            base_survey = await self.generate_survey(request)

            # 使用综述生成引擎增强内容
            content_generator = SurveyContentGenerator()
            quality_assessor = SurveyQualityAssessor()

            # 准备AutoSurvey结果数据
            autosurvey_result = {
                "content": base_survey.content,
                "outline": base_survey.outline,
                "references": base_survey.references,
                "metadata": {
                    "generation_time": base_survey.generation_time.isoformat(),
                    "survey_id": base_survey.survey_id
                }
            }

            # 生成增强内容
            enhanced_content_result = content_generator.generate_survey_content(
                topic=request.topic,
                autosurvey_result=autosurvey_result,
                algorithm_lineage=algorithm_lineage,
                template_name="academic_standard"
            )

            # 评估质量
            quality_scores = quality_assessor.assess_survey_quality(
                enhanced_content_result["content"],
                enhanced_content_result["metadata"]
            )

            # 创建增强版结果
            enhanced_survey = SurveyResult(
                survey_id=base_survey.survey_id,
                topic=request.topic,
                content=enhanced_content_result["content"],
                outline=enhanced_content_result.get("sections", base_survey.outline),
                references=base_survey.references,
                algorithm_lineage=algorithm_lineage,
                generation_time=datetime.now(),
                quality_metrics=quality_scores
            )

            self.logger.info(f"增强版综述生成完成: {request.topic}")
            return enhanced_survey

        except Exception as e:
            self.logger.error(f"生成增强版综述失败: {str(e)}")
            # 如果增强失败，返回基础版本
            return await self.generate_survey(request)

    def _calculate_quality_metrics(self, response: Dict[str, Any]) -> Dict[str, float]:
        """计算质量指标"""
        content = response.get("content", "")
        references = response.get("references", [])

        return {
            "content_length": len(content),
            "reference_count": len(references),
            "section_count": len(response.get("outline", {}).get("sections", [])),
            "completeness_score": min(1.0, len(content) / 5000),  # 基于内容长度的完整性评分
            "reference_density": len(references) / max(1, len(content.split()))  # 引用密度
        }

class AlgorithmLineageAnalyzer:
    """算法脉络分析器模块"""

    def __init__(self, db_manager):
        self.db_manager = db_manager
        self.logger = logging.getLogger(__name__)

    def analyze_algorithm_lineage(self, task_data: TaskData) -> Dict[str, Any]:
        """分析算法脉络"""
        try:
            # 构建算法图谱
            algorithm_graph = self._build_algorithm_graph(task_data)

            # 识别关键节点
            key_nodes = self._identify_key_nodes(algorithm_graph)

            # 分析发展路径
            development_paths = self._analyze_development_paths(algorithm_graph)

            # 计算影响力指标
            influence_metrics = self._calculate_influence_metrics(algorithm_graph)

            return {
                "algorithm_graph": algorithm_graph,
                "key_nodes": key_nodes,
                "development_paths": development_paths,
                "influence_metrics": influence_metrics,
                "analysis_summary": self._generate_analysis_summary(key_nodes, development_paths)
            }

        except Exception as e:
            self.logger.error(f"算法脉络分析失败: {str(e)}")
            return {}

    def _build_algorithm_graph(self, task_data: TaskData) -> Dict[str, Any]:
        """构建算法图谱（增强版）"""
        nodes = {}
        edges = []
        node_attributes = {}

        # 添加算法节点
        for entity in task_data.entities:
            if entity.entity_type == "Algorithm":
                node_id = entity.entity_id

                # 基础节点信息
                nodes[node_id] = {
                    "id": node_id,
                    "name": entity.name,
                    "year": entity.year,
                    "authors": entity.authors or [],
                    "type": "algorithm",
                    "title": entity.title or entity.name,
                    "description": entity.description or "",
                    "source": entity.source
                }

                # 计算节点属性
                node_attributes[node_id] = self._calculate_node_attributes(entity, task_data)

        # 添加关系边（包括非算法实体的关系）
        edge_weights = {}

        for relation in task_data.relations:
            source_id = relation.from_entity
            target_id = relation.to_entity

            # 只处理算法之间的关系
            if source_id in nodes and target_id in nodes:
                edge_key = f"{source_id}->{target_id}"

                # 如果已存在相同的边，合并权重
                if edge_key in edge_weights:
                    edge_weights[edge_key]["confidence"] = max(
                        edge_weights[edge_key]["confidence"],
                        relation.confidence
                    )
                    edge_weights[edge_key]["types"].append(relation.relation_type)
                else:
                    edge_weights[edge_key] = {
                        "source": source_id,
                        "target": target_id,
                        "confidence": relation.confidence,
                        "types": [relation.relation_type],
                        "detail": relation.detail,
                        "evidence": relation.evidence
                    }

        # 转换为边列表
        for edge_data in edge_weights.values():
            edges.append({
                "source": edge_data["source"],
                "target": edge_data["target"],
                "type": edge_data["types"][0],  # 主要关系类型
                "all_types": edge_data["types"],  # 所有关系类型
                "confidence": edge_data["confidence"],
                "weight": self._calculate_edge_weight(edge_data),
                "detail": edge_data["detail"],
                "evidence": edge_data["evidence"]
            })

        # 计算图的拓扑属性
        graph_metrics = self._calculate_graph_metrics(nodes, edges)

        # 识别算法家族/集群
        algorithm_clusters = self._identify_algorithm_clusters(nodes, edges)

        return {
            "nodes": nodes,
            "edges": edges,
            "node_attributes": node_attributes,
            "graph_metrics": graph_metrics,
            "algorithm_clusters": algorithm_clusters,
            "metadata": {
                "node_count": len(nodes),
                "edge_count": len(edges),
                "analysis_time": datetime.now().isoformat(),
                "density": len(edges) / max(1, len(nodes) * (len(nodes) - 1)),
                "avg_degree": sum(len([e for e in edges if e["source"] == n or e["target"] == n]) for n in nodes) / max(1, len(nodes))
            }
        }

    def _calculate_node_attributes(self, entity: EntityData, task_data: TaskData) -> Dict[str, Any]:
        """计算节点属性"""
        node_id = entity.entity_id

        # 计算度数
        in_degree = sum(1 for r in task_data.relations if r.to_entity == node_id)
        out_degree = sum(1 for r in task_data.relations if r.from_entity == node_id)
        total_degree = in_degree + out_degree

        # 计算年份相关属性
        year_score = 0.0
        if entity.year:
            # 年份越新，分数越高（以2000年为基准）
            year_score = max(0, (entity.year - 2000) / 25.0)

        # 计算作者影响力（简化版）
        author_score = min(1.0, len(entity.authors or []) * 0.2) if entity.authors else 0.0

        # 计算描述完整性
        description_score = 0.0
        if entity.description:
            description_score = min(1.0, len(entity.description) / 200.0)

        # 综合重要性评分
        importance_score = (
            total_degree * 0.4 +
            year_score * 0.2 +
            author_score * 0.2 +
            description_score * 0.2
        )

        return {
            "in_degree": in_degree,
            "out_degree": out_degree,
            "total_degree": total_degree,
            "year_score": year_score,
            "author_score": author_score,
            "description_score": description_score,
            "importance_score": importance_score,
            "is_root": in_degree == 0,  # 根节点（没有前驱）
            "is_leaf": out_degree == 0,  # 叶节点（没有后继）
            "is_hub": total_degree >= 3   # 枢纽节点（度数较高）
        }

    def _calculate_edge_weight(self, edge_data: Dict[str, Any]) -> float:
        """计算边权重"""
        base_weight = edge_data["confidence"]

        # 根据关系类型调整权重
        type_weights = {
            "improve": 0.9,
            "extend": 0.8,
            "based_on": 0.7,
            "compare": 0.5,
            "similar": 0.4,
            "related": 0.3
        }

        max_type_weight = 0.0
        for rel_type in edge_data["types"]:
            for key, weight in type_weights.items():
                if key in rel_type.lower():
                    max_type_weight = max(max_type_weight, weight)
                    break

        # 如果有证据，增加权重
        evidence_bonus = 0.1 if edge_data.get("evidence") else 0.0

        return min(1.0, base_weight + max_type_weight + evidence_bonus)

    def _calculate_graph_metrics(self, nodes: Dict[str, Any], edges: List[Dict[str, Any]]) -> Dict[str, Any]:
        """计算图的拓扑指标"""
        if not nodes:
            return {}

        # 构建邻接表
        adj_list = {node_id: [] for node_id in nodes}
        for edge in edges:
            adj_list[edge["source"]].append(edge["target"])

        # 计算连通分量
        connected_components = self._find_connected_components(nodes, edges)

        # 计算最长路径
        longest_paths = self._find_longest_paths(nodes, edges)

        # 计算中心性指标
        centrality_metrics = self._calculate_centrality_metrics(nodes, edges)

        return {
            "connected_components": len(connected_components),
            "largest_component_size": max([len(comp) for comp in connected_components]) if connected_components else 0,
            "longest_path_length": max([len(path) for path in longest_paths]) if longest_paths else 0,
            "average_path_length": self._calculate_average_path_length(longest_paths),
            "centrality_metrics": centrality_metrics,
            "is_dag": self._is_directed_acyclic_graph(nodes, edges),
            "has_cycles": self._has_cycles(nodes, edges)
        }

    def _identify_algorithm_clusters(self, nodes: Dict[str, Any], edges: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """识别算法集群/家族"""
        clusters = []

        # 基于年份聚类
        year_clusters = self._cluster_by_year(nodes)

        # 基于作者聚类
        author_clusters = self._cluster_by_authors(nodes)

        # 基于连通性聚类
        connectivity_clusters = self._cluster_by_connectivity(nodes, edges)

        # 合并聚类结果
        all_clusters = {
            "temporal_clusters": year_clusters,
            "author_clusters": author_clusters,
            "connectivity_clusters": connectivity_clusters
        }

        return all_clusters

    def _cluster_by_year(self, nodes: Dict[str, Any]) -> List[Dict[str, Any]]:
        """基于年份聚类"""
        year_groups = {}

        for node_id, node_data in nodes.items():
            year = node_data.get("year")
            if year:
                # 按5年为一个时间段分组
                period = (year // 5) * 5
                if period not in year_groups:
                    year_groups[period] = []
                year_groups[period].append(node_id)

        clusters = []
        for period, node_ids in year_groups.items():
            if len(node_ids) >= 2:  # 至少2个节点才形成集群
                clusters.append({
                    "type": "temporal",
                    "period": f"{period}-{period+4}",
                    "nodes": node_ids,
                    "size": len(node_ids)
                })

        return sorted(clusters, key=lambda x: x["period"])

    def _cluster_by_authors(self, nodes: Dict[str, Any]) -> List[Dict[str, Any]]:
        """基于作者聚类"""
        author_groups = {}

        for node_id, node_data in nodes.items():
            authors = node_data.get("authors", [])
            for author in authors:
                if author not in author_groups:
                    author_groups[author] = []
                author_groups[author].append(node_id)

        clusters = []
        for author, node_ids in author_groups.items():
            if len(node_ids) >= 2:  # 同一作者至少有2个算法
                clusters.append({
                    "type": "author",
                    "author": author,
                    "nodes": list(set(node_ids)),  # 去重
                    "size": len(set(node_ids))
                })

        return sorted(clusters, key=lambda x: x["size"], reverse=True)

    def _cluster_by_connectivity(self, nodes: Dict[str, Any], edges: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """基于连通性聚类"""
        # 使用简化的连通分量算法
        visited = set()
        clusters = []

        def dfs(node_id, component):
            if node_id in visited:
                return
            visited.add(node_id)
            component.append(node_id)

            # 找到所有相邻节点
            for edge in edges:
                if edge["source"] == node_id and edge["target"] not in visited:
                    dfs(edge["target"], component)
                elif edge["target"] == node_id and edge["source"] not in visited:
                    dfs(edge["source"], component)

        for node_id in nodes:
            if node_id not in visited:
                component = []
                dfs(node_id, component)
                if len(component) >= 2:
                    clusters.append({
                        "type": "connectivity",
                        "nodes": component,
                        "size": len(component)
                    })

        return sorted(clusters, key=lambda x: x["size"], reverse=True)

    def _find_connected_components(self, nodes: Dict[str, Any], edges: List[Dict[str, Any]]) -> List[List[str]]:
        """查找连通分量"""
        visited = set()
        components = []

        # 构建无向图的邻接表
        adj_list = {node_id: [] for node_id in nodes}
        for edge in edges:
            adj_list[edge["source"]].append(edge["target"])
            adj_list[edge["target"]].append(edge["source"])

        def dfs(node_id, component):
            if node_id in visited:
                return
            visited.add(node_id)
            component.append(node_id)

            for neighbor in adj_list[node_id]:
                if neighbor not in visited:
                    dfs(neighbor, component)

        for node_id in nodes:
            if node_id not in visited:
                component = []
                dfs(node_id, component)
                components.append(component)

        return components

    def _find_longest_paths(self, nodes: Dict[str, Any], edges: List[Dict[str, Any]]) -> List[List[str]]:
        """查找最长路径"""
        # 构建有向图的邻接表
        adj_list = {node_id: [] for node_id in nodes}
        in_degree = {node_id: 0 for node_id in nodes}

        for edge in edges:
            adj_list[edge["source"]].append(edge["target"])
            in_degree[edge["target"]] += 1

        # 找到所有根节点（入度为0）
        root_nodes = [node_id for node_id, degree in in_degree.items() if degree == 0]

        longest_paths = []

        def dfs_longest_path(node_id, current_path):
            current_path = current_path + [node_id]

            # 如果没有后继节点，这是一条完整路径
            if not adj_list[node_id]:
                longest_paths.append(current_path)
                return

            # 继续探索所有后继节点
            for neighbor in adj_list[node_id]:
                dfs_longest_path(neighbor, current_path)

        # 从每个根节点开始搜索
        for root in root_nodes:
            dfs_longest_path(root, [])

        return longest_paths

    def _calculate_average_path_length(self, paths: List[List[str]]) -> float:
        """计算平均路径长度"""
        if not paths:
            return 0.0

        total_length = sum(len(path) for path in paths)
        return total_length / len(paths)

    def _calculate_centrality_metrics(self, nodes: Dict[str, Any], edges: List[Dict[str, Any]]) -> Dict[str, Any]:
        """计算中心性指标"""
        if not nodes:
            return {}

        # 度中心性
        degree_centrality = {}
        for node_id in nodes:
            degree = sum(1 for edge in edges if edge["source"] == node_id or edge["target"] == node_id)
            degree_centrality[node_id] = degree / max(1, len(nodes) - 1)

        # 找到最高中心性的节点
        max_degree_node = max(degree_centrality, key=degree_centrality.get) if degree_centrality else None

        # 入度和出度中心性
        in_degree_centrality = {}
        out_degree_centrality = {}

        for node_id in nodes:
            in_degree = sum(1 for edge in edges if edge["target"] == node_id)
            out_degree = sum(1 for edge in edges if edge["source"] == node_id)

            in_degree_centrality[node_id] = in_degree / max(1, len(nodes) - 1)
            out_degree_centrality[node_id] = out_degree / max(1, len(nodes) - 1)

        return {
            "degree_centrality": degree_centrality,
            "in_degree_centrality": in_degree_centrality,
            "out_degree_centrality": out_degree_centrality,
            "max_degree_node": max_degree_node,
            "max_degree_value": degree_centrality.get(max_degree_node, 0.0) if max_degree_node else 0.0,
            "average_degree_centrality": sum(degree_centrality.values()) / len(degree_centrality) if degree_centrality else 0.0
        }

    def _is_directed_acyclic_graph(self, nodes: Dict[str, Any], edges: List[Dict[str, Any]]) -> bool:
        """检查是否为有向无环图（DAG）"""
        # 使用拓扑排序检测环
        in_degree = {node_id: 0 for node_id in nodes}
        adj_list = {node_id: [] for node_id in nodes}

        for edge in edges:
            adj_list[edge["source"]].append(edge["target"])
            in_degree[edge["target"]] += 1

        # 队列存储入度为0的节点
        queue = [node_id for node_id, degree in in_degree.items() if degree == 0]
        processed_count = 0

        while queue:
            node_id = queue.pop(0)
            processed_count += 1

            for neighbor in adj_list[node_id]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)

        # 如果所有节点都被处理，则无环
        return processed_count == len(nodes)

    def _has_cycles(self, nodes: Dict[str, Any], edges: List[Dict[str, Any]]) -> bool:
        """检查是否有环"""
        return not self._is_directed_acyclic_graph(nodes, edges)

    def generate_lineage_summary(self, graph_data: Dict[str, Any]) -> str:
        """生成算法脉络摘要"""
        nodes = graph_data.get("nodes", {})
        edges = graph_data.get("edges", [])
        clusters = graph_data.get("algorithm_clusters", {})
        metrics = graph_data.get("graph_metrics", {})

        summary_parts = []

        # 基本统计
        summary_parts.append(f"算法图谱包含 {len(nodes)} 个算法节点和 {len(edges)} 个演进关系。")

        # 连通性分析
        if metrics.get("connected_components", 0) > 1:
            summary_parts.append(f"图谱分为 {metrics['connected_components']} 个连通分量，最大分量包含 {metrics.get('largest_component_size', 0)} 个节点。")
        else:
            summary_parts.append("所有算法节点形成一个连通的演进网络。")

        # 路径分析
        longest_path = metrics.get("longest_path_length", 0)
        if longest_path > 0:
            summary_parts.append(f"最长演进路径包含 {longest_path} 个算法节点，平均路径长度为 {metrics.get('average_path_length', 0):.1f}。")

        # 集群分析
        temporal_clusters = clusters.get("temporal_clusters", [])
        if temporal_clusters:
            summary_parts.append(f"识别出 {len(temporal_clusters)} 个时间段集群，显示了算法发展的阶段性特征。")

        author_clusters = clusters.get("author_clusters", [])
        if author_clusters:
            top_author_cluster = author_clusters[0]
            summary_parts.append(f"发现 {len(author_clusters)} 个作者研究集群，其中 {top_author_cluster['author']} 贡献了 {top_author_cluster['size']} 个相关算法。")

        # 拓扑特征
        if metrics.get("is_dag"):
            summary_parts.append("算法演进图谱呈现有向无环结构，符合技术发展的时序特征。")
        elif metrics.get("has_cycles"):
            summary_parts.append("算法演进图谱中存在循环引用，表明某些技术之间存在相互影响。")

        # 中心性分析
        centrality = metrics.get("centrality_metrics", {})
        max_degree_node = centrality.get("max_degree_node")
        if max_degree_node and max_degree_node in nodes:
            node_name = nodes[max_degree_node]["name"]
            degree_value = centrality.get("max_degree_value", 0)
            summary_parts.append(f"算法 '{node_name}' 具有最高的中心性 ({degree_value:.2f})，在演进网络中起到关键枢纽作用。")

        return " ".join(summary_parts)

    def _identify_key_nodes(self, graph: Dict[str, Any]) -> List[Dict[str, Any]]:
        """识别关键节点（增强版）"""
        nodes = graph["nodes"]
        edges = graph["edges"]
        node_attributes = graph.get("node_attributes", {})

        # 计算各种中心性指标
        centrality_scores = self._calculate_comprehensive_centrality(nodes, edges)

        # 识别不同类型的关键节点
        milestone_nodes = self._identify_milestone_nodes(nodes, edges, node_attributes)
        breakthrough_nodes = self._identify_breakthrough_nodes(nodes, edges, node_attributes)
        influential_nodes = self._identify_influential_nodes(nodes, edges, centrality_scores)
        bridge_nodes = self._identify_bridge_nodes(nodes, edges)

        # 合并所有关键节点
        all_key_nodes = {}

        # 添加里程碑节点
        for node in milestone_nodes:
            node_id = node["node_id"]
            if node_id not in all_key_nodes:
                all_key_nodes[node_id] = node
                all_key_nodes[node_id]["types"] = []
            all_key_nodes[node_id]["types"].append("milestone")

        # 添加突破性节点
        for node in breakthrough_nodes:
            node_id = node["node_id"]
            if node_id not in all_key_nodes:
                all_key_nodes[node_id] = node
                all_key_nodes[node_id]["types"] = []
            all_key_nodes[node_id]["types"].append("breakthrough")

        # 添加影响力节点
        for node in influential_nodes:
            node_id = node["node_id"]
            if node_id not in all_key_nodes:
                all_key_nodes[node_id] = node
                all_key_nodes[node_id]["types"] = []
            all_key_nodes[node_id]["types"].append("influential")

        # 添加桥接节点
        for node in bridge_nodes:
            node_id = node["node_id"]
            if node_id not in all_key_nodes:
                all_key_nodes[node_id] = node
                all_key_nodes[node_id]["types"] = []
            all_key_nodes[node_id]["types"].append("bridge")

        # 计算综合重要性评分
        for node_id, node_data in all_key_nodes.items():
            node_data["comprehensive_score"] = self._calculate_comprehensive_importance(
                node_data, centrality_scores.get(node_id, {}), node_attributes.get(node_id, {})
            )

        # 按综合评分排序
        key_nodes_list = list(all_key_nodes.values())
        key_nodes_list.sort(key=lambda x: x["comprehensive_score"], reverse=True)

        return key_nodes_list

    def _calculate_comprehensive_centrality(self, nodes: Dict[str, Any], edges: List[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
        """计算综合中心性指标"""
        centrality_scores = {}

        for node_id in nodes:
            # 度中心性
            degree = sum(1 for edge in edges if edge["source"] == node_id or edge["target"] == node_id)
            degree_centrality = degree / max(1, len(nodes) - 1)

            # 入度和出度中心性
            in_degree = sum(1 for edge in edges if edge["target"] == node_id)
            out_degree = sum(1 for edge in edges if edge["source"] == node_id)

            in_degree_centrality = in_degree / max(1, len(nodes) - 1)
            out_degree_centrality = out_degree / max(1, len(nodes) - 1)

            # 加权度中心性（考虑边的权重）
            weighted_degree = sum(edge.get("weight", 1.0) for edge in edges
                                if edge["source"] == node_id or edge["target"] == node_id)
            weighted_degree_centrality = weighted_degree / max(1, len(edges))

            # 接近中心性（简化版）
            closeness_centrality = self._calculate_closeness_centrality(node_id, nodes, edges)

            # 介数中心性（简化版）
            betweenness_centrality = self._calculate_betweenness_centrality(node_id, nodes, edges)

            centrality_scores[node_id] = {
                "degree_centrality": degree_centrality,
                "in_degree_centrality": in_degree_centrality,
                "out_degree_centrality": out_degree_centrality,
                "weighted_degree_centrality": weighted_degree_centrality,
                "closeness_centrality": closeness_centrality,
                "betweenness_centrality": betweenness_centrality
            }

        return centrality_scores

    def _identify_milestone_nodes(self, nodes: Dict[str, Any], edges: List[Dict[str, Any]],
                                node_attributes: Dict[str, Any]) -> List[Dict[str, Any]]:
        """识别里程碑节点"""
        milestone_nodes = []

        # 按年份排序节点
        nodes_by_year = []
        for node_id, node_data in nodes.items():
            if node_data.get("year"):
                nodes_by_year.append((node_data["year"], node_id, node_data))

        nodes_by_year.sort(key=lambda x: x[0])

        # 识别每个时期的重要节点
        time_periods = {}
        for year, node_id, node_data in nodes_by_year:
            period = (year // 5) * 5  # 5年为一个时期
            if period not in time_periods:
                time_periods[period] = []
            time_periods[period].append((node_id, node_data))

        for period, period_nodes in time_periods.items():
            if len(period_nodes) >= 2:  # 至少有2个节点的时期
                # 选择该时期最重要的节点作为里程碑
                best_node = max(period_nodes, key=lambda x: node_attributes.get(x[0], {}).get("importance_score", 0))

                node_id, node_data = best_node
                attrs = node_attributes.get(node_id, {})

                milestone_nodes.append({
                    "node_id": node_id,
                    "name": node_data["name"],
                    "year": node_data["year"],
                    "period": f"{period}-{period+4}",
                    "importance_score": attrs.get("importance_score", 0),
                    "milestone_type": "temporal_leader",
                    "description": f"{period}年代的代表性算法"
                })

        return milestone_nodes

    def _identify_breakthrough_nodes(self, nodes: Dict[str, Any], edges: List[Dict[str, Any]],
                                   node_attributes: Dict[str, Any]) -> List[Dict[str, Any]]:
        """识别突破性节点"""
        breakthrough_nodes = []

        for node_id, node_data in nodes.items():
            attrs = node_attributes.get(node_id, {})

            # 突破性节点的特征
            is_breakthrough = False
            breakthrough_reasons = []

            # 1. 高出度（影响了很多后续算法）
            if attrs.get("out_degree", 0) >= 3:
                is_breakthrough = True
                breakthrough_reasons.append("高影响力")

            # 2. 早期重要节点（根节点且有后续发展）
            if attrs.get("is_root") and attrs.get("out_degree", 0) >= 2:
                is_breakthrough = True
                breakthrough_reasons.append("开创性")

            # 3. 连接不同时期的桥梁节点
            if attrs.get("in_degree", 0) >= 2 and attrs.get("out_degree", 0) >= 2:
                is_breakthrough = True
                breakthrough_reasons.append("承上启下")

            # 4. 高重要性评分
            if attrs.get("importance_score", 0) >= 0.8:
                is_breakthrough = True
                breakthrough_reasons.append("综合重要性高")

            if is_breakthrough:
                breakthrough_nodes.append({
                    "node_id": node_id,
                    "name": node_data["name"],
                    "year": node_data["year"],
                    "importance_score": attrs.get("importance_score", 0),
                    "breakthrough_reasons": breakthrough_reasons,
                    "in_degree": attrs.get("in_degree", 0),
                    "out_degree": attrs.get("out_degree", 0)
                })

        return breakthrough_nodes

    def _identify_influential_nodes(self, nodes: Dict[str, Any], edges: List[Dict[str, Any]],
                                  centrality_scores: Dict[str, Dict[str, float]]) -> List[Dict[str, Any]]:
        """识别影响力节点"""
        influential_nodes = []

        # 计算各种中心性的阈值（前25%）
        all_degree_centrality = [scores.get("degree_centrality", 0) for scores in centrality_scores.values()]
        all_betweenness_centrality = [scores.get("betweenness_centrality", 0) for scores in centrality_scores.values()]
        all_closeness_centrality = [scores.get("closeness_centrality", 0) for scores in centrality_scores.values()]

        degree_threshold = sorted(all_degree_centrality, reverse=True)[len(all_degree_centrality)//4] if all_degree_centrality else 0
        betweenness_threshold = sorted(all_betweenness_centrality, reverse=True)[len(all_betweenness_centrality)//4] if all_betweenness_centrality else 0
        closeness_threshold = sorted(all_closeness_centrality, reverse=True)[len(all_closeness_centrality)//4] if all_closeness_centrality else 0

        for node_id, node_data in nodes.items():
            scores = centrality_scores.get(node_id, {})

            # 检查是否在任何中心性指标上表现突出
            high_centrality_count = 0
            centrality_types = []

            if scores.get("degree_centrality", 0) >= degree_threshold:
                high_centrality_count += 1
                centrality_types.append("度中心性")

            if scores.get("betweenness_centrality", 0) >= betweenness_threshold:
                high_centrality_count += 1
                centrality_types.append("介数中心性")

            if scores.get("closeness_centrality", 0) >= closeness_threshold:
                high_centrality_count += 1
                centrality_types.append("接近中心性")

            # 至少在两个中心性指标上表现突出
            if high_centrality_count >= 2:
                influential_nodes.append({
                    "node_id": node_id,
                    "name": node_data["name"],
                    "year": node_data["year"],
                    "centrality_scores": scores,
                    "high_centrality_types": centrality_types,
                    "influence_score": sum(scores.values()) / len(scores) if scores else 0
                })

        return influential_nodes

    def _identify_bridge_nodes(self, nodes: Dict[str, Any], edges: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """识别桥接节点（连接不同集群的节点）"""
        bridge_nodes = []

        # 简化的桥接节点识别：找到连接不同年份段的节点
        year_groups = {}
        for node_id, node_data in nodes.items():
            year = node_data.get("year")
            if year:
                period = (year // 10) * 10  # 10年为一个大时期
                if period not in year_groups:
                    year_groups[period] = set()
                year_groups[period].add(node_id)

        for node_id, node_data in nodes.items():
            # 检查该节点是否连接了不同时期的节点
            connected_periods = set()

            for edge in edges:
                if edge["source"] == node_id:
                    target_year = nodes[edge["target"]].get("year")
                    if target_year:
                        target_period = (target_year // 10) * 10
                        connected_periods.add(target_period)
                elif edge["target"] == node_id:
                    source_year = nodes[edge["source"]].get("year")
                    if source_year:
                        source_period = (source_year // 10) * 10
                        connected_periods.add(source_period)

            # 如果连接了3个或更多不同时期，认为是桥接节点
            if len(connected_periods) >= 3:
                bridge_nodes.append({
                    "node_id": node_id,
                    "name": node_data["name"],
                    "year": node_data["year"],
                    "connected_periods": list(connected_periods),
                    "bridge_score": len(connected_periods)
                })

        return bridge_nodes

    def _calculate_closeness_centrality(self, node_id: str, nodes: Dict[str, Any], edges: List[Dict[str, Any]]) -> float:
        """计算接近中心性（简化版）"""
        # 构建邻接表
        adj_list = {nid: [] for nid in nodes}
        for edge in edges:
            adj_list[edge["source"]].append(edge["target"])
            adj_list[edge["target"]].append(edge["source"])  # 无向图

        # BFS计算到所有其他节点的最短距离
        distances = {nid: float('inf') for nid in nodes}
        distances[node_id] = 0
        queue = [node_id]

        while queue:
            current = queue.pop(0)
            for neighbor in adj_list[current]:
                if distances[neighbor] == float('inf'):
                    distances[neighbor] = distances[current] + 1
                    queue.append(neighbor)

        # 计算接近中心性
        reachable_distances = [d for d in distances.values() if d != float('inf') and d > 0]
        if not reachable_distances:
            return 0.0

        avg_distance = sum(reachable_distances) / len(reachable_distances)
        return 1.0 / avg_distance if avg_distance > 0 else 0.0

    def _calculate_betweenness_centrality(self, node_id: str, nodes: Dict[str, Any], edges: List[Dict[str, Any]]) -> float:
        """计算介数中心性（简化版）"""
        # 简化实现：计算通过该节点的边数占总边数的比例
        passing_edges = 0
        total_edges = len(edges)

        for edge in edges:
            # 如果该节点是边的源或目标，认为边通过该节点
            if edge["source"] == node_id or edge["target"] == node_id:
                passing_edges += 1

        return passing_edges / max(1, total_edges)

    def _calculate_comprehensive_importance(self, node_data: Dict[str, Any],
                                         centrality_scores: Dict[str, float],
                                         node_attributes: Dict[str, Any]) -> float:
        """计算综合重要性评分"""
        score = 0.0

        # 基础重要性评分
        base_importance = node_attributes.get("importance_score", 0.0)
        score += base_importance * 0.3

        # 中心性评分
        degree_centrality = centrality_scores.get("degree_centrality", 0.0)
        betweenness_centrality = centrality_scores.get("betweenness_centrality", 0.0)
        closeness_centrality = centrality_scores.get("closeness_centrality", 0.0)

        centrality_avg = (degree_centrality + betweenness_centrality + closeness_centrality) / 3
        score += centrality_avg * 0.4

        # 节点类型加分
        node_types = node_data.get("types", [])
        type_bonus = 0.0

        if "milestone" in node_types:
            type_bonus += 0.1
        if "breakthrough" in node_types:
            type_bonus += 0.15
        if "influential" in node_types:
            type_bonus += 0.1
        if "bridge" in node_types:
            type_bonus += 0.05

        score += type_bonus

        # 年份加分（较新的算法获得额外分数）
        year = node_data.get("year")
        if year and year >= 2015:
            year_bonus = min(0.1, (year - 2015) * 0.02)
            score += year_bonus

        return min(1.0, score)  # 限制在0-1范围内

    def _calculate_importance_score(self, node_data: Dict[str, Any], in_degree: int, out_degree: int) -> float:
        """计算节点重要性评分"""
        # 基于度数、年份等因素计算重要性
        degree_score = (in_degree + out_degree) * 0.4
        year_score = (node_data.get("year", 2000) - 1990) * 0.01  # 年份越新分数越高
        return degree_score + year_score

    def _analyze_development_paths(self, graph: Dict[str, Any]) -> List[Dict[str, Any]]:
        """分析发展路径（增强版）"""
        nodes = graph["nodes"]
        edges = graph["edges"]

        # 1. 找到所有可能的发展路径
        all_paths = self._find_all_development_paths(nodes, edges)

        # 2. 分析路径特征
        analyzed_paths = []
        for path in all_paths:
            path_analysis = self._analyze_single_path(path, nodes, edges)
            analyzed_paths.append(path_analysis)

        # 3. 识别主要发展路径
        main_paths = self._identify_main_development_paths(analyzed_paths)

        # 4. 识别技术分支和融合点
        branches_and_merges = self._identify_branches_and_merges(nodes, edges)

        # 5. 分析发展趋势
        development_trends = self._analyze_development_trends(analyzed_paths, nodes)

        return {
            "all_paths": analyzed_paths,
            "main_paths": main_paths,
            "branches_and_merges": branches_and_merges,
            "development_trends": development_trends,
            "path_statistics": self._calculate_path_statistics(analyzed_paths)
        }

    def _find_all_development_paths(self, nodes: Dict[str, Any], edges: List[Dict[str, Any]]) -> List[List[str]]:
        """找到所有发展路径"""
        # 构建有向图邻接表
        adj_list = {node_id: [] for node_id in nodes}
        in_degree = {node_id: 0 for node_id in nodes}

        for edge in edges:
            adj_list[edge["source"]].append(edge["target"])
            in_degree[edge["target"]] += 1

        # 找到所有根节点（入度为0）
        root_nodes = [node_id for node_id, degree in in_degree.items() if degree == 0]

        all_paths = []

        def dfs_all_paths(node_id, current_path, visited):
            current_path = current_path + [node_id]

            # 如果没有后继节点，这是一条完整路径
            if not adj_list[node_id]:
                if len(current_path) > 1:  # 至少包含2个节点
                    all_paths.append(current_path)
                return

            # 继续探索所有后继节点
            for neighbor in adj_list[node_id]:
                if neighbor not in visited:  # 避免环路
                    new_visited = visited | {node_id}
                    dfs_all_paths(neighbor, current_path, new_visited)

        # 从每个根节点开始搜索
        for root in root_nodes:
            dfs_all_paths(root, [], set())

        # 如果没有根节点（可能有环），从度数最高的节点开始
        if not all_paths and nodes:
            # 计算每个节点的总度数
            node_degrees = {}
            for node_id in nodes:
                degree = sum(1 for edge in edges if edge["source"] == node_id or edge["target"] == node_id)
                node_degrees[node_id] = degree

            # 从度数最高的节点开始
            start_node = max(node_degrees, key=node_degrees.get)
            dfs_all_paths(start_node, [], set())

        return all_paths

    def _analyze_single_path(self, path: List[str], nodes: Dict[str, Any], edges: List[Dict[str, Any]]) -> Dict[str, Any]:
        """分析单条路径"""
        if len(path) < 2:
            return {"path": path, "valid": False}

        # 基本信息
        path_nodes = [nodes[node_id] for node_id in path]
        years = [node.get("year") for node in path_nodes if node.get("year")]

        # 时间跨度分析
        time_span = {
            "start_year": min(years) if years else None,
            "end_year": max(years) if years else None,
            "duration": max(years) - min(years) if len(years) >= 2 else 0,
            "chronological": self._is_chronological_path(path_nodes)
        }

        # 路径中的关系类型
        path_relations = []
        for i in range(len(path) - 1):
            source_id = path[i]
            target_id = path[i + 1]

            # 找到对应的边
            for edge in edges:
                if edge["source"] == source_id and edge["target"] == target_id:
                    path_relations.append({
                        "from": source_id,
                        "to": target_id,
                        "type": edge["type"],
                        "confidence": edge.get("confidence", 0.0)
                    })
                    break

        # 路径特征分析
        path_features = self._analyze_path_features(path, path_nodes, path_relations)

        # 路径重要性评分
        importance_score = self._calculate_path_importance(path, path_nodes, path_relations)

        return {
            "path": path,
            "path_nodes": [{"id": node_id, "name": nodes[node_id]["name"], "year": nodes[node_id].get("year")}
                          for node_id in path],
            "length": len(path),
            "time_span": time_span,
            "relations": path_relations,
            "features": path_features,
            "importance_score": importance_score,
            "valid": True
        }

    def _is_chronological_path(self, path_nodes: List[Dict[str, Any]]) -> bool:
        """检查路径是否按时间顺序排列"""
        years = [node.get("year") for node in path_nodes if node.get("year")]
        if len(years) < 2:
            return True

        return all(years[i] <= years[i+1] for i in range(len(years)-1))

    def _analyze_path_features(self, path: List[str], path_nodes: List[Dict[str, Any]],
                             path_relations: List[Dict[str, Any]]) -> Dict[str, Any]:
        """分析路径特征"""
        # 关系类型分布
        relation_types = [rel["type"] for rel in path_relations]
        relation_type_counts = {}
        for rel_type in relation_types:
            relation_type_counts[rel_type] = relation_type_counts.get(rel_type, 0) + 1

        # 主要关系类型
        dominant_relation = max(relation_type_counts, key=relation_type_counts.get) if relation_type_counts else None

        # 平均置信度
        confidences = [rel["confidence"] for rel in path_relations if rel["confidence"] > 0]
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0

        # 作者重叠分析
        all_authors = []
        for node in path_nodes:
            authors = node.get("authors", [])
            if authors:
                all_authors.extend(authors)

        unique_authors = list(set(all_authors))
        author_overlap_ratio = (len(all_authors) - len(unique_authors)) / max(1, len(all_authors))

        # 路径连续性（时间间隔分析）
        years = [node.get("year") for node in path_nodes if node.get("year")]
        year_gaps = []
        if len(years) >= 2:
            for i in range(len(years) - 1):
                gap = years[i+1] - years[i]
                year_gaps.append(gap)

        avg_year_gap = sum(year_gaps) / len(year_gaps) if year_gaps else 0

        return {
            "relation_type_distribution": relation_type_counts,
            "dominant_relation_type": dominant_relation,
            "average_confidence": avg_confidence,
            "author_overlap_ratio": author_overlap_ratio,
            "unique_authors_count": len(unique_authors),
            "average_year_gap": avg_year_gap,
            "max_year_gap": max(year_gaps) if year_gaps else 0,
            "continuity_score": 1.0 / (1.0 + avg_year_gap) if avg_year_gap > 0 else 1.0
        }

    def _calculate_path_importance(self, path: List[str], path_nodes: List[Dict[str, Any]],
                                 path_relations: List[Dict[str, Any]]) -> float:
        """计算路径重要性评分"""
        score = 0.0

        # 路径长度加分
        length_score = min(1.0, len(path) / 10.0)  # 最多10个节点得满分
        score += length_score * 0.3

        # 时间跨度加分
        years = [node.get("year") for node in path_nodes if node.get("year")]
        if len(years) >= 2:
            time_span = max(years) - min(years)
            time_score = min(1.0, time_span / 20.0)  # 20年跨度得满分
            score += time_score * 0.2

        # 关系质量加分
        confidences = [rel["confidence"] for rel in path_relations if rel["confidence"] > 0]
        if confidences:
            avg_confidence = sum(confidences) / len(confidences)
            score += avg_confidence * 0.3

        # 节点重要性加分（假设有importance_score属性）
        node_importance_scores = []
        for node in path_nodes:
            # 这里需要从node_attributes中获取重要性评分
            # 简化处理：基于年份和名称长度估算
            year = node.get("year", 2000)
            name_length = len(node.get("name", ""))
            estimated_importance = min(1.0, (year - 2000) * 0.02 + name_length * 0.01)
            node_importance_scores.append(estimated_importance)

        if node_importance_scores:
            avg_node_importance = sum(node_importance_scores) / len(node_importance_scores)
            score += avg_node_importance * 0.2

        return min(1.0, score)

    def _identify_main_development_paths(self, analyzed_paths: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """识别主要发展路径"""
        if not analyzed_paths:
            return []

        # 按重要性评分排序
        sorted_paths = sorted(analyzed_paths, key=lambda x: x["importance_score"], reverse=True)

        # 选择前几条最重要的路径
        main_paths = []
        max_main_paths = min(5, len(sorted_paths))

        for i in range(max_main_paths):
            path = sorted_paths[i]
            if path["importance_score"] >= 0.5:  # 重要性阈值
                main_paths.append({
                    **path,
                    "rank": i + 1,
                    "is_main_path": True
                })

        return main_paths

    def _identify_branches_and_merges(self, nodes: Dict[str, Any], edges: List[Dict[str, Any]]) -> Dict[str, Any]:
        """识别技术分支和融合点"""
        branches = []
        merges = []

        # 计算每个节点的入度和出度
        in_degree = {node_id: 0 for node_id in nodes}
        out_degree = {node_id: 0 for node_id in nodes}

        for edge in edges:
            out_degree[edge["source"]] += 1
            in_degree[edge["target"]] += 1

        # 识别分支点（出度 >= 2）
        for node_id, degree in out_degree.items():
            if degree >= 2:
                node_data = nodes[node_id]

                # 获取分支的目标节点
                branch_targets = []
                for edge in edges:
                    if edge["source"] == node_id:
                        target_node = nodes[edge["target"]]
                        branch_targets.append({
                            "node_id": edge["target"],
                            "name": target_node["name"],
                            "year": target_node.get("year"),
                            "relation_type": edge["type"]
                        })

                branches.append({
                    "branch_node": {
                        "node_id": node_id,
                        "name": node_data["name"],
                        "year": node_data.get("year")
                    },
                    "branch_count": degree,
                    "targets": branch_targets,
                    "branch_type": self._classify_branch_type(branch_targets)
                })

        # 识别融合点（入度 >= 2）
        for node_id, degree in in_degree.items():
            if degree >= 2:
                node_data = nodes[node_id]

                # 获取融合的源节点
                merge_sources = []
                for edge in edges:
                    if edge["target"] == node_id:
                        source_node = nodes[edge["source"]]
                        merge_sources.append({
                            "node_id": edge["source"],
                            "name": source_node["name"],
                            "year": source_node.get("year"),
                            "relation_type": edge["type"]
                        })

                merges.append({
                    "merge_node": {
                        "node_id": node_id,
                        "name": node_data["name"],
                        "year": node_data.get("year")
                    },
                    "merge_count": degree,
                    "sources": merge_sources,
                    "merge_type": self._classify_merge_type(merge_sources)
                })

        return {
            "branches": sorted(branches, key=lambda x: x["branch_count"], reverse=True),
            "merges": sorted(merges, key=lambda x: x["merge_count"], reverse=True),
            "branch_statistics": {
                "total_branches": len(branches),
                "max_branch_count": max([b["branch_count"] for b in branches]) if branches else 0,
                "avg_branch_count": sum([b["branch_count"] for b in branches]) / len(branches) if branches else 0
            },
            "merge_statistics": {
                "total_merges": len(merges),
                "max_merge_count": max([m["merge_count"] for m in merges]) if merges else 0,
                "avg_merge_count": sum([m["merge_count"] for m in merges]) / len(merges) if merges else 0
            }
        }

    def _classify_branch_type(self, targets: List[Dict[str, Any]]) -> str:
        """分类分支类型"""
        if not targets:
            return "unknown"

        # 基于关系类型分类
        relation_types = [t.get("relation_type", "") for t in targets]

        if all("improve" in rt.lower() for rt in relation_types):
            return "improvement_branch"
        elif all("extend" in rt.lower() for rt in relation_types):
            return "extension_branch"
        elif any("compare" in rt.lower() for rt in relation_types):
            return "comparison_branch"
        else:
            return "mixed_branch"

    def _classify_merge_type(self, sources: List[Dict[str, Any]]) -> str:
        """分类融合类型"""
        if not sources:
            return "unknown"

        # 基于时间和关系类型分类
        years = [s.get("year") for s in sources if s.get("year")]
        relation_types = [s.get("relation_type", "") for s in sources]

        if len(set(years)) <= 1:  # 同一时期的融合
            return "concurrent_merge"
        elif any("combine" in rt.lower() or "integrate" in rt.lower() for rt in relation_types):
            return "integration_merge"
        else:
            return "evolutionary_merge"

    def _calculate_path_statistics(self, analyzed_paths: List[Dict[str, Any]]) -> Dict[str, Any]:
        """计算路径统计信息"""
        if not analyzed_paths:
            return {}

        # 基本统计
        path_lengths = [p["length"] for p in analyzed_paths]
        importance_scores = [p["importance_score"] for p in analyzed_paths]

        # 时间跨度统计
        time_spans = []
        for path in analyzed_paths:
            time_span = path.get("time_span", {})
            duration = time_span.get("duration", 0)
            if duration > 0:
                time_spans.append(duration)

        return {
            "total_paths": len(analyzed_paths),
            "average_path_length": sum(path_lengths) / len(path_lengths),
            "max_path_length": max(path_lengths),
            "min_path_length": min(path_lengths),
            "average_importance_score": sum(importance_scores) / len(importance_scores),
            "average_time_span": sum(time_spans) / len(time_spans) if time_spans else 0,
            "max_time_span": max(time_spans) if time_spans else 0,
            "paths_with_time_data": len(time_spans)
        }

class DataFormatConverter:
    """数据格式转换器"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def convert_to_autosurvey_format(self, task_data: TaskData, topic: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """转换为AutoSurvey标准输入格式"""
        try:
            self.logger.info(f"开始转换数据格式，主题: {topic}")

            # 基础转换
            autosurvey_data = task_data.to_autosurvey_input()

            # 增强实体数据
            enhanced_entities = self._enhance_entities(autosurvey_data["entities"])

            # 增强关系数据
            enhanced_relations = self._enhance_relations(autosurvey_data["relations"])

            # 构建完整的AutoSurvey请求
            request_data = {
                "topic": topic,
                "entities": enhanced_entities,
                "relations": enhanced_relations,
                "metadata": {
                    **autosurvey_data["metadata"],
                    "conversion_time": datetime.now().isoformat(),
                    "topic": topic,
                    "data_quality": self._assess_data_quality(enhanced_entities, enhanced_relations)
                },
                "generation_params": self._prepare_generation_params(params),
                "context": self._build_context_information(enhanced_entities, enhanced_relations)
            }

            # 验证转换结果
            self._validate_converted_data(request_data)

            self.logger.info(f"数据格式转换完成: {len(enhanced_entities)} 实体, {len(enhanced_relations)} 关系")
            return request_data

        except Exception as e:
            self.logger.error(f"数据格式转换失败: {str(e)}")
            raise

    def _enhance_entities(self, entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """增强实体数据"""
        enhanced = []

        for entity in entities:
            try:
                enhanced_entity = {
                    **entity,
                    "enhanced_metadata": {
                        "importance_score": self._calculate_entity_importance(entity),
                        "completeness_score": self._calculate_entity_completeness(entity),
                        "keywords": self._extract_entity_keywords(entity),
                        "domain": self._infer_entity_domain(entity)
                    }
                }
                enhanced.append(enhanced_entity)
            except Exception as e:
                self.logger.warning(f"增强实体数据失败 {entity.get('id', 'unknown')}: {str(e)}")
                enhanced.append(entity)  # 使用原始数据

        return enhanced

    def _enhance_relations(self, relations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """增强关系数据"""
        enhanced = []

        for relation in relations:
            try:
                enhanced_relation = {
                    **relation,
                    "enhanced_metadata": {
                        "strength": self._calculate_relation_strength(relation),
                        "temporal_order": self._infer_temporal_order(relation),
                        "relation_category": self._categorize_relation(relation),
                        "bidirectional": self._check_bidirectional(relation)
                    }
                }
                enhanced.append(enhanced_relation)
            except Exception as e:
                self.logger.warning(f"增强关系数据失败: {str(e)}")
                enhanced.append(relation)  # 使用原始数据

        return enhanced

    def _calculate_entity_importance(self, entity: Dict[str, Any]) -> float:
        """计算实体重要性评分"""
        score = 0.0

        # 基于实体类型
        entity_type = entity.get("type", "").lower()
        if entity_type == "algorithm":
            score += 0.4
        elif entity_type == "dataset":
            score += 0.3
        elif entity_type == "metric":
            score += 0.2

        # 基于年份（越新越重要）
        year = entity.get("year")
        if year and isinstance(year, int):
            if year >= 2020:
                score += 0.3
            elif year >= 2015:
                score += 0.2
            elif year >= 2010:
                score += 0.1

        # 基于作者数量
        authors = entity.get("authors", [])
        if authors:
            score += min(0.2, len(authors) * 0.05)

        # 基于描述完整性
        if entity.get("description"):
            score += 0.1

        return min(1.0, score)

    def _calculate_entity_completeness(self, entity: Dict[str, Any]) -> float:
        """计算实体数据完整性评分"""
        required_fields = ["id", "name", "type"]
        optional_fields = ["title", "year", "authors", "description"]

        score = 0.0

        # 必需字段
        for field in required_fields:
            if entity.get(field):
                score += 0.3

        # 可选字段
        for field in optional_fields:
            if entity.get(field):
                score += 0.1

        return min(1.0, score)

    def _extract_entity_keywords(self, entity: Dict[str, Any]) -> List[str]:
        """提取实体关键词"""
        keywords = []

        # 从名称提取
        name = entity.get("name", "")
        if name:
            keywords.extend(name.split())

        # 从标题提取
        title = entity.get("title", "")
        if title:
            # 简单的关键词提取（实际项目中可以使用更复杂的NLP技术）
            import re
            words = re.findall(r'\b[A-Za-z]{3,}\b', title)
            keywords.extend(words)

        # 去重并转换为小写
        keywords = list(set([kw.lower() for kw in keywords if len(kw) >= 3]))

        return keywords[:10]  # 限制关键词数量

    def _infer_entity_domain(self, entity: Dict[str, Any]) -> str:
        """推断实体所属领域"""
        # 基于关键词推断领域
        keywords = self._extract_entity_keywords(entity)

        domain_keywords = {
            "machine_learning": ["learning", "neural", "network", "deep", "model", "training"],
            "computer_vision": ["vision", "image", "visual", "detection", "recognition", "cnn"],
            "natural_language": ["language", "text", "nlp", "word", "sentence", "bert"],
            "data_mining": ["mining", "data", "clustering", "classification", "pattern"],
            "optimization": ["optimization", "algorithm", "search", "genetic", "evolution"]
        }

        domain_scores = {}
        for domain, domain_kws in domain_keywords.items():
            score = sum(1 for kw in keywords if any(dkw in kw for dkw in domain_kws))
            if score > 0:
                domain_scores[domain] = score

        if domain_scores:
            return max(domain_scores, key=domain_scores.get)
        else:
            return "general"

    def _calculate_relation_strength(self, relation: Dict[str, Any]) -> float:
        """计算关系强度"""
        strength = relation.get("confidence", 0.0)

        # 基于关系类型调整强度
        relation_type = relation.get("type", "").lower()
        if "improve" in relation_type or "extend" in relation_type:
            strength += 0.2
        elif "compare" in relation_type or "similar" in relation_type:
            strength += 0.1

        # 基于证据长度
        evidence = relation.get("evidence", "")
        if evidence:
            strength += min(0.2, len(evidence) / 500)

        return min(1.0, strength)

    def _infer_temporal_order(self, relation: Dict[str, Any]) -> Optional[str]:
        """推断关系的时间顺序"""
        relation_type = relation.get("type", "").lower()

        if any(word in relation_type for word in ["improve", "extend", "build", "based"]):
            return "forward"  # 源实体在前，目标实体在后
        elif any(word in relation_type for word in ["inspire", "derive", "origin"]):
            return "backward"  # 目标实体在前，源实体在后
        else:
            return "simultaneous"  # 同时期或无明确时间顺序

    def _categorize_relation(self, relation: Dict[str, Any]) -> str:
        """关系分类"""
        relation_type = relation.get("type", "").lower()

        if any(word in relation_type for word in ["improve", "enhance", "optimize"]):
            return "improvement"
        elif any(word in relation_type for word in ["extend", "expand", "generalize"]):
            return "extension"
        elif any(word in relation_type for word in ["compare", "versus", "against"]):
            return "comparison"
        elif any(word in relation_type for word in ["combine", "merge", "integrate"]):
            return "combination"
        else:
            return "other"

    def _check_bidirectional(self, relation: Dict[str, Any]) -> bool:
        """检查关系是否为双向"""
        relation_type = relation.get("type", "").lower()

        # 某些关系类型通常是双向的
        bidirectional_types = ["similar", "compare", "related", "equivalent"]
        return any(bt in relation_type for bt in bidirectional_types)

    def _prepare_generation_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """准备生成参数"""
        default_params = {
            "section_num": 7,
            "subsection_len": 700,
            "rag_num": 60,
            "outline_reference_num": 1500,
            "model": "gpt-4o-2024-05-13"
        }

        # 合并用户参数
        generation_params = {**default_params, **params}

        # 参数验证和调整
        generation_params["section_num"] = max(3, min(15, generation_params.get("section_num", 7)))
        generation_params["subsection_len"] = max(300, min(1500, generation_params.get("subsection_len", 700)))
        generation_params["rag_num"] = max(20, min(200, generation_params.get("rag_num", 60)))

        return generation_params

    def _build_context_information(self, entities: List[Dict[str, Any]], relations: List[Dict[str, Any]]) -> Dict[str, Any]:
        """构建上下文信息"""
        # 统计信息
        entity_types = {}
        relation_types = {}
        domains = {}

        for entity in entities:
            entity_type = entity.get("type", "unknown")
            entity_types[entity_type] = entity_types.get(entity_type, 0) + 1

            domain = entity.get("enhanced_metadata", {}).get("domain", "unknown")
            domains[domain] = domains.get(domain, 0) + 1

        for relation in relations:
            relation_type = relation.get("type", "unknown")
            relation_types[relation_type] = relation_types.get(relation_type, 0) + 1

        # 时间范围
        years = [e.get("year") for e in entities if e.get("year")]
        time_range = {
            "start_year": min(years) if years else None,
            "end_year": max(years) if years else None,
            "span_years": max(years) - min(years) if years else 0
        }

        return {
            "entity_distribution": entity_types,
            "relation_distribution": relation_types,
            "domain_distribution": domains,
            "temporal_context": time_range,
            "data_density": len(relations) / max(1, len(entities)),
            "primary_domain": max(domains, key=domains.get) if domains else "unknown"
        }

    def _assess_data_quality(self, entities: List[Dict[str, Any]], relations: List[Dict[str, Any]]) -> Dict[str, float]:
        """评估数据质量"""
        if not entities:
            return {"overall": 0.0, "entity_quality": 0.0, "relation_quality": 0.0, "completeness": 0.0}

        # 实体质量
        entity_completeness_scores = [
            e.get("enhanced_metadata", {}).get("completeness_score", 0.0) for e in entities
        ]
        entity_quality = sum(entity_completeness_scores) / len(entity_completeness_scores)

        # 关系质量
        if relations:
            relation_confidence_scores = [r.get("confidence", 0.0) for r in relations]
            relation_quality = sum(relation_confidence_scores) / len(relation_confidence_scores)
        else:
            relation_quality = 0.0

        # 完整性
        completeness = min(1.0, len(relations) / max(1, len(entities)))

        # 总体质量
        overall = (entity_quality * 0.4 + relation_quality * 0.4 + completeness * 0.2)

        return {
            "overall": overall,
            "entity_quality": entity_quality,
            "relation_quality": relation_quality,
            "completeness": completeness
        }

    def _validate_converted_data(self, data: Dict[str, Any]):
        """验证转换后的数据"""
        required_fields = ["topic", "entities", "relations", "metadata", "generation_params"]

        for field in required_fields:
            if field not in data:
                raise ValueError(f"转换后的数据缺少必需字段: {field}")

        if not data["entities"]:
            raise ValueError("转换后的数据没有实体")

        if not data["topic"].strip():
            raise ValueError("综述主题不能为空")

        self.logger.info("数据格式转换验证通过")

class AsyncTaskManager:
    """异步任务管理器"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.running_tasks = {}
        self.task_results = {}
        self.max_concurrent_tasks = 5

    async def submit_task(self, task_id: str, task_func, *args, **kwargs):
        """提交异步任务"""
        if len(self.running_tasks) >= self.max_concurrent_tasks:
            raise Exception("并发任务数量已达上限")

        # 创建任务
        task = asyncio.create_task(self._run_task_with_monitoring(task_id, task_func, *args, **kwargs))
        self.running_tasks[task_id] = task

        self.logger.info(f"异步任务已提交: {task_id}")
        return task_id

    async def _run_task_with_monitoring(self, task_id: str, task_func, *args, **kwargs):
        """运行任务并监控状态"""
        try:
            self.logger.info(f"开始执行任务: {task_id}")
            result = await task_func(*args, **kwargs)

            # 保存结果
            self.task_results[task_id] = {
                "status": "completed",
                "result": result,
                "completed_at": datetime.now().isoformat()
            }

            self.logger.info(f"任务完成: {task_id}")

        except Exception as e:
            self.logger.error(f"任务失败 {task_id}: {str(e)}")
            self.task_results[task_id] = {
                "status": "failed",
                "error": str(e),
                "failed_at": datetime.now().isoformat()
            }
        finally:
            # 清理运行中的任务
            if task_id in self.running_tasks:
                del self.running_tasks[task_id]

    def get_task_status(self, task_id: str) -> Dict[str, Any]:
        """获取任务状态"""
        if task_id in self.running_tasks:
            return {"status": "running", "task_id": task_id}
        elif task_id in self.task_results:
            return self.task_results[task_id]
        else:
            return {"status": "not_found", "task_id": task_id}

    def cancel_task(self, task_id: str) -> bool:
        """取消任务"""
        if task_id in self.running_tasks:
            task = self.running_tasks[task_id]
            task.cancel()
            del self.running_tasks[task_id]

            self.task_results[task_id] = {
                "status": "cancelled",
                "cancelled_at": datetime.now().isoformat()
            }

            self.logger.info(f"任务已取消: {task_id}")
            return True
        return False

    def cleanup_old_results(self, max_age_hours: int = 24):
        """清理旧的任务结果"""
        cutoff_time = datetime.now().timestamp() - (max_age_hours * 3600)

        to_remove = []
        for task_id, result in self.task_results.items():
            completed_at = result.get("completed_at") or result.get("failed_at") or result.get("cancelled_at")
            if completed_at:
                try:
                    result_time = datetime.fromisoformat(completed_at).timestamp()
                    if result_time < cutoff_time:
                        to_remove.append(task_id)
                except:
                    to_remove.append(task_id)  # 无效时间格式，也删除

        for task_id in to_remove:
            del self.task_results[task_id]

        if to_remove:
            self.logger.info(f"清理了 {len(to_remove)} 个旧任务结果")

class ProgressTracker:
    """进度跟踪器"""

    def __init__(self, task_id: str, total_steps: int = 100):
        self.task_id = task_id
        self.total_steps = total_steps
        self.current_step = 0
        self.current_stage = "初始化"
        self.start_time = datetime.now()
        self.logs = []
        self.logger = logging.getLogger(__name__)

    def update(self, step: int, stage: str, message: str = ""):
        """更新进度"""
        self.current_step = min(step, self.total_steps)
        self.current_stage = stage

        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "step": self.current_step,
            "stage": stage,
            "message": message,
            "percentage": (self.current_step / self.total_steps) * 100
        }

        self.logs.append(log_entry)

        # 保留最近50条日志
        if len(self.logs) > 50:
            self.logs = self.logs[-50:]

        self.logger.info(f"任务 {self.task_id} 进度更新: {stage} ({self.current_step}/{self.total_steps})")

    def get_progress(self) -> Dict[str, Any]:
        """获取当前进度"""
        elapsed_time = (datetime.now() - self.start_time).total_seconds()

        return {
            "task_id": self.task_id,
            "current_step": self.current_step,
            "total_steps": self.total_steps,
            "percentage": (self.current_step / self.total_steps) * 100,
            "current_stage": self.current_stage,
            "elapsed_time": elapsed_time,
            "logs": self.logs[-10:],  # 返回最近10条日志
            "start_time": self.start_time.isoformat()
        }

    def add_log(self, message: str, level: str = "info"):
        """添加日志"""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "level": level,
            "message": message
        }

        self.logs.append(log_entry)

        if len(self.logs) > 50:
            self.logs = self.logs[-50:]

class RetryManager:
    """重试管理器"""

    def __init__(self, max_retries: int = 3, backoff_factor: float = 2.0):
        self.max_retries = max_retries
        self.backoff_factor = backoff_factor
        self.logger = logging.getLogger(__name__)

    async def retry_async(self, func, *args, **kwargs):
        """异步重试执行函数"""
        last_exception = None

        for attempt in range(self.max_retries + 1):
            try:
                if attempt > 0:
                    delay = self.backoff_factor ** (attempt - 1)
                    self.logger.info(f"重试第 {attempt} 次，等待 {delay:.1f} 秒...")
                    await asyncio.sleep(delay)

                result = await func(*args, **kwargs)

                if attempt > 0:
                    self.logger.info(f"重试成功，尝试次数: {attempt + 1}")

                return result

            except Exception as e:
                last_exception = e
                self.logger.warning(f"尝试 {attempt + 1} 失败: {str(e)}")

                if attempt == self.max_retries:
                    self.logger.error(f"所有重试都失败了，最后错误: {str(e)}")
                    break

        raise last_exception

class ResourceMonitor:
    """资源监控器"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.start_time = datetime.now()
        self.peak_memory = 0

    def check_memory_usage(self) -> Dict[str, Any]:
        """检查内存使用情况"""
        try:
            import psutil
            process = psutil.Process()
            memory_info = process.memory_info()

            current_memory = memory_info.rss / 1024 / 1024  # MB
            self.peak_memory = max(self.peak_memory, current_memory)

            return {
                "current_memory_mb": current_memory,
                "peak_memory_mb": self.peak_memory,
                "memory_percent": process.memory_percent()
            }
        except ImportError:
            return {"error": "psutil not available"}
        except Exception as e:
            return {"error": str(e)}

    def check_disk_space(self, path: str = ".") -> Dict[str, Any]:
        """检查磁盘空间"""
        try:
            import shutil
            total, used, free = shutil.disk_usage(path)

            return {
                "total_gb": total / (1024**3),
                "used_gb": used / (1024**3),
                "free_gb": free / (1024**3),
                "usage_percent": (used / total) * 100
            }
        except Exception as e:
            return {"error": str(e)}

    def get_runtime_stats(self) -> Dict[str, Any]:
        """获取运行时统计"""
        runtime = (datetime.now() - self.start_time).total_seconds()

        return {
            "runtime_seconds": runtime,
            "runtime_formatted": self._format_duration(runtime),
            "start_time": self.start_time.isoformat()
        }

    def _format_duration(self, seconds: float) -> str:
        """格式化持续时间"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)

        if hours > 0:
            return f"{hours}h {minutes}m {secs}s"
        elif minutes > 0:
            return f"{minutes}m {secs}s"
        else:
            return f"{secs}s"

# 全局实例
task_manager = AsyncTaskManager()
progress_trackers = {}

def get_progress_tracker(task_id: str, total_steps: int = 100) -> ProgressTracker:
    """获取或创建进度跟踪器"""
    if task_id not in progress_trackers:
        progress_trackers[task_id] = ProgressTracker(task_id, total_steps)
    return progress_trackers[task_id]

def cleanup_progress_tracker(task_id: str):
    """清理进度跟踪器"""
    if task_id in progress_trackers:
        del progress_trackers[task_id]

    def _dfs_longest_path(self, node: str, adj_list: Dict[str, List[str]],
                         nodes: Dict[str, Any], visited: List[str]) -> List[str]:
        """DFS寻找最长路径"""
        if node in visited:
            return visited

        visited = visited + [node]
        longest_path = visited

        for neighbor in adj_list[node]:
            path = self._dfs_longest_path(neighbor, adj_list, nodes, visited)
            if len(path) > len(longest_path):
                longest_path = path

        return longest_path

    def _calculate_influence_metrics(self, graph: Dict[str, Any]) -> Dict[str, float]:
        """计算影响力指标"""
        nodes = graph["nodes"]
        edges = graph["edges"]

        return {
            "network_density": len(edges) / max(1, len(nodes) * (len(nodes) - 1)),
            "average_degree": sum(len(adj) for adj in self._build_adjacency_list(edges).values()) / max(1, len(nodes)),
            "clustering_coefficient": self._calculate_clustering_coefficient(graph),
            "centralization": self._calculate_centralization(graph)
        }

    def _build_adjacency_list(self, edges: List[Dict[str, Any]]) -> Dict[str, List[str]]:
        """构建邻接表"""
        adj_list = {}
        for edge in edges:
            source = edge["source"]
            target = edge["target"]
            if source not in adj_list:
                adj_list[source] = []
            if target not in adj_list:
                adj_list[target] = []
            adj_list[source].append(target)
        return adj_list

    def _calculate_clustering_coefficient(self, graph: Dict[str, Any]) -> float:
        """计算聚类系数"""
        # 简化实现
        return 0.5  # 占位符

    def _calculate_centralization(self, graph: Dict[str, Any]) -> float:
        """计算中心化程度"""
        # 简化实现
        return 0.3  # 占位符

    def _generate_analysis_summary(self, key_nodes: List[Dict[str, Any]],
                                 development_paths: List[Dict[str, Any]]) -> str:
        """生成分析摘要"""
        summary = f"算法脉络分析发现了 {len(key_nodes)} 个关键算法节点"

        if key_nodes:
            most_important = key_nodes[0]
            summary += f"，其中最重要的是 {most_important['name']} (重要性评分: {most_important['importance_score']:.2f})"

        if development_paths:
            longest_path = development_paths[0]
            summary += f"。发现了 {len(development_paths)} 条主要发展路径，最长路径包含 {longest_path['length']} 个算法节点"
            if longest_path['start_year'] and longest_path['end_year']:
                summary += f"，时间跨度从 {longest_path['start_year']} 年到 {longest_path['end_year']} 年"

        return summary + "。"
