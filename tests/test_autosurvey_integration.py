"""
AutoSurvey集成功能测试
"""

import pytest
import asyncio
import tempfile
import os
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime

from app.modules.autosurvey_integration import (
    TaskSelector, EntityRelationExtractor, DataFormatConverter,
    AutoSurveyConnector, AutoSurveyConfig, AlgorithmLineageAnalyzer,
    TaskData, EntityData, RelationData, SurveyGenerationRequest
)
from app.modules.survey_storage_manager import SurveyStorageManager
from app.modules.survey_generation_engine import SurveyContentGenerator
from app.modules.lineage_description_generator import AlgorithmLineageDescriptionGenerator

class TestTaskSelector:
    """任务选择器测试"""
    
    def setup_method(self):
        self.mock_db_manager = Mock()
        self.task_selector = TaskSelector(self.mock_db_manager)
    
    def test_get_available_tasks(self):
        """测试获取可用任务"""
        # 模拟数据库返回
        mock_tasks = [
            {
                "task_id": "task1",
                "task_name": "测试任务1",
                "status": "completed",
                "entity_count": 10,
                "relation_count": 20,
                "created_at": "2024-01-01"
            }
        ]
        
        self.mock_db_manager.db_utils.fetch_all.return_value = mock_tasks
        
        result = self.task_selector.get_available_tasks()
        
        assert len(result) == 1
        assert result[0]["task_id"] == "task1"
        assert result[0]["entity_count"] == 10
    
    def test_validate_task_selection(self):
        """测试任务选择验证"""
        task_ids = ["task1", "task2"]
        
        # 模拟任务存在
        self.mock_db_manager.db_utils.fetch_one.return_value = {"task_id": "task1"}
        
        is_valid, issues = self.task_selector.validate_task_selection(task_ids)
        
        assert is_valid == True
        assert len(issues) == 0

class TestEntityRelationExtractor:
    """实体关系提取器测试"""
    
    def setup_method(self):
        self.mock_db_manager = Mock()
        self.extractor = EntityRelationExtractor(self.mock_db_manager)
    
    def test_extract_task_data(self):
        """测试提取任务数据"""
        task_id = "test_task"
        
        # 模拟数据库返回
        mock_entities = [
            {
                "algorithm_id": "alg1",
                "name": "测试算法",
                "year": "2020",
                "authors": '["作者1", "作者2"]',
                "title": "算法标题",
                "task": "分类",
                "source": "论文"
            }
        ]
        
        mock_relations = [
            {
                "from_entity": "alg1",
                "to_entity": "alg2",
                "relation_type": "improve",
                "confidence": "0.8",
                "detail": "改进关系",
                "evidence": "实验证据"
            }
        ]
        
        # 模拟方法返回
        with patch.object(self.extractor, '_get_algorithms_by_task', return_value=mock_entities), \
             patch.object(self.extractor, '_get_datasets_by_task', return_value=[]), \
             patch.object(self.extractor, '_get_metrics_by_task', return_value=[]), \
             patch.object(self.extractor, '_get_relations_by_task', return_value=mock_relations), \
             patch.object(self.extractor, '_get_task_info', return_value={"task_name": "测试任务"}):
            
            result = self.extractor.extract_task_data(task_id)
            
            assert isinstance(result, TaskData)
            assert result.task_id == task_id
            assert len(result.entities) == 1
            assert len(result.relations) == 1
            assert result.entities[0].name == "测试算法"

class TestDataFormatConverter:
    """数据格式转换器测试"""
    
    def setup_method(self):
        self.converter = DataFormatConverter()
    
    def test_convert_to_autosurvey_format(self):
        """测试转换为AutoSurvey格式"""
        # 创建测试数据
        entities = [
            EntityData(
                entity_id="alg1",
                entity_type="Algorithm",
                name="测试算法",
                year=2020,
                authors=["作者1"],
                description="测试描述"
            )
        ]
        
        relations = [
            RelationData(
                from_entity="alg1",
                to_entity="alg2",
                relation_type="improve",
                confidence=0.8
            )
        ]
        
        task_data = TaskData(
            task_id="test_task",
            task_name="测试任务",
            entities=entities,
            relations=relations,
            created_at=datetime.now()
        )
        
        topic = "测试主题"
        params = {"model": "gpt-4"}
        
        result = self.converter.convert_to_autosurvey_format(task_data, topic, params)
        
        assert "topic" in result
        assert "entities" in result
        assert "relations" in result
        assert result["topic"] == topic
        assert len(result["entities"]) == 1
        assert len(result["relations"]) == 1

class TestAutoSurveyConnector:
    """AutoSurvey连接器测试"""
    
    def setup_method(self):
        self.config = AutoSurveyConfig()
        self.connector = AutoSurveyConnector(self.config)
    
    @pytest.mark.asyncio
    async def test_generate_survey_mock(self):
        """测试生成综述（模拟模式）"""
        # 创建测试请求
        task_data = TaskData(
            task_id="test_task",
            task_name="测试任务",
            entities=[],
            relations=[],
            created_at=datetime.now()
        )
        
        request = SurveyGenerationRequest(
            topic="测试主题",
            task_data=task_data,
            parameters={}
        )
        
        async with self.connector:
            result = await self.connector.generate_survey(request)
            
            assert result.topic == "测试主题"
            assert result.content is not None
            assert len(result.content) > 0

class TestAlgorithmLineageAnalyzer:
    """算法脉络分析器测试"""
    
    def setup_method(self):
        self.mock_db_manager = Mock()
        self.analyzer = AlgorithmLineageAnalyzer(self.mock_db_manager)
    
    def test_analyze_algorithm_lineage(self):
        """测试算法脉络分析"""
        # 创建测试数据
        entities = [
            EntityData(
                entity_id="alg1",
                entity_type="Algorithm",
                name="算法1",
                year=2020,
                authors=["作者1"]
            ),
            EntityData(
                entity_id="alg2",
                entity_type="Algorithm",
                name="算法2",
                year=2021,
                authors=["作者2"]
            )
        ]
        
        relations = [
            RelationData(
                from_entity="alg1",
                to_entity="alg2",
                relation_type="improve",
                confidence=0.8
            )
        ]
        
        task_data = TaskData(
            task_id="test_task",
            task_name="测试任务",
            entities=entities,
            relations=relations,
            created_at=datetime.now()
        )
        
        result = self.analyzer.analyze_algorithm_lineage(task_data)
        
        assert "algorithm_graph" in result
        assert "key_nodes" in result
        assert "development_paths" in result
        assert "analysis_summary" in result

class TestSurveyStorageManager:
    """综述存储管理器测试"""
    
    def setup_method(self):
        # 使用临时目录
        self.temp_dir = tempfile.mkdtemp()
        self.storage_manager = SurveyStorageManager(self.temp_dir)
    
    def teardown_method(self):
        # 清理临时目录
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_store_survey(self):
        """测试存储综述"""
        survey_id = "test_survey"
        topic = "测试主题"
        content = "# 测试综述\n\n这是测试内容。"
        task_ids = ["task1"]
        metadata = {"word_count": 100}
        
        record = self.storage_manager.store_survey(
            survey_id=survey_id,
            topic=topic,
            content=content,
            task_ids=task_ids,
            metadata=metadata,
            formats=["markdown"]
        )
        
        assert record.survey_id == survey_id
        assert record.topic == topic
        assert "markdown" in record.file_paths
    
    def test_get_survey(self):
        """测试获取综述"""
        # 先存储一个综述
        survey_id = "test_survey"
        self.storage_manager.store_survey(
            survey_id=survey_id,
            topic="测试主题",
            content="测试内容",
            task_ids=["task1"],
            metadata={}
        )
        
        # 获取综述
        record = self.storage_manager.get_survey(survey_id)
        
        assert record is not None
        assert record.survey_id == survey_id
    
    def test_list_surveys(self):
        """测试列出综述"""
        # 存储多个综述
        for i in range(3):
            self.storage_manager.store_survey(
                survey_id=f"survey_{i}",
                topic=f"主题{i}",
                content="内容",
                task_ids=[f"task_{i}"],
                metadata={}
            )
        
        # 列出综述
        surveys = self.storage_manager.list_surveys()
        
        assert len(surveys) == 3

class TestSurveyContentGenerator:
    """综述内容生成器测试"""
    
    def setup_method(self):
        self.generator = SurveyContentGenerator()
    
    def test_generate_survey_content(self):
        """测试生成综述内容"""
        topic = "测试主题"
        
        autosurvey_result = {
            "content": "# 测试内容\n\n这是AutoSurvey生成的内容。",
            "references": [
                {"title": "论文1", "authors": ["作者1"], "year": 2020}
            ]
        }
        
        algorithm_lineage = {
            "key_nodes": [
                {"name": "算法1", "year": 2020, "importance_score": 0.8}
            ],
            "development_paths": []
        }
        
        result = self.generator.generate_survey_content(
            topic=topic,
            autosurvey_result=autosurvey_result,
            algorithm_lineage=algorithm_lineage
        )
        
        assert "content" in result
        assert "sections" in result
        assert "metadata" in result
        assert len(result["content"]) > 0

class TestLineageDescriptionGenerator:
    """脉络描述生成器测试"""
    
    def setup_method(self):
        self.generator = AlgorithmLineageDescriptionGenerator()
    
    def test_generate_lineage_description(self):
        """测试生成脉络描述"""
        topic = "测试主题"
        
        algorithm_lineage = {
            "key_nodes": [
                {
                    "name": "算法1",
                    "year": 2020,
                    "importance_score": 0.8,
                    "in_degree": 0,
                    "out_degree": 2
                },
                {
                    "name": "算法2", 
                    "year": 2021,
                    "importance_score": 0.6,
                    "in_degree": 1,
                    "out_degree": 1
                }
            ],
            "algorithm_graph": {
                "nodes": {
                    "alg1": {"name": "算法1", "year": 2020},
                    "alg2": {"name": "算法2", "year": 2021}
                },
                "edges": [
                    {"source": "alg1", "target": "alg2", "type": "improve"}
                ]
            },
            "algorithm_clusters": {
                "temporal_clusters": [
                    {"period": "2020-2024", "size": 2}
                ]
            }
        }
        
        result = self.generator.generate_lineage_description(algorithm_lineage, topic)
        
        assert result.timeline_description is not None
        assert result.technical_relationships is not None
        assert result.influence_analysis is not None
        assert result.development_summary is not None
        assert len(result.key_insights) > 0

class TestIntegrationWorkflow:
    """集成工作流测试"""
    
    def setup_method(self):
        self.temp_dir = tempfile.mkdtemp()
    
    def teardown_method(self):
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    @pytest.mark.asyncio
    async def test_complete_workflow(self):
        """测试完整工作流"""
        # 这是一个集成测试，测试整个流程
        
        # 1. 模拟任务数据
        task_data = TaskData(
            task_id="integration_test",
            task_name="集成测试任务",
            entities=[
                EntityData(
                    entity_id="alg1",
                    entity_type="Algorithm",
                    name="测试算法",
                    year=2020,
                    authors=["测试作者"]
                )
            ],
            relations=[],
            created_at=datetime.now()
        )
        
        # 2. 数据格式转换
        converter = DataFormatConverter()
        autosurvey_input = converter.convert_to_autosurvey_format(
            task_data, "集成测试主题", {}
        )
        
        assert "topic" in autosurvey_input
        assert "entities" in autosurvey_input
        
        # 3. 算法脉络分析
        mock_db_manager = Mock()
        analyzer = AlgorithmLineageAnalyzer(mock_db_manager)
        lineage_analysis = analyzer.analyze_algorithm_lineage(task_data)
        
        assert "key_nodes" in lineage_analysis
        
        # 4. 综述内容生成
        generator = SurveyContentGenerator()
        
        mock_autosurvey_result = {
            "content": "# 集成测试\n\n测试内容",
            "references": []
        }
        
        content_result = generator.generate_survey_content(
            topic="集成测试主题",
            autosurvey_result=mock_autosurvey_result,
            algorithm_lineage=lineage_analysis
        )
        
        assert "content" in content_result
        
        # 5. 结果存储
        storage_manager = SurveyStorageManager(self.temp_dir)
        
        storage_record = storage_manager.store_survey(
            survey_id="integration_test_survey",
            topic="集成测试主题",
            content=content_result["content"],
            task_ids=["integration_test"],
            metadata=content_result["metadata"],
            formats=["markdown"]
        )
        
        assert storage_record.survey_id == "integration_test_survey"
        assert "markdown" in storage_record.file_paths
        
        # 验证文件确实被创建
        markdown_path = storage_record.file_paths["markdown"]
        assert os.path.exists(markdown_path)

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
