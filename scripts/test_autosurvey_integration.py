#!/usr/bin/env python3
"""
AutoSurvey集成功能端到端测试脚本
"""

import sys
import os
import asyncio
import tempfile
import json
from datetime import datetime

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.modules.autosurvey_integration import (
    TaskData, EntityData, RelationData,
    DataFormatConverter, AlgorithmLineageAnalyzer,
    AutoSurveyConnector, AutoSurveyConfig, SurveyGenerationRequest
)
from app.modules.survey_generation_engine import SurveyContentGenerator
from app.modules.survey_storage_manager import SurveyStorageManager
from app.modules.lineage_description_generator import AlgorithmLineageDescriptionGenerator
from app.modules.survey_formatter import SurveyFormatter, FormattingOptions

class MockDatabaseManager:
    """模拟数据库管理器"""
    
    def __init__(self):
        self.db_utils = MockDBUtils()

class MockDBUtils:
    """模拟数据库工具"""
    
    def fetch_all(self, sql, params=None):
        return []
    
    def fetch_one(self, sql, params=None):
        return None

def create_test_data():
    """创建测试数据"""
    print("📊 创建测试数据...")
    
    # 创建测试实体
    entities = [
        EntityData(
            entity_id="bert_2018",
            entity_type="Algorithm",
            name="BERT",
            title="BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding",
            year=2018,
            authors=["Jacob Devlin", "Ming-Wei Chang", "Kenton Lee", "Kristina Toutanova"],
            description="双向编码器表示的Transformer模型",
            source="Google AI"
        ),
        EntityData(
            entity_id="gpt_2018",
            entity_type="Algorithm", 
            name="GPT",
            title="Improving Language Understanding by Generative Pre-Training",
            year=2018,
            authors=["Alec Radford", "Karthik Narasimhan"],
            description="生成式预训练Transformer模型",
            source="OpenAI"
        ),
        EntityData(
            entity_id="transformer_2017",
            entity_type="Algorithm",
            name="Transformer",
            title="Attention Is All You Need",
            year=2017,
            authors=["Ashish Vaswani", "Noam Shazeer", "Niki Parmar"],
            description="基于注意力机制的序列到序列模型",
            source="Google Brain"
        ),
        EntityData(
            entity_id="glue_2018",
            entity_type="Dataset",
            name="GLUE",
            description="通用语言理解评估基准",
            year=2018,
            authors=["Alex Wang", "Amanpreet Singh"],
            source="NYU"
        )
    ]
    
    # 创建测试关系
    relations = [
        RelationData(
            from_entity="transformer_2017",
            to_entity="bert_2018",
            relation_type="基础架构",
            from_entity_type="Algorithm",
            to_entity_type="Algorithm",
            detail="BERT基于Transformer架构",
            evidence="BERT论文明确提到使用Transformer编码器",
            confidence=0.95
        ),
        RelationData(
            from_entity="transformer_2017",
            to_entity="gpt_2018",
            relation_type="基础架构",
            from_entity_type="Algorithm",
            to_entity_type="Algorithm",
            detail="GPT基于Transformer解码器",
            evidence="GPT论文使用Transformer解码器架构",
            confidence=0.95
        ),
        RelationData(
            from_entity="bert_2018",
            to_entity="gpt_2018",
            relation_type="对比研究",
            from_entity_type="Algorithm",
            to_entity_type="Algorithm",
            detail="BERT和GPT代表不同的预训练方向",
            evidence="两者在预训练目标和架构上有显著差异",
            confidence=0.8
        )
    ]
    
    # 创建任务数据
    task_data = TaskData(
        task_id="nlp_pretrain_test",
        task_name="自然语言处理预训练模型测试",
        entities=entities,
        relations=relations,
        created_at=datetime.now()
    )
    
    print(f"✅ 创建了 {len(entities)} 个实体和 {len(relations)} 个关系")
    return task_data

def test_data_format_conversion(task_data):
    """测试数据格式转换"""
    print("\n🔄 测试数据格式转换...")
    
    converter = DataFormatConverter()
    topic = "自然语言处理中的预训练模型发展"
    params = {
        "model": "gpt-4o-2024-05-13",
        "section_num": 7,
        "subsection_len": 700
    }
    
    autosurvey_input = converter.convert_to_autosurvey_format(task_data, topic, params)
    
    print(f"✅ 转换完成，包含 {len(autosurvey_input['entities'])} 个实体")
    print(f"✅ 数据质量评分: {autosurvey_input['metadata']['data_quality']['overall']:.2f}")
    
    return autosurvey_input

def test_algorithm_lineage_analysis(task_data):
    """测试算法脉络分析"""
    print("\n🌳 测试算法脉络分析...")
    
    mock_db = MockDatabaseManager()
    analyzer = AlgorithmLineageAnalyzer(mock_db)
    
    lineage_analysis = analyzer.analyze_algorithm_lineage(task_data)
    
    print(f"✅ 识别出 {len(lineage_analysis['key_nodes'])} 个关键节点")
    
    # 显示关键节点
    for i, node in enumerate(lineage_analysis['key_nodes'][:3], 1):
        print(f"   {i}. {node['name']} ({node.get('year', '未知年份')}) - 重要性: {node.get('importance_score', 0):.2f}")
    
    # 显示发展路径
    dev_paths = lineage_analysis.get('development_paths', {})
    if isinstance(dev_paths, dict) and dev_paths.get('main_paths'):
        print(f"✅ 发现 {len(dev_paths['main_paths'])} 条主要发展路径")
    
    return lineage_analysis

async def test_autosurvey_connector(autosurvey_input, task_data):
    """测试AutoSurvey连接器"""
    print("\n🤖 测试AutoSurvey连接器...")
    
    config = AutoSurveyConfig()
    
    request = SurveyGenerationRequest(
        topic="自然语言处理中的预训练模型发展",
        task_data=task_data,
        parameters={}
    )
    
    async with AutoSurveyConnector(config) as connector:
        survey_result = await connector.generate_survey(request)
        
        print(f"✅ 生成综述成功，ID: {survey_result.survey_id}")
        print(f"✅ 内容长度: {len(survey_result.content)} 字符")
        print(f"✅ 参考文献: {len(survey_result.references)} 篇")
        
        return survey_result

def test_survey_content_generation(autosurvey_result, lineage_analysis):
    """测试综述内容生成"""
    print("\n📝 测试综述内容生成...")
    
    generator = SurveyContentGenerator()
    
    # 模拟AutoSurvey结果
    mock_autosurvey_result = {
        "content": autosurvey_result.content,
        "references": autosurvey_result.references,
        "outline": autosurvey_result.outline
    }
    
    enhanced_content = generator.generate_survey_content(
        topic="自然语言处理中的预训练模型发展",
        autosurvey_result=mock_autosurvey_result,
        algorithm_lineage=lineage_analysis,
        template_name="academic_standard"
    )
    
    print(f"✅ 增强内容生成完成")
    print(f"✅ 章节数: {len(enhanced_content['sections'])}")
    print(f"✅ 字数: {enhanced_content['metadata']['word_count']}")
    
    return enhanced_content

def test_lineage_description_generation(lineage_analysis):
    """测试脉络描述生成"""
    print("\n📖 测试脉络描述生成...")
    
    generator = AlgorithmLineageDescriptionGenerator()
    
    description = generator.generate_lineage_description(
        lineage_analysis, 
        "自然语言处理中的预训练模型发展"
    )
    
    print(f"✅ 脉络描述生成完成")
    print(f"✅ 关键洞察: {len(description.key_insights)} 个")
    
    # 显示部分洞察
    for i, insight in enumerate(description.key_insights[:2], 1):
        print(f"   {i}. {insight}")
    
    return description

def test_survey_formatting(content):
    """测试综述格式化"""
    print("\n🎨 测试综述格式化...")
    
    formatter = SurveyFormatter()
    
    # 测试多种格式
    formats = ["markdown", "html"]
    results = {}
    
    for format_type in formats:
        options = FormattingOptions(
            format_type=format_type,
            style="academic",
            include_toc=True,
            include_references=True
        )
        
        result = formatter.format_survey(content, {}, options)
        results[format_type] = result
        
        print(f"✅ {format_type.upper()} 格式化完成")
    
    return results

def test_survey_storage(content, metadata):
    """测试综述存储"""
    print("\n💾 测试综述存储...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        storage_manager = SurveyStorageManager(temp_dir)
        
        # 存储综述
        record = storage_manager.store_survey(
            survey_id="test_survey_001",
            topic="自然语言处理中的预训练模型发展",
            content=content,
            task_ids=["nlp_pretrain_test"],
            metadata=metadata,
            formats=["markdown", "html"]
        )
        
        print(f"✅ 综述存储完成，ID: {record.survey_id}")
        print(f"✅ 可用格式: {list(record.file_paths.keys())}")
        
        # 测试检索
        retrieved = storage_manager.get_survey(record.survey_id)
        assert retrieved is not None
        print(f"✅ 综述检索成功")
        
        # 测试列表
        surveys = storage_manager.list_surveys()
        print(f"✅ 综述列表: {len(surveys)} 个")
        
        # 测试搜索
        search_results = storage_manager.search_surveys("预训练")
        print(f"✅ 搜索结果: {len(search_results)} 个")
        
        return record

async def run_integration_test():
    """运行集成测试"""
    print("🚀 开始AutoSurvey集成功能端到端测试\n")
    
    try:
        # 1. 创建测试数据
        task_data = create_test_data()
        
        # 2. 测试数据格式转换
        autosurvey_input = test_data_format_conversion(task_data)
        
        # 3. 测试算法脉络分析
        lineage_analysis = test_algorithm_lineage_analysis(task_data)
        
        # 4. 测试AutoSurvey连接器
        autosurvey_result = await test_autosurvey_connector(autosurvey_input, task_data)
        
        # 5. 测试综述内容生成
        enhanced_content = test_survey_content_generation(autosurvey_result, lineage_analysis)
        
        # 6. 测试脉络描述生成
        lineage_description = test_lineage_description_generation(lineage_analysis)
        
        # 7. 测试综述格式化
        formatted_results = test_survey_formatting(enhanced_content["content"])
        
        # 8. 测试综述存储
        storage_record = test_survey_storage(
            enhanced_content["content"], 
            enhanced_content["metadata"]
        )
        
        print("\n🎉 所有测试通过！AutoSurvey集成功能正常工作")
        
        # 输出测试摘要
        print("\n📋 测试摘要:")
        print(f"   • 处理实体: {len(task_data.entities)} 个")
        print(f"   • 处理关系: {len(task_data.relations)} 个")
        print(f"   • 关键节点: {len(lineage_analysis['key_nodes'])} 个")
        print(f"   • 综述字数: {enhanced_content['metadata']['word_count']} 字")
        print(f"   • 支持格式: {len(formatted_results)} 种")
        print(f"   • 存储路径: {storage_record.file_paths}")
        
        return True
        
    except Exception as e:
        print(f"\n❌ 测试失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = asyncio.run(run_integration_test())
    sys.exit(0 if success else 1)
