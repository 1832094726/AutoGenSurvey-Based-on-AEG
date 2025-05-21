import unittest
import logging
import os
import json
import sys
from app.modules.db_manager import db_manager
from app.modules.data_processing import normalize_entities
from app.modules.knowledge_graph import build_knowledge_graph, export_graph_to_json

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename='test_combined.log'
)

class TestDBManager(unittest.TestCase):
    """数据库管理器测试"""
    
    def setUp(self):
        """测试前准备"""
        self.db_manager = db_manager
        self.test_entity = {
            "entity_id": "test_algo_001",
            "name": "测试算法",
            "entity_type": "Algorithm",
            "tags": ["测试", "算法"],
            "description": "这是一个用于测试的算法实体"
        }
        self.test_relation = {
            "relation_id": "test_rel_001",
            "from_entity": "test_algo_001",
            "to_entity": "test_algo_002",
            "relation_type": "Improvement",
            "description": "算法改进关系"
        }
    
    def test_entity_operations(self):
        """测试实体操作"""
        # 存储实体
        result = self.db_manager.store_entity(self.test_entity)
        self.assertTrue(result)
        
        # 获取实体
        entity = self.db_manager.get_entity_by_id(self.test_entity["entity_id"])
        self.assertIsNotNone(entity)
        self.assertEqual(entity["name"], self.test_entity["name"])
        
        # 更新实体
        updated_entity = self.test_entity.copy()
        updated_entity["name"] = "更新后的算法名称"
        update_result = self.db_manager.update_entity(updated_entity)
        self.assertTrue(update_result)
        
        # 验证更新结果
        entity = self.db_manager.get_entity_by_id(self.test_entity["entity_id"])
        self.assertEqual(entity["name"], "更新后的算法名称")
        
        # 删除实体
        delete_result = self.db_manager.delete_entity(self.test_entity["entity_id"])
        self.assertTrue(delete_result)
        
        # 验证删除结果
        entity = self.db_manager.get_entity_by_id(self.test_entity["entity_id"])
        self.assertIsNone(entity)
    
    def test_relation_operations(self):
        """测试关系操作"""
        # 先存储两个实体
        entity1 = self.test_entity.copy()
        entity2 = self.test_entity.copy()
        entity2["entity_id"] = "test_algo_002"
        entity2["name"] = "测试算法2"
        
        self.db_manager.store_entity(entity1)
        self.db_manager.store_entity(entity2)
        
        # 存储关系
        result = self.db_manager.store_relation(self.test_relation)
        self.assertTrue(result)
        
        # 获取关系
        relation = self.db_manager.get_relation_by_id(self.test_relation["relation_id"])
        self.assertIsNotNone(relation)
        self.assertEqual(relation["relation_type"], self.test_relation["relation_type"])
        
        # 清理测试数据
        self.db_manager.delete_relation(self.test_relation["relation_id"])
        self.db_manager.delete_entity(entity1["entity_id"])
        self.db_manager.delete_entity(entity2["entity_id"])

class TestEntityNormalization(unittest.TestCase):
    """实体规范化测试"""
    
    def setUp(self):
        """测试前准备"""
        self.raw_entities = [
            {
                "algorithm_entity": {
                    "name": "BERT",
                    "description": "Bidirectional Encoder Representations from Transformers",
                    "keywords": ["NLP", "Transformer", "Pre-training"],
                    "year": 2018
                }
            },
            {
                "algorithm_entity": {
                    "name": "ResNet",
                    "description": "Residual Network for image classification",
                    "keywords": ["CNN", "Deep Learning", "Computer Vision"],
                    "year": 2015
                }
            }
        ]
    
    def test_normalize_entities(self):
        """测试实体规范化"""
        normalized = normalize_entities(self.raw_entities)
        
        # 验证规范化结果
        self.assertEqual(len(normalized), 2)
        self.assertTrue("entity_id" in normalized[0])
        self.assertTrue("entity_type" in normalized[0])
        self.assertEqual(normalized[0]["entity_type"], "Algorithm")
        
        # 验证原始属性保留
        self.assertEqual(normalized[0]["name"], "BERT")
        self.assertEqual(normalized[1]["name"], "ResNet")

class TestKnowledgeGraph(unittest.TestCase):
    """知识图谱测试"""
    
    def setUp(self):
        """测试前准备"""
        self.entities = [
            {
                "entity_id": "algo_001",
                "name": "算法A",
                "entity_type": "Algorithm",
                "description": "测试算法A"
            },
            {
                "entity_id": "algo_002",
                "name": "算法B",
                "entity_type": "Algorithm",
                "description": "测试算法B"
            }
        ]
        
        self.relations = [
            {
                "relation_id": "rel_001",
                "from_entity": "algo_001",
                "to_entity": "algo_002",
                "relation_type": "Improvement",
                "description": "算法B改进了算法A"
            }
        ]
    
    def test_build_graph(self):
        """测试图构建"""
        graph = build_knowledge_graph(self.entities, self.relations)
        
        # 验证图结构
        self.assertEqual(len(graph.nodes), 2)
        self.assertEqual(len(graph.edges), 1)
        
        # 验证节点属性
        for node in graph.nodes:
            node_data = graph.nodes[node]
            self.assertTrue("name" in node_data)
            self.assertTrue("entity_type" in node_data)
        
        # 验证边属性
        for edge in graph.edges:
            edge_data = graph.edges[edge]
            self.assertTrue("relation_type" in edge_data)
    
    def test_export_graph(self):
        """测试图导出"""
        graph = build_knowledge_graph(self.entities, self.relations)
        
        # 导出为JSON
        export_path = "test_graph_export.json"
        graph_data = export_graph_to_json(graph, export_path)
        
        # 验证导出结果
        self.assertTrue(os.path.exists(export_path))
        self.assertTrue("nodes" in graph_data)
        self.assertTrue("edges" in graph_data)
        self.assertEqual(len(graph_data["nodes"]), 2)
        self.assertEqual(len(graph_data["edges"]), 1)
        
        # 清理测试文件
        if os.path.exists(export_path):
            os.remove(export_path)

class TestEntityExtraction(unittest.TestCase):
    """实体提取测试"""
    
    def test_entity_format(self):
        """测试实体格式"""
        # 模拟从文件提取的实体
        test_entity = {
            "algorithm_entity": {
                "name": "测试算法",
                "description": "这是一个测试算法",
                "keywords": ["测试", "算法"],
                "year": 2023
            }
        }
        
        # 验证实体格式
        self.assertTrue("algorithm_entity" in test_entity)
        self.assertTrue("name" in test_entity["algorithm_entity"])
        self.assertTrue("description" in test_entity["algorithm_entity"])

if __name__ == '__main__':
    unittest.main() 