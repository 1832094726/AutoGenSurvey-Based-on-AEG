from flask import Blueprint, request, jsonify
import logging
from app.modules.db_manager import db_manager
from app.modules.knowledge_graph import build_knowledge_graph, visualize_graph, export_graph_to_json
from app.config import Config
import os

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# 创建蓝图
relation_api = Blueprint('relations', __name__)

@relation_api.route('', methods=['GET'])
def get_relations():
    """
    获取所有演化关系的信息。
    """
    try:
        relations = db_manager.get_all_relations()
        return jsonify(relations)
    except Exception as e:
        logging.error(f"获取演化关系时出错: {str(e)}")
        return jsonify({'error': str(e)}), 500

@relation_api.route('/<relation_id>', methods=['GET'])
def get_relation(relation_id):
    """
    获取指定演化关系的信息。
    """
    try:
        relations = db_manager.get_all_relations()
        
        # 查找指定ID的关系
        for relation in relations:
            # SQLite模式下，relation_id是一个整数
            if db_manager.db_type == 'sqlite' and 'relation_id' in relation and str(relation['relation_id']) == relation_id:
                return jsonify(relation)
            # Neo4j模式下，使用复合键
            elif db_manager.db_type == 'neo4j' and relation.get('from_entity') == relation_id.split(':')[0] and relation.get('to_entity') == relation_id.split(':')[1]:
                return jsonify(relation)
        
        return jsonify({'error': f'关系 {relation_id} 不存在'}), 404
    except Exception as e:
        logging.error(f"获取关系 {relation_id} 时出错: {str(e)}")
        return jsonify({'error': str(e)}), 500

@relation_api.route('', methods=['POST'])
def add_relation():
    """
    添加新的演化关系。
    """
    try:
        relation_data = request.json
        
        if not relation_data:
            return jsonify({'error': '请求中没有数据'}), 400
        
        # 保存到数据库
        success = db_manager.add_relation(relation_data)
        
        if success:
            # 更新图
            update_graph()
            
            return jsonify({'success': True, 'message': '关系添加成功'})
        else:
            return jsonify({'error': '添加关系失败'}), 500
    except Exception as e:
        logging.error(f"添加关系时出错: {str(e)}")
        return jsonify({'error': str(e)}), 500

@relation_api.route('/<relation_id>', methods=['PUT'])
def modify_relation(relation_id):
    """
    修改现有的演化关系。
    """
    try:
        updated_data = request.json
        
        if not updated_data:
            return jsonify({'error': '请求中没有数据'}), 400
        
        # 处理不同数据库类型下的关系ID
        if db_manager.db_type == 'sqlite':
            # SQLite模式下，relation_id是一个整数
            db_relation_id = int(relation_id)
        else:  # neo4j
            # Neo4j模式下，使用复合键
            parts = relation_id.split(':')
            if len(parts) != 2:
                return jsonify({'error': '无效的关系ID格式'}), 400
                
            db_relation_id = {
                'from_entity': parts[0],
                'to_entity': parts[1]
            }
        
        # 更新数据库
        success = db_manager.modify_relation(db_relation_id, updated_data)
        
        if success:
            # 更新图
            update_graph()
            
            return jsonify({'success': True, 'message': f'关系 {relation_id} 更新成功'})
        else:
            return jsonify({'error': f'更新关系 {relation_id} 失败'}), 404
    except Exception as e:
        logging.error(f"修改关系 {relation_id} 时出错: {str(e)}")
        return jsonify({'error': str(e)}), 500

@relation_api.route('/<relation_id>', methods=['DELETE'])
def delete_relation(relation_id):
    """
    删除指定的演化关系。
    """
    try:
        # 处理不同数据库类型下的关系ID
        if db_manager.db_type == 'sqlite':
            # SQLite模式下，relation_id是一个整数
            db_relation_id = int(relation_id)
        else:  # neo4j
            # Neo4j模式下，使用复合键
            parts = relation_id.split(':')
            if len(parts) != 2:
                return jsonify({'error': '无效的关系ID格式'}), 400
                
            db_relation_id = {
                'from_entity': parts[0],
                'to_entity': parts[1]
            }
        
        # 删除关系
        success = db_manager.delete_relation(db_relation_id)
        
        if success:
            # 更新图
            update_graph()
            
            return jsonify({'success': True, 'message': f'关系 {relation_id} 已删除'})
        else:
            return jsonify({'error': f'删除关系 {relation_id} 失败'}), 404
    except Exception as e:
        logging.error(f"删除关系 {relation_id} 时出错: {str(e)}")
        return jsonify({'error': str(e)}), 500

def update_graph():
    """
    更新知识图谱。
    """
    try:
        # 获取最新数据
        entities = db_manager.get_all_entities()
        relations = db_manager.get_all_relations()
        
        # 构建图并可视化
        graph = build_knowledge_graph(entities, relations)
        graph_image_path = os.path.join(Config.GRAPH_DATA_DIR, 'graph.png')
        visualize_graph(graph, output_path=graph_image_path)
        
        # 导出图数据供前端使用
        graph_json_path = os.path.join(Config.GRAPH_DATA_DIR, 'graph.json')
        export_graph_to_json(graph, graph_json_path)
        
        return True
    except Exception as e:
        logging.error(f"更新图时出错: {str(e)}")
        return False 