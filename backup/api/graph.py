import os
import json
import logging
from flask import Blueprint, jsonify, send_file, current_app, request
from app.modules.db_manager import DatabaseManager
from app.modules.knowledge_graph import build_knowledge_graph, visualize_graph, export_graph_to_json

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 创建Blueprint
graph_api = Blueprint('graph_api', __name__)

# 初始化数据库管理器
db_manager = DatabaseManager()

@graph_api.route('/data', methods=['GET'])
def get_graph_data():
    """
    获取图数据，用于前端渲染
    返回格式：
    {
        "nodes": [
            {"id": "1", "label": "Algorithm A", "type": "Algorithm", ...},
            ...
        ],
        "edges": [
            {"source": "1", "target": "2", "label": "Improve", "relation_type": "Improve", ...},
            ...
        ]
    }
    """
    try:
        # 检查是否强制刷新
        force_refresh = request.args.get('refresh', '0') == '1'
        logger.info(f"[graph_api] 获取图数据，强制刷新: {force_refresh}")
        
        # 从数据库获取实体和关系
        entities = db_manager.get_all_entities()
        relations = db_manager.get_all_relations()
        
        logger.info(f"从数据库获取到 {len(entities)} 个实体和 {len(relations)} 个关系")
        
        # 构建图
        graph = build_knowledge_graph(entities, relations)
        
        # 导出为JSON
        graph_data_dir = current_app.config.get('GRAPH_DATA_DIR', 'data/graph')
        os.makedirs(graph_data_dir, exist_ok=True)
        graph_json_path = os.path.join(graph_data_dir, 'graph_data.json')
        
        # 导出图数据到JSON，确保包含所有属性
        graph_data = export_graph_to_json(graph, graph_json_path)
        
        logger.info(f"生成图数据: {len(graph_data.get('nodes', []))}个节点, {len(graph_data.get('edges', []))}个边")
        
        return jsonify(graph_data)
    
    except Exception as e:
        logger.error(f"获取图数据时出错: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return jsonify({"error": f"获取图数据失败: {str(e)}"}), 500

@graph_api.route('/image', methods=['GET'])
def get_graph_image():
    """
    获取图像文件
    """
    try:
        # 图像文件路径
        image_path = os.path.join(current_app.config['GRAPH_DATA_DIR'], 'graph_image.png')
        
        # 如果图像不存在，生成它
        if not os.path.exists(image_path):
            logger.info("图像不存在，正在生成...")
            
            # 从数据库获取数据
            entities = db_manager.get_all_entities()
            relations = db_manager.get_all_relations()
            
            # 构建图
            graph = build_knowledge_graph(entities, relations)
            
            # 可视化图并保存
            os.makedirs(os.path.dirname(image_path), exist_ok=True)
            visualize_graph(graph, output_path=image_path, show=False)
        
        # 返回图像文件
        return send_file(image_path, mimetype='image/png')
    
    except Exception as e:
        logger.error(f"获取图像时出错: {str(e)}")
        return jsonify({"error": f"获取图像失败: {str(e)}"}), 500

@graph_api.route('/update', methods=['POST'])
def update_graph():
    """
    更新图数据和图像
    通常在数据库更新后调用，以更新缓存的JSON和图像文件
    """
    try:
        # 删除旧的JSON文件和图像
        json_path = os.path.join(current_app.config['GRAPH_DATA_DIR'], 'graph_data.json')
        image_path = os.path.join(current_app.config['GRAPH_DATA_DIR'], 'graph_image.png')
        
        if os.path.exists(json_path):
            os.remove(json_path)
        if os.path.exists(image_path):
            os.remove(image_path)
        
        # 从数据库获取数据
        entities = db_manager.get_all_entities()
        relations = db_manager.get_all_relations()
        
        # 构建图
        graph = build_knowledge_graph(entities, relations)
        
        # 导出为JSON
        os.makedirs(os.path.dirname(json_path), exist_ok=True)
        export_graph_to_json(graph, json_path)
        
        # 可视化图并保存
        visualize_graph(graph, output_path=image_path, show=False)
        
        return jsonify({"success": True, "message": "图数据和图像已更新"})
    
    except Exception as e:
        logger.error(f"更新图数据时出错: {str(e)}")
        return jsonify({"error": f"更新图数据失败: {str(e)}"}), 500

@graph_api.route('/node/<entity_id>', methods=['GET'])
def get_node_details(entity_id):
    """
    获取指定节点的详细信息
    """
    try:
        # 从数据库获取实体详情
        entity = db_manager.get_entity_by_id(entity_id)
        
        if not entity:
            return jsonify({"error": f"未找到实体: {entity_id}"}), 404
        
        # 返回完整的实体信息
        return jsonify(entity)
    
    except Exception as e:
        logger.error(f"获取节点详情时出错: {str(e)}")
        return jsonify({"error": f"获取节点详情失败: {str(e)}"}), 500 