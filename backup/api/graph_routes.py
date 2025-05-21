from flask import Blueprint, jsonify, request
from app.modules.db_manager import db_manager
import logging

# 将蓝图重命名为graph_bp
graph_bp = Blueprint('graph_bp', __name__)

@graph_bp.route('/graph_data', methods=['GET'])
def get_graph_data():
    """
    获取图形数据，包括算法实体、数据集、评估指标和它们之间的演化关系
    """
    try:
        # 获取所有实体
        algorithms = db_manager.get_all_entities()
        datasets = db_manager.get_all_datasets()
        metrics = db_manager.get_all_metrics()
        
        # 获取所有演化关系
        all_relations = db_manager.get_all_relations()
        
        # 创建节点集合
        nodes = []
        node_ids = set()  # 用于跟踪已添加的节点
        
        # 添加算法节点
        for algo in algorithms:
            if algo['algorithm_id'] not in node_ids:
                nodes.append({
                    'id': algo['algorithm_id'],
                    'name': algo['name'] or algo['algorithm_id'],
                    'type': 'Algorithm',
                    'year': algo.get('year', ''),
                    'authors': algo.get('authors', ''),
                    'task': algo.get('task', '')
                })
                node_ids.add(algo['algorithm_id'])
        
        # 添加数据集节点
        for dataset in datasets:
            if dataset['dataset_id'] not in node_ids:
                nodes.append({
                    'id': dataset['dataset_id'],
                    'name': dataset['name'] or dataset['dataset_id'],
                    'type': 'Dataset',
                    'year': dataset.get('year', ''),
                    'creators': dataset.get('creators', ''),
                    'domain': dataset.get('domain', '')
                })
                node_ids.add(dataset['dataset_id'])
        
        # 添加评估指标节点
        for metric in metrics:
            if metric['metric_id'] not in node_ids:
                nodes.append({
                    'id': metric['metric_id'],
                    'name': metric['name'] or metric['metric_id'],
                    'type': 'Metric',
                    'category': metric.get('category', ''),
                    'description': metric.get('description', '')
                })
                node_ids.add(metric['metric_id'])
        
        # 创建边集合
        edges = []
        for relation in all_relations:
            # 验证来源和目标实体是否在节点列表中
            from_entity = relation['from_entity']
            to_entity = relation['to_entity']
            
            # 确保节点存在（可能有些关系引用了不存在的节点）
            if from_entity not in node_ids or to_entity not in node_ids:
                continue
                
            # 添加边
            edges.append({
                'id': relation['relation_id'],
                'source': from_entity,
                'target': to_entity,
                'type': relation['relation_type'],
                'label': relation['relation_type'],
                'detail': relation.get('detail', ''),
                'structure': relation.get('structure', ''),
                'from_type': relation.get('from_entity_type', 'Algorithm'),
                'to_type': relation.get('to_entity_type', 'Algorithm')
            })
        
        # 返回图形数据
        return jsonify({
            'nodes': nodes,
            'edges': edges
        })
    except Exception as e:
        logging.error(f"获取图形数据时出错: {str(e)}")
        import traceback
        logging.error(traceback.format_exc())
        return jsonify({'error': str(e)}), 500

@graph_bp.route('/entity_details/<entity_id>', methods=['GET'])
def get_entity_details(entity_id):
    """
    获取实体详细信息，包括与其相关的演化关系
    
    Args:
        entity_id (str): 实体ID
    """
    try:
        # 获取实体类型
        entity_type = request.args.get('type', 'Algorithm')
        
        # 根据实体类型获取详细信息
        if entity_type == 'Algorithm':
            entity = db_manager.get_entity_by_id(entity_id)
        elif entity_type == 'Dataset':
            entity = db_manager.get_dataset_by_id(entity_id)
        elif entity_type == 'Metric':
            entity = db_manager.get_metric_by_id(entity_id)
        else:
            return jsonify({'error': f'不支持的实体类型: {entity_type}'}), 400
            
        if not entity:
            return jsonify({'error': f'实体不存在: {entity_id}'}), 404
        
        # 获取实体的演化关系
        incoming_relations = db_manager.get_incoming_relations(entity_id)
        outgoing_relations = db_manager.get_outgoing_relations(entity_id)
        
        # 组织返回数据
        result = {
            'entity': entity,
            'incoming_relations': incoming_relations,
            'outgoing_relations': outgoing_relations
        }
        
        return jsonify(result)
    except Exception as e:
        logging.error(f"获取实体详细信息时出错: {str(e)}")
        import traceback
        logging.error(traceback.format_exc())
        return jsonify({'error': str(e)}), 500

@graph_bp.route('/relation_types', methods=['GET'])
def get_relation_types():
    """获取所有演化关系类型"""
    try:
        relation_types = db_manager.get_relation_types()
        return jsonify({
            'types': relation_types
        })
    except Exception as e:
        logging.error(f"获取关系类型时出错: {str(e)}")
        return jsonify({'error': str(e)}), 500

@graph_bp.route('/entity_types', methods=['GET'])
def get_entity_types():
    """获取所有实体类型"""
    try:
        return jsonify({
            'types': ['Algorithm', 'Dataset', 'Metric']
        })
    except Exception as e:
        logging.error(f"获取实体类型时出错: {str(e)}")
        return jsonify({'error': str(e)}), 500 