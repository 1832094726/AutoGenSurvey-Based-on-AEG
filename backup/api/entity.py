from flask import Blueprint, request, jsonify
import logging
from app.modules.db_manager import db_manager

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# 创建蓝图
entity_bp = Blueprint('entity', __name__)

@entity_bp.route('/<entity_id>', methods=['GET'])
def get_entity(entity_id):
    """
    获取指定ID的实体详情，支持算法、数据集和评价指标
    
    Args:
        entity_id (str): 实体ID
    """
    try:
        logging.info(f"请求获取实体详情: {entity_id}")
        
        # 从数据库获取实体
        entity = db_manager.get_entity_by_id(entity_id)
        
        if entity:
            return jsonify(entity)
        else:
            logging.warning(f"未找到实体: {entity_id}")
            return jsonify({
                'success': False,
                'message': f'未找到实体: {entity_id}'
            }), 404
    
    except Exception as e:
        logging.error(f"获取实体详情时出错: {str(e)}")
        import traceback
        logging.error(traceback.format_exc())
        return jsonify({
            'success': False,
            'message': f'获取实体详情时出错: {str(e)}'
        }), 500

@entity_bp.route('/search', methods=['GET'])
def search_entities():
    """
    搜索实体，支持按名称、类型和其他属性搜索
    """
    try:
        # 获取查询参数
        name = request.args.get('name', '')
        entity_type = request.args.get('type', '')
        
        logging.info(f"搜索实体，名称关键词: {name}, 类型: {entity_type}")
        
        # 根据条件搜索实体
        entities = db_manager.search_entities(name, entity_type)
        
        # 规范化实体格式
        processed_entities = []
        for entity in entities:
            if 'algorithm_entity' in entity:
                algorithm = entity['algorithm_entity']
                algorithm['entity_type'] = 'Algorithm'
                processed_entities.append(algorithm)
            elif 'dataset_entity' in entity:
                dataset = entity['dataset_entity']
                dataset['entity_type'] = 'Dataset'
                processed_entities.append(dataset)
            elif 'metric_entity' in entity:
                metric = entity['metric_entity']
                metric['entity_type'] = 'Metric'
                processed_entities.append(metric)
            elif 'entity_type' in entity:
                processed_entities.append(entity)
        
        return jsonify({
            'success': True,
            'entities': processed_entities,
            'count': len(processed_entities)
        })
        
    except Exception as e:
        logging.error(f"搜索实体时出错: {str(e)}")
        import traceback
        logging.error(traceback.format_exc())
        return jsonify({
            'success': False,
            'message': f'搜索实体时出错: {str(e)}'
        }), 500 