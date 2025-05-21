from flask import Blueprint, request, jsonify
import logging
from app.modules.db_manager import db_manager

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# 创建蓝图
entity_api = Blueprint('entities', __name__)

@entity_api.route('', methods=['GET'])
def get_entities():
    """
    获取所有实体的信息，包括算法、数据集和评价指标。
    返回统一的格式。
    """
    try:
        logging.info("请求获取所有实体数据")
        
        # 从数据库获取所有实体
        entities = db_manager.get_all_entities()
        
        # 处理响应格式
        processed_entities = []
        
        for entity in entities:
            # 处理不同类型的实体
            if 'algorithm_entity' in entity:
                # 算法实体，直接使用内部数据
                algorithm = entity['algorithm_entity']
                algorithm['entity_type'] = 'Algorithm'
                processed_entities.append(algorithm)
                
            elif 'dataset_entity' in entity:
                # 数据集实体，直接使用内部数据
                dataset = entity['dataset_entity']
                dataset['entity_type'] = 'Dataset'
                processed_entities.append(dataset)
                
            elif 'metric_entity' in entity:
                # 指标实体，直接使用内部数据
                metric = entity['metric_entity']
                metric['entity_type'] = 'Metric'
                processed_entities.append(metric)
            
            # 直接格式的实体已经有entity_type，不需要额外处理
            elif 'entity_type' in entity:
                processed_entities.append(entity)
                
            else:
                logging.warning(f"未知实体格式: {entity.keys()}")
        
        # 调试输出
        logging.info(f"返回 {len(processed_entities)} 个实体，包括: "
                    f"算法: {len([e for e in processed_entities if e.get('entity_type') == 'Algorithm'])}，"
                    f"数据集: {len([e for e in processed_entities if e.get('entity_type') == 'Dataset'])}，"
                    f"评价指标: {len([e for e in processed_entities if e.get('entity_type') == 'Metric'])}")
                
        return jsonify({
            'success': True,
            'count': len(processed_entities),
            'entities': processed_entities
        })
    except Exception as e:
        logging.error(f"获取实体时出错: {str(e)}")
        import traceback
        tb = traceback.format_exc()
        logging.error(tb)
        return jsonify({
            'success': False,
            'error': str(e),
            'traceback': tb
        }), 500

@entity_api.route('/<entity_id>', methods=['GET'])
def get_entity(entity_id):
    """
    获取指定ID的实体详情，支持算法、数据集和评价指标。
    """
    try:
        # 添加详细日志，帮助调试
        logging.info(f"正在获取实体详情: {entity_id}")
        
        # 直接使用db_manager的方法获取实体详情
        entity = db_manager.get_entity_by_id(entity_id)
        
        if entity:
            # 记录成功获取的实体类型
            entity_type = entity.get('entity_type', 'Unknown')
            logging.info(f"成功获取实体 {entity_id}，类型: {entity_type}")
            
            # 返回实体数据
            return jsonify(entity)
        else:
            logging.warning(f"未找到实体: {entity_id}")
            return jsonify({'error': f'实体 {entity_id} 不存在'}), 404
    except Exception as e:
        logging.error(f"获取实体 {entity_id} 时出错: {str(e)}")
        import traceback
        tb = traceback.format_exc()
        logging.error(tb)
        return jsonify({
            'error': str(e),
            'traceback': tb
        }), 500

@entity_api.route('', methods=['POST'])
def add_entity():
    """
    添加新的实体，支持算法、数据集和评价指标。
    """
    try:
        entity_data = request.json
        
        if not entity_data:
            return jsonify({'error': '请求中没有数据'}), 400
        
        # 检查是否含有实体类型字段
        entity_type = entity_data.get('entity_type', '').lower()
        
        # 根据实体类型进行处理
        if entity_type == 'algorithm' or 'algorithm_id' in entity_data:
            # 处理算法实体
            if 'algorithm_entity' not in entity_data and 'entity_type' in entity_data:
                # 直接是算法对象，包装成标准格式
                entity_data = {'algorithm_entity': entity_data}
                
        elif entity_type == 'dataset' or 'dataset_id' in entity_data:
            # 处理数据集实体
            if 'dataset_entity' not in entity_data and 'entity_type' in entity_data:
                # 直接是数据集对象，包装成标准格式
                entity_data = {'dataset_entity': entity_data}
                
        elif entity_type == 'metric' or 'metric_id' in entity_data:
            # 处理评价指标实体
            if 'metric_entity' not in entity_data and 'entity_type' in entity_data:
                # 直接是评价指标对象，包装成标准格式
                entity_data = {'metric_entity': entity_data}
                
        # 如果还没有任何包装且没有明确类型，默认作为算法实体处理
        if not any(key in entity_data for key in ['algorithm_entity', 'dataset_entity', 'metric_entity']):
            entity_data = {'algorithm_entity': entity_data}
            logging.warning(f"未指定实体类型，默认作为算法实体处理")
        
        # 保存到数据库
        db_manager.store_entities([entity_data])
        
        return jsonify({'success': True, 'message': '实体添加成功'})
    except Exception as e:
        logging.error(f"添加实体时出错: {str(e)}")
        import traceback
        tb = traceback.format_exc()
        logging.error(tb)
        return jsonify({'error': str(e), 'traceback': tb}), 500

@entity_api.route('', methods=['PUT'])
def update_entity_no_id():
    """
    处理没有实体ID的更新请求。支持所有实体类型。
    """
    try:
        data = request.json
        if not data:
            return jsonify({'error': '请求中没有数据'}), 400
            
        # 尝试从不同的ID字段获取实体ID
        entity_id = None
        
        if 'algorithm_id' in data:
            entity_id = data['algorithm_id']
        elif 'dataset_id' in data:
            entity_id = data['dataset_id']
        elif 'metric_id' in data:
            entity_id = data['metric_id']
        elif 'entity_id' in data:
            entity_id = data['entity_id']
            
        if not entity_id:
            return jsonify({'error': '请求中缺少实体ID字段（algorithm_id/dataset_id/metric_id/entity_id）'}), 400
        
        return update_entity(entity_id)
    except Exception as e:
        logging.error(f"更新实体时出错: {str(e)}")
        import traceback
        tb = traceback.format_exc()
        logging.error(tb)
        return jsonify({'error': str(e), 'traceback': tb}), 500

@entity_api.route('/<entity_id>', methods=['PUT'])
def update_entity(entity_id):
    """
    更新指定实体的信息。支持所有实体类型。
    """
    try:
        updated_data = request.json
        
        if not updated_data:
            return jsonify({'error': '请求中没有数据'}), 400
        
        # 确保实体ID一致（检查各种可能的ID字段）
        if ('algorithm_id' in updated_data and updated_data['algorithm_id'] != entity_id) or \
           ('dataset_id' in updated_data and updated_data['dataset_id'] != entity_id) or \
           ('metric_id' in updated_data and updated_data['metric_id'] != entity_id) or \
           ('entity_id' in updated_data and updated_data['entity_id'] != entity_id):
            return jsonify({'error': '请求体中的ID与URL中的实体ID不匹配'}), 400
        
        # 获取实体类型
        entity_type = updated_data.get('entity_type', 'Algorithm').lower()
        
        # 根据实体类型调用相应的更新方法
        if entity_type == 'dataset':
            # TODO: 实现数据集更新方法
            success = False  # db_manager.update_dataset(entity_id, updated_data)
            
            if not success:
                logging.warning(f"数据集实体更新方法尚未实现，使用通用方法")
                success = db_manager.update_entity(entity_id, updated_data)
                
        elif entity_type == 'metric':
            # TODO: 实现评价指标更新方法
            success = False  # db_manager.update_metric(entity_id, updated_data)
            
            if not success:
                logging.warning(f"评价指标实体更新方法尚未实现，使用通用方法")
                success = db_manager.update_entity(entity_id, updated_data)
        else:
            # 默认作为算法实体更新
            success = db_manager.update_entity(entity_id, updated_data)
        
        if success:
            return jsonify({'success': True, 'message': f'实体 {entity_id} 更新成功'})
        else:
            return jsonify({'error': f'更新实体 {entity_id} 失败，实体可能不存在'}), 404
    except Exception as e:
        logging.error(f"更新实体 {entity_id} 时出错: {str(e)}")
        import traceback
        tb = traceback.format_exc()
        logging.error(tb)
        return jsonify({'error': str(e), 'traceback': tb}), 500

@entity_api.route('/<entity_id>', methods=['DELETE'])
def delete_entity(entity_id):
    """
    删除指定实体，支持算法、数据集和评价指标。
    """
    try:
        logging.info(f"请求删除实体 {entity_id}")
        
        # 首先尝试查找实体，以确定实体类型
        entity = db_manager.get_entity_by_id(entity_id)
        
        if not entity:
            logging.warning(f"要删除的实体不存在: {entity_id}")
            return jsonify({'error': f'实体 {entity_id} 不存在'}), 404
        
        entity_type = entity.get('entity_type', 'Algorithm').lower()
        logging.info(f"实体类型: {entity_type}")
        
        # 根据实体类型调用相应的删除方法
        if entity_type == 'dataset':
            # TODO: 实现数据集删除方法
            success = False  # db_manager.delete_dataset(entity_id)
            
            if not success:
                logging.warning(f"数据集实体删除方法尚未实现，使用通用方法")
                success = db_manager.delete_entity(entity_id)
                
        elif entity_type == 'metric':
            # TODO: 实现评价指标删除方法
            success = False  # db_manager.delete_metric(entity_id)
            
            if not success:
                logging.warning(f"评价指标实体删除方法尚未实现，使用通用方法")
                success = db_manager.delete_entity(entity_id)
        else:
            # 默认作为算法实体删除
            success = db_manager.delete_entity(entity_id)
        
        if success:
            logging.info(f"成功删除实体 {entity_id}")
            return jsonify({'success': True, 'message': f'实体 {entity_id} 及其关联关系已删除'})
        else:
            logging.warning(f"删除实体失败: {entity_id}")
            return jsonify({'error': f'删除实体 {entity_id} 失败'}), 404
    except Exception as e:
        logging.error(f"删除实体 {entity_id} 时出错: {str(e)}")
        import traceback
        tb = traceback.format_exc()
        logging.error(tb)
        return jsonify({'error': str(e), 'traceback': tb}), 500 