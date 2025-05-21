from flask import Blueprint, request, jsonify
import logging
from app.modules.db_manager import db_manager
from app.modules.agents import extract_evolution_relations

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# 创建蓝图
evolution_bp = Blueprint('evolution', __name__)

@evolution_bp.route('/relations', methods=['GET'])
def get_relations():
    """
    获取所有演化关系
    """
    try:
        logging.info("请求获取所有演化关系")
        
        # 从数据库获取所有演化关系
        relations = db_manager.get_all_relations()
        
        return jsonify({
            'success': True,
            'relations': relations,
            'count': len(relations)
        })
        
    except Exception as e:
        logging.error(f"获取演化关系时出错: {str(e)}")
        import traceback
        logging.error(traceback.format_exc())
        return jsonify({
            'success': False,
            'message': f'获取演化关系时出错: {str(e)}'
        }), 500

@evolution_bp.route('/relations/<entity_id>', methods=['GET'])
def get_entity_relations(entity_id):
    """
    获取指定实体的演化关系
    
    Args:
        entity_id (str): 实体ID
    """
    try:
        logging.info(f"请求获取实体 {entity_id} 的演化关系")
        
        # 获取作为源和目标的关系
        incoming = db_manager.get_incoming_relations(entity_id)
        outgoing = db_manager.get_outgoing_relations(entity_id)
        
        return jsonify({
            'success': True,
            'entity_id': entity_id,
            'incoming_relations': incoming,
            'outgoing_relations': outgoing,
            'incoming_count': len(incoming),
            'outgoing_count': len(outgoing)
        })
        
    except Exception as e:
        logging.error(f"获取实体关系时出错: {str(e)}")
        import traceback
        logging.error(traceback.format_exc())
        return jsonify({
            'success': False,
            'message': f'获取实体关系时出错: {str(e)}'
        }), 500

@evolution_bp.route('/generate', methods=['POST'])
def generate_relations():
    """
    根据实体信息生成演化关系
    """
    try:
        data = request.json
        
        if not data or 'entities' not in data:
            return jsonify({
                'success': False,
                'message': '未提供实体数据'
            }), 400
            
        entities = data.get('entities', [])
        pdf_path = data.get('pdf_path')
        
        logging.info(f"请求为 {len(entities)} 个实体生成演化关系")
        if pdf_path:
            logging.info(f"使用PDF文件: {pdf_path}")
            
        # 调用agents模块生成演化关系
        relations = extract_evolution_relations(entities, pdf_path)
        
        # 保存关系到数据库
        if relations:
            db_manager.store_relations(relations)
            
        return jsonify({
            'success': True,
            'message': f'成功生成 {len(relations)} 条演化关系',
            'relations': relations
        })
        
    except Exception as e:
        logging.error(f"生成演化关系时出错: {str(e)}")
        import traceback
        logging.error(traceback.format_exc())
        return jsonify({
            'success': False,
            'message': f'生成演化关系时出错: {str(e)}'
        }), 500

@evolution_bp.route('/relations', methods=['POST'])
def add_relation():
    """
    添加新的演化关系
    """
    try:
        relation_data = request.json
        
        if not relation_data:
            return jsonify({
                'success': False,
                'message': '未提供关系数据'
            }), 400
            
        # 检查必要字段
        required_fields = ['from_entity', 'to_entity', 'relation_type']
        for field in required_fields:
            if field not in relation_data:
                return jsonify({
                    'success': False,
                    'message': f'缺少必要字段: {field}'
                }), 400
                
        # 保存到数据库
        db_manager.store_relations([relation_data])
        
        return jsonify({
            'success': True,
            'message': '关系添加成功',
            'relation': relation_data
        })
        
    except Exception as e:
        logging.error(f"添加演化关系时出错: {str(e)}")
        import traceback
        logging.error(traceback.format_exc())
        return jsonify({
            'success': False,
            'message': f'添加演化关系时出错: {str(e)}'
        }), 500

@evolution_bp.route('/relations/<relation_id>', methods=['DELETE'])
def delete_relation(relation_id):
    """
    删除指定的演化关系
    
    Args:
        relation_id (str): 关系ID
    """
    try:
        logging.info(f"请求删除关系: {relation_id}")
        
        # 从数据库删除关系
        success = db_manager.delete_relation(relation_id)
        
        if success:
            return jsonify({
                'success': True,
                'message': f'关系删除成功: {relation_id}'
            })
        else:
            return jsonify({
                'success': False,
                'message': f'关系不存在或删除失败: {relation_id}'
            }), 404
            
    except Exception as e:
        logging.error(f"删除关系时出错: {str(e)}")
        import traceback
        logging.error(traceback.format_exc())
        return jsonify({
            'success': False,
            'message': f'删除关系时出错: {str(e)}'
        }), 500 