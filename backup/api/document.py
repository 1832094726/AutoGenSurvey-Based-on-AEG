from flask import Blueprint, request, jsonify, send_from_directory
import logging
import os
from app.modules.db_manager import db_manager
from app.config import Config
from werkzeug.utils import secure_filename

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# 创建蓝图
document_bp = Blueprint('document', __name__)

@document_bp.route('/upload', methods=['POST'])
def upload_document():
    """
    上传文档文件（PDF等）
    """
    try:
        # 检查是否有文件
        if 'file' not in request.files:
            return jsonify({
                'success': False,
                'message': '没有上传文件'
            }), 400
            
        file = request.files['file']
        
        # 检查文件名是否为空
        if file.filename == '':
            return jsonify({
                'success': False,
                'message': '未选择文件'
            }), 400
            
        # 确保上传目录存在
        os.makedirs(Config.UPLOAD_FOLDER, exist_ok=True)
        
        # 安全地保存文件
        filename = secure_filename(file.filename)
        file_path = os.path.join(Config.UPLOAD_FOLDER, filename)
        file.save(file_path)
        
        # 记录文件信息到数据库
        file_info = {
            'filename': filename,
            'original_name': file.filename,
            'file_path': file_path,
            'file_type': file.content_type,
            'upload_time': 'NOW()'  # 使用MySQL的NOW()函数
        }
        
        # 提示：实现数据库存储代码
        # db_manager.store_document(file_info)
        
        return jsonify({
            'success': True,
            'message': '文件上传成功',
            'file_info': {
                'filename': filename,
                'path': file_path
            }
        })
        
    except Exception as e:
        logging.error(f"上传文件时出错: {str(e)}")
        import traceback
        logging.error(traceback.format_exc())
        return jsonify({
            'success': False,
            'message': f'上传文件时出错: {str(e)}'
        }), 500

@document_bp.route('/list', methods=['GET'])
def list_documents():
    """
    获取已上传的文档列表
    """
    try:
        # 确保上传目录存在
        os.makedirs(Config.UPLOAD_FOLDER, exist_ok=True)
        
        # 获取目录中的所有文件
        files = []
        for filename in os.listdir(Config.UPLOAD_FOLDER):
            file_path = os.path.join(Config.UPLOAD_FOLDER, filename)
            if os.path.isfile(file_path):
                # 获取文件大小和修改时间
                file_stats = os.stat(file_path)
                files.append({
                    'filename': filename,
                    'path': file_path,
                    'size': file_stats.st_size,
                    'modified_time': file_stats.st_mtime
                })
        
        return jsonify({
            'success': True,
            'files': files,
            'count': len(files)
        })
        
    except Exception as e:
        logging.error(f"获取文件列表时出错: {str(e)}")
        import traceback
        logging.error(traceback.format_exc())
        return jsonify({
            'success': False,
            'message': f'获取文件列表时出错: {str(e)}'
        }), 500

@document_bp.route('/download/<filename>', methods=['GET'])
def download_document(filename):
    """
    下载指定文件
    
    Args:
        filename (str): 文件名
    """
    try:
        return send_from_directory(Config.UPLOAD_FOLDER, filename, as_attachment=True)
    except Exception as e:
        logging.error(f"下载文件时出错: {str(e)}")
        import traceback
        logging.error(traceback.format_exc())
        return jsonify({
            'success': False,
            'message': f'下载文件时出错: {str(e)}'
        }), 500

@document_bp.route('/delete/<filename>', methods=['DELETE'])
def delete_document(filename):
    """
    删除指定文件
    
    Args:
        filename (str): 文件名
    """
    try:
        file_path = os.path.join(Config.UPLOAD_FOLDER, filename)
        
        # 检查文件是否存在
        if not os.path.exists(file_path):
            return jsonify({
                'success': False,
                'message': f'文件不存在: {filename}'
            }), 404
            
        # 删除文件
        os.remove(file_path)
        
        # 提示：实现数据库删除记录代码
        # db_manager.delete_document(filename)
        
        return jsonify({
            'success': True,
            'message': f'文件删除成功: {filename}'
        })
        
    except Exception as e:
        logging.error(f"删除文件时出错: {str(e)}")
        import traceback
        logging.error(traceback.format_exc())
        return jsonify({
            'success': False,
            'message': f'删除文件时出错: {str(e)}'
        }), 500 