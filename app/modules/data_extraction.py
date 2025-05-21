import os
import logging
import arxiv
import json
import re
from pathlib import Path
import shutil
import tempfile
from app.config import Config
from app.modules.db_manager import db_manager
import time
import datetime
import requests
from app.modules.agents import extract_evolution_relations  # 添加导入演化关系提取函数

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def parse_review_paper(file_path, task_id=None):
    """
    解析综述文章，提取引用文献列表。
    
    Args:
        file_path (str): 综述文章的文件路径
        task_id (str, optional): 处理任务ID，用于更新处理状态
        
    Returns:
        List[str]: 引用文献的DOI或唯一标识列表
    """
    try:
        # 确保缓存目录存在
        os.makedirs(Config.CACHE_DIR, exist_ok=True)
        
        # 生成缓存文件名
        file_basename = os.path.basename(file_path)
        filename_without_ext = os.path.splitext(file_basename)[0]
        
        # 移除可能的任务ID前缀
        if task_id and filename_without_ext.startswith(f"task_{task_id}_"):
            filename_without_ext = filename_without_ext[len(f"task_{task_id}_"):]
            
        # 移除日期时间前缀格式 (如: 20250520_164839_)
        date_time_prefix_pattern = r"^\d{8}_\d{6}_"
        if re.match(date_time_prefix_pattern, filename_without_ext):
            logging.info(f"检测到日期时间前缀: {filename_without_ext}")
            filename_without_ext = re.sub(date_time_prefix_pattern, "", filename_without_ext)
            logging.info(f"移除前缀后的文件名: {filename_without_ext}")
        
        cache_file = os.path.join(Config.CACHE_DIR, f"references_{filename_without_ext}.json")
        
        # 检查缓存是否存在
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'r', encoding='utf-8') as f:
                    cached_data = json.load(f)
                    logging.info(f"找到缓存文件: {cache_file}")
                    
                    # 检查缓存是否完整
                    if 'completed' in cached_data and cached_data['completed']:
                        references = cached_data.get('references', [])
                        logging.info(f"从缓存中加载了 {len(references)} 条完整引用")
                        return references
                    
                    # 检查是否有部分引用且数量大于1
                    if 'references' in cached_data and len(cached_data['references']) > 1:
                        references = cached_data.get('references', [])
                        logging.info(f"从缓存中加载了 {len(references)} 条引用（部分）")
                        return references
                    
                    # 如果有部分响应数据，后续会使用它继续处理
                    logging.info("发现不完整的缓存，将用于继续处理")
            except Exception as e:
                logging.error(f"读取缓存文件出错: {str(e)}")
        
        # 更新处理状态：开始解析
        if task_id:
            db_manager.update_processing_status(
                task_id=task_id,
                current_stage='解析综述文章',
                progress=0.15,
                current_file=os.path.basename(file_path),
                message=f'解析综述文章: {os.path.basename(file_path)}'
            )
        
        # 检查文件是否存在且可读
        if not os.path.exists(file_path):
            logging.error(f"文件不存在: {file_path}")
            return []
        
        # 使用qwen API解析PDF文件提取引用文献
        # 更新处理状态：调用API
        if task_id:
            db_manager.update_processing_status(
                task_id=task_id,
                current_stage='调用AI模型解析',
                progress=0.35,
                message='正在调用千问模型解析引用文献'
            )
        
        # 使用qwen API进行文件解析
        from app.modules.agents import extract_paper_entities_openai
        from openai import OpenAI
        import re
        
        # 创建临时文件副本以避免权限问题
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
            temp_path = temp_file.name
        
        try:
            shutil.copy2(file_path, temp_path)
            logging.info(f"成功创建临时文件副本: {temp_path}")
            
            # 初始化qwen客户端
            client = OpenAI(
                api_key=Config.QWEN_API_KEY,
                base_url=Config.QWEN_BASE_URL
            )
            
            # 上传PDF文件进行解析
            file = client.files.create(file=Path(temp_path), purpose="file-extract")
            logging.info(f"文件上传成功，file_id: {file.id}")
            
            # 读取现有缓存中的部分响应（如果有）
            previous_references = []
            continuation_prompt = ""
            
            if os.path.exists(cache_file):
                try:
                    with open(cache_file, 'r', encoding='utf-8') as f:
                        cached_data = json.load(f)
                        if 'partial_response' in cached_data:
                            continuation_prompt = f"以下是已经提取的部分内容，请在此基础上继续提取：\n{cached_data['partial_response']}\n\n"
                            logging.info("将使用缓存中的部分响应作为提示继续提取")
                        if 'references' in cached_data:
                            previous_references = cached_data.get('references', [])
                            logging.info(f"缓存中已有 {len(previous_references)} 条引用")
                except Exception as e:
                    logging.error(f"读取缓存的部分响应时出错: {str(e)}")
            
            # 设置最大尝试次数，避免无限循环
            max_attempts = 3
            current_attempt = 0
            all_references = previous_references.copy()
            
            while current_attempt < max_attempts:
                current_attempt += 1
                logging.info(f"开始第 {current_attempt}/{max_attempts} 次提取尝试")
                
                # 构建提示词
                system_message = '你是一个专注于文献分析的AI助手。'
                file_content_message = f'fileid://{file.id}'
                
                base_user_message = '''请从提供的文本中提取所有引用的文献，并以JSON数组的格式返回。
                
                每个文献条目应包括以下信息（如可获取）：
                1. id（文献编号或DOI）
                2. authors（作者列表）
                3. title（标题）
                4. year（发表年份）
                5. venue（发表期刊或会议）
                
                直接返回有效的JSON数组，不要包含任何其他文本。'''
                
                user_message = continuation_prompt + base_user_message
                
                if all_references:
                    # 构造一个提示，告知已经收集的文献，以避免重复提取
                    collected_ids = [ref.get('id', '') for ref in all_references if 'id' in ref]
                    collected_titles = [ref.get('title', '') for ref in all_references if 'title' in ref]
                    
                    # 最多展示10个已收集的文献，避免提示词过长
                    if collected_titles:
                        sample_titles = collected_titles[:10]
                        user_message += f"\n\n已经收集了 {len(all_references)} 条文献，包括：\n" + "\n".join([f"- {title}" for title in sample_titles])
                        if len(collected_titles) > 10:
                            user_message += f"\n以及其他 {len(collected_titles) - 10} 条..."
                        user_message += "\n\n重要提示：请只提取尚未收集的其他文献，不要重复返回已有文献。只返回新文献的JSON数组，不要包含已收集的文献。"
                
                # 输出提示词
                logging.info("提交给千问API的提示词:")
                logging.info(f"系统消息: {system_message}")
                logging.info(f"文件引用: {file_content_message}")
                logging.info(f"用户消息: {user_message[:100]}...")
                
                # 使用qwen模型提取引用文献
                logging.info(f"调用千问API提取引用文献 (尝试 {current_attempt}/{max_attempts})")
                
                completion = client.chat.completions.create(
                    model="qwen-long",
                    messages=[
                        {
                            'role': 'system',
                            'content': system_message
                        },
                        {
                            'role': 'system',
                            'content': file_content_message
                        },
                        {
                            'role': 'user',
                            'content': user_message
                        }
                    ],
                    stream=True
                )
                
                # 收集流式响应内容
                response_content = ""
                chunk_count = 0
                
                for chunk in completion:
                    chunk_count += 1
                    if chunk.choices and chunk.choices[0].delta.content:
                        content_piece = chunk.choices[0].delta.content
                        response_content += content_piece
                        if chunk_count % 10 == 0:  # 每10个块记录一次，避免日志过多
                            logging.info(f"收到响应块 #{chunk_count}，当前响应长度: {len(response_content)}")
                
                logging.info(f"响应接收完成，共 {chunk_count} 个响应块，总长度: {len(response_content)}")
                
                # 直接保存完整响应内容到缓存，无论格式是否正确
                temp_cached_data = {
                    'partial_response': response_content,
                    'references': all_references,
                    'completed': False,
                    'timestamp': datetime.datetime.now().isoformat(),
                    'attempt': current_attempt
                }
                
                with open(cache_file, 'w', encoding='utf-8') as f:
                    json.dump(temp_cached_data, f, ensure_ascii=False, indent=2)
                logging.info(f"已保存原始响应内容到缓存文件")
                
                # 提取JSON部分
                try:
                    json_match = re.search(r'```json\s*(.*?)\s*```', response_content, re.DOTALL)
                    if json_match:
                        json_str = json_match.group(1)
                        logging.info("从响应中提取到JSON格式数据")
                    else:
                        json_str = response_content
                        logging.info("响应中未找到JSON代码块，尝试将整个响应解析为JSON")
                    
                    # 尝试清理和解析JSON
                    json_str = _clean_json_string(json_str)
                    new_references = json.loads(json_str)
                    
                    # 确保references是列表
                    if not isinstance(new_references, list):
                        new_references = [new_references]
                    
                    logging.info(f"本次成功解析出 {len(new_references)} 条引用")
                    
                    # 去重合并新旧引用
                    if new_references:
                        # 使用标题作为主要去重依据
                        existing_titles = {ref.get('title', '').lower() for ref in all_references if 'title' in ref}
                        
                        for new_ref in new_references:
                            new_title = new_ref.get('title', '').lower()
                            if new_title and new_title not in existing_titles:
                                all_references.append(new_ref)
                                existing_titles.add(new_title)
                        
                        logging.info(f"去重后总共有 {len(all_references)} 条引用")
                    
                    # 判断是否需要继续提取
                    if len(new_references) < 5 and current_attempt > 1:
                        # 如果获取的新引用很少且不是第一次尝试，认为提取已完成
                        logging.info("获取的新引用很少，认为提取已完成")
                        break
                    
                    # 更新部分提示，继续下一轮提取
                    continuation_prompt = f"我们已成功提取了 {len(all_references)} 条引用，但文档可能还有其他引用未被提取。请继续提取这些未包含的引用。请注意：\n\n1. 只返回新的、尚未提取的引用\n2. 不要重复之前已提取的内容\n3. 直接返回JSON数组格式\n\n"
                    
                except Exception as e:
                    logging.error(f"解析JSON时出错: {str(e)}")
                    # 如果JSON解析失败，保存原始响应以供后续分析
                    # 已经在循环开始时保存了原始响应，这里不需要再次保存
                    
                    if current_attempt < max_attempts:
                        logging.info(f"将在下一次尝试中继续提取")
                        # 使用部分响应作为提示词的一部分
                        continuation_prompt += f"上次响应解析失败，请尝试提供格式正确的JSON。\n\n"
                    else:
                        logging.warning(f"达到最大尝试次数，使用已提取的 {len(all_references)} 条引用")
            
            # 保存最终结果到缓存
            cached_data = {
                'references': all_references,
                'completed': True,
                'timestamp': datetime.datetime.now().isoformat(),
                'attempt_count': current_attempt
            }
            
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(cached_data, f, ensure_ascii=False, indent=2)
            
            logging.info(f"成功解析引用文献，共找到 {len(all_references)} 条引用")
            
            # 更新处理状态：解析完成
            if task_id:
                db_manager.update_processing_status(
                    task_id=task_id,
                    current_stage='引用文献解析完成',
                    progress=0.45,
                    message=f'成功解析引用文献，找到 {len(all_references)} 条引用'
                )
            
            return all_references
            
        finally:
            # 清理临时文件
            try:
                if os.path.exists(temp_path):
                    os.unlink(temp_path)
                    logging.info(f"临时文件已删除: {temp_path}")
            except Exception as e:
                logging.warning(f"清理临时文件时出错: {str(e)}")
        
    except Exception as e:
        logging.error(f"解析综述文章时出错: {str(e)}")
        import traceback
        logging.error(traceback.format_exc())
        
        # 保存错误信息到缓存
        with open(cache_file, 'w', encoding='utf-8') as f:
            json.dump({
                'error': str(e),
                'completed': False,
                'timestamp': datetime.datetime.now().isoformat()
            }, f, ensure_ascii=False, indent=2)
        
        return []

def retrieve_referenced_papers(citation_list, download_dir=None):
    """
    根据引用列表检索并下载相关文献的PDF文件。
    
    Args:
        citation_list (List[str]): 引用文献列表
        download_dir (str): 下载目录路径
        
    Returns:
        List[str]: 下载的PDF文件路径列表
    """
    if download_dir is None:
        download_dir = Config.CITED_PAPERS_DIR
        
    if not os.path.exists(download_dir):
        os.makedirs(download_dir)
    
    downloaded_files = []
    client = arxiv.Client()
    
    for citation in citation_list:
        try:
            # 尝试从arXiv检索文章
            search = arxiv.Search(
                query=citation,
                max_results=1
            )
            
            for result in client.results(search):
                # 使用临时文件下载
                with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
                    temp_path = temp_file.name
                
                try:
                    result.download_pdf(filename=temp_path)
                    
                    # 生成目标文件名
                    final_filename = f"{download_dir}/{result.get_short_id()}.pdf"
                    
                    # 确保目标目录存在
                    os.makedirs(os.path.dirname(final_filename), exist_ok=True)
                    
                    # 复制到最终位置
                    shutil.copy2(temp_path, final_filename)
                    downloaded_files.append(final_filename)
                    logging.info(f"已下载: {final_filename}")
                    break
                finally:
                    # 清理临时文件
                    try:
                        if os.path.exists(temp_path):
                            os.unlink(temp_path)
                    except Exception as e:
                        logging.warning(f"清理临时文件时出错: {str(e)}")
        except Exception as e:
            logging.error(f"下载文献 {citation} 时出错: {str(e)}")
    
    return downloaded_files

def get_entity_from_cache(paper_id):
    """从缓存获取已处理的实体信息"""
    cache_file = os.path.join(Config.CACHE_DIR, f"{paper_id}.json")
    if os.path.exists(cache_file):
        try:
            with open(cache_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logging.warning(f"读取缓存失败: {str(e)}")
    return None

def save_entity_to_cache(paper_id, entity_data):
    """保存实体信息到缓存"""
    os.makedirs(Config.CACHE_DIR, exist_ok=True)
    cache_file = os.path.join(Config.CACHE_DIR, f"{paper_id}.json")
    try:
        with open(cache_file, 'w', encoding='utf-8') as f:
            json.dump(entity_data, f, ensure_ascii=False, indent=2)
        logging.info(f"实体信息已缓存: {paper_id}")
    except Exception as e:
        logging.warning(f"保存缓存失败: {str(e)}")

def extract_entities_from_paper(pdf_path, task_id=None, sub_progress=None):
    """
    从论文中提取算法实体和要素
    
    Args:
        pdf_path (str): PDF文件路径
        task_id (str, optional): 任务ID，用于缓存标识
        sub_progress (tuple, optional): 进度范围 (start, end)，用于在分批处理中更新进度
        
    Returns:
        list: 提取的实体列表
    """
    # 标准化路径分隔符
    pdf_path = os.path.normpath(pdf_path)
    
    if not os.path.exists(pdf_path):
        logging.error(f"文件不存在: {pdf_path}")
        return []
        
    # 尝试从缓存获取
    basename = os.path.basename(pdf_path)
    filename_without_ext = os.path.splitext(basename)[0]
    
    # 移除可能的任务ID前缀
    if task_id and filename_without_ext.startswith(f"task_{task_id}_"):
        filename_without_ext = filename_without_ext[len(f"task_{task_id}_"):]
    
    # 移除日期时间前缀格式 (如: 20250520_164839_)
    date_time_prefix_pattern = r"^\d{8}_\d{6}_"
    if re.match(date_time_prefix_pattern, filename_without_ext):
        logging.info(f"检测到日期时间前缀: {filename_without_ext}")
        filename_without_ext = re.sub(date_time_prefix_pattern, "", filename_without_ext)
        logging.info(f"移除前缀后的文件名: {filename_without_ext}")
    
    # 创建缓存目录
    cache_dir = os.path.join(Config.CACHE_DIR, "entities")
    os.makedirs(cache_dir, exist_ok=True)
    
    # 生成缓存文件名
    cache_key = f"{filename_without_ext}_entities.json"
    cache_path = os.path.join(cache_dir, cache_key)
    
    # 检查缓存是否存在
    if os.path.exists(cache_path):
        try:
            with open(cache_path, 'r', encoding='utf-8') as f:
                cache_data = json.load(f)
                logging.info(f"从缓存加载实体数据: {cache_path}")
                
                # 检查实体提取是否完整
                is_complete = True
                for entity in cache_data:
                    if entity.get('is_complete', False) == False:
                        is_complete = False
                        break
                
                if is_complete:
                    logging.info(f"所有实体数据已完成提取，共 {len(cache_data)} 个实体")
                    return cache_data
                else:
                    logging.info(f"实体数据提取不完整，将继续提取...")
                    # 此处不直接返回，而是继续处理，使用已有数据作为历史记录
        except Exception as e:
            logging.error(f"读取缓存文件时出错: {str(e)}")
            
    # 更新处理状态
    if task_id and sub_progress:
        start_progress, end_progress = sub_progress
        current_progress = start_progress + (end_progress - start_progress) * 0.3
        db_manager.update_processing_status(
            task_id=task_id,
            current_stage='提取实体',
            progress=current_progress,
            current_file=basename,
            message=f"正在提取实体: {basename}"
        )
    
    # 使用agents模块中的方法提取实体
    from app.modules.agents import extract_paper_entities
    entities, is_complete = extract_paper_entities(
        pdf_path, 
        model_name=Config.DEFAULT_MODEL,
        task_id=task_id
    )
    
    # 如果提取成功
    if entities:
        # 缓存提取的实体
        with open(cache_path, 'w', encoding='utf-8') as f:
            json.dump(entities, f, ensure_ascii=False, indent=2)
        logging.info(f"已提取 {len(entities)} 个实体，已缓存到: {cache_path}")
        
        # 更新处理状态
        if task_id and sub_progress:
            start_progress, end_progress = sub_progress
            current_progress = start_progress + (end_progress - start_progress) * 0.7
            db_manager.update_processing_status(
                task_id=task_id,
                current_stage='实体提取完成',
                progress=current_progress,
                current_file=basename,
                message=f"已从 {basename} 中提取 {len(entities)} 个实体"
            )
    else:
        logging.warning(f"未能从 {pdf_path} 提取任何实体")
    
    return entities

def _clean_json_string(json_str):
    """清理和修复常见的JSON格式错误"""
    # 移除开头和结尾的非JSON字符
    json_str = re.sub(r'^[^{\[]+', '', json_str)
    json_str = re.sub(r'[^}\]]+$', '', json_str)
    
    # 检查JSON是否完整
    open_brackets = sum(1 for c in json_str if c in '{[')
    close_brackets = sum(1 for c in json_str if c in '}]')
    
    # 添加缺失的括号
    if open_brackets > close_brackets:
        missing_close = open_brackets - close_brackets
        # 判断需要添加的括号类型
        last_open = None
        for c in reversed(json_str):
            if c in '{[':
                last_open = c
                break
        
        if last_open == '{':
            json_str += '}' * missing_close
        elif last_open == '[':
            json_str += ']' * missing_close
        else:
            # 如果无法确定，默认添加方括号和花括号的组合
            json_str += ']}' * (missing_close // 2)
            if missing_close % 2 == 1:
                json_str += '}'
                
        logging.warning(f"JSON不完整，添加了 {missing_close} 个缺失的闭合括号")
    elif close_brackets > open_brackets:
        # 如果闭合括号过多，从末尾移除多余的闭合括号
        excess = close_brackets - open_brackets
        for _ in range(excess):
            last_index = max(json_str.rfind('}'), json_str.rfind(']'))
            if last_index > 0:
                json_str = json_str[:last_index] + json_str[last_index+1:]
        logging.warning(f"JSON格式错误，移除了 {excess} 个多余的闭合括号")
    
    # 修复常见的格式错误
    # 1. 修复缺少逗号的问题
    json_str = re.sub(r'}\s*{', '},{', json_str)
    json_str = re.sub(r']\s*{', '],{', json_str)
    json_str = re.sub(r'}\s*\[', '},\[', json_str)
    json_str = re.sub(r'"(\w+)"(\s*):', r'"\1":', json_str)  # 修复属性周围多余的空格
    
    # 2. 修复常见的转义字符问题
    json_str = re.sub(r'\\([^"\\/bfnrtu])', r'\1', json_str)
    
    # 3. 修复未闭合的引号
    # 计算引号数量，确保是偶数
    quote_count = json_str.count('"')
    if quote_count % 2 != 0:
        # 尝试定位未闭合的引号并修复
        in_quote = False
        fixed_str = ""
        for i, c in enumerate(json_str):
            if c == '"' and (i == 0 or json_str[i-1] != '\\'):
                in_quote = not in_quote
            fixed_str += c
            
            # 如果到了JSON末尾还在引号内，添加一个引号
            if i == len(json_str) - 1 and in_quote:
                fixed_str += '"'
                logging.warning("检测到未闭合的引号，已修复")
        
        json_str = fixed_str
    
    # 4. 修复常见的尾部截断问题
    if json_str.rstrip().endswith(','):
        if '[' in json_str:
            json_str = json_str.rstrip().rstrip(',') + ']'
            logging.warning("检测到数组尾部截断，已修复")
        elif '{' in json_str:
            json_str = json_str.rstrip().rstrip(',') + '}'
            logging.warning("检测到对象尾部截断，已修复")
    
    # 5. 处理非标准JSON（例如有注释或尾随逗号）
    # 移除JavaScript风格的注释
    json_str = re.sub(r'//.*?$', '', json_str, flags=re.MULTILINE)
    json_str = re.sub(r'/\*.*?\*/', '', json_str, flags=re.DOTALL)
    
    # 移除尾随逗号
    json_str = re.sub(r',\s*([}\]])', r'\1', json_str)
    
    # 6. 检查是否是孤立的数组元素，如果是，则包装为数组
    if not (json_str.startswith('[') or json_str.startswith('{')):
        if json_str.startswith('"') or json_str.startswith('{'):
            json_str = '[' + json_str + ']'
            logging.warning("检测到孤立的JSON元素，已包装为数组")
    
    # 7. 尝试修复缺少键名的情况
    json_str = re.sub(r':\s*,', ': "",', json_str)  # 空值
    json_str = re.sub(r':\s*}', ': ""}', json_str)  # 末尾空值
    
    # 记录清理后的长度
    logging.info(f"JSON字符串清理完成，长度: {len(json_str)}")
    
    return json_str

def process_papers_and_extract_data(review_pdf_path, task_id=None, citation_paths=None):
    """
    处理综述文章和引用文献，提取实体和关系数据。
    
    Args:
        review_pdf_path (str): 综述文章的PDF文件路径
        task_id (str, optional): 处理任务ID，用于更新处理状态
        citation_paths (list, optional): 引用文献的PDF文件路径列表
        
    Returns:
        Tuple[List[Dict], List[Dict]]: 提取的实体列表和关系列表
    """
    # 标准化路径分隔符
    review_pdf_path = os.path.normpath(review_pdf_path)
    if citation_paths:
        citation_paths = [os.path.normpath(path) for path in citation_paths]
    
    # 确保所有必要的目录存在
    for directory in [Config.UPLOAD_DIR, Config.CACHE_DIR, Config.CITED_PAPERS_DIR]:
        os.makedirs(directory, exist_ok=True)
    
    # 初始化关系列表
    relations = []
    
    # 检查是否有进度文件，尝试恢复之前的处理
    progress_file = None
    if task_id:
        progress_file = os.path.join(Config.CACHE_DIR, f"progress_{task_id}.json")
        if os.path.exists(progress_file):
            try:
                with open(progress_file, 'r', encoding='utf-8') as f:
                    progress_data = json.load(f)
                    logging.info(f"找到现有进度文件: {progress_file}")
                    
                    # 从缓存中恢复已处理的实体列表
                    entities = progress_data.get('entities', [])
                    processed_files = set(progress_data.get('processed_files', []))
                    
                    # 更新处理状态
                    db_manager.update_processing_status(
                        task_id=task_id,
                        current_stage='恢复处理',
                        progress=progress_data.get('progress', 0.3),
                        message=f'从上次中断处恢复处理，已处理 {len(processed_files)} 个文件'
                    )
                    
                    # 检查主综述是否已处理
                    review_basename = os.path.basename(review_pdf_path)
                    if review_basename in processed_files:
                        logging.info("综述文章已处理，跳过")
                    else:
                        # 需要处理综述文章
                        logging.info("需要处理综述文章")
                        processed_files = set()  # 重置处理列表
                        entities = []  # 重置实体列表
            except Exception as e:
                logging.error(f"读取进度文件时出错: {str(e)}")
                processed_files = set()
                entities = []
        else:
            processed_files = set()
            entities = []
    else:
        processed_files = set()
        entities = []
    
    # 检查文件是否存在
    if not os.path.exists(review_pdf_path):
        logging.error(f"文件不存在: {review_pdf_path}")
        return [], []
    
    # 处理综述文章（如果尚未处理）
    review_basename = os.path.basename(review_pdf_path)
    review_entities = []
    
    if review_basename not in processed_files:
        # 更新处理状态
        if task_id:
            message = f"正在分析主要综述文章: {review_basename}"
            db_manager.update_processing_status(
                task_id=task_id,
                current_stage='分析综述',
                progress=0.2,
                current_file=review_basename,
                message=message
            )
        
        # 更新处理状态
        if task_id:
            message = f"正在提取文章中的算法实体: {review_basename}"
            db_manager.update_processing_status(
                task_id=task_id,
                current_stage='提取实体',
                progress=0.3,
                message=message
            )
        
        # 从综述文章中提取实体
        review_entities = extract_entities_from_paper(review_pdf_path, task_id)
        if review_entities:
            entities.extend(review_entities)
            processed_files.add(review_basename)
            
            # 保存进度
            if task_id:
                save_progress(task_id, entities, processed_files, 0.3)
    
    # 收集所有需要处理的引用文献
    citation_pdfs = []
    
    # 如果提供了引用文献列表
    if citation_paths:
        for path in citation_paths:
            if os.path.exists(path) and path.endswith('.pdf'):
                citation_pdfs.append(path)
    
    # 检查上传目录中的其他PDF文件
    if not citation_pdfs and os.path.exists(Config.UPLOAD_DIR):
        for filename in os.listdir(Config.UPLOAD_DIR):
            if filename.endswith('.pdf'):
                pdf_path = os.path.join(Config.UPLOAD_DIR, filename)
                if pdf_path != review_pdf_path and os.path.exists(pdf_path):
                    citation_pdfs.append(pdf_path)
    
    # 检查引用文献目录
    if not citation_pdfs and os.path.exists(Config.CITED_PAPERS_DIR):
        for filename in os.listdir(Config.CITED_PAPERS_DIR):
            if filename.endswith('.pdf'):
                pdf_path = os.path.join(Config.CITED_PAPERS_DIR, filename)
                if pdf_path != review_pdf_path and os.path.exists(pdf_path):
                    citation_pdfs.append(pdf_path)
    
    # 过滤掉已处理的文件
    remaining_pdfs = []
    for pdf_path in citation_pdfs:
        if os.path.basename(pdf_path) not in processed_files:
            remaining_pdfs.append(pdf_path)
    
    # 更新处理状态
    if task_id and remaining_pdfs:
        message = f"正在处理 {len(remaining_pdfs)} 篇相关论文（共 {len(citation_pdfs)} 篇，{len(processed_files)} 篇已处理）"
        db_manager.update_processing_status(
            task_id=task_id,
            current_stage='处理相关论文',
            progress=0.5,
            message=message
        )
    
    # 批量处理引用文献
    citation_entities = []
    if remaining_pdfs:
        # 分批处理，每次处理10个文件
        batch_size = 10
        total_batches = (len(remaining_pdfs) + batch_size - 1) // batch_size
        
        for batch_index in range(total_batches):
            start_idx = batch_index * batch_size
            end_idx = min(start_idx + batch_size, len(remaining_pdfs))
            batch_pdfs = remaining_pdfs[start_idx:end_idx]
            
            if task_id:
                progress = 0.5 + (batch_index / total_batches * 0.3)
                db_manager.update_processing_status(
                    task_id=task_id,
                    current_stage=f'处理引用文献（批次 {batch_index+1}/{total_batches}）',
                    progress=progress,
                    message=f'正在处理第 {batch_index+1}/{total_batches} 批引用文献，共 {len(batch_pdfs)} 个文件'
                )
            
            # 使用优化的批量处理方法
            from app.modules.agents import extract_paper_entities
            batch_entities, _ = extract_paper_entities(batch_pdfs, model_name=Config.DEFAULT_MODEL, task_id=task_id)
            
            if batch_entities:
                citation_entities.extend(batch_entities)
                for pdf_path in batch_pdfs:
                    processed_files.add(os.path.basename(pdf_path))
                
                # 合并到总实体列表
                entities.extend(batch_entities)
                
                # 定期保存进度
                if task_id:
                    progress = 0.5 + ((batch_index + 1) / total_batches * 0.3)
                    save_progress(task_id, entities, processed_files, progress)
    
    # 检查是否需要提取关系
    # 创建关系缓存文件名
    relations_cache_file = None
    if task_id:
        review_basename = os.path.basename(review_pdf_path)
        review_filename_without_ext = os.path.splitext(review_basename)[0]
        
        # 移除可能的任务ID前缀
        if task_id and review_filename_without_ext.startswith(f"task_{task_id}_"):
            review_filename_without_ext = review_filename_without_ext[len(f"task_{task_id}_"):]
            
        relations_cache_file = os.path.join(Config.CACHE_DIR, f"relations_{review_filename_without_ext}.json")
        if os.path.exists(relations_cache_file):
            try:
                with open(relations_cache_file, 'r', encoding='utf-8') as f:
                    relations = json.load(f)
                    logging.info(f"从缓存加载关系数据: {relations_cache_file}, 共 {len(relations)} 条关系")
                    
                    # 检查关系是否完整
                    is_complete = True
                    for relation in relations:
                        if not relation.get('is_complete', False):
                            is_complete = False
                            break
                    
                    if is_complete:
                        logging.info("所有关系已完整提取")
                    else:
                        logging.info("关系提取不完整，将继续完善")
                        # 使用已有的关系数据作为上下文继续提取
                        relations = extract_relationships_with_context(entities, review_pdf_path, task_id, relations)
            except Exception as e:
                logging.error(f"读取关系缓存文件出错: {str(e)}")
                # 重新提取关系
                relations = extract_relationships_with_context(entities, review_pdf_path, task_id)
        else:
            # 提取所有实体之间的关系
            relations = extract_relationships_with_context(entities, review_pdf_path, task_id)
    else:
        # 提取所有实体之间的关系
        relations = extract_relationships_with_context(entities, review_pdf_path, task_id)
    
    # 保存所有实体到数据库
    stored_count = 0
    skipped_count = 0
    logging.info("开始将提取的实体保存到数据库...")
    
    # 记录所有实体类型的计数
    entity_type_counts = {
        'Algorithm': 0,
        'Dataset': 0,
        'Metric': 0,
        'Unknown': 0
    }
    
    # 处理所有实体
    for entity_data in entities:
        try:
            # 只保存没有明显错误的实体
            if not entity_data or 'error' in entity_data:
                skipped_count += 1
                continue
            
            # 确定实体类型
            entity_type = 'Unknown'
            if 'algorithm_entity' in entity_data:
                entity_type = 'Algorithm'
                entity_type_counts['Algorithm'] += 1
            elif 'dataset_entity' in entity_data:
                entity_type = 'Dataset'
                entity_type_counts['Dataset'] += 1
            elif 'metric_entity' in entity_data:
                entity_type = 'Metric'
                entity_type_counts['Metric'] += 1
            else:
                entity_type_counts['Unknown'] += 1
            
            # 保存到数据库
            if task_id:
                if entity_type != 'Unknown':
                    db_manager.save_entity(entity_data, task_id, entity_type)
                    stored_count += 1
                else:
                    skipped_count += 1
        except Exception as e:
            logging.error(f"保存实体到数据库时出错: {str(e)}")
            skipped_count += 1
    
    # 保存所有关系到数据库
    relation_count = 0
    if relations and task_id:
        for relation in relations:
            try:
                if 'source_id' in relation and 'target_id' in relation:
                    db_manager.save_relation(relation, task_id)
                    relation_count += 1
            except Exception as e:
                logging.error(f"保存关系到数据库时出错: {str(e)}")
    
    # 更新处理状态
    if task_id:
        entities_msg = ", ".join([f"{key}: {value}" for key, value in entity_type_counts.items() if value > 0])
        message = f"处理完成，共提取 {len(entities)} 个实体 ({entities_msg})，{relation_count} 条关系，保存 {stored_count} 项，跳过 {skipped_count} 项"
        db_manager.update_processing_status(
            task_id=task_id,
            current_stage='处理完成',
            progress=1.0,
            message=message,
            status='completed'
        )
    
    logging.info(f"处理完成，共提取 {len(entities)} 个实体，{relation_count} 条关系")
    return entities, relations

def extract_relationships_with_context(entities, pdf_path, task_id=None, existing_relations=None):
    """
    提取实体间的关系，支持保留上下文进行断点续传
    
    Args:
        entities (list): 实体列表
        pdf_path (str): PDF文件路径，用于提取关系
        task_id (str, optional): 任务ID
        existing_relations (list, optional): 已有的关系数据，用于断点续传
        
    Returns:
        list: 提取的关系列表
    """
    if not entities or len(entities) < 2:
        logging.info("实体数量不足，无法提取关系")
        return []
    
    from app.modules.agents import extract_evolution_relations_from_paper
    
    # 更新处理状态
    if task_id:
        db_manager.update_processing_status(
            task_id=task_id,
            current_stage='提取关系',
            progress=0.8,
            message=f'正在提取 {len(entities)} 个实体之间的关系'
        )
    
    # 使用现有关系作为上下文继续提取
    if existing_relations:
        # 创建一个包含已存在关系信息的提示
        relations = extract_evolution_relations_from_paper(
            pdf_path, 
            entities, 
            task_id=task_id, 
            previous_relations=existing_relations
        )
    else:
        # 第一次提取
        relations = extract_evolution_relations_from_paper(
            pdf_path, 
            entities, 
            task_id=task_id
        )
    
    # 缓存关系数据
    if relations and task_id:
        # 保存关系到缓存
        basename = os.path.basename(pdf_path)
        filename_without_ext = os.path.splitext(basename)[0]
        
        # 移除可能的任务ID前缀
        if task_id and filename_without_ext.startswith(f"task_{task_id}_"):
            filename_without_ext = filename_without_ext[len(f"task_{task_id}_"):]
            
        relations_cache_file = os.path.join(Config.CACHE_DIR, f"relations_{filename_without_ext}.json")
        with open(relations_cache_file, 'w', encoding='utf-8') as f:
            json.dump(relations, f, ensure_ascii=False, indent=4)
        logging.info(f"已保存 {len(relations)} 条关系到缓存文件: {relations_cache_file}")
    
    return relations

def calculate_comparison_metrics(review_entities, citation_entities, relations):
    """
    计算各种比较指标
    
    Args:
        review_entities (list): 从综述中提取的实体列表
        citation_entities (list): 从引用文献中提取的实体列表
        relations (list): 提取的演化关系列表
    
    Returns:
        dict: 包含各种指标的字典
    """
    metrics = {
        'entity_stats': {},
        'relation_stats': {},
        'clustering': {}
    }
    
    # 分类计数实体
    review_algo_count = len([e for e in review_entities if 'algorithm_entity' in e])
    review_dataset_count = len([e for e in review_entities if 'dataset_entity' in e])
    review_metric_count = len([e for e in review_entities if 'metric_entity' in e])
    
    citation_algo_count = len([e for e in citation_entities if 'algorithm_entity' in e])
    citation_dataset_count = len([e for e in citation_entities if 'dataset_entity' in e])
    citation_metric_count = len([e for e in citation_entities if 'metric_entity' in e])
    
    # 提取实体ID
    review_algo_ids = set()
    for entity in review_entities:
        if 'algorithm_entity' in entity:
            algo_id = entity['algorithm_entity'].get('algorithm_id', entity['algorithm_entity'].get('entity_id', ''))
            if algo_id:
                review_algo_ids.add(algo_id.upper())
    
    citation_algo_ids = set()
    for entity in citation_entities:
        if 'algorithm_entity' in entity:
            algo_id = entity['algorithm_entity'].get('algorithm_id', entity['algorithm_entity'].get('entity_id', ''))
            if algo_id:
                citation_algo_ids.add(algo_id.upper())
    
    # 计算交集
    common_algo_ids = review_algo_ids.intersection(citation_algo_ids)
    
    # 计算实体精确率和召回率
    entity_precision = len(common_algo_ids) / citation_algo_count if citation_algo_count > 0 else 0
    entity_recall = len(common_algo_ids) / review_algo_count if review_algo_count > 0 else 0
    
    # 统计关系类型
    relation_types = {
        'improve_count': 0,
        'optimize_count': 0,
        'extend_count': 0,
        'replace_count': 0,
        'use_count': 0
    }
    
    for relation in relations:
        relation_type = relation.get('relation_type', '').lower()
        if 'improve' in relation_type:
            relation_types['improve_count'] += 1
        elif 'optimize' in relation_type:
            relation_types['optimize_count'] += 1
        elif 'extend' in relation_type:
            relation_types['extend_count'] += 1
        elif 'replace' in relation_type:
            relation_types['replace_count'] += 1
        elif 'use' in relation_type:
            relation_types['use_count'] += 1
    
    # 计算关系覆盖率（假设review_relations是标准覆盖关系数量）
    # 这里简化为假设每个实体应有至少一个关系
    estimated_standard_relations = review_algo_count
    relation_coverage = len(relations) / estimated_standard_relations if estimated_standard_relations > 0 else 0
    
    # 计算聚类指标
    # 构建算法关系图
    algorithm_clusters = build_algorithm_clusters(relations)
    
    # 假设的标准聚类（这里简化，实际应从金标准数据中提取）
    standard_clusters = []  # 应从金标准数据中提取
    
    # 计算聚类精确率和召回率
    cluster_precision = 0.0
    cluster_recall = 0.0
    
    # 填充实体统计
    metrics['entity_stats'] = {
        'algorithm_count_review': review_algo_count,
        'dataset_count_review': review_dataset_count,
        'metric_count_review': review_metric_count,
        'algorithm_count_citations': citation_algo_count,
        'dataset_count_citations': citation_dataset_count,
        'metric_count_citations': citation_metric_count,
        'entity_precision': entity_precision,
        'entity_recall': entity_recall
    }
    
    # 填充关系统计
    metrics['relation_stats'] = {
        **relation_types,
        'relation_coverage': relation_coverage
    }
    
    # 填充聚类统计
    metrics['clustering'] = {
        'clusters': algorithm_clusters,
        'precision': cluster_precision,
        'recall': cluster_recall
    }
    
    return metrics

def build_algorithm_clusters(relations):
    """
    根据算法之间的演化关系构建聚类
    
    Args:
        relations (list): 演化关系列表
        
    Returns:
        list: 聚类列表，每个聚类是一组相关的算法ID
    """
    # 使用图算法来找出连通分量
    import networkx as nx
    
    # 创建一个无向图
    G = nx.Graph()
    
    # 添加边（关系）
    for relation in relations:
        from_entity = relation.get('from_entity', '')
        to_entity = relation.get('to_entity', '')
        relation_type = relation.get('relation_type', '')
        
        # 只考虑算法之间的关系
        from_type = relation.get('from_entity_type', 'Algorithm')
        to_type = relation.get('to_entity_type', 'Algorithm')
        
        if from_type == 'Algorithm' and to_type == 'Algorithm':
            G.add_edge(from_entity, to_entity, type=relation_type)
    
    # 获取连通分量（聚类）
    clusters = list(nx.connected_components(G))
    
    # 转换为列表格式
    return [list(cluster) for cluster in clusters]

def save_progress(task_id, entities, processed_files, progress):
    """
    保存处理进度到文件
    
    Args:
        task_id (str): 处理任务ID
        entities (list): 已提取的实体列表
        processed_files (set): 已处理的文件集合
        progress (float): 当前进度(0-1)
    """
    progress_file = os.path.join(Config.CACHE_DIR, f"progress_{task_id}.json")
    try:
        progress_data = {
            'task_id': task_id,
            'entities': entities,
            'processed_files': list(processed_files),  # 转换set为list以便序列化
            'progress': progress,
            'timestamp': datetime.datetime.now().isoformat()
        }
        
        with open(progress_file, 'w', encoding='utf-8') as f:
            json.dump(progress_data, f, ensure_ascii=False, indent=2)
        
        logging.info(f"已保存进度，已处理 {len(processed_files)} 个文件，进度 {progress:.2f}")
    except Exception as e:
        logging.error(f"保存进度文件时出错: {str(e)}") 