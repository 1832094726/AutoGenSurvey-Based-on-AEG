#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import logging
import json
import re
from pathlib import Path

# 设置项目根目录
current_dir = Path(__file__).resolve().parent
ROOT_DIR = current_dir.parent
sys.path.insert(0, str(ROOT_DIR))

# 手动定义函数而不是导入，避免复杂的导入依赖
def check_extraction_complete(content):
    """
    检查提取响应中是否包含完成标志
    
    Args:
        content (str): API响应内容
    
    Returns:
        bool: 提取是否完成
    """
    # 使用更强大的模式匹配，适应不同格式的完成标志
    patterns = [
        r'EXTRACTION_COMPLETE:\s*true',
        r'EXTRACTION[_\s-]*COMPLETE\s*[:：]\s*true',
        r'提取\s*完成\s*[:：]\s*true',
        r'完成\s*提取\s*[:：]\s*true',
        r'提取.*?已.*?完成',
        r'已.*?完成.*?提取'
    ]
    
    # 检查所有可能的模式
    is_complete = False
    for pattern in patterns:
        if re.search(pattern, content.lower()):
            is_complete = True
            break
    
    # 同时检查是否包含未完成标志
    incomplete_patterns = [
        r'EXTRACTION_COMPLETE:\s*false',
        r'EXTRACTION[_\s-]*COMPLETE\s*[:：]\s*false',
        r'提取\s*未\s*完成',
        r'需要\s*继续\s*提取'
    ]
    
    # 如果同时找到完成和未完成标志，以未完成为准
    for pattern in incomplete_patterns:
        if re.search(pattern, content.lower()):
            is_complete = False
            break
    
    logging.info(f"提取完成状态: {'完成' if is_complete else '未完成'}")
    return is_complete

def is_balanced(text):
    """检查文本中的括号是否匹配平衡"""
    stack = []
    brackets = {')': '(', '}': '{', ']': '['}
    
    for char in text:
        if char in '({[':
            stack.append(char)
        elif char in ')}]':
            if not stack or stack.pop() != brackets[char]:
                return False
    
    return len(stack) == 0

def is_json_valid(text):
    """检查文本是否是有效的JSON"""
    try:
        json.loads(text)
        return True
    except Exception:
        return False

def extract_json_from_text(text):
    """
    从文本中提取JSON内容
    
    Args:
        text (str): 包含JSON的文本
        
    Returns:
        str or None: 提取的JSON字符串，或者None
    """
    if not text:
        return None
    
    logging.debug(f"开始从文本中提取JSON，文本长度：{len(text)} 字符")
    
    # 首先，检查是否有标准的代码块格式
    json_match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', text, re.DOTALL)
    if json_match:
        extracted = json_match.group(1).strip()
        logging.debug(f"从代码块提取到JSON，长度: {len(extracted)}")
        
        # 验证提取的内容是否是有效的JSON
        if is_json_valid(extracted):
            return extracted
        else:
            logging.warning("从代码块提取的内容不是有效的JSON，尝试其他方法")
    
    # 尝试查找最长的有效JSON数组
    array_matches = re.findall(r'\[\s*\{[\s\S]*?\}\s*(?:,\s*\{[\s\S]*?\}\s*)*\]', text, re.DOTALL)
    if array_matches:
        # 按长度排序，取最长的匹配
        array_matches.sort(key=len, reverse=True)
        for potential_json in array_matches:
            if is_json_valid(potential_json):
                logging.debug(f"找到有效的JSON数组，长度: {len(potential_json)}")
                return potential_json
    
    # 尝试查找完整JSON对象
    object_matches = re.findall(r'(\{[\s\S]*\})', text, re.DOTALL)
    if object_matches:
        # 按长度排序，取最长的匹配
        object_matches.sort(key=len, reverse=True)
        for potential_json in object_matches:
            if is_balanced(potential_json) and is_json_valid(potential_json):
                logging.debug(f"找到有效的JSON对象，长度: {len(potential_json)}")
                return potential_json
    
    # 尝试更复杂的嵌套括号匹配
    # 对于数组
    if '[' in text and ']' in text:
        all_array_matches = []
        for match in re.finditer(r'\[', text):
            start_idx = match.start()
            try:
                # 使用括号栈找到匹配的结束位置
                stack = 1  # 已找到一个 '['
                for i in range(start_idx + 1, len(text)):
                    if text[i] == '[':
                        stack += 1
                    elif text[i] == ']':
                        stack -= 1
                        if stack == 0:  # 找到匹配的结束括号
                            potential_json = text[start_idx:i+1]
                            all_array_matches.append(potential_json)
                            break
            except Exception as e:
                logging.warning(f"尝试匹配数组时出错: {str(e)}")
        
        # 按长度排序，验证每个匹配
        all_array_matches.sort(key=len, reverse=True)
        for potential_json in all_array_matches:
            if is_json_valid(potential_json):
                logging.debug(f"通过括号匹配找到JSON数组，长度: {len(potential_json)}")
                return potential_json
    
    logging.warning("未能从文本中提取有效的JSON内容")
    return None

def test_json_extraction_and_completion_status(text):
    """
    测试JSON提取和完成状态检测功能
    
    Args:
        text (str): 要测试的文本
        
    Returns:
        tuple: (提取的JSON字符串, 完成状态)
    """
    logging.info("开始测试JSON提取和完成状态检测")
    
    # 检查完成状态
    is_complete = check_extraction_complete(text)
    logging.info(f"完成状态检测结果: {is_complete}")
    
    # 提取JSON
    json_str = extract_json_from_text(text)
    if json_str:
        logging.info(f"成功提取JSON，长度: {len(json_str)}")
        json_preview = json_str[:200] + "..." if len(json_str) > 200 else json_str
        logging.info(f"JSON预览: {json_preview}")
    else:
        logging.error("未能提取到有效的JSON")
        
    # 如果提取到了JSON，尝试解析
    entities = []
    if json_str:
        try:
            entities = json.loads(json_str)
            if not isinstance(entities, list):
                entities = [entities]
            logging.info(f"解析到 {len(entities)} 个实体")
        except Exception as e:
            logging.error(f"JSON解析错误: {str(e)}")
    
    return json_str, is_complete, entities

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('extraction_test.log', encoding='utf-8')
    ]
)

def test_from_file(file_path):
    """从文件测试提取功能"""
    logging.info(f"从文件 {file_path} 测试提取功能")
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
        logging.info(f"文件内容长度: {len(content)} 字符")
        json_str, is_complete, entities = test_json_extraction_and_completion_status(content)
        
        # 输出测试结果
        print("\n=== 测试结果 ===")
        print(f"完成状态: {'完成' if is_complete else '未完成'}")
        print(f"提取的实体数量: {len(entities)}")
        
        # 如果提取成功，保存结果
        if json_str:
            output_file = file_path + ".extracted.json"
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(entities, f, ensure_ascii=False, indent=2)
            print(f"已保存提取结果到: {output_file}")
            
    except Exception as e:
        logging.error(f"测试过程中出错: {str(e)}")
        import traceback
        logging.error(traceback.format_exc())

def test_from_string(text):
    """从字符串测试提取功能"""
    logging.info(f"从字符串测试提取功能，长度: {len(text)} 字符")
    
    try:
        json_str, is_complete, entities = test_json_extraction_and_completion_status(text)
        
        # 输出测试结果
        print("\n=== 测试结果 ===")
        print(f"完成状态: {'完成' if is_complete else '未完成'}")
        print(f"提取的实体数量: {len(entities)}")
        
        # 如果提取成功，保存结果
        if json_str:
            output_file = "string_test_result.json"
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(entities, f, ensure_ascii=False, indent=2)
            print(f"已保存提取结果到: {output_file}")
            
    except Exception as e:
        logging.error(f"测试过程中出错: {str(e)}")
        import traceback
        logging.error(traceback.format_exc())

def main():
    """主函数"""
    if len(sys.argv) < 2:
        print("用法: python test_extraction.py <文件路径或字符串>")
        sys.exit(1)
        
    arg = sys.argv[1]
    
    # 判断是文件路径还是字符串
    if os.path.exists(arg):
        test_from_file(arg)
    else:
        # 如果第一个参数不是文件，则将所有参数作为字符串处理
        text = ' '.join(sys.argv[1:])
        test_from_string(text)

if __name__ == "__main__":
    main() 