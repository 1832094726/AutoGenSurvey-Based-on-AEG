import re

def check_extraction_complete(text):
    """
    检查API响应中是否包含完成标志
    
    Args:
        text (str): API响应文本
        
    Returns:
        bool: 是否已完成提取
    """
    # 寻找完成标志
    completion_patterns = [
        r'EXTRACTION_COMPLETE:\s*true',
        r'extraction_complete"?\s*:\s*true',
        r'{"extraction_complete"?\s*:\s*true}',
        r'提取完成'
    ]
    
    for pattern in completion_patterns:
        if re.search(pattern, text, re.IGNORECASE):
            return True
            
    # 检查是否明确未完成
    incomplete_patterns = [
        r'EXTRACTION_COMPLETE:\s*false',
        r'extraction_complete"?\s*:\s*false',
        r'{"extraction_complete"?\s*:\s*false}',
        r'需要继续提取',
        r'提取未完成'
    ]
    
    for pattern in incomplete_patterns:
        if re.search(pattern, text, re.IGNORECASE):
            return False
            
    # 默认为未完成
    return False

# 测试
with open(r"E:\program development\Automatic Generation of AI Algorithm Reviews Based on Algorithmic Evolution Knowledge\data\cache\test\original_text_1748152579.txt", "r", encoding="utf-8") as file:
    text = file.read()
print(check_extraction_complete(text))
