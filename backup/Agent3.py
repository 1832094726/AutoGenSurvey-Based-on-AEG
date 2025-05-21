import os
import logging
import arxiv
import pprint
import json
import re
from qwen_agent.agents import Assistant
from qwen_agent.tools.base import BaseTool, register_tool
from qwen_agent.agents.doc_qa import ParallelDocQA
import pandas as pd
# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# ## 通过千问Qwen-Long的调用示例：

# from pathlib import Path

# from openai import OpenAI

# client = OpenAI(
#     api_key="sk-0fe80fc99c3045dfaa4c2921910245c1",  # 替换成实际DashScope的API_KEY
#     base_url="https://dashscope.aliyuncs.com/compatible-mode/v1", # 填写DashScope服务的base_url
# )

# # data.pdf 是一个示例文件

# file = client.files.create(file=Path("data.pdf"), purpose="file-extract")

# completion = client.chat.completions.create(
#     model="qwen-long",
#     messages=[
#         {
#             'role': 'system',
#             'content': 'You are a helpful assistant.'
#         },
#         {
#             'role': 'system',
#             'content': f'fileid://{file.id}'
#         },
#         {
#             'role': 'user',
#             'content': '这篇文章讲了什么？'
#         }
#     ],
#     stream=False
# )

# print(completion.choices[0].message.dict())

# Add a new tool to the agent named'elements_gen'
@register_tool('elements_gen')
class SueveyElementGen(BaseTool):
    description = 'Paper Elements提取，输入文本描述，返回基于文本信息的文章内容。'
    parameters = [{
        'name': 'prompt',
        'type': 'string',
        'description': '需要得到的文章的内容',
        'required': True
    }]

    def call(self, params: str, **kwargs) -> str:
        import json5
        import urllib.parse
        prompt = json5.loads(params)['prompt']
        prompt = urllib.parse.quote(prompt)
        return json5.dumps(
            [prompt],
            ensure_ascii=False)

def download_arxiv_pdfs(query, max_results, directory='pdfs'):
    if not os.path.exists(directory):
        os.makedirs(directory)

    client = arxiv.Client()
    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.Relevance
    )

    for result in client.results(search):
        try:
            filename = f"{directory}/{result.get_short_id()}.pdf"
            result.download_pdf(filename=filename)
            logging.info(f"Downloaded: {filename}")
        except Exception as e:
            logging.error(f"Error downloading {result.get_short_id()}: {str(e)}")


def setup_qwen_agent(pdf_directory='pdfs'):
    llm_cfg = {
        'model': 'qwen-long',
        'model_server': 'https://dashscope.aliyuncs.com/compatible-mode/v1',
        'api_key': 'sk-0fe80fc99c3045dfaa4c2921910245c1',
        # 'base_url': "https://dashscope.aliyuncs.com/compatible-mode/v1",
        'generate_cfg': {
            'top_p': 0.7
        }
    }

    system_instruction = '''你是一个乐于助人的AI助手。
    在收到用户的请求后，你应该：
    - 首先分析论文，提取出关键改进点。
    - 然后识别出论文之间的改进关系。
    - 最后生成要点和论文之间的关系图
    你总是用中文回复用户。'''

    tools = ['elements_gen', 'code_interpreter']
    files = [os.path.join(pdf_directory, f) for f in os.listdir(pdf_directory) if f.endswith('.pdf')]

    return Assistant(llm=llm_cfg,
                     system_message=system_instruction,
                     function_list=tools,
                     files=files)

def table_string_to_dict(content):
    """
    通用表格提取函数，能够从文本中提取表格数据
    """
    # 实体提取表可能的标识符
    entity_table_patterns = [
        "表格1:", "表格 1:", "表格1：", "表格 1：", 
        "实体提取表", "模式类型，实体名称，引文"
    ]
    
    # 进化模式表可能的标识符
    evolution_table_patterns = [
        "表格2:", "表格 2:", "表格2：", "表格 2：", 
        "进化模式表", "进化模式表格"
    ]
    
    entity_extraction_dict = []
    evolution_pattern_dict = []
    
    # 分割内容为行
    lines = content.splitlines()
    
    # 查找表格
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        
        # 查找第一个表格(实体提取表)
        entity_table_found = False
        for pattern in entity_table_patterns:
            if pattern in line:
                entity_table_found = True
                i += 1
                # 跳过表头行
                while i < len(lines) and ('|' not in lines[i] or '模式类型' not in lines[i]):
                    i += 1
                
                if i < len(lines):
                    i += 1  # 跳过表头行
                
                # 跳过分隔行
                if i < len(lines) and '|--' in lines[i]:
                    i += 1
                
                # 提取表格数据
                while i < len(lines) and '|' in lines[i] and len(lines[i].strip()) > 1:
                    row = lines[i].strip()
                    parts = row.split('|')
                    if len(parts) >= 4:
                        entity_extraction_dict.append({
                            "模式类型": parts[1].strip(),
                            "实体名称": parts[2].strip(),
                            "引文": parts[3].strip()
                        })
                    i += 1
                break
        
        if not entity_table_found:
            i += 1
            continue
        
        # 查找第二个表格(进化模式表)
        while i < len(lines):
            line = lines[i].strip()
            evolution_table_found = False
            
            for pattern in evolution_table_patterns:
                if pattern in line:
                    evolution_table_found = True
                    i += 1
                    # 跳过表头行
                    while i < len(lines) and ('|' not in lines[i] or '实体A' not in lines[i]):
                        i += 1
                    
                    if i < len(lines):
                        i += 1  # 跳过表头行
                    
                    # 跳过分隔行
                    if i < len(lines) and '|--' in lines[i]:
                        i += 1
                    
                    # 提取表格数据
                    while i < len(lines) and '|' in lines[i] and len(lines[i].strip()) > 1:
                        row = lines[i].strip()
                        parts = row.split('|')
                        if len(parts) >= 6:
                            evolution_pattern_dict.append({
                                "实体A": parts[1].strip(),
                                "改进类型": parts[2].strip(),
                                "改进模式层级": parts[3].strip(),
                                "改进内容": parts[4].strip(),
                                "实体B": parts[5].strip()
                            })
                        i += 1
                    break
            
            if evolution_table_found:
                break
            
            i += 1
    
    print(f"提取到 {len(entity_extraction_dict)} 个实体和 {len(evolution_pattern_dict)} 个进化模式")
    return entity_extraction_dict, evolution_pattern_dict


def main():
    pdf_directory = 'pdfs'

    # 确保文件夹存在
    if not os.path.exists(pdf_directory):
        os.makedirs(pdf_directory)

    # 检查指定文件夹中的 PDF 文件
    if not any(f.endswith('.pdf') for f in os.listdir(pdf_directory)):
        logging.warning("没有找到 PDF 文件，将使用测试示例进行处理。")
        # 创建一个简单的测试文件
        with open(os.path.join(pdf_directory, 'test.pdf'), 'w') as f:
            f.write("This is a test PDF file for demonstration purposes.")

    bot = setup_qwen_agent(pdf_directory)

    result1list = []
    result2list = []
    messages = []
    
    query="""你是一名研究人员。请你阅读文献列表[ ]后，用原文英文生成。按照算法进化模式[组件，架构，特征，方法，任务，评估]（每个模式下又有细分方面，例如Components：具体组件或模块的优化主要两方面，layer和机制如门机制。Architecture:   算法整体结构的创新包含组件与连接方式。Methodology: 算法的训练和优化例如参数挑战和训练策略。Features:输入数据的处理方式，算法更好地理解、利用和数据Task：针对特定任务的定制优化）
    完成以下任务。
    1 提取每个文献中各模式实体名称，你可以分段提取，尽可能完整的提取实体名称，并标注引文，需要着重关注每个引文相关的上下文，尤其注意带有大写或代表性名称，表格内或加黑的实体[例子:RNN,CNN,LSTM等]。尤其注意请生成全部实体，例如如果有130个引文，尽量生成120个实体。
    生成表格[模式类型，实体名称，引文（若本文实体则引文为本文）]，实体名称只要一个实体名称，不需要多余的描述。
    2 根据实体名称生成各模式算法之间的改进模式，每对实体之间的改进类型和对应的改进词特征[improve，enhance，replace，introduce，optimize，boost]等。
    生成进化模式表格【实体A,改进类型，改进模式层级，改进内容，实体B】。"""
    
    messages.append({'role': 'user', 'content': query})
    response = []
    
    try:
        for response in bot.run(messages=messages):
            print('机器人回应:')
            pprint.pprint(response, indent=2)
        
        if response:
            content = response[0]["content"]
            print("开始提取表格数据...")
            entity_extraction_dict, evolution_pattern_dict = table_string_to_dict(content)
            
            # 把entity_extraction_dict, evolution_pattern_dict写入json文件
            with open('entity_extraction_dict.json', 'w', encoding='utf-8') as f:
                json.dump(entity_extraction_dict, f, ensure_ascii=False, indent=4)
            with open('evolution_pattern_dict.json', 'w', encoding='utf-8') as f:
                json.dump(evolution_pattern_dict, f, ensure_ascii=False, indent=4)
            
            # 将每次的结果列表追加到总的列表里中
            result1list.extend(entity_extraction_dict)
            print("result1list:", result1list)
            result2list.extend(evolution_pattern_dict)
            print("result2list:", result2list)
        else:
            logging.error("没有收到API响应或响应为空")
        
    except Exception as e:
        logging.error(f"处理请求时出错: {str(e)}")
        print(f"抱歉，处理您的请求时出现了错误: {str(e)}")

    # 总的列表解析后写入excel
    if result1list:
        df1 = pd.DataFrame(result1list)
        df1.to_excel('entity_extraction_dict.xlsx', index=False)
        print("实体提取数据已写入到 entity_extraction_dict.xlsx 文件中。")
    else:
        print("没有提取到实体数据，无法生成entity_extraction_dict.xlsx文件。")
        
    if result2list:
        df2 = pd.DataFrame(result2list)
        df2.to_excel('evolution_pattern_dict.xlsx', index=False)
        print("进化模式数据已写入到 evolution_pattern_dict.xlsx 文件中。")
    else:
        print("没有提取到进化模式数据，无法生成evolution_pattern_dict.xlsx文件。")


if __name__ == "__main__":
    main()
