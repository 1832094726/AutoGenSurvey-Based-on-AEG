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
            {prompt},
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
        'api_key': 'sk-e24e5880be494af4b08dbb2668918634',
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
    # 分割内容
    tables = content.split("### ")

    # 提取实体提取表
    entity_extraction_table = tables[1].split("### 进化模式表")[0].strip().splitlines()[4:]
    # 创建字典:列表里面放了多个字典
    entity_extraction_dict = []
    for row in entity_extraction_table:
        if row.strip():
            parts = row.split("|")
            if len(parts) >= 4:  # 确保有足够的列
                entity_extraction_dict.append({
                    "模式类型": parts[1].strip(),
                    "实体名称": parts[2].strip(),
                    "引文": parts[3].strip()
                })

    # 提取进化模式表
    evolution_pattern_table = tables[2].strip().splitlines()[4:]
    # 创建字典:列表里面放了多个字典
    evolution_pattern_dict = []
    for row in evolution_pattern_table:
        if row.strip():
            parts = row.split("|")
            if len(parts) >= 6:  # 确保有足够的列
                evolution_pattern_dict.append({
                    "实体A": parts[1].strip(),
                    "改进类型": parts[2].strip(),
                    "改进模式层级": parts[3].strip(),
                    "改进内容": parts[4].strip(),
                    "实体B": parts[5].strip()
                })

    return entity_extraction_dict, evolution_pattern_dict


def main():
    pdf_directory = 'pdfs'

    # 检查指定文件夹中的 PDF 文件
    if not os.path.exists(pdf_directory) or not any(f.endswith('.pdf') for f in os.listdir(pdf_directory)):
        logging.error("没有找到 PDF 文件，确保 pdfs 文件夹中有 PDF 文件。")
        return

    bot = setup_qwen_agent(pdf_directory)

    result1list = []
    result2list = []
    messages = []
    while True:
        try:
            query = input('用户请求: ')
            if query.lower() in ['退出', 'exit', 'quit']:
                break

            messages.append({'role': 'user', 'content': query})
            response = []
            for response in bot.run(messages=messages):
                print('机器人回应:')
                pprint.pprint(response, indent=2)
            messages.extend(response)
        except Exception as e:
            logging.error(f"处理请求时出错: {str(e)}")
            print("抱歉，处理您的请求时出现了错误。请再试一次。")

        content = response[0]["content"]
        entity_extraction_dict, evolution_pattern_dict = table_string_to_dict(content)

        # 把entity_extraction_dict, evolution_pattern_dict写入json文件
        # 可以编号后都放到一个文件夹里面，以免中途出错，前功尽弃
        with open('entity_extraction_dict.json', 'w', encoding='utf-8') as f:
            json.dump(entity_extraction_dict, f, ensure_ascii=False, indent=4)
        with open('evolution_pattern_dict.json', 'w', encoding='utf-8') as f:
            json.dump(evolution_pattern_dict, f, ensure_ascii=False, indent=4)

        # 将每次的结果列表[{}, {}, {}],追加到总的列表里中[{}, {}, {}]
        result1list.extend(entity_extraction_dict)  # 使用 extend 将列表中的元素追加
        print("result1list:", result1list)
        result2list.extend(evolution_pattern_dict)
        print("result2list:", result2list)

    # 总的列表解析后写入excel
    df1 = pd.DataFrame(result1list)
    df2 = pd.DataFrame(result2list)

    df1.to_excel('entity_extraction_dict.xlsx', index=False)
    df2.to_excel('evolution_pattern_dict.xlsx', index=False)
    print("数据已写入到 entity_extraction_dict.xlsx 和 evolution_pattern_dict.xlsx 文件中。")


if __name__ == "__main__":
    main()
