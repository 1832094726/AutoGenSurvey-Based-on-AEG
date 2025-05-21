import sys
import logging
import json

# 设置日志级别
logging.basicConfig(level=logging.INFO)

# 添加当前目录到sys.path
sys.path.append('.')

# 导入所需模块
from app.modules.knowledge_graph import build_knowledge_graph
from app.modules.db_manager import db_manager

def main():
    # 从数据库获取所有实体和关系
    logging.info("从数据库获取实体...")
    entities = db_manager.get_all_entities()
    logging.info(f"获取到 {len(entities)} 个实体")
    
    logging.info("从数据库获取关系...")
    relations = db_manager.get_all_relations()
    logging.info(f"获取到 {len(relations)} 个关系")
    
    # 检查关系格式
    if relations:
        logging.info(f"示例关系: {json.dumps(relations[0], indent=2)}")
        # 检查是否有from_entity和to_entity字段
        if 'from_entity' in relations[0] and 'to_entity' in relations[0]:
            logging.info("关系使用正确的数据库格式")
        else:
            logging.warning("关系数据格式可能不符合数据库要求")
    
    # 构建知识图谱
    logging.info("开始构建知识图谱...")
    try:
        graph = build_knowledge_graph(entities, relations)
        logging.info(f"成功构建图谱: 节点数量 {len(graph.nodes)}, 边数量 {len(graph.edges)}")
        
        # 显示一些节点和边
        if graph.nodes:
            logging.info(f"部分节点: {list(graph.nodes)[:5]}")
        if graph.edges:
            logging.info(f"部分边: {list(graph.edges)[:5]}")
            
        return True
    except Exception as e:
        logging.error(f"构建图谱时出错: {str(e)}", exc_info=True)
        return False
    
if __name__ == "__main__":
    success = main()
    if success:
        print("测试成功!")
    else:
        print("测试失败!")
        sys.exit(1) 