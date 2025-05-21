import logging
import networkx as nx
import matplotlib.pyplot as plt
import os
import json
from app.config import Config
from matplotlib.colors import LinearSegmentedColormap
import random

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def build_knowledge_graph(entities, relations):
    """
    根据实体和关系构建知识图谱
    
    Args:
        entities (List[Dict]): 实体列表，可能包含算法、数据集和评价指标
        relations (List[Dict]): 演化关系列表，使用数据库格式，每个关系包含from_entity和to_entity
        
    Returns:
        nx.DiGraph: 构建的有向图
    """
    # 创建有向图
    G = nx.DiGraph()
    
    # 添加节点
    node_ids = set()  # 用于跟踪已添加的节点ID
    
    for entity_wrapper in entities:
        # 处理算法实体
        if 'algorithm_entity' in entity_wrapper:
            entity = entity_wrapper['algorithm_entity']
            # 统一使用大写作为节点ID
            node_id = str(entity["algorithm_id"]).upper()
            G.add_node(
                node_id, 
                name=entity["name"],
                year=entity.get("year", ""),
                task=entity.get("task", ""),
                node_type="Algorithm"
            )
            node_ids.add(node_id)
        # 处理数据集实体
        elif 'dataset_entity' in entity_wrapper:
            entity = entity_wrapper['dataset_entity']
            # 统一使用大写作为节点ID
            node_id = str(entity["dataset_id"]).upper()
            G.add_node(
                node_id, 
                name=entity["name"],
                description=entity.get("description", ""),
                domain=entity.get("domain", ""),
                node_type="Dataset"
            )
            node_ids.add(node_id)
        # 处理评价指标实体
        elif 'metric_entity' in entity_wrapper:
            entity = entity_wrapper['metric_entity']
            # 统一使用大写作为节点ID
            node_id = str(entity["metric_id"]).upper()
            G.add_node(
                node_id, 
                name=entity["name"],
                description=entity.get("description", ""),
                category=entity.get("category", ""),
                node_type="Metric"
            )
            node_ids.add(node_id)
        # 处理直接的实体对象（可能在其他地方创建的实体）
        elif 'entity_id' in entity_wrapper:
            entity = entity_wrapper
            node_id = str(entity["entity_id"]).upper()
            G.add_node(
                node_id,
                name=entity.get("name", node_id),
                node_type=entity.get("entity_type", "Unknown")
            )
            node_ids.add(node_id)
    
    # 添加边之前检查是否有缺失的节点
    missing_nodes = set()
    
    for relation in relations:
        # 确保关系数据使用数据库格式（from_entity/to_entity）
        if "from_entity" not in relation or "to_entity" not in relation:
            logging.warning(f"关系数据格式不正确，缺少from_entity或to_entity字段: {relation}")
            continue
            
        # 统一使用大写转换
        from_id = str(relation["from_entity"]).upper()
        to_id = str(relation["to_entity"]).upper()
        
        if from_id not in node_ids:
            missing_nodes.add(from_id)
        if to_id not in node_ids:
            missing_nodes.add(to_id)
    
    # 添加缺失的节点
    for node_id in missing_nodes:
        G.add_node(
            node_id,
            name=node_id,  # 使用ID作为名称
            node_type="Unknown"  # 未知类型
        )
        node_ids.add(node_id)
        logging.warning(f"为关系添加了缺失的节点: {node_id}")
    
    # 添加边
    for relation in relations:
        # 确保关系数据使用数据库格式
        if "from_entity" not in relation or "to_entity" not in relation:
            continue
            
        # 统一使用大写转换
        from_id = str(relation["from_entity"]).upper()
        to_id = str(relation["to_entity"]).upper()
        
        # 确保relation_type存在
        relation_type = relation.get("relation_type")
        if not relation_type:
            logging.warning(f"关系缺少relation_type字段，跳过: {relation}")
            continue
        
        G.add_edge(
            from_id, 
            to_id,
            relation_type=relation_type,
            structure=relation.get("structure", ""),
            detail=relation.get("detail", ""),
            evidence=relation.get("evidence", ""),
            confidence=relation.get("confidence", 0.0)
        )
    
    return G

def update_knowledge_graph(graph, updates):
    """
    根据用户的修改更新知识图谱。
    
    Args:
        graph (nx.DiGraph): 当前知识图谱
        updates (Dict): 有关更新的字典，包括添加/删除的节点和边
        
    Returns:
        nx.DiGraph: 更新后的知识图谱
    """
    # 复制当前图以进行修改
    updated_graph = graph.copy()
    
    # 处理节点更新
    for node_update in updates.get('nodes', []):
        action = node_update.get('action', '')
        node_id = node_update.get('id', '')
        
        if not node_id:
            continue
            
        if action == 'add':
            # 添加节点
            label = node_update.get('label', node_id)
            entity_type = node_update.get('entity_type', 'Algorithm')
            
            updated_graph.add_node(node_id, label=label, entity_type=entity_type)
            
        elif action == 'update':
            # 更新节点属性
            if node_id in updated_graph:
                for key, value in node_update.items():
                    if key not in ['action', 'id']:
                        updated_graph.nodes[node_id][key] = value
                        
        elif action == 'delete':
            # 删除节点
            if node_id in updated_graph:
                updated_graph.remove_node(node_id)
    
    # 处理边更新
    for edge_update in updates.get('edges', []):
        action = edge_update.get('action', '')
        from_id = edge_update.get('from', '')
        to_id = edge_update.get('to', '')
        
        if not from_id or not to_id:
            continue
            
        if action == 'add':
            # 添加边
            # 确保节点存在
            if not updated_graph.has_node(from_id):
                updated_graph.add_node(from_id, label=from_id, entity_type="Unknown")
            if not updated_graph.has_node(to_id):
                updated_graph.add_node(to_id, label=to_id, entity_type="Unknown")
                
            relation_type = edge_update.get('relation_type', 'Unknown')
            structure = edge_update.get('structure', '')
            detail = edge_update.get('detail', '')
            evidence = edge_update.get('evidence', '')
            confidence = edge_update.get('confidence', 0.5)
            
            edge_label = f"{relation_type}"
            if structure:
                edge_label += f" on {structure}"
            if detail:
                edge_label += f"\n{detail}"
            
            updated_graph.add_edge(from_id, to_id, 
                                  label=edge_label,
                                  relation_type=relation_type,
                                  structure=structure,
                                  detail=detail,
                                  evidence=evidence,
                                  confidence=confidence)
                                  
        elif action == 'update':
            # 更新边属性
            if updated_graph.has_edge(from_id, to_id):
                for key, value in edge_update.items():
                    if key not in ['action', 'from', 'to']:
                        updated_graph[from_id][to_id][key] = value
                
                # 重新计算边标签
                relation_type = updated_graph[from_id][to_id].get('relation_type', 'Unknown')
                structure = updated_graph[from_id][to_id].get('structure', '')
                detail = updated_graph[from_id][to_id].get('detail', '')
                
                edge_label = f"{relation_type}"
                if structure:
                    edge_label += f" on {structure}"
                if detail:
                    edge_label += f"\n{detail}"
                
                updated_graph[from_id][to_id]['label'] = edge_label
                
        elif action == 'delete':
            # 删除边
            if updated_graph.has_edge(from_id, to_id):
                updated_graph.remove_edge(from_id, to_id)
    
    return updated_graph

def visualize_graph(graph, output_path=None, show=True):
    """
    可视化知识图谱
    
    Args:
        graph (nx.DiGraph): 知识图谱
        output_path (str, optional): 输出图像路径
        show (bool, optional): 是否显示图像
        
    Returns:
        None
    """
    # 如果图为空，则返回
    if len(graph) == 0:
        logging.warning("图为空，无法可视化")
        return
    
    # 创建绘图
    plt.figure(figsize=(20, 15))
    
    # 定义节点位置 - 使用spring布局
    pos = nx.spring_layout(graph, k=0.3, iterations=50)
    
    # 定义节点颜色映射
    node_colors = []
    for node in graph.nodes():
        if 'node_type' in graph.nodes[node]:
            if graph.nodes[node]['node_type'] == 'Algorithm':
                node_colors.append('lightblue')
            else:
                node_colors.append('lightgreen')
        else:
            node_colors.append('lightgray')
    
    # 绘制节点
    nx.draw_networkx_nodes(graph, pos, 
                          node_size=800, 
                          node_color=node_colors, 
                          alpha=0.8)
    
    # 绘制边
    edge_colors = []
    edge_widths = []
    
    for u, v, data in graph.edges(data=True):
        relation_type = data.get('relation_type', '')
        confidence = data.get('confidence', 0.5)
        
        # 根据关系类型设置颜色
        if relation_type == 'Improve':
            edge_colors.append('green')
        elif relation_type == 'Replace':
            edge_colors.append('red')
        elif relation_type == 'Extend':
            edge_colors.append('blue')
        else:
            edge_colors.append('gray')
        
        # 根据置信度设置宽度
        edge_widths.append(1 + 3 * confidence)
    
    nx.draw_networkx_edges(graph, pos,
                          width=edge_widths,
                          edge_color=edge_colors,
                          arrowsize=15,
                          alpha=0.7)
    
    # 添加节点标签
    labels = {}
    for node in graph.nodes():
        if 'name' in graph.nodes[node]:
            labels[node] = graph.nodes[node]['name']
        else:
            labels[node] = node
    
    nx.draw_networkx_labels(graph, pos, labels=labels, font_size=10, font_family='sans-serif')
    
    # 添加边标签
    edge_labels = {(u, v): data.get('relation_type', '') 
                  for u, v, data in graph.edges(data=True)}
    
    nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels, font_size=8)
    
    # 设置布局
    plt.axis('off')
    plt.tight_layout()
    
    # 保存图像
    if output_path:
        plt.savefig(output_path, format="png", dpi=300, bbox_inches='tight')
        logging.info(f"图像已保存到: {output_path}")
    
    # 显示图像
    if show:
        plt.show()
    
    plt.close()

def export_graph_to_json(graph, output_path):
    """
    将图导出为JSON格式
    
    Args:
        graph (nx.DiGraph): 知识图谱
        output_path (str): 输出文件路径
        
    Returns:
        dict: 导出的图数据字典
    """
    # 准备JSON数据
    json_data = {
        "nodes": [],
        "edges": []
    }
    
    # 添加节点
    for node, data in graph.nodes(data=True):
        node_id = str(node)  # 确保ID是字符串
        # 创建基本节点数据结构
        node_data = {
            "id": node_id,
            "label": data.get("name", node_id),
            "type": data.get("node_type", "Unknown")
        }
        
        # 添加所有属性到节点数据中
        for key, value in data.items():
            if key not in ["name"]:  # name已经映射到label
                node_data[key] = value
                
        json_data["nodes"].append(node_data)
    
    # 添加边
    for source, target, data in graph.edges(data=True):
        source_id = str(source)  # 确保ID是字符串
        target_id = str(target)  # 确保ID是字符串
        
        edge_data = {
            "source": source_id,
            "target": target_id,
            "label": data.get("relation_type", ""),
            "relation_type": data.get("relation_type", ""),
            "structure": data.get("structure", ""),
            "detail": data.get("detail", ""),
            "evidence": data.get("evidence", ""),
            "confidence": data.get("confidence", 0.0)
        }
        
        # 添加所有边属性
        for key, value in data.items():
            if key not in ["label", "relation_type", "structure", "detail", "evidence", "confidence"]:
                edge_data[key] = value
                
        json_data["edges"].append(edge_data)
    
    # 保存为JSON文件
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, ensure_ascii=False, indent=2)
    
    logging.info(f"图数据已导出到: {output_path}，共 {len(json_data['nodes'])} 个节点和 {len(json_data['edges'])} 条边")
    
    return json_data  # 返回导出的数据

def load_graph_from_json(input_path):
    """
    从JSON文件加载知识图谱。
    
    Args:
        input_path (str): 输入JSON文件的路径
        
    Returns:
        nx.DiGraph: 加载的知识图谱
    """
    try:
        with open(input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 创建图
        G = nx.DiGraph()
        
        # 添加节点
        for node_data in data.get("nodes", []):
            node_id = node_data.get("id", "")
            if node_id:
                label = node_data.get("label", node_id)
                entity_type = node_data.get("type", "Unknown")
                G.add_node(node_id, label=label, entity_type=entity_type)
        
        # 添加边
        for edge_data in data.get("edges", []):
            source = edge_data.get("source", "")
            target = edge_data.get("target", "")
            
            if source and target:
                label = edge_data.get("label", "")
                relation_type = edge_data.get("relation_type", "Unknown")
                structure = edge_data.get("structure", "")
                detail = edge_data.get("detail", "")
                evidence = edge_data.get("evidence", "")
                confidence = edge_data.get("confidence", 0.5)
                
                G.add_edge(source, target,
                          label=label,
                          relation_type=relation_type,
                          structure=structure,
                          detail=detail,
                          evidence=evidence,
                          confidence=confidence)
        
        logging.info(f"从 {input_path} 加载了知识图谱")
        return G
    except Exception as e:
        logging.error(f"加载知识图谱时出错: {str(e)}")
        return nx.DiGraph() 