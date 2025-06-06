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

def build_knowledge_graph(entities, relations, task_id=None):
    """
    根据实体和关系构建知识图谱并转换为前端可用的格式
    
    Args:
        entities (List[Dict]): 实体列表，可能包含算法、数据集和评价指标
        relations (List[Dict]): 演化关系列表，使用数据库格式，每个关系包含from_entity和to_entity
        task_id (str, optional): 任务ID，用于前端展示
        
    Returns:
        Dict: 包含节点和边的字典，格式为前端Cytoscape可用的格式
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
    
    # 转换为前端所需的格式
    # 节点集合
    nodes = []
    node_map = {}
    
    # 处理实体
    for entity in entities:
        if "algorithm_entity" in entity:
            entity_data = entity["algorithm_entity"]
            node_type = "Algorithm"
            entity_id = entity_data.get("algorithm_id") or entity_data.get("entity_id")
            name = entity_data.get("name", "")
        elif "dataset_entity" in entity:
            entity_data = entity["dataset_entity"]
            node_type = "Dataset"
            entity_id = entity_data.get("dataset_id") or entity_data.get("entity_id")
            name = entity_data.get("name", "")
        elif "metric_entity" in entity:
            entity_data = entity["metric_entity"]
            node_type = "Metric"
            entity_id = entity_data.get("metric_id") or entity_data.get("entity_id")
            name = entity_data.get("name", "")
        else:
            continue
            
        # 避免重复节点
        if entity_id in node_map:
            continue
            
        node = {
            "id": entity_id,
            "label": name,
            "type": node_type,
            "entity_type": node_type,
            "data": entity_data
        }
        nodes.append(node)
        node_map[entity_id] = node
        
    # 先收集关系中提到的所有实体ID
    missing_entity_ids = set()
    for relation in relations:
        from_entity = relation.get("from_entity")
        to_entity = relation.get("to_entity")
        
        if from_entity and from_entity not in node_map:
            missing_entity_ids.add(from_entity)
        if to_entity and to_entity not in node_map:
            missing_entity_ids.add(to_entity)
            
    # 为缺失的节点创建占位符节点
    for entity_id in missing_entity_ids:
        logging.info(f"为关系创建占位符节点: {entity_id}")
        node = {
            "id": entity_id,
            "label": entity_id,  # 使用ID作为标签
            "type": "Unknown",   # 类型未知
            "entity_type": "Unknown",   # 类型未知
            "data": {"name": entity_id, "placeholder": True, "entity_type": "Unknown"}
        }
        nodes.append(node)
        node_map[entity_id] = node
    
    # 处理关系，支持多关系边
    edges = []
    edge_map = {}  # 用于跟踪已创建的边
    
    skipped_count = 0
    for relation in relations:
        from_entity = relation.get("from_entity")
        to_entity = relation.get("to_entity")
        relation_type = relation.get("relation_type", "Unknown")
        
        # 跳过缺少起点或终点的关系
        if not from_entity or not to_entity:
            skipped_count += 1
            continue
            
        # 由于已经为所有缺失的节点创建了占位符，这个检查应该不会再跳过任何关系
        if from_entity not in node_map or to_entity not in node_map:
            logging.warning(f"关系节点仍然不存在: {from_entity} -> {to_entity}")
            skipped_count += 1
            continue
        
        # 构建唯一的边键
        edge_key = f"{from_entity}_{to_entity}"
        
        # 如果边已存在，添加到已有边的关系列表中
        if edge_key in edge_map:
            edge = edge_map[edge_key]
            relation_list = edge["data"]["relations"]
            
            # 添加新关系
            relation_list.append({
                "type": relation_type,
                "structure": relation.get("structure", ""),
                "detail": relation.get("detail", ""),
                "problem_addressed": relation.get("problem_addressed", ""),
                "evidence": relation.get("evidence", ""),
                "confidence": relation.get("confidence", 0.5)
            })
            
            # 更新边标签，包含所有关系类型
            relation_types = set([rel["type"] for rel in relation_list])
            edge["label"] = ", ".join(relation_types)
            
        else:
            # 创建新边
            edge = {
                "id": f"edge_{len(edges)}",
                "source": from_entity,
                "target": to_entity,
                "label": relation_type,
                "relation_type": relation_type,
                "data": {
                    "relations": [{
                        "type": relation_type,
                        "structure": relation.get("structure", ""),
                        "detail": relation.get("detail", ""),
                        "problem_addressed": relation.get("problem_addressed", ""),
                        "evidence": relation.get("evidence", ""),
                        "confidence": relation.get("confidence", 0.5)
                    }]
                }
            }
            edges.append(edge)
            edge_map[edge_key] = edge
    
    if skipped_count > 0:
        logging.warning(f"跳过了 {skipped_count} 条不完整的关系")
        
    # 返回包含原始网络图和前端格式数据的结果
    graph_data = {
        "nodes": nodes,
        "edges": edges,
        "task_id": task_id,
        "networkx_graph": G  # 保留networkx图，以便其他函数使用
    }
    
    logging.info(f"构建图形数据完成: {len(nodes)} 个节点, {len(edges)} 条边")
    return graph_data

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
        graph (nx.DiGraph或dict): 知识图谱或包含networkx_graph的字典
        output_path (str, optional): 输出图像路径
        show (bool, optional): 是否显示图像
        
    Returns:
        None
    """
    # 处理graph为字典的情况（新格式）
    if isinstance(graph, dict) and "networkx_graph" in graph:
        G = graph["networkx_graph"]
    elif isinstance(graph, nx.DiGraph):
        G = graph
    else:
        logging.warning("传入的图形数据格式不正确，无法可视化")
        return
    
    # 如果图为空，则返回
    if len(G) == 0:
        logging.warning("图为空，无法可视化")
        return
    
    # 创建绘图
    plt.figure(figsize=(20, 15))
    
    # 定义节点位置 - 使用spring布局
    pos = nx.spring_layout(G, k=0.3, iterations=50)
    
    # 定义节点颜色映射
    node_colors = []
    for node in G.nodes():
        if 'node_type' in G.nodes[node]:
            if G.nodes[node]['node_type'] == 'Algorithm':
                node_colors.append('lightblue')
            else:
                node_colors.append('lightgreen')
        else:
            node_colors.append('lightgray')
    
    # 绘制节点
    nx.draw_networkx_nodes(G, pos, 
                          node_size=800, 
                          node_color=node_colors, 
                          alpha=0.8)
    
    # 绘制边
    edge_colors = []
    edge_widths = []
    
    for u, v, data in G.edges(data=True):
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
    
    nx.draw_networkx_edges(G, pos,
                          width=edge_widths,
                          edge_color=edge_colors,
                          arrowsize=15,
                          alpha=0.7)
    
    # 添加节点标签
    labels = {}
    for node in G.nodes():
        if 'name' in G.nodes[node]:
            labels[node] = G.nodes[node]['name']
        else:
            labels[node] = node
    
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=10, font_family='sans-serif')
    
    # 添加边标签
    edge_labels = {(u, v): data.get('relation_type', '') 
                  for u, v, data in G.edges(data=True)}
    
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8)
    
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
    将图导出为JSON格式，供前端可视化使用
    
    Args:
        graph (nx.DiGraph或dict): 知识图谱或包含networkx_graph的字典
        output_path (str): 输出路径
        
    Returns:
        dict: 图的JSON表示
    """
    # 如果已经是新格式的数据字典，直接使用其中的nodes和edges
    if isinstance(graph, dict) and 'nodes' in graph and 'edges' in graph:
        data = {
            'nodes': graph['nodes'],
            'edges': graph['edges']
        }
        # 如果有task_id，也包含进去
        if 'task_id' in graph:
            data['task_id'] = graph['task_id']
        
        # 写入文件
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        logging.info(f"图数据已保存到: {output_path}")
        return data
    
    # 如果是networkx图，需要转换
    elif isinstance(graph, nx.DiGraph):
        nodes = []
        edges = []
        
        # 添加节点
        for node in graph.nodes():
            node_data = graph.nodes[node]
            node_type = node_data.get('node_type', 'Unknown')
            name = node_data.get('name', str(node))
            
            nodes.append({
                'id': str(node),
                'label': name,
                'type': node_type,
                'data': node_data
            })
        
        # 添加边
        edge_ids = 0
        for u, v, data in graph.edges(data=True):
            relation_type = data.get('relation_type', 'Unknown')
            structure = data.get('structure', '')
            detail = data.get('detail', '')
            evidence = data.get('evidence', '')
            confidence = data.get('confidence', 0.0)
            
            # 创建边对象
            edge = {
                'id': f'edge_{edge_ids}',
                'source': str(u),
                'target': str(v),
                'label': relation_type,
                'relation_type': relation_type,
                'data': {
                    'relations': [{
                        'type': relation_type,
                        'structure': structure,
                        'detail': detail,
                        'evidence': evidence,
                        'confidence': confidence
                    }]
                }
            }
            
            edges.append(edge)
            edge_ids += 1
        
        # 创建最终数据
        data = {
            'nodes': nodes,
            'edges': edges
        }
        
        # 写入文件
        with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        
                logging.info(f"图数据已保存到: {output_path}")
                return data
    
    else:
        logging.warning("无效的图格式，无法导出")
        return {'nodes': [], 'edges': []}

def load_graph_from_json(input_path):
    """
    从JSON文件加载图
    
    Args:
        input_path (str): JSON文件路径
        
    Returns:
        dict: 包含networkx_graph和前端数据的图结构
    """
    if not os.path.exists(input_path):
        logging.warning(f"找不到图文件: {input_path}")
        return None
        
    try:
        with open(input_path, 'r', encoding='utf-8') as f:
            graph_data = json.load(f)
        
        # 创建NetworkX图形
        G = nx.DiGraph()
        
        # 如果已经是新格式，则保留原始格式
        
        # 添加节点
        for node_data in graph_data.get('nodes', []):
            node_id = node_data.get('id')
            if not node_id:
                continue
                
            # 节点属性
            attrs = {}
            # 从data字段提取属性
            if 'data' in node_data and isinstance(node_data['data'], dict):
                for key, value in node_data['data'].items():
                    attrs[key] = value
                    
            # 顶层属性
            for key, value in node_data.items():
                if key not in ['id', 'data']:
                    attrs[key] = value
                    
            # 确保name和node_type存在
            if 'name' not in attrs and 'label' in node_data:
                attrs['name'] = node_data['label']
                
            if 'node_type' not in attrs and 'type' in node_data:
                attrs['node_type'] = node_data['type']
                
            # 添加节点
            G.add_node(node_id, **attrs)
            
        # 添加边
        for edge_data in graph_data.get('edges', []):
            source = edge_data.get('source')
            target = edge_data.get('target')
            
            if not source or not target:
                continue
                
            # 边属性
            attrs = {}
            
            # 从data和relations字段提取属性
            if 'data' in edge_data and isinstance(edge_data['data'], dict):
                # 处理多关系数据
                if 'relations' in edge_data['data'] and isinstance(edge_data['data']['relations'], list) and edge_data['data']['relations']:
                    relation = edge_data['data']['relations'][0]  # 取第一个关系
                    for key, value in relation.items():
                        attrs[key] = value
                
                # 其他data字段
                for key, value in edge_data['data'].items():
                    if key != 'relations':
                        attrs[key] = value
                        
            # 顶层属性
            for key, value in edge_data.items():
                if key not in ['source', 'target', 'id', 'data']:
                    attrs[key] = value
                    
            # 确保relation_type存在
            if 'relation_type' not in attrs and 'label' in edge_data:
                attrs['relation_type'] = edge_data['label']
                
            # 添加边
            G.add_edge(source, target, **attrs)
                
        # 返回完整的图数据结构
        result = {
            'nodes': graph_data.get('nodes', []),
            'edges': graph_data.get('edges', []),
            'networkx_graph': G,
        }
        
        # 如果有task_id，也包含进去
        if 'task_id' in graph_data:
            result['task_id'] = graph_data['task_id']
            
        logging.info(f"从 {input_path} 加载了图形数据，包含 {len(result['nodes'])} 个节点和 {len(result['edges'])} 条边")
        return result
        
    except Exception as e:
        logging.error(f"加载图形数据时出错: {str(e)}")
        import traceback
        logging.error(traceback.format_exc())
        return None 