import json
import networkx as nx
import matplotlib.pyplot as plt

# 示例：多篇论文的结构化 JSON 数据（你可以替换为文件读取）
data = [
    {
        "algorithm_entity": {
            "algorithm_id": "Zhang2016_TemplateSolver",
            "entity_type": "Algorithm",
            "name": "TemplateSolver",
            "year": 2016,
            "architecture": {
                "components": ["TemplateParser"],
                "mechanisms": ["Rule-based Matching"]
            },
            "evolution_relations": []
        }
    },
    {
        "algorithm_entity": {
            "algorithm_id": "Huang2017_NeuralSolver",
            "entity_type": "Algorithm",
            "name": "NeuralSolver",
            "year": 2017,
            "dataset": ["Math23K"],
            "architecture": {
                "components": ["Encoder", "Decoder"],
                "mechanisms": ["GRU"]
            },
            "evolution_relations": [
                {
                    "from_entity": "Zhang2016_TemplateSolver",
                    "to_entity": "Huang2017_NeuralSolver",
                    "relation_type": "Replace",
                    "structure": "Architecture.Mechanism",
                    "detail": "Template parser → neural encoder-decoder",
                    "evidence": "We replace Zhang's parser with a GRU-based encoder-decoder.",
                    "effect": "Improved adaptability to diverse structures.",
                    "confidence": 0.91
                },
                {
                    "from_entity": "Math23K",
                    "to_entity": "Huang2017_NeuralSolver",
                    "relation_type": "Use",
                    "structure": "Evaluation.Dataset",
                    "detail": "Math23K for training and testing",
                    "evidence": "We use Math23K as benchmark dataset.",
                    "effect": "Enables standardized evaluation.",
                    "confidence": 0.98
                },
                {
                    "from_entity": "Accuracy",
                    "to_entity": "Huang2017_NeuralSolver",
                    "relation_type": "Optimize",
                    "structure": "Evaluation.Metric",
                    "detail": "Achieved 84.2% accuracy",
                    "evidence": "Outperforms previous work with 84.2% accuracy.",
                    "effect": "Better accuracy on Math23K.",
                    "confidence": 0.93
                }
            ]
        }
    },
    {
        "algorithm_entity": {
            "algorithm_id": "Wang2018_MathDQN",
            "entity_type": "Algorithm",
            "name": "MathDQN",
            "year": 2018,
            "architecture": {
                "components": ["Encoder", "Decoder", "RL-Agent"],
                "mechanisms": ["Deep Q-Network"]
            },
            "evolution_relations": [
                {
                    "from_entity": "Huang2017_NeuralSolver",
                    "to_entity": "Wang2018_MathDQN",
                    "relation_type": "Improve",
                    "structure": "Architecture.Mechanism",
                    "detail": "GRU → Deep Q-Network",
                    "evidence": "We improve GRU by using DQN agent in decoding.",
                    "effect": "Improved action-based reasoning.",
                    "confidence": 0.94
                },
                {
                    "from_entity": "Accuracy",
                    "to_entity": "Wang2018_MathDQN",
                    "relation_type": "Optimize",
                    "structure": "Evaluation.Metric",
                    "detail": "Reached 86.5% accuracy",
                    "evidence": "Achieved 86.5% accuracy on Math23K.",
                    "effect": "State-of-the-art performance.",
                    "confidence": 0.96
                }
            ]
        }
    }
]

# 构建图
G = nx.DiGraph()

for entry in data:
    algo = entry["algorithm_entity"]
    algo_id = algo["algorithm_id"]
    algo_name = algo["name"]
    G.add_node(algo_id, label=algo_name, entity_type="Algorithm")

    for rel in algo.get("evolution_relations", []):
        from_id = rel["from_entity"]
        to_id = rel["to_entity"]

        # 自动添加非算法节点
        if not G.has_node(from_id):
            if "dataset" in from_id.lower():
                entity_type = "Dataset"
            elif from_id.lower() in ["accuracy", "f1", "bleu"]:
                entity_type = "Metric"
            else:
                entity_type = "Algorithm"
            G.add_node(from_id, label=from_id, entity_type=entity_type)

        label = f"{rel['relation_type']} on {rel['structure']} ({rel['detail']})\nEffect: {rel['effect']}"
        G.add_edge(from_id, to_id, label=label)

# 布局与绘图
pos = nx.spring_layout(G, seed=42)
plt.figure(figsize=(14, 10))

# 节点颜色
node_colors = []
for _, data in G.nodes(data=True):
    if data["entity_type"] == "Algorithm":
        node_colors.append("skyblue")
    elif data["entity_type"] == "Dataset":
        node_colors.append("lightgreen")
    elif data["entity_type"] == "Metric":
        node_colors.append("orange")
    else:
        node_colors.append("gray")

# 画节点和标签
nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=1800)
nx.draw_networkx_labels(G, pos, labels={n: G.nodes[n]["label"] for n in G.nodes()}, font_size=9)

# 画边和边标签
nx.draw_networkx_edges(G, pos, arrows=True, arrowstyle="->", width=2)
edge_labels = {(u, v): d["label"] for u, v, d in G.edges(data=True)}
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=7)

plt.title("Algorithm Evolution Roadmap (Multi-Entity)", fontsize=14)
plt.axis("off")
plt.tight_layout()
plt.show()
