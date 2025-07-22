# 调试信息记录器
# 文件：app/modules/debug_logger.py

import os
import json
import logging
from datetime import datetime
from typing import Dict, Any, Optional
from app.config import Config

class DebugLogger:
    """论文分析调试信息记录器"""
    
    def __init__(self, analysis_id: str):
        self.analysis_id = analysis_id
        self.debug_dir = os.path.join(Config.CACHE_DIR, f"debug_{analysis_id}")
        self.ensure_debug_dir()
        
        # 创建步骤计数器
        self.step_counter = 0
        
        # 创建主调试文件
        self.debug_log_path = os.path.join(self.debug_dir, "debug_log.json")
        self.init_debug_log()
    
    def ensure_debug_dir(self):
        """确保调试目录存在"""
        if not os.path.exists(self.debug_dir):
            os.makedirs(self.debug_dir)
            logging.info(f"创建调试目录: {self.debug_dir}")
    
    def init_debug_log(self):
        """初始化调试日志文件"""
        debug_info = {
            "analysis_id": self.analysis_id,
            "start_time": datetime.now().isoformat(),
            "steps": [],
            "summary": {
                "total_steps": 0,
                "total_llm_calls": 0,
                "total_tokens_used": 0,
                "errors": []
            }
        }
        
        with open(self.debug_log_path, 'w', encoding='utf-8') as f:
            json.dump(debug_info, f, ensure_ascii=False, indent=2)
    
    def log_step(self, step_name: str, step_type: str, description: str, **kwargs):
        """记录分析步骤"""
        self.step_counter += 1
        
        step_info = {
            "step_number": self.step_counter,
            "step_name": step_name,
            "step_type": step_type,  # prompt, llm_call, processing, result
            "description": description,
            "timestamp": datetime.now().isoformat(),
            "details": kwargs
        }
        
        # 更新主调试文件
        with open(self.debug_log_path, 'r', encoding='utf-8') as f:
            debug_info = json.load(f)
        
        debug_info["steps"].append(step_info)
        debug_info["summary"]["total_steps"] = self.step_counter
        
        with open(self.debug_log_path, 'w', encoding='utf-8') as f:
            json.dump(debug_info, f, ensure_ascii=False, indent=2)
        
        logging.info(f"调试步骤 {self.step_counter}: {step_name}")
    
    def save_prompt(self, step_name: str, prompt: str, model_name: str, **kwargs):
        """保存提示词"""
        filename = f"step_{self.step_counter:02d}_{step_name}_prompt.txt"
        filepath = os.path.join(self.debug_dir, filename)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(f"=== 提示词信息 ===\n")
            f.write(f"步骤: {step_name}\n")
            f.write(f"模型: {model_name}\n")
            f.write(f"时间: {datetime.now().isoformat()}\n")
            f.write(f"长度: {len(prompt)} 字符\n")
            f.write(f"\n=== 提示词内容 ===\n")
            f.write(prompt)
            
            if kwargs:
                f.write(f"\n\n=== 额外参数 ===\n")
                for key, value in kwargs.items():
                    f.write(f"{key}: {value}\n")
        
        self.log_step(
            step_name=f"{step_name}_prompt",
            step_type="prompt",
            description=f"保存{step_name}的提示词",
            model_name=model_name,
            prompt_length=len(prompt),
            prompt_file=filename,
            **kwargs
        )
    
    def save_llm_response(self, step_name: str, response: str, model_name: str, **kwargs):
        """保存大模型响应"""
        filename = f"step_{self.step_counter:02d}_{step_name}_response.txt"
        filepath = os.path.join(self.debug_dir, filename)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(f"=== 响应信息 ===\n")
            f.write(f"步骤: {step_name}\n")
            f.write(f"模型: {model_name}\n")
            f.write(f"时间: {datetime.now().isoformat()}\n")
            f.write(f"长度: {len(response)} 字符\n")
            f.write(f"\n=== 响应内容 ===\n")
            f.write(response)
            
            if kwargs:
                f.write(f"\n\n=== 额外信息 ===\n")
                for key, value in kwargs.items():
                    f.write(f"{key}: {value}\n")
        
        # 更新LLM调用统计
        with open(self.debug_log_path, 'r', encoding='utf-8') as f:
            debug_info = json.load(f)
        
        debug_info["summary"]["total_llm_calls"] += 1
        
        with open(self.debug_log_path, 'w', encoding='utf-8') as f:
            json.dump(debug_info, f, ensure_ascii=False, indent=2)
        
        self.log_step(
            step_name=f"{step_name}_response",
            step_type="llm_call",
            description=f"保存{step_name}的大模型响应",
            model_name=model_name,
            response_length=len(response),
            response_file=filename,
            **kwargs
        )
    
    def save_json_result(self, step_name: str, data: Dict[Any, Any], description: str = ""):
        """保存JSON格式的结果"""
        filename = f"step_{self.step_counter:02d}_{step_name}_result.json"
        filepath = os.path.join(self.debug_dir, filename)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        self.log_step(
            step_name=f"{step_name}_result",
            step_type="result",
            description=description or f"保存{step_name}的JSON结果",
            result_file=filename,
            data_keys=list(data.keys()) if isinstance(data, dict) else "non-dict",
            data_size=len(str(data))
        )
    
    def save_graph_data(self, step_name: str, graph_data: Dict[Any, Any], description: str = ""):
        """保存图数据"""
        filename = f"step_{self.step_counter:02d}_{step_name}_graph.json"
        filepath = os.path.join(self.debug_dir, filename)
        
        # 保存图数据
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(graph_data, f, ensure_ascii=False, indent=2)
        
        # 如果有连通图信息，保存可视化数据
        if 'connected_components' in graph_data:
            viz_filename = f"step_{self.step_counter:02d}_{step_name}_graph_viz.txt"
            viz_filepath = os.path.join(self.debug_dir, viz_filename)
            
            with open(viz_filepath, 'w', encoding='utf-8') as f:
                f.write(f"=== 连通图分析 ===\n")
                f.write(f"步骤: {step_name}\n")
                f.write(f"时间: {datetime.now().isoformat()}\n")
                f.write(f"连通分量数量: {len(graph_data.get('connected_components', []))}\n\n")
                
                for i, component in enumerate(graph_data.get('connected_components', [])):
                    f.write(f"=== 连通分量 {i+1} ===\n")
                    f.write(f"节点数量: {len(component.get('nodes', []))}\n")
                    f.write(f"边数量: {len(component.get('edges', []))}\n")
                    f.write(f"节点列表: {', '.join(component.get('nodes', []))}\n\n")
        
        self.log_step(
            step_name=f"{step_name}_graph",
            step_type="processing",
            description=description or f"保存{step_name}的图数据",
            graph_file=filename,
            nodes_count=len(graph_data.get('nodes', [])),
            edges_count=len(graph_data.get('edges', [])),
            components_count=len(graph_data.get('connected_components', []))
        )
    
    def save_error(self, step_name: str, error: Exception, context: str = ""):
        """保存错误信息"""
        filename = f"step_{self.step_counter:02d}_{step_name}_error.txt"
        filepath = os.path.join(self.debug_dir, filename)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(f"=== 错误信息 ===\n")
            f.write(f"步骤: {step_name}\n")
            f.write(f"时间: {datetime.now().isoformat()}\n")
            f.write(f"错误类型: {type(error).__name__}\n")
            f.write(f"错误消息: {str(error)}\n")
            f.write(f"上下文: {context}\n")
            
            # 如果有traceback信息
            import traceback
            f.write(f"\n=== 详细错误信息 ===\n")
            f.write(traceback.format_exc())
        
        # 更新错误统计
        with open(self.debug_log_path, 'r', encoding='utf-8') as f:
            debug_info = json.load(f)
        
        debug_info["summary"]["errors"].append({
            "step": step_name,
            "error_type": type(error).__name__,
            "error_message": str(error),
            "timestamp": datetime.now().isoformat()
        })
        
        with open(self.debug_log_path, 'w', encoding='utf-8') as f:
            json.dump(debug_info, f, ensure_ascii=False, indent=2)
        
        self.log_step(
            step_name=f"{step_name}_error",
            step_type="error",
            description=f"{step_name}发生错误",
            error_type=type(error).__name__,
            error_message=str(error),
            error_file=filename,
            context=context
        )
    
    def finalize(self, final_result: Optional[Dict[Any, Any]] = None):
        """完成调试记录"""
        # 保存最终结果
        if final_result:
            self.save_json_result("final_result", final_result, "最终分析结果")
        
        # 更新调试日志
        with open(self.debug_log_path, 'r', encoding='utf-8') as f:
            debug_info = json.load(f)
        
        debug_info["end_time"] = datetime.now().isoformat()
        debug_info["summary"]["total_steps"] = self.step_counter
        
        with open(self.debug_log_path, 'w', encoding='utf-8') as f:
            json.dump(debug_info, f, ensure_ascii=False, indent=2)
        
        # 创建调试摘要
        self.create_debug_summary()
        
        logging.info(f"调试信息记录完成，保存在: {self.debug_dir}")
    
    def create_debug_summary(self):
        """创建调试摘要"""
        summary_path = os.path.join(self.debug_dir, "README.md")
        
        with open(self.debug_log_path, 'r', encoding='utf-8') as f:
            debug_info = json.load(f)
        
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write(f"# 论文分析调试信息\n\n")
            f.write(f"**分析ID**: {self.analysis_id}\n")
            f.write(f"**开始时间**: {debug_info.get('start_time', 'N/A')}\n")
            f.write(f"**结束时间**: {debug_info.get('end_time', 'N/A')}\n")
            f.write(f"**总步骤数**: {debug_info['summary']['total_steps']}\n")
            f.write(f"**LLM调用次数**: {debug_info['summary']['total_llm_calls']}\n")
            f.write(f"**错误数量**: {len(debug_info['summary']['errors'])}\n\n")
            
            f.write(f"## 文件说明\n\n")
            f.write(f"- `debug_log.json`: 完整的调试日志\n")
            f.write(f"- `step_XX_*_prompt.txt`: 各步骤的提示词\n")
            f.write(f"- `step_XX_*_response.txt`: 大模型响应\n")
            f.write(f"- `step_XX_*_result.json`: 处理结果\n")
            f.write(f"- `step_XX_*_graph.json`: 图数据\n")
            f.write(f"- `step_XX_*_error.txt`: 错误信息\n\n")
            
            f.write(f"## 步骤概览\n\n")
            for step in debug_info.get('steps', []):
                f.write(f"**步骤 {step['step_number']}**: {step['step_name']} ({step['step_type']})\n")
                f.write(f"- 描述: {step['description']}\n")
                f.write(f"- 时间: {step['timestamp']}\n\n")
            
            if debug_info['summary']['errors']:
                f.write(f"## 错误信息\n\n")
                for error in debug_info['summary']['errors']:
                    f.write(f"- **{error['step']}**: {error['error_type']} - {error['error_message']}\n")
