"""
综述结果存储和管理系统
支持版本管理、多格式导出、结果检索等功能
"""

import logging
import os
import json
import hashlib
import shutil
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass, asdict
import sqlite3
from pathlib import Path

from app.modules.survey_formatter import SurveyFormatter, FormattingOptions

@dataclass
class SurveyRecord:
    """综述记录"""
    survey_id: str
    topic: str
    task_ids: List[str]
    generation_time: datetime
    content_hash: str
    file_paths: Dict[str, str]  # format -> file_path
    metadata: Dict[str, Any]
    version: int
    status: str  # draft, published, archived
    tags: List[str]

class SurveyStorageManager:
    """综述存储管理器"""
    
    def __init__(self, storage_root: str = "./survey_storage"):
        self.storage_root = Path(storage_root)
        self.storage_root.mkdir(exist_ok=True)
        
        self.db_path = self.storage_root / "surveys.db"
        self.files_dir = self.storage_root / "files"
        self.files_dir.mkdir(exist_ok=True)
        
        self.logger = logging.getLogger(__name__)
        self.formatter = SurveyFormatter()
        
        self._init_database()
    
    def _init_database(self):
        """初始化数据库"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS surveys (
                    survey_id TEXT PRIMARY KEY,
                    topic TEXT NOT NULL,
                    task_ids TEXT NOT NULL,
                    generation_time TEXT NOT NULL,
                    content_hash TEXT NOT NULL,
                    file_paths TEXT NOT NULL,
                    metadata TEXT NOT NULL,
                    version INTEGER NOT NULL DEFAULT 1,
                    status TEXT NOT NULL DEFAULT 'draft',
                    tags TEXT NOT NULL DEFAULT '[]',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_topic ON surveys(topic)
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_generation_time ON surveys(generation_time)
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_status ON surveys(status)
            """)
    
    def store_survey(self, 
                    survey_id: str,
                    topic: str,
                    content: str,
                    task_ids: List[str],
                    metadata: Dict[str, Any],
                    formats: List[str] = None) -> SurveyRecord:
        """存储综述"""
        try:
            self.logger.info(f"开始存储综述: {survey_id}")
            
            if formats is None:
                formats = ["markdown", "html", "pdf"]
            
            # 计算内容哈希
            content_hash = self._calculate_content_hash(content)
            
            # 检查是否已存在相同内容
            existing_record = self._find_by_content_hash(content_hash)
            if existing_record:
                self.logger.info(f"发现相同内容的综述: {existing_record.survey_id}")
                return existing_record
            
            # 创建文件目录
            survey_dir = self.files_dir / survey_id
            survey_dir.mkdir(exist_ok=True)
            
            # 生成多种格式的文件
            file_paths = {}
            for format_type in formats:
                file_path = self._generate_format_file(
                    content, metadata, format_type, survey_dir, survey_id
                )
                if file_path:
                    file_paths[format_type] = str(file_path)
            
            # 创建综述记录
            record = SurveyRecord(
                survey_id=survey_id,
                topic=topic,
                task_ids=task_ids,
                generation_time=datetime.now(),
                content_hash=content_hash,
                file_paths=file_paths,
                metadata=metadata,
                version=1,
                status="draft",
                tags=[]
            )
            
            # 保存到数据库
            self._save_to_database(record)
            
            self.logger.info(f"综述存储完成: {survey_id}")
            return record
            
        except Exception as e:
            self.logger.error(f"存储综述失败: {str(e)}")
            raise
    
    def _calculate_content_hash(self, content: str) -> str:
        """计算内容哈希"""
        return hashlib.sha256(content.encode('utf-8')).hexdigest()
    
    def _find_by_content_hash(self, content_hash: str) -> Optional[SurveyRecord]:
        """根据内容哈希查找综述"""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(
                "SELECT * FROM surveys WHERE content_hash = ?",
                (content_hash,)
            )
            row = cursor.fetchone()
            
            if row:
                return self._row_to_record(row)
            return None
    
    def _generate_format_file(self, 
                             content: str, 
                             metadata: Dict[str, Any], 
                             format_type: str, 
                             survey_dir: Path, 
                             survey_id: str) -> Optional[Path]:
        """生成指定格式的文件"""
        try:
            options = FormattingOptions(
                format_type=format_type,
                style="academic",
                include_toc=True,
                include_references=True
            )
            
            result = self.formatter.format_survey(content, metadata, options)
            formatted_content = result["content"]
            
            # 确定文件扩展名
            extensions = {
                "markdown": ".md",
                "html": ".html",
                "latex": ".tex",
                "pdf": ".pdf",
                "word": ".docx"
            }
            
            ext = extensions.get(format_type, ".txt")
            file_path = survey_dir / f"{survey_id}{ext}"
            
            # 保存文件
            if format_type in ["pdf", "word"]:
                # 二进制格式
                with open(file_path, 'wb') as f:
                    f.write(formatted_content)
            else:
                # 文本格式
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(formatted_content)
            
            return file_path
            
        except Exception as e:
            self.logger.error(f"生成{format_type}格式文件失败: {str(e)}")
            return None
    
    def _save_to_database(self, record: SurveyRecord):
        """保存记录到数据库"""
        with sqlite3.connect(self.db_path) as conn:
            now = datetime.now().isoformat()
            
            conn.execute("""
                INSERT OR REPLACE INTO surveys (
                    survey_id, topic, task_ids, generation_time, content_hash,
                    file_paths, metadata, version, status, tags, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                record.survey_id,
                record.topic,
                json.dumps(record.task_ids),
                record.generation_time.isoformat(),
                record.content_hash,
                json.dumps(record.file_paths),
                json.dumps(record.metadata),
                record.version,
                record.status,
                json.dumps(record.tags),
                now,
                now
            ))
    
    def get_survey(self, survey_id: str) -> Optional[SurveyRecord]:
        """获取综述记录"""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(
                "SELECT * FROM surveys WHERE survey_id = ?",
                (survey_id,)
            )
            row = cursor.fetchone()
            
            if row:
                return self._row_to_record(row)
            return None
    
    def list_surveys(self, 
                    topic_filter: str = None,
                    status_filter: str = None,
                    limit: int = 50,
                    offset: int = 0) -> List[SurveyRecord]:
        """列出综述记录"""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            
            query = "SELECT * FROM surveys WHERE 1=1"
            params = []
            
            if topic_filter:
                query += " AND topic LIKE ?"
                params.append(f"%{topic_filter}%")
            
            if status_filter:
                query += " AND status = ?"
                params.append(status_filter)
            
            query += " ORDER BY generation_time DESC LIMIT ? OFFSET ?"
            params.extend([limit, offset])
            
            cursor = conn.execute(query, params)
            rows = cursor.fetchall()
            
            return [self._row_to_record(row) for row in rows]
    
    def _row_to_record(self, row: sqlite3.Row) -> SurveyRecord:
        """将数据库行转换为记录对象"""
        return SurveyRecord(
            survey_id=row["survey_id"],
            topic=row["topic"],
            task_ids=json.loads(row["task_ids"]),
            generation_time=datetime.fromisoformat(row["generation_time"]),
            content_hash=row["content_hash"],
            file_paths=json.loads(row["file_paths"]),
            metadata=json.loads(row["metadata"]),
            version=row["version"],
            status=row["status"],
            tags=json.loads(row["tags"])
        )
    
    def get_file_content(self, survey_id: str, format_type: str) -> Optional[bytes]:
        """获取文件内容"""
        record = self.get_survey(survey_id)
        if not record:
            return None
        
        file_path = record.file_paths.get(format_type)
        if not file_path or not os.path.exists(file_path):
            return None
        
        try:
            with open(file_path, 'rb') as f:
                return f.read()
        except Exception as e:
            self.logger.error(f"读取文件失败 {file_path}: {str(e)}")
            return None
    
    def update_survey_status(self, survey_id: str, status: str) -> bool:
        """更新综述状态"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute(
                    "UPDATE surveys SET status = ?, updated_at = ? WHERE survey_id = ?",
                    (status, datetime.now().isoformat(), survey_id)
                )
                return cursor.rowcount > 0
        except Exception as e:
            self.logger.error(f"更新状态失败: {str(e)}")
            return False
    
    def add_tags(self, survey_id: str, tags: List[str]) -> bool:
        """添加标签"""
        try:
            record = self.get_survey(survey_id)
            if not record:
                return False
            
            # 合并标签
            existing_tags = set(record.tags)
            new_tags = existing_tags.union(set(tags))
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute(
                    "UPDATE surveys SET tags = ?, updated_at = ? WHERE survey_id = ?",
                    (json.dumps(list(new_tags)), datetime.now().isoformat(), survey_id)
                )
                return cursor.rowcount > 0
        except Exception as e:
            self.logger.error(f"添加标签失败: {str(e)}")
            return False
    
    def create_new_version(self, 
                          original_survey_id: str, 
                          new_content: str, 
                          metadata: Dict[str, Any]) -> Optional[SurveyRecord]:
        """创建新版本"""
        try:
            original_record = self.get_survey(original_survey_id)
            if not original_record:
                return None
            
            # 生成新的survey_id
            new_survey_id = f"{original_survey_id}_v{original_record.version + 1}"
            
            # 创建新版本
            new_record = self.store_survey(
                survey_id=new_survey_id,
                topic=original_record.topic,
                content=new_content,
                task_ids=original_record.task_ids,
                metadata=metadata
            )
            
            # 更新版本号
            new_record.version = original_record.version + 1
            self._save_to_database(new_record)
            
            return new_record
            
        except Exception as e:
            self.logger.error(f"创建新版本失败: {str(e)}")
            return None
    
    def delete_survey(self, survey_id: str) -> bool:
        """删除综述"""
        try:
            record = self.get_survey(survey_id)
            if not record:
                return False
            
            # 删除文件
            survey_dir = self.files_dir / survey_id
            if survey_dir.exists():
                shutil.rmtree(survey_dir)
            
            # 删除数据库记录
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute(
                    "DELETE FROM surveys WHERE survey_id = ?",
                    (survey_id,)
                )
                return cursor.rowcount > 0
                
        except Exception as e:
            self.logger.error(f"删除综述失败: {str(e)}")
            return False
    
    def export_survey(self, survey_id: str, format_type: str, output_path: str) -> bool:
        """导出综述到指定路径"""
        try:
            content = self.get_file_content(survey_id, format_type)
            if not content:
                return False
            
            with open(output_path, 'wb') as f:
                f.write(content)
            
            return True
            
        except Exception as e:
            self.logger.error(f"导出综述失败: {str(e)}")
            return False
    
    def search_surveys(self, 
                      query: str, 
                      search_fields: List[str] = None) -> List[SurveyRecord]:
        """搜索综述"""
        if search_fields is None:
            search_fields = ["topic", "metadata"]
        
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            
            conditions = []
            params = []
            
            for field in search_fields:
                if field == "topic":
                    conditions.append("topic LIKE ?")
                    params.append(f"%{query}%")
                elif field == "metadata":
                    conditions.append("metadata LIKE ?")
                    params.append(f"%{query}%")
            
            if not conditions:
                return []
            
            sql = f"SELECT * FROM surveys WHERE ({' OR '.join(conditions)}) ORDER BY generation_time DESC"
            
            cursor = conn.execute(sql, params)
            rows = cursor.fetchall()
            
            return [self._row_to_record(row) for row in rows]
    
    def get_storage_stats(self) -> Dict[str, Any]:
        """获取存储统计信息"""
        with sqlite3.connect(self.db_path) as conn:
            # 总数统计
            total_count = conn.execute("SELECT COUNT(*) FROM surveys").fetchone()[0]
            
            # 状态统计
            status_stats = {}
            cursor = conn.execute("SELECT status, COUNT(*) FROM surveys GROUP BY status")
            for row in cursor:
                status_stats[row[0]] = row[1]
            
            # 最近生成统计
            recent_count = conn.execute(
                "SELECT COUNT(*) FROM surveys WHERE generation_time >= datetime('now', '-7 days')"
            ).fetchone()[0]
            
            # 存储空间统计
            total_size = 0
            for file_path in self.files_dir.rglob("*"):
                if file_path.is_file():
                    total_size += file_path.stat().st_size
            
            return {
                "total_surveys": total_count,
                "status_distribution": status_stats,
                "recent_surveys": recent_count,
                "storage_size_mb": total_size / (1024 * 1024),
                "storage_path": str(self.storage_root)
            }
