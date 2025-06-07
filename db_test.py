"""
数据库连接池测试示例
"""

import logging
import json
from app.modules.db_pool import db_utils

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def test_select():
    """测试查询操作"""
    try:
        # 查询任务状态示例
        sql = """
        SELECT id, task_id, task_name, status, progress, current_stage, message, 
               start_time, update_time, end_time, completed 
        FROM ProcessingStatus 
        LIMIT 5
        """
        results = db_utils.select_all(sql)
        logging.info(f"查询到 {len(results)} 条任务记录")
        
        # 打印第一条记录的信息
        if results:
            logging.info(f"第一条记录: {json.dumps(results[0], ensure_ascii=False, default=str)}")
        
        # 查询单条记录示例
        if results:
            task_id = results[0]['task_id']
            task_sql = "SELECT * FROM ProcessingStatus WHERE task_id = %s"
            task = db_utils.select_one(task_sql, (task_id,))
            logging.info(f"查询到任务: {task['task_name'] if task else 'None'}")
        
        # 查询算法实体示例
        algo_sql = "SELECT * FROM Algorithms LIMIT 3"
        algos = db_utils.select_all(algo_sql)
        logging.info(f"查询到 {len(algos)} 个算法实体")
        
        return True
    except Exception as e:
        logging.error(f"测试查询操作时出错: {str(e)}")
        import traceback
        logging.error(traceback.format_exc())
        return False

def test_insert():
    """测试插入操作"""
    try:
        # 创建测试任务记录
        import datetime
        task_id = f"test_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}"
        
        insert_sql = """
        INSERT INTO ProcessingStatus 
        (task_id, task_name, status, current_stage, progress, message, start_time, update_time)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        """
        params = (
            task_id,
            "连接池测试任务",
            "running",
            "初始化",
            0.5,
            json.dumps({"test": True}, ensure_ascii=False),
            datetime.datetime.now(),
            datetime.datetime.now()
        )
        
        rows = db_utils.insert_one(insert_sql, params)
        logging.info(f"插入测试任务记录，影响 {rows} 行")
        
        # 查询刚插入的记录
        select_sql = "SELECT * FROM ProcessingStatus WHERE task_id = %s"
        task = db_utils.select_one(select_sql, (task_id,))
        logging.info(f"查询插入的测试任务: {task['task_name'] if task else 'None'}")
        
        return task_id
    except Exception as e:
        logging.error(f"测试插入操作时出错: {str(e)}")
        import traceback
        logging.error(traceback.format_exc())
        return None

def test_update(task_id):
    """测试更新操作"""
    if not task_id:
        logging.warning("未提供任务ID，跳过更新测试")
        return False
        
    try:
        # 更新测试任务
        update_sql = """
        UPDATE ProcessingStatus 
        SET status = %s, progress = %s, current_stage = %s, update_time = %s
        WHERE task_id = %s
        """
        import datetime
        params = (
            "completed",
            1.0,
            "测试完成",
            datetime.datetime.now(),
            task_id
        )
        
        rows = db_utils.update_one(update_sql, params)
        logging.info(f"更新测试任务记录，影响 {rows} 行")
        
        # 查询更新后的记录
        select_sql = "SELECT * FROM ProcessingStatus WHERE task_id = %s"
        task = db_utils.select_one(select_sql, (task_id,))
        logging.info(f"查询更新后的测试任务: {task['status'] if task else 'None'}")
        
        return True
    except Exception as e:
        logging.error(f"测试更新操作时出错: {str(e)}")
        import traceback
        logging.error(traceback.format_exc())
        return False

def test_delete(task_id):
    """测试删除操作"""
    if not task_id:
        logging.warning("未提供任务ID，跳过删除测试")
        return False
        
    try:
        # 删除测试任务
        delete_sql = "DELETE FROM ProcessingStatus WHERE task_id = %s"
        
        rows = db_utils.delete_one(delete_sql, (task_id,))
        logging.info(f"删除测试任务记录，影响 {rows} 行")
        
        # 查询是否删除成功
        select_sql = "SELECT * FROM ProcessingStatus WHERE task_id = %s"
        task = db_utils.select_one(select_sql, (task_id,))
        
        if task:
            logging.warning(f"删除失败，仍能查询到测试任务: {task['task_name']}")
            return False
        else:
            logging.info("删除成功，无法查询到测试任务")
            return True
    except Exception as e:
        logging.error(f"测试删除操作时出错: {str(e)}")
        import traceback
        logging.error(traceback.format_exc())
        return False

def main():
    """主测试函数"""
    logging.info("开始数据库连接池测试...")
    
    # 测试查询
    logging.info("===== 测试查询操作 =====")
    test_select()
    
    # 测试插入
    logging.info("===== 测试插入操作 =====")
    task_id = test_insert()
    
    # 测试更新
    logging.info("===== 测试更新操作 =====")
    test_update(task_id)
    
    # 测试删除
    logging.info("===== 测试删除操作 =====")
    test_delete(task_id)
    
    logging.info("数据库连接池测试完成")

if __name__ == "__main__":
    main() 