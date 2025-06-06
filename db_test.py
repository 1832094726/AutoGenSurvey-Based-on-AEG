import sys
import traceback

print("开始执行数据库测试脚本")

try:
    import mysql.connector
    print("成功导入 mysql.connector")
    
    try:
        # 创建数据库连接
        conn = mysql.connector.connect(
            host="localhost",
            user="root",
            password="123456",
            database="algorithm_evolution",
            charset='utf8mb4'
        )
        
        print("成功连接到数据库")
        
        # 创建游标
        cursor = conn.cursor(dictionary=True)
        print("成功创建游标")
        
        # 执行简单查询
        cursor.execute("SELECT 1 as test")
        result = cursor.fetchone()
        print(f"查询结果: {result}")
        
        # 关闭连接
        cursor.close()
        conn.close()
        print("连接已关闭")
        
    except Exception as e:
        print(f"数据库操作发生错误: {str(e)}")
        traceback.print_exc()
        sys.exit(1)
        
except ImportError as e:
    print(f"导入 mysql.connector 失败: {str(e)}")
    traceback.print_exc()
    sys.exit(1)

print("数据库测试完成") 