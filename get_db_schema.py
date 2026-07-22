import sqlite3
import json

try:
    conn = sqlite3.connect('Inno3D_Data/inspection.db')
    cursor = conn.cursor()
    cursor.execute("SELECT name, sql FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()
    
    schema_info = {}
    for table_name, table_sql in tables:
        schema_info[table_name] = table_sql
        
    print(json.dumps(schema_info, indent=2))
except Exception as e:
    print(f"Error: {e}")
finally:
    if 'conn' in locals():
        conn.close()
