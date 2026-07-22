import sys, dis, marshal

def process():
    try:
        f = open('__pycache__/ai_3d_tab.cpython-312.pyc', 'rb')
        f.read(16)
        code = marshal.load(f)
        
        for cls_code in code.co_consts:
            if hasattr(cls_code, 'co_name') and cls_code.co_name == 'SliceViewer':
                for method_code in cls_code.co_consts:
                    if hasattr(method_code, 'co_name') and method_code.co_name == 'update_view':
                        print(f"Found {method_code.co_name}")
                        dis.dis(method_code)
                        return
    except Exception as e:
        print(f"Error: {e}")

process()
