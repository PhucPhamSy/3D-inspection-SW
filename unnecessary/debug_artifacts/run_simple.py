"""
Simple usage - truyền path DLL trực tiếp
"""
import bumpvoid

print("=" * 50)
print("  BumpVoid DLL - Simple Python Usage")
print("=" * 50)

# ★ Truyền path DLL trực tiếp
bumpvoid.load_dll(r"E:\semiconductor\DEMO\src_frontend_backend_v3")

# Info
print(f"/nDLL Version: {bumpvoid.get_version()}")
print(f"Threads: {bumpvoid.get_thread_count()}")

# Load config
config = bumpvoid.load_config(r"E:/semiconductor/DEMO/src_frontend_backend_v3/config_HBM.txt")

print("\nConfig:")
bumpvoid.print_config(config)

# Run
print("\nProcessing...")
result = bumpvoid.process(config)

# Result
print("\nResult:")
bumpvoid.print_result(result)