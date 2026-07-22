# convert_logo.py
import base64
import os

def convert_image_to_base64(image_path):
    """Convert image to base64 string"""
    with open(image_path, 'rb') as f:
        data = f.read()
        base64_str = base64.b64encode(data).decode('utf-8')
    return base64_str

# Convert logo công ty
logo_path = r"E:\semiconductor\DEMO\images\company_logo_v1.webp"
if os.path.exists(logo_path):
    logo_base64 = convert_image_to_base64(logo_path)
    
    # Tạo file Python chứa logo
    with open('logo_data.py', 'w') as f:
        f.write(f'# Logo data embedded\n')
        f.write(f'LOGO_BASE64 = """{logo_base64}"""\n')
    
    print("✅ Created logo_data.py")
else:
    print(f"❌ Logo not found: {logo_path}")