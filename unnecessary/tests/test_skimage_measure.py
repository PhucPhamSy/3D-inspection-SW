import tifffile
from skimage import measure

#for class 1 (TGV)
# image_3d = tifffile.imread(r"E:\semiconductor\TGV_YMT\TestClient1_by_syphuc_ver1\input\Results\Total16_real_dll_test_TGV3DFillHole.tif") 

#for class 2 (void)
image_3d = tifffile.imread(r"E:\semiconductor\TGV_YMT\TestClient1_by_syphuc_ver1\input\Results\Total16_real_dll_test_voidsOnlyHQ.tif") 

label_image = measure.label(image_3d)


props = measure.regionprops(label_image)

for obj in props:
    print(f"Object ID: {obj.label}")
    print(f"Volume: {obj.area}") 
    print(f"Bounding Box: {obj.bbox}")