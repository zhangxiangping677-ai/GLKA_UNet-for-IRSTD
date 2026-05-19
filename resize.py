import os
from PIL import Image

def resize_images_recursive(root_folder, size=(420, 420)):
    for dirpath, _, filenames in os.walk(root_folder):
        for filename in filenames:
            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff')):
                image_path = os.path.join(dirpath, filename)
                try:
                    with Image.open(image_path) as img:
                        resized_img = img.resize(size, Image.Resampling.LANCZOS)
                        resized_img.save(image_path)
                        print(f"已处理：{image_path}")
                except Exception as e:
                    print(f"处理 {image_path} 时出错：{e}")

# 示例调用
folder_path = 'cam/appendix'
resize_images_recursive(folder_path)

