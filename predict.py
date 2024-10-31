import os
from cog import BasePredictor, Input, Path

import numpy as np
import torch
import torch.nn.functional as F
from torchvision.transforms.functional import normalize
from huggingface_hub import hf_hub_download
import gradio as gr
from briarmbg import BriaRMBG
import PIL
from PIL import Image
from typing import List
import cv2
    
def resize_image(image):
    image = image.convert('RGB')
    model_input_size = (1024, 1024)
    image = image.resize(model_input_size, Image.BILINEAR)
    return image

def inference(image, net):
    orig_image = Image.fromarray(image)
    w,h = orig_im_size = orig_image.size
    image = resize_image(orig_image)
    im_np = np.array(image)
    im_tensor = torch.tensor(im_np, dtype=torch.float32).permute(2,0,1)
    im_tensor = torch.unsqueeze(im_tensor,0)
    im_tensor = torch.divide(im_tensor,255.0)
    im_tensor = normalize(im_tensor,[0.5,0.5,0.5],[1.0,1.0,1.0])
    if torch.cuda.is_available():
        im_tensor=im_tensor.cuda()

    result=net(im_tensor)
    result = torch.squeeze(F.interpolate(result[0][0], size=(h,w), mode='bilinear') ,0)
    ma = torch.max(result)
    mi = torch.min(result)
    result = (result-mi)/(ma-mi)    
    im_array = (result*255).cpu().data.numpy().astype(np.uint8)
    pil_im = Image.fromarray(np.squeeze(im_array))
    new_im = Image.new("RGBA", pil_im.size, (0,0,0,0))
    new_im.paste(orig_image, mask=pil_im)
    return new_im

class Predictor(BasePredictor):
    def setup(self) -> None:
        self.net=BriaRMBG()
        model_path = hf_hub_download("briaai/RMBG-1.4", 'model.pth')
        if torch.cuda.is_available():
            self.net.load_state_dict(torch.load(model_path))
            self.net=self.net.cuda()
        else:
            self.net.load_state_dict(torch.load(model_path,map_location="cpu"))
        self.net.eval()
        
    def predict(
        self,
        input_image: Path = Input(description="Input Image"),
        contract_pixels: int = Input(description="Number of pixels to contract the edge", default=1, ge=1, le=5)
    ) -> List[Path]:
        # 使用PIL Image读取图像
        pil_image = Image.open(str(input_image))
        # 转换为numpy数组
        image = np.array(pil_image)
    
        # 调用inference函数处理图像得到第一张输出图片
        output_image1 = inference(image, self.net)
    
        # 将output_image1保存为临时文件
        temp_path = "temp_inference.png"
        output_image1.save(temp_path)
    
        # 处理inference的输出得到第二张图片
        processed_image = extract_precise_edge(temp_path, "temp.png", contract_pixels=contract_pixels)
        output_image2 = Image.fromarray(processed_image)
    
        # 保存输出图像
        output_path1 = Path("output1.png")
        output_path2 = Path("output2.png")
    
        output_image1.save(str(output_path1))
        output_image2.save(str(output_path2))
    
        # 删除临时文件
        import os
        if os.path.exists(temp_path):
            os.remove(temp_path)
    
        return [output_path1, output_path2]



def extract_precise_edge(image_path, output_path, contract_pixels=1):
    # 使用PIL Image读取图像
    pil_image = Image.open(image_path)
    img = np.array(pil_image)
    
    # 确保图像有alpha通道
    if img.shape[2] == 3:
        alpha = np.full(img.shape[:2], 255, dtype=np.uint8)
        img = np.dstack((img, alpha))
    
    if img is None or img.shape[2] != 4:
        print("错误：无法读取图像或图像不包含alpha通道")
        return None
    
    # 后续处理保持不变
    bgr = img[:, :, :3]
    alpha = img[:, :, 3].copy()
    
    # 预处理：降噪
    alpha_denoised = cv2.fastNlMeansDenoising(alpha)
    
    # 获取复杂边缘
    edges = get_complex_edge(alpha_denoised)
    
    # 创建核心结构元素
    kernel = np.ones((3,3), np.uint8)
    
    # 边缘优化
    dilated_edges = cv2.dilate(edges, kernel, iterations=1)
    edge_mask = np.zeros_like(alpha)
    edge_mask[dilated_edges > 0] = 255
    
    # 向内收缩
    contracted_mask = cv2.erode(alpha, kernel, iterations=contract_pixels)
    
    # 应用mask
    new_alpha = contracted_mask.copy()
    
    # 合并结果
    cleaned_image = np.dstack((bgr, new_alpha))
    
    # 保存结果
    cv2.imwrite(output_path, cleaned_image)
    
    return cleaned_image

def get_complex_edge(alpha):
    # Canny边缘检测
    canny_edges = cv2.Canny(alpha, 100, 200)
    
    # Sobel边缘检测
    sobelx = cv2.Sobel(alpha, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(alpha, cv2.CV_64F, 0, 1, ksize=3)
    sobel_edges = np.sqrt(sobelx**2 + sobely**2)
    sobel_edges = cv2.normalize(sobel_edges, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    # Laplacian边缘检测
    laplacian = cv2.Laplacian(alpha, cv2.CV_64F)
    laplacian = np.uint8(np.absolute(laplacian))
    
    # 自适应阈值
    adaptive_thresh = cv2.adaptiveThreshold(alpha, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                          cv2.THRESH_BINARY, 11, 2)
    
    # 合并所有边缘检测结果
    combined_edges = cv2.bitwise_or(canny_edges, sobel_edges)
    combined_edges = cv2.bitwise_or(combined_edges, laplacian)
    combined_edges = cv2.bitwise_or(combined_edges, adaptive_thresh)
    
    # 使用形态学操作优化边缘
    kernel = np.ones((3,3), np.uint8)
    combined_edges = cv2.morphologyEx(combined_edges, cv2.MORPH_CLOSE, kernel)
    combined_edges = cv2.morphologyEx(combined_edges, cv2.MORPH_OPEN, kernel)
    
    return combined_edges

