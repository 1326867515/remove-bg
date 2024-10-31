import numpy as np
import torch
import torch.nn.functional as F
from torchvision.transforms.functional import normalize
from huggingface_hub import hf_hub_download
from PIL import Image
from cog import BasePredictor, Input, Path
from briarmbg import BriaRMBG

class Predictor(BasePredictor):
    def setup(self):
        """Load the model into memory"""
        self.net = BriaRMBG()
        model_path = "model.pth"
        
        if torch.cuda.is_available():
            self.net.load_state_dict(torch.load(model_path))
            self.net = self.net.cuda()
        else:
            self.net.load_state_dict(torch.load(model_path, map_location="cpu"))
        
        self.net.eval()

    def resize_image(self, image):
        image = image.convert('RGB')
        model_input_size = (1024, 1024)
        image = image.resize(model_input_size, Image.BILINEAR)
        return image

    def predict(
        self,
        image: Path = Input(description="Input image to remove background from"),
    ) -> Path:
        """Run a single prediction on the model"""
        # 读取输入图像
        orig_image = Image.open(image)
        w, h = orig_im_size = orig_image.size
        
        # 预处理
        image = self.resize_image(orig_image)
        im_np = np.array(image)
        im_tensor = torch.tensor(im_np, dtype=torch.float32).permute(2,0,1)
        im_tensor = torch.unsqueeze(im_tensor,0)
        im_tensor = torch.divide(im_tensor,255.0)
        im_tensor = normalize(im_tensor,[0.5,0.5,0.5],[1.0,1.0,1.0])
        
        if torch.cuda.is_available():
            im_tensor = im_tensor.cuda()

        # 推理
        with torch.no_grad():
            result = self.net(im_tensor)
            
        # 后处理
        result = torch.squeeze(F.interpolate(result[0][0], size=(h,w), mode='bilinear'), 0)
        ma = torch.max(result)
        mi = torch.min(result)
        result = (result-mi)/(ma-mi)    
        
        # 转换为PIL图像
        im_array = (result*255).cpu().data.numpy().astype(np.uint8)
        pil_im = Image.fromarray(np.squeeze(im_array))
        
        # 创建透明背景
        new_im = Image.new("RGBA", pil_im.size, (0,0,0,0))
        new_im.paste(orig_image, mask=pil_im)
        
        # 保存结果
        output_path = Path("output.png")
        new_im.save(output_path, "PNG")
        
        return output_path
