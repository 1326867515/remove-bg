import numpy as np
import torch
import torch.nn.functional as F
from torchvision.transforms.functional import normalize
from huggingface_hub import hf_hub_download
from PIL import Image
from cog import BasePredictor, Input, Path
from briarmbg import BriaRMBG
import logging

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class Predictor(BasePredictor):
    def setup(self):
        """Load the model into memory"""
        logger.info("Starting model setup...")
        try:
            self.net = BriaRMBG()
            model_path = "model.pth"
            
            if torch.cuda.is_available():
                logger.info("CUDA is available. Loading model to GPU...")
                self.net.load_state_dict(torch.load(model_path))
                self.net = self.net.cuda()
            else:
                logger.info("CUDA not available. Loading model to CPU...")
                self.net.load_state_dict(torch.load(model_path, map_location="cpu"))
            
            self.net.eval()
            logger.info("Model setup completed successfully")
        except Exception as e:
            logger.error(f"Error during model setup: {str(e)}")
            raise

    def resize_image(self, image):
        try:
            logger.info("Resizing image...")
            image = image.convert('RGB')
            model_input_size = (1024, 1024)
            image = image.resize(model_input_size, Image.BILINEAR)
            logger.info(f"Image resized to {model_input_size}")
            return image
        except Exception as e:
            logger.error(f"Error during image resizing: {str(e)}")
            raise

    def predict(
        self,
        image: Path = Input(description="Input image to remove background from"),
    ) -> Path:
        """Run a single prediction on the model"""
        try:
            logger.info("Starting prediction...")
            
            # 读取输入图像
            logger.info(f"Loading input image from {image}")
            orig_image = Image.open(image)
            w, h = orig_im_size = orig_image.size
            logger.info(f"Original image size: {orig_im_size}")
            
            # 预处理
            logger.info("Preprocessing image...")
            image = self.resize_image(orig_image)
            im_np = np.array(image)
            im_tensor = torch.tensor(im_np, dtype=torch.float32).permute(2,0,1)
            im_tensor = torch.unsqueeze(im_tensor,0)
            im_tensor = torch.divide(im_tensor,255.0)
            im_tensor = normalize(im_tensor,[0.5,0.5,0.5],[1.0,1.0,1.0])
            
            if torch.cuda.is_available():
                im_tensor = im_tensor.cuda()
                logger.info("Input tensor moved to GPU")

            # 推理
            logger.info("Running inference...")
            with torch.no_grad():
                result = self.net(im_tensor)
            logger.info("Inference completed")
                
            # 后处理
            logger.info("Post-processing results...")
            result = torch.squeeze(F.interpolate(result[0][0], size=(h,w), mode='bilinear'), 0)
            ma = torch.max(result)
            mi = torch.min(result)
            result = (result-mi)/(ma-mi)    
            
            # 转换为PIL图像
            logger.info("Converting result to PIL image...")
            im_array = (result*255).cpu().data.numpy().astype(np.uint8)
            pil_im = Image.fromarray(np.squeeze(im_array))
            
            # 创建透明背景
            logger.info("Creating transparent background...")
            new_im = Image.new("RGBA", pil_im.size, (0,0,0,0))
            new_im.paste(orig_image, mask=pil_im)
            
            # 保存结果
            output_path = Path("output.png")
            logger.info(f"Saving result to {output_path}")
            new_im.save(output_path, "PNG")
            
            logger.info("Prediction completed successfully")
            return output_path
            
        except Exception as e:
            logger.error(f"Error during prediction: {str(e)}")
            raise

