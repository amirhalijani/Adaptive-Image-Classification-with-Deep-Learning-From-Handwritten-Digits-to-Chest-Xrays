import io
from ts.torch_handler.base_handler import BaseHandler
from torchvision import transforms
from PIL import Image
import torch

class ChestXrayHandler(BaseHandler):
    def __init__(self):
        super(ChestXrayHandler, self).__init__()
        self.transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])

    def preprocess_one_image(self, req):
        image = req.get("data")
        if image is None:
            image = req.get("body")
        image = Image.open(io.BytesIO(image))
        image = self.transform(image)
        image = image.unsqueeze(0)
        return image

    def preprocess(self, requests):
        images = [self.preprocess_one_image(req) for req in requests]
        images = torch.cat(images)
        return images

    def postprocess(self, data):
        return data.argmax(1).tolist()
