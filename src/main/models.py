import logging
from django.db import models
from pathlib import Path
from PIL import Image
import torch
import torchvision.transforms as transforms
import torchvision.models as tv_models
import torch.nn as nn

logger = logging.getLogger(__name__)

# === Global Model Config ===
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / 'resnet18_lung_cancer_classifier.pth'
CLASS_NAMES = ['adenocarcinoma', 'large_cell_carcinoma', 'normal', 'squamous_cell_carcinoma']

# === Load PyTorch Model ===
def load_lung_model():
    try:
        # Create model with EXACT architecture used during training
        model = tv_models.resnet18(weights=None)
        num_ftrs = model.fc.in_features

        # Match your training architecture: 2 linear layers only
        model.fc = nn.Sequential(
            nn.Linear(num_ftrs, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, len(CLASS_NAMES)),
        )

        # Load state dict with error handling
        state_dict = torch.load(MODEL_PATH, map_location=torch.device('cpu'))

        # Remove 'module.' prefix if present (from DataParallel)
        new_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}

        model.load_state_dict(new_state_dict)
        model.eval()
        logger.info("Model loaded successfully.")
        return model
    except Exception as e:
        logger.error(f"Failed to load model: {e}", exc_info=True)
        return None

MODEL = load_lung_model()

# === Image Preprocessing Transform ===
image_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    # Normalization used during training — do NOT comment this out
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# === CT Scan Record Model ===
class CTScanRecord(models.Model):
    pimage = models.ImageField(upload_to='uploads/')
    classified = models.CharField(max_length=200, blank=True)
    confidence = models.FloatField(null=True, blank=True)  # Store confidence score
    uploaded = models.DateTimeField(auto_now_add=True)

    def classify_image(self):
        """Run classification on the uploaded image"""
        if MODEL is None:
            return "Model unavailable", 0.0

        try:
            img_path = Path(self.pimage.path)
            if not img_path.exists():
                logger.warning("Image file not found: %s", img_path)
                return "Image not found", 0.0

            img = Image.open(img_path).convert("RGB")
            img_tensor = image_transform(img).unsqueeze(0)  # Add batch dimension

            with torch.inference_mode():
                outputs = MODEL(img_tensor)
                probabilities = torch.softmax(outputs, dim=1)
                confidence, predicted_idx = torch.max(probabilities, 1)
                confidence = confidence.item()

                return CLASS_NAMES[predicted_idx.item()], confidence

        except Exception as e:
            logger.error(f"Classification error: {e}", exc_info=True)
            return "Classification failed", 0.0

    def save(self, *args, **kwargs):
        # On creation, classify image and save results
        is_new = self._state.adding
        super().save(*args, **kwargs)

        if is_new:
            class_name, confidence = self.classify_image()
            self.classified = class_name
            self.confidence = confidence
            # Save only classification fields
            super().save(update_fields=["classified", "confidence"])

    def __str__(self):
        return self.classified or "Pending"
