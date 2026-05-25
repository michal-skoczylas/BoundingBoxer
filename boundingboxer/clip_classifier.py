import numpy as np
import torch
from PIL import Image

from .config import CLASS_MAP, CLIP_MODEL

TEXTS = [
    "a photo of a closed fist",
    "a photo of an open palm",
    "a photo of no hand, background only",
]


class ClipClassifier:
    def __init__(self, model_name=CLIP_MODEL, device=None):
        from transformers import CLIPModel, CLIPProcessor

        self._device = device or ("mps" if torch.backends.mps.is_available() else "cpu")
        self._model = CLIPModel.from_pretrained(model_name).to(self._device)
        self._processor = CLIPProcessor.from_pretrained(model_name)
        self._model.eval()

        self._texts = TEXTS
        self._class_names = ["closed_fist", "open_palm", "none"]

    def classify(self, hand_crop):
        if hand_crop is None or hand_crop.size == 0 or min(hand_crop.shape[:2]) < 10:
            return ("none", CLASS_MAP["none"], 0.3)

        if hand_crop.shape[-1] == 3:
            pil_img = Image.fromarray(
                hand_crop[:, :, ::-1].copy()
            )
        else:
            pil_img = Image.fromarray(hand_crop).convert("RGB")

        inputs = self._processor(
            text=self._texts,
            images=pil_img,
            return_tensors="pt",
            padding=True,
        ).to(self._device)

        with torch.no_grad():
            outputs = self._model(**inputs)

        image_embeds = outputs.image_embeds / outputs.image_embeds.norm(
            p=2, dim=-1, keepdim=True
        )
        text_embeds = outputs.text_embeds / outputs.text_embeds.norm(
            p=2, dim=-1, keepdim=True
        )

        logits = image_embeds @ text_embeds.T
        logit_scale = (
            outputs.logit_scale if hasattr(outputs, "logit_scale") else
            torch.tensor(1.0, device=self._device)
        )
        probs = (logits * logit_scale.exp()).softmax(dim=-1)[0].cpu().numpy()

        best = int(probs.argmax())
        class_name = self._class_names[best]
        class_id = CLASS_MAP[class_name]
        confidence = float(probs[best])

        return (class_name, class_id, confidence)

    def classify_file(self, image_path):
        img = np.array(Image.open(image_path).convert("RGB"))
        return self.classify(img)
