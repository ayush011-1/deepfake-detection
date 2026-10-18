import os
import sys
import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image

# Setup backward compatibility aliases for older pickled timm models
try:
    import timm
    import timm.layers
    import timm.layers.adaptive_avgmax_pool
    sys.modules['timm.models.layers.adaptive_avgmax_pool'] = timm.layers.adaptive_avgmax_pool
except Exception as e:
    pass

MODEL_PATHS = {
    'celebdf': os.path.join(os.path.dirname(__file__), 'models', 'CelebDF_model_20_epochs_99acc.pt'),
    'ffpp': os.path.join(os.path.dirname(__file__), 'models', 'FFPP_model_20_epochs_99acc.pt')
}

class DeepfakeDetector:
    def __init__(self, model_type='celebdf', device=None):
        """
        model_type: 'celebdf' or 'ffpp'
        """
        self.model_type = model_type.lower()
        if self.model_type not in MODEL_PATHS:
            raise ValueError(f"Unknown model_type '{model_type}'. Choose 'celebdf' or 'ffpp'.")
        
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_path = MODEL_PATHS[self.model_type]
        self._load_model()
        
        # Standard input transform matching training resolution (128x128)
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((128, 128)),
            transforms.ToTensor()
        ])
        
        # Load OpenCV Haar face detector
        haar_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        self.face_cascade = cv2.CascadeClassifier(haar_path)

    def _load_model(self):
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model checkpoint not found at: {self.model_path}")
        
        checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=False)
        self.model = checkpoint['model'].to(self.device)
        self.model.eval()

        # Disable in-place operations for ReLU to allow clean backward hooks for Grad-CAM
        for module in self.model.modules():
            if isinstance(module, torch.nn.ReLU):
                module.inplace = False

    def detect_and_crop_faces(self, image_bgr, scale_factor=1.3):
        """
        Detects faces using Haar cascade and expands bounding box by scale_factor.
        Returns list of (cropped_face_bgr, bbox_tuple (x, y, w, h)).
        """
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=4,
            minSize=(40, 40)
        )
        
        crops = []
        h_img, w_img = image_bgr.shape[:2]
        
        for (x, y, w, h) in faces:
            new_w = int(w * scale_factor)
            new_h = int(h * scale_factor)
            delta_x = (new_w - w) // 2
            delta_y = (new_h - h) // 2
            
            x1 = max(0, x - delta_x)
            y1 = max(0, y - delta_y)
            x2 = min(w_img, x + w + delta_x)
            y2 = min(h_img, y + h + delta_y)
            
            crop = image_bgr[y1:y2, x1:x2]
            if crop.size > 0:
                crops.append((crop, (x1, y1, x2 - x1, y2 - y1)))
                
        return crops

    def _generate_gradcam(self, tensor_input, target_layer_name='block7'):
        """
        Generates Grad-CAM heatmap tensor for the input tensor.
        """
        activations = None
        gradients = None

        def forward_hook(module, inp, out):
            nonlocal activations
            activations = out

        def backward_hook(module, grad_in, grad_out):
            nonlocal gradients
            gradients = grad_out[0]

        target_layer = getattr(self.model, target_layer_name, self.model.block7)
        h_fwd = target_layer.register_forward_hook(forward_hook)
        h_bwd = target_layer.register_full_backward_hook(backward_hook)

        self.model.zero_grad()
        output = self.model(tensor_input)
        output.backward(retain_graph=False)

        h_fwd.remove()
        h_bwd.remove()

        if activations is None or gradients is None:
            return None

        # Global average pool of gradients
        pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])
        weighted_act = activations.clone()
        for i in range(weighted_act.shape[1]):
            weighted_act[:, i, :, :] *= pooled_gradients[i]

        heatmap = torch.mean(weighted_act, dim=1).squeeze().cpu().detach().numpy()
        heatmap = np.maximum(heatmap, 0)
        if np.max(heatmap) > 0:
            heatmap = heatmap / np.max(heatmap)
            
        return heatmap

    def predict_face(self, face_bgr, return_gradcam=True, target_layer='block7'):
        """
        Inference on an already cropped face image (BGR).
        """
        rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
        tensor = self.transform(rgb).unsqueeze(0).to(self.device)

        with torch.no_grad():
            raw_logit = self.model(tensor).item()
            prob_fake = torch.sigmoid(torch.tensor(raw_logit)).item()

        label = 'FAKE' if prob_fake > 0.5 else 'REAL'
        confidence = prob_fake if label == 'FAKE' else (1.0 - prob_fake)

        gradcam_overlay = None
        gradcam_raw = None

        if return_gradcam:
            tensor.requires_grad = True
            heatmap = self._generate_gradcam(tensor, target_layer_name=target_layer)
            if heatmap is not None:
                gradcam_raw = heatmap
                h, w = face_bgr.shape[:2]
                heatmap_resized = cv2.resize(heatmap, (w, h))
                heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
                gradcam_overlay = cv2.addWeighted(face_bgr, 0.65, heatmap_colored, 0.35, 0)

        return {
            'label': label,
            'confidence': float(confidence),
            'prob_fake': float(prob_fake),
            'raw_logit': float(raw_logit),
            'gradcam_overlay': gradcam_overlay,
            'gradcam_raw': gradcam_raw
        }

    def predict_image(self, image_path_or_array, return_gradcam=True):
        """
        Inference on an image file path or numpy BGR array.
        Handles face detection; if no face detected, runs on entire image.
        """
        if isinstance(image_path_or_array, str):
            image = cv2.imread(image_path_or_array)
            if image is None:
                raise ValueError(f"Could not read image from {image_path_or_array}")
        else:
            image = image_path_or_array.copy()

        face_crops = self.detect_and_crop_faces(image)
        results = []

        if len(face_crops) == 0:
            # Fallback: assume the input is already a cropped face
            pred = self.predict_face(image, return_gradcam=return_gradcam)
            pred['bbox'] = (0, 0, image.shape[1], image.shape[0])
            pred['face_crop'] = image
            results.append(pred)
        else:
            for crop_img, bbox in face_crops:
                pred = self.predict_face(crop_img, return_gradcam=return_gradcam)
                pred['bbox'] = bbox
                pred['face_crop'] = crop_img
                results.append(pred)

        # Build full annotated visualization
        annotated_image = image.copy()
        for res in results:
            x, y, w, h = res['bbox']
            color = (0, 0, 255) if res['label'] == 'FAKE' else (0, 255, 0)
            cv2.rectangle(annotated_image, (x, y), (x + w, y + h), color, 2)
            caption = f"{res['label']}: {res['confidence']*100:.1f}%"
            cv2.putText(annotated_image, caption, (x, max(20, y - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)

        # Primary prediction is the highest-confidence face
        primary = max(results, key=lambda r: r['prob_fake']) if any(r['label'] == 'FAKE' for r in results) else results[0]

        return {
            'model': self.model_type,
            'faces_count': len(results),
            'primary': primary,
            'all_faces': results,
            'annotated_image': annotated_image
        }

    def predict_video(self, video_path, frame_skip=15, max_frames=50):
        """
        Inference on a video file (.mp4, .avi, etc.).
        Subsamples frames and aggregates frame-level predictions.
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")

        frame_idx = 0
        analyzed_frames = 0
        fake_counts = 0
        real_counts = 0
        frame_results = []

        while cap.isOpened() and analyzed_frames < max_frames:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % frame_skip == 0:
                pred = self.predict_image(frame, return_gradcam=False)
                if pred['faces_count'] > 0:
                    primary = pred['primary']
                    if primary['label'] == 'FAKE':
                        fake_counts += 1
                    else:
                        real_counts += 1
                    frame_results.append({
                        'frame_idx': frame_idx,
                        'label': primary['label'],
                        'prob_fake': primary['prob_fake'],
                        'confidence': primary['confidence']
                    })
                    analyzed_frames += 1

            frame_idx += 1

        cap.release()

        total = fake_counts + real_counts
        overall_label = 'FAKE' if fake_counts > real_counts else 'REAL'
        overall_prob = (fake_counts / total) if total > 0 else 0.0

        return {
            'model': self.model_type,
            'total_analyzed_frames': analyzed_frames,
            'overall_label': overall_label,
            'fake_percentage': float(overall_prob * 100),
            'real_percentage': float((1.0 - overall_prob) * 100),
            'frame_details': frame_results
        }
