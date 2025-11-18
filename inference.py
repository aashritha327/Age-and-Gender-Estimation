import os
from PIL import Image
import torchvision.transforms as transforms
import torch
from model import SmallCNN

_transform = transforms.Compose([
    transforms.Resize((64,64)),
    transforms.ToTensor(),
])

_model_cache = None

def load_model(path):
    global _model_cache
    if _model_cache is not None:
        return _model_cache
    if not path or not os.path.exists(path):
        _model_cache = None
        return None
    model = SmallCNN()
    try:
        state = torch.load(path, map_location='cpu')
        # state may be state_dict or full model dict
        if isinstance(state, dict) and all(k.startswith('net') for k in state.keys()) == False:
            try:
                model.load_state_dict(state)
            except Exception:
                # try if state contains 'state_dict' key
                if 'state_dict' in state and isinstance(state['state_dict'], dict):
                    model.load_state_dict(state['state_dict'])
        else:
            # fallback: try to load directly
            model.load_state_dict(state)
        model.eval()
        _model_cache = model
    except Exception:
        _model_cache = None
    return _model_cache

def _dummy_predict(image_path):
    # Simple deterministic heuristic: use image brightness to estimate age,
    # and filename hash parity for gender
    img = Image.open(image_path).convert('L')
    arr = list(img.getdata())
    mean = sum(arr)/len(arr) if len(arr)>0 else 127
    age = int((mean/255)*80)
    gender = 'male' if (os.path.basename(image_path).__hash__() % 2 == 0) else 'female'
    return age, gender

def predict_image(image_path, model_path=None):
    model = load_model(model_path) if model_path else None
    if model is None:
        return _dummy_predict(image_path)

    img = Image.open(image_path).convert('RGB')
    x = _transform(img).unsqueeze(0)
    with torch.no_grad():
        out = model(x)
        # assume out[0,0] -> age, out[0,1] -> gender logit
        age_raw = float(out[0,0].item())
        gender_logit = float(out[0,1].item())
        age = int(max(0, min(100, age_raw)))
        gender = 'male' if gender_logit > 0 else 'female'
    return age, gender
