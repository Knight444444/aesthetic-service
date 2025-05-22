import io, base64
from fastapi import FastAPI
from pydantic import BaseModel
import torch, clip, numpy as np
from PIL import Image
import torch.nn as nn
import aiohttp

class MLP(nn.Module):
    def __init__(self, input_size=768):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, 1024),
            nn.Dropout(0.2),
            nn.Linear(1024, 128),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.Dropout(0.1),
            nn.Linear(64, 16),
            nn.Linear(16, 1)
        )
    def forward(self, x):
        return self.layers(x)

def normalize(arr: np.ndarray) -> np.ndarray:
    l2 = np.linalg.norm(arr, axis=-1, keepdims=True)
    l2[l2 == 0] = 1
    return arr / l2

class ImagesIn(BaseModel):
    images: list[str]

app = FastAPI()

@app.on_event("startup")
async def load_models():
    global device, clip_model, clip_preprocess, mlp
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)
    mlp = MLP().to(device)
    mlp.load_state_dict(torch.load("ava+logos-l14-linearMSE.pth", map_location=device))
    mlp.eval()
    try:
        mlp = torch.compile(mlp)
    except:
        pass

@app.post("/predict")
async def predict(payload: ImagesIn):
    tensors = []
    async with aiohttp.ClientSession() as session:
        for img in payload.images:
            if img.startswith("data:image"):
                _, b64 = img.split(",", 1)
                data = base64.b64decode(b64)
                pil = Image.open(io.BytesIO(data)).convert("RGB")
            else:
                resp = await session.get(img)
                data = await resp.read()
                pil = Image.open(io.BytesIO(data)).convert("RGB")
            tensors.append(clip_preprocess(pil))
    batch = torch.stack(tensors).to(device)
    with torch.no_grad():
        feats = clip_model.encode_image(batch)
    embs = normalize(feats.cpu().numpy())
    emb_t = torch.from_numpy(embs).to(device).float()
    with torch.no_grad():
        out = mlp(emb_t).squeeze(-1).cpu().numpy().tolist()
    return {"scores": out}
