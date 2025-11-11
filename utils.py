import torch
from transformers import pipeline
import numpy as np
from tqdm import tqdm
import pandas as pd
import open_clip
from PIL import Image
from torchvision.transforms import v2

# google/siglip2-base-patch16-224     0.788
# google/siglip2-so400m-patch16-512   0.781
# google/siglip2-large-patch16-256   0.781


def load_Siglip2(model="google/siglip2-large-patch16-256"):
    pipeline_instance = pipeline(
        task="zero-shot-image-classification",
        model=model,
        device="cuda",
        dtype=torch.bfloat16,
    )

    return pipeline_instance


def load_Siglip():
    tokenizer = torch.load(
        "/home/janweimer/Documents/github/ml-services/services/vector-store-rules/add_data/tokenizer_fashionSigLIP.pth",
        weights_only=False,
    )

    model, _, preprocessor = open_clip.create_model_and_transforms(
        "ViT-B-16-SigLIP",
        pretrained="/home/janweimer/Documents/github/ml-services/services/vector-store-rules/artifacts/marqo_fashionSigLIP/open_clip_pytorch_model.bin",
        image_mean=(0.5, 0.5, 0.5),
        image_std=(0.5, 0.5, 0.5),
        image_resize_mode="squash",
        image_interpolation="bicubic",
    )

    model = model.to("cuda")
    return model, tokenizer, preprocessor


def Load_Dino3(resize_size: int = 516):
    to_tensor = v2.ToImage()
    resize = v2.Resize((resize_size, resize_size), antialias=True)
    to_float = v2.ToDtype(torch.float32, scale=True)
    normalize = v2.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
    )
    preprocessor = v2.Compose([to_tensor, resize, to_float, normalize])

    # DINOv3
    model, tokenizer = torch.hub.load(
        "dinov3/dinov3",
        "dinov3_vitl16_dinotxt_tet1280d20h24l",
        source="local",
        weights="dinov3/dinov3_vitl16_dinotxt_vision_head_and_text_encoder-a442d8f5.pth",
        backbone_weights="dinov3/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth",
    )
    model = model.to("cuda")
    return model, tokenizer, preprocessor


def apply_model_on_ds(
    ds,
    pipeline_instance=None,
    model=None,
    tokenizer=None,
    preprocessor=None,
    target_field="Category three",
    candidate_labels=None,
):
    results = []
    labels = []
    file_ids = []

    if candidate_labels is None:
        candidate_labels = ds.distinct(target_field)
        candidate_labels = [
            "a garment of the type " + x.lower() for x in candidate_labels
        ]

    if model:
        product_type_features = tokenizer(candidate_labels)

        with torch.no_grad():
            product_type_features = model.encode_text(
                product_type_features.to("cuda"), normalize=True
            )

    for sample in tqdm(ds):
        file_id = sample.filepath.split("/")[-1]
        image_path = f"data/samples/{file_id}"

        # IMPORTANT: we pass `padding=max_length` and `max_length=64` since the model was trained with this

        if pipeline_instance:
            res = pipeline_instance(image_path, candidate_labels=candidate_labels)

            res = str(
                np.array([x["label"] for x in res])[
                    np.argmax(np.array([x["score"] for x in res]))
                ]
            )

        if model:
            encoding_image = model.encode_image(
                preprocessor(Image.open(image_path)).to("cuda").unsqueeze(0),
                normalize=True,
            )
            text_probs = (encoding_image @ product_type_features.T).softmax(dim=-1)
            product_type_pos = torch.argmax(text_probs).item()
            res = candidate_labels[product_type_pos]

        if target_field == "Category three":
            results.append(res.split(" ")[-1])
        else:
            results.append(res)

        labels.append(sample[target_field][0])
        file_ids.append(file_id)

    return pd.DataFrame({"file_id": file_ids, "label": labels, "pred": results})
