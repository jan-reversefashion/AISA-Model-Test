import torch
from transformers import pipeline
import numpy as np
from tqdm import tqdm
import pandas as pd
import open_clip
from PIL import Image
from torchvision.transforms import v2
from transformers import AutoModel, AutoProcessor
from transformers import GemmaTokenizerFast

# google/siglip2-base-patch16-224     0.788
# google/siglip2-so400m-patch16-512   0.781
# google/siglip2-large-patch16-256   0.781


def load_Siglip2(model="google/siglip2-large-patch16-256"):
    ckpt = "google/siglip2-so400m-patch14-384"
    model = AutoModel.from_pretrained(ckpt, device_map="auto").eval()
    processor = AutoProcessor.from_pretrained(ckpt)

    tokenizer = GemmaTokenizerFast.from_pretrained(
        "google/siglip2-so400m-patch14-384", max_length=64, padding="max_length"
    )

    tokenizer.encode("Hello this is a test")
    return model, tokenizer, processor


def apply_model_on_ds(
    ds,
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
        product_type_features = tokenizer(candidate_labels, return_tensors="pt").to(
            model.device
        )

        with torch.no_grad():
            product_type_features = model.get_text_features(
                **product_type_features, normalize=True
            )

    for sample in tqdm(ds):
        if sample[target_field] is None:
            continue
        if len(sample[target_field]) == 0:
            continue

        file_id = sample.filepath.split("/")[-1]
        image_path = f"data/samples/{file_id}"

        # IMPORTANT: we pass `padding=max_length` and `max_length=64` since the model was trained with this

        if model:
            encoding_image = model.get_image_features(
                **preprocessor(images=[Image.open(image_path)], return_tensors="pt").to(
                    model.device
                )
            )
            text_probs = (encoding_image @ product_type_features.T).softmax(dim=-1)
            product_type_pos = torch.argmax(text_probs).item()
            res = candidate_labels[product_type_pos]

        results.append(res)
        labels.append(sample[target_field][0])
        file_ids.append(file_id)

    return pd.DataFrame({"file_id": file_ids, "label": labels, "pred": results})
