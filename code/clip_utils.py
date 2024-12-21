import torch
import clip
from tqdm.auto import tqdm
import numpy as np


def get_clip_features(raw_text: str):
    '''
    return clip features for a given text prompt
    '''

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _ = clip.load('ViT-L/14@336px', device=device)
    model.eval()
    text = clip.tokenize([raw_text]).to(device)
    with torch.no_grad():
        text_features = model.encode_text(text)
    # normalize the features
    text_features /= torch.norm(text_features, dim=-1, keepdim=True)
    assert text_features.shape == (1, 768)

    return text_features.detach().float()


def get_clip_features_loop(language_motion_dict: dict, range_start, range_end):

    language_motion_dict['clip_features'] = np.zeros((len(language_motion_dict['raw_text']), 768))

    for idx in tqdm(range(range_start, range_end)):
        text = language_motion_dict['raw_text'][idx]
        text_features = get_clip_features(raw_text=text).cpu().numpy()
        language_motion_dict['clip_features'][[idx]] = text_features
        print(idx, flush=True)

    print(f'language_motion_dict["clip_features"] shape: {language_motion_dict["clip_features"].shape}', flush=True)

    assert len(language_motion_dict.keys()) == 4, 'language_motion_dict should have 4 keys'

    return language_motion_dict
