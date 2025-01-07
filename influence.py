import os
import json
import torch
import numpy as np
from PIL import Image
import pickle
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn

from trijoint import im2recipe
from args import get_parser

os.environ['CUDA_VISIBLE_DEVICES']='0'

# =============================================================================
parser = get_parser()
opts = parser.parse_args()
# =============================================================================

torch.manual_seed(opts.seed)
np.random.seed(opts.seed)

if not(torch.cuda.device_count()):
    device = torch.device('cpu')
else:
    torch.cuda.manual_seed(opts.seed)
    device = torch.device('cuda:0')

def main():
    # 加载模型
    model = im2recipe()
    model.visionMLP = torch.nn.DataParallel(model.visionMLP)
    model.to(device)

    print("=> loading checkpoint '{}'".format(opts.model_path))
    if device.type == 'cpu':
        checkpoint = torch.load(opts.model_path, encoding='latin1', map_location='cpu')
    else:
        checkpoint = torch.load(opts.model_path, encoding='latin1')
    opts.start_epoch = checkpoint['epoch']
    model.load_state_dict(checkpoint['state_dict'])
    print("=> loaded checkpoint '{}' (epoch {})".format(opts.model_path, checkpoint['epoch']))

    model.eval()

    # 定义图像预处理
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])

    # 从opts.img_path中取出前10张图片
    image_files = sorted(os.listdir(opts.img_path))
    # 确保目录下有至少10张图片
    image_files = image_files[:]

    # 加载文本embedding和ID
    rec_emb_path = r"/root/im2recipe-Pytorch/results"
    rec_ids_path = r"/root/im2recipe-Pytorch/results"
    with open(os.path.join(rec_emb_path, 'rec_embeds.pkl'), 'rb') as f:
        rec_embeds = pickle.load(f)  # shape (N, embDim)
    with open(os.path.join(rec_ids_path, 'rec_ids.pkl'), 'rb') as f:
        rec_ids = pickle.load(f)     # shape (N, )

    # 加载layer1.json
    id2recipe = load_id2recipe('/root/im2recipe-Pytorch/data/layer1.json')

    # 对每张图片进行embedding并检索
    for img_name in image_files:
        img_path = os.path.join(opts.img_path, img_name)
        img_emb = extract_image_embedding(model, img_path, transform, device)

        # 与rec_embeds计算相似度
        # 假设rec_embeds已归一化，img_emb也已归一化，那么点积可以直接视为相似度
        sims = np.dot(rec_embeds, img_emb[0])  # img_emb是(1, D)，所以使用img_emb[0]
        best_idx = np.argmax(sims)
        best_rec_id = rec_ids[best_idx]

        print("\n========== Image: {} ==========".format(img_name))
        # 输出对应文本信息
        if best_rec_id in id2recipe:
            recipe = id2recipe[best_rec_id]
            title = recipe.get('title', 'No Title')
            ingredients = recipe.get('ingredients', [])
            instructions = recipe.get('instructions', [])

            print("Title:", title)
            print("Ingredients:")
            for ing in ingredients:
                print("-", ing['text'])
            print("Instructions:")
            for step in instructions:
                print("-", step['text'])
        else:
            print("No recipe found for id:", best_rec_id)


def extract_image_embedding(model, img_path, transform, device):
    """对单张图像提取embedding"""
    img = Image.open(img_path).convert('RGB')
    img = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        # 提取视觉特征
        visual_emb = model.visionMLP(img)
        visual_emb = visual_emb.view(visual_emb.size(0), -1)
        visual_emb = model.visual_embedding(visual_emb)
        # 归一化
        visual_emb = visual_emb / visual_emb.norm(p=2, dim=1, keepdim=True)

    return visual_emb.cpu().numpy()


def load_id2recipe(json_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    id2recipe = {}
    for rec in data:
        id2recipe[rec['id']] = rec
    return id2recipe


if __name__ == '__main__':
    main()
