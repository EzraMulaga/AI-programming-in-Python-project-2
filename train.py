import argparse
import torch
from torchvision import transforms
import torch.nn.functional as F
import numpy as np
from PIL import Image
from devil import load_checkpoint, load_cat_names

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('checkpoint', action='store', default='checkpoint.pth')
    parser.add_argument('--top_k', default=3, type=int)
    parser.add_argument('--filepath', default='flowers/test/1/image_06743.jpg')
    parser.add_argument('--category_names', default='cat_to_name.json')
    parser.add_argument('--gpu', action='store', default='gpu')
    return parser.parse_args()

def process_image(image_path):
    img_pil = Image.open(image_path)
    adjustments = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return adjustments(img_pil)

def predict(image_path, model, topk=3, gpu='gpu'):
    if gpu == 'gpu' and torch.cuda.is_available():
        model = model.cuda()
        device = torch.device('cuda')
    else:
        model = model.cpu()
        device = torch.device('cpu')

    img_torch = process_image(image_path).unsqueeze_(0).to(device)

    with torch.no_grad():
        output = model(img_torch)
    probability = F.softmax(output, dim=1)
    top_probs, top_indices = probability.topk(topk)

    top_probs = top_probs.cpu().numpy().flatten()
    top_classes = [model.class_to_idx[str(i)] for i in top_indices.cpu().numpy().flatten()]

    return top_probs, top_classes

def main():
    args = parse_args()
    model = load_checkpoint(args.checkpoint)
    cat_to_name = load_cat_names(args.category_names)

    probs, classes = predict(args.filepath, model, args.top_k, args.gpu)
    labels = [cat_to_name[str(index)] for index in classes]

    print(f"File selected: {args.filepath}")
    for label, prob in zip(labels, probs):
        print(f"{label} with a probability of {prob:.2f}")

if __name__ == "__main__":
    main()
