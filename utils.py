import torch
from PIL import Image
import matplotlib.pyplot as plt

cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)


class FeatureExtractor(torch.nn.Module):
    def __init__(self, model):
        super(FeatureExtractor, self).__init__()
        self.features = list(model.features)
        self.features = torch.nn.Sequential(*self.features)
        self.pooling = model.avgpool
        self.flatten = torch.nn.Flatten()
        self.fc = model.classifier[0]

    def forward(self, x):
        out = self.features(x)
        out = self.pooling(out)
        out = self.flatten(out)
        out = self.fc(out)
        return out


def prepare_frame_for_inference(img_array, transform):
    img = Image.fromarray(img_array)
    img_tensor = transform(img).unsqueeze(0)

    return img_tensor


def get_feature_vector(img_tensor, device, model):
    img_tensor = img_tensor.to(device)
    with torch.no_grad():
        feature_vec = model(img_tensor)
    feature_vec = feature_vec.cpu().detach()

    return feature_vec


def compute_similarity(feature_list, anch, reverse = True):
    similarity_list = list()
    for i in range(len(feature_list)):
        sim = cos(anch, feature_list[i])
        similarity_list.append((sim, i))
    similarity_list.sort(key=lambda i:i[0], reverse = reverse)
    return similarity_list


def compute_similarity_index_known(feature_list, similarity_list_temp, anch):
    similarity_list_new = list()
    for i in range(len(similarity_list_temp)):
        idx = similarity_list_temp[i][1]
        sim = cos(anch, feature_list[idx])
        similarity_list_new.append((sim, idx))
    similarity_list_new.sort(key=lambda i:i[0], reverse=True)
    return similarity_list_new


def imshow_frame(image, ax=None, title=None):
    if ax is None:
        fig, ax = plt.subplots()

    ax.imshow(image)
    plt.axis('off')

    return ax, image











