from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from utils import *

color_list = ['r', 'b', 'y', 'g', 'c', 'k', 'm', 'teal', 'dodgerblue',
                      'indigo', 'deeppink', 'pink', 'peru', 'brown', 'lime', 'darkorange']

def reduce_embed_dim(model):
    o_embed = model.get_top_embeddings().detach().cpu().numpy()
    reduce_embed = TSNE(n_components=2, learning_rate='auto',
                   init='random', perplexity=30).fit_transform(o_embed)
    return reduce_embed

def visualiza_embed(model, image_path, epoch, epoch_adv):
    reduce_embed = reduce_embed_dim(model)
    for i in range(len(model.user_tags)):
        plt.figure()
        labels = model.user_tags[i]
        num_groups = max(labels)
        for j in range(num_groups+1):
            target = reduce_embed[labels == j]
            plt.plot(target[:, 0], target[:,1],
                         'o', color=color_list[j])

        plt.savefig(image_path+f'/epoch_{epoch}_adv_{epoch_adv}_attribute_{i}.png')

