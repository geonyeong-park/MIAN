from __future__ import division

import scipy.misc
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("whitegrid")
import pandas as pd
import imageio
import os
from skimage import img_as_ubyte

def plot_embedding(X, y, d, save_path, title=None):
    """Plot an embedding X with the class label y colored by the domain d."""
    x_min, x_max = np.min(X, 0), np.max(X, 0)
    X = (X - x_min) / (x_max - x_min)
    num_color = np.max(d) + 1
    cmap = plt.cm.get_cmap('rainbow', num_color)

    # Plot colors numbers
    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(111)

    for i in range(X.shape[0]):
        # plot colored number
        plt.text(X[i, 0], X[i, 1], str(y[i]),
                 color=cmap(d[i]),
                 fontdict={'weight': 'bold', 'size': 9})

    plt.xticks([]), plt.yticks([])
    if title is not None:
        plt.title(title)
    #legend = ['source domain {}'.format(i+1) for i in range(min(d), max(d))]
    #legend[-1] = ['target domain']
    #plt.legend(legend)

    fig.savefig(save_path)
    plt.close('all')

def save_translated_target_img(translated_img, whole_img, whole_label, sourced, targetd, offset):
    sourceTotargetPath = 'data/{}2{}'.format(sourced, targetd)
    if not os.path.exists(sourceTotargetPath):
        os.makedirs(sourceTotargetPath)

    for i, img in enumerate(translated_img):
        ithPath = os.path.join(sourceTotargetPath, '{:1}_{:0>5}_fake.png'.format(whole_label[i].argmax(), i+offset))
        imageio.imwrite(ithPath, recover(img))

        ithPath = os.path.join(sourceTotargetPath, '{:1}_{:0>5}_real.png'.format(whole_label[i].argmax(), i+offset))
        imageio.imwrite(ithPath, recover(whole_img[i]))
        if (i+1) % 5000 == 0:
            print('saving {}-th image'.format(i))

def save_projected_Qz(Qz_source, Qz_target, Pz, image_path, seaborn=False):
    df = pd.DataFrame(columns=['distribution', 'x', 'y'])
    data = [Qz_source, Qz_target, Pz]
    for i, d in enumerate(data):
        for j, row in enumerate(d):
            if i == 0:
                dist = 'source'
            elif i == 1:
                dist = 'target'
            else:
                dist = 'prior'
            df = df.append({'distribution': dist, 'x': row[0], 'y': row[1]}, ignore_index=True)

    fig, ax = plt.subplots()
    sns.scatterplot(x='x', y='y', hue='distribution', data=df, legend='full',
                    palette={'source': 'darkred', 'target': 'darkblue', 'prior': 'dimgray'}, ax=ax)

    xmin, xmax = df.min()['x'] - 0.3 * abs(df.max()['x'] - df.min()['x']), df.max()['x'] + 0.3 * abs(df.max()['x'] - df.min()['x'])
    ymin, ymax = df.min()['y'] - 0.3 * abs(df.max()['y'] - df.min()['y']), df.max()['y'] + 0.3 * abs(df.max()['y'] - df.min()['y'])

    plt.xlim([xmin, xmax])
    plt.ylim([ymin, ymax])
    fig.savefig(image_path)


def save_reconstruction(images, size, image_path):
    return imsave(recover(images), size, image_path)


def recover(images):
    return img_as_ubyte(0.5 * (images + 1.))


def imsave(images, size, path):
  image = np.squeeze(merge(images, size))
  print(path)
  return imageio.imwrite(path, image)


def merge(images, size):
    h, w = images.shape[1], images.shape[2]

    if images.shape[3] in (3, 4):
        c = images.shape[3]
        img = np.zeros((h * size[0], w * size[1], c))
        for idx, image in enumerate(images):
          i = idx % size[1]
          j = idx // size[1]
          img[j * h:j * h + h, i * w:i * w + w, :] = image
        return img

    elif images.shape[3] == 1:
        img = np.zeros((h * size[0], w * size[1]))
        for idx, image in enumerate(images):
          i = idx % size[1]
          j = idx // size[1]
          img[j * h:j * h + h, i * w:i * w + w] = image[:, :, 0]
        return img

    else:
        raise ValueError('in merge(images,size) images parameter must have dimensions: HxW or HxWx3 or HxWx4')


def image_manifold_size(num_images):
    manifold_h = int(np.floor(np.sqrt(num_images)))
    manifold_w = int(np.ceil(np.sqrt(num_images)))
    assert manifold_h * manifold_w == num_images
    return manifold_h, manifold_w


def closest_divisor(number):
    divisors = []
    t_num = int(number / 2)

    divisors.append(number)
    while t_num >= 1:
        if number % t_num == 0:
            divisors.append(t_num)
        t_num -= 1

    if np.mod(len(divisors), 2) == 1:
        w, h = divisors[len(divisors)//2], divisors[len(divisors)//2]
    else:
        w, h = divisors[len(divisors)//2-1], divisors[len(divisors)//2]
    return w, h
