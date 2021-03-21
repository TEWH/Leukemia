#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import json
from pprint import pprint
import io
import pickle
import os
import PIL 
import PIL.Image as pImage
from fastai.vision.all import *
import random
print("done")


# In[2]:


def dataset_creator(path):
  file_names = []
  for filename in os.listdir(path):
      if filename.endswith(".bmp"):
          file_names.append(os.path.join(path, filename))
      else:
          continue
  return file_names


# In[3]:


data_path = "/work/07925/alaukik/frontera/LeukemiaDataset/C-NMC_training_data/CombinedData"
leuk_data = dataset_creator(data_path + '/all')
no_leuk_data = dataset_creator(data_path + '/hem')


# In[8]:


# Selects random images from each list and concatenates list to create overall dataset
#fix this!!!
random.seed(77)
leuk_data_rand = random.sample(set(leuk_data), 3389)
no_leuk_data_rand = random.sample(set(no_leuk_data), 3389)
full_data = list(leuk_data_rand) + list(no_leuk_data_rand)
labels_ls = ['all']*3389 + ['hem']*3389

#Checking that labels match files
print(full_data[250])
print(full_data[251])
print(full_data[249])
print(labels_ls[250], labels_ls[251], labels_ls[249])


# In[9]:


def image_to_arr_of_numpy_arr(data_array):
  try:
    i=0
    matrix_2d = []
    '''
    matrix_2d = np.array(Image.open(data_array[0]))
    while i < len(data_array):
      if matrix_2d.shape == (257,257,3):
        break
      matrix_2d = np.array(Image.open(data_array[i]))
      i += 1
    '''

    #print(matrix_2d.shape)
    while i < len(data_array):
      image = np.array(pImage.open(data_array[i]))
      if image.shape == (257,257,3):
        matrix_2d.append(image)
      i += 1
    matrix_2d = np.array(matrix_2d)
    #print(matrix_2d.shape)
    
    return matrix_2d

  except TypeError:
    print('input Data array cannot be empty or null')
    return


# In[10]:


image = mpimg.imread(data_path + '/all/UID_1_1_1_all.bmp')
print(type)
plt.imshow(image)
plt.show()


# In[11]:


np.random.seed(32)
data = ImageDataLoaders.from_lists(data_path, full_data, labels=labels_ls, valid_pct=0.2,
      batch_tfms= [*aug_transforms(),Normalize.from_stats(*imagenet_stats)], bs = 64, num_workers=4) 
#TRY THIS FOR LABELS (2/5 Meeting): https://docs.fast.ai/vision.data.html#ImageDataLoaders.from_lists 


# In[12]:


data.show_batch(nrows=3, figsize=(7,8))
#data.label_list


# In[13]:


#number of classifications
print(data.c)


# In[33]:


learn = cnn_learner(data, models.resnet34, metrics=error_rate) #metrics = [] 


# In[34]:


learn.fit_one_cycle(10)


# In[35]:


learn.save('stage-1')


# In[36]:


learn.recorder.plot_loss()
learn.unfreeze()


# In[37]:


learn.lr_find()


# In[38]:


#figure out better way for lr
#learn.recorder.plot(suggestion = True)
#minlr = learn.recorder.min_grad_lr
learn.fit_one_cycle(10,10e-5)


# In[39]:


learn.recorder.plot_loss()


# In[40]:


learn.save('stage-2')


# In[41]:


learn.load('stage-2')


# In[42]:


interp = ClassificationInterpretation.from_learner(learn)


# In[43]:


interp.plot_confusion_matrix()


# In[31]:


from fastai.widgets import *


# In[ ]:


ds, idxs = DatasetFormatter().from_toplosses(learn)
ImageCleaner(ds, idxs, data_path)


# In[ ]:


def visualize_cnn_by_cam(learn, data_index): ## learn is your learner, change this to whatever name you gave yours
  x, _y = learn.data.valid_ds[data_index]
  y = _y.data
  if not isinstance(y, (list, np.ndarray)): # single label -> one hot encoding
    y = np.eye(learn.data.valid_ds.c)[y]
  m = learn.model.eval()
  xb,_ = learn.data.one_item(x)
  xb_im = Image(learn.data.denorm(xb)[0])
  xb = xb.cuda()

def hooked_backward(cat):
    with hook_output(m[0]) as hook_a: 
      with hook_output(m[0], grad=True) as hook_g:
        preds = m(xb)
        preds[0,int(cat)].backward()
    return hook_a,hook_g

#matplotlib
def show_heatmap(img, hm, label):
    _,axs = plt.subplots(1, 2)
    axs[0].set_title(label)
    img.show(axs[0])
    axs[1].set_title(f'CAM of {label}')
    img.show(axs[1])
    axs[1].imshow(hm, alpha=0.6, extent=(0,img.shape[1],img.shape[1],0),
                  interpolation='bilinear', cmap='magma');
    plt.show()
    #pytorch
  for y_i in np.where(y > 0)[0]:
    hook_a,hook_g = hooked_backward(cat=y_i)
    acts = hook_a.stored[0].cpu()
    grad = hook_g.stored[0][0].cpu()
    grad_chan = grad.mean(1).mean(1)
    mult = (acts*grad_chan[...,None,None]).mean(0)
    show_heatmap(img=xb_im, hm=mult, label=str(learn.data.valid_ds.y[data_index]))
#learn.data.valid.ds - translate

num_leuk = len (leuk_data)
num_no_leuk = len (no_leuk_data)       
for idx in range(10): ## Range can be whatever you want, or select a specific image of interest given its index
    visualize_cnn_by_cam(learn, idx)

