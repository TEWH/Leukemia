#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from torchvision import transforms
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

defaults.device = torch.device('cuda:0')
def dataset_creator(path):
  file_names = []
  for filename in os.listdir(path):
      if filename.endswith(".bmp"):
          file_names.append(os.path.join(path, filename))
      else:
          continue
  return file_names


# In[3]:


data_path = "/work/07925/alaukik/LeukemiaDataset/C-NMC_training_data/CombinedData"
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


#image = mpimg.imread(data_path + '/all/UID_1_1_1_all.bmp')
#print(type)
#plt.imshow(image)
#plt.show()


# In[11]:

tfms = [Normalize.from_stats(*imagenet_stats)]
np.random.seed(32)
data = ImageDataLoaders.from_lists(data_path, full_data, labels=labels_ls, valid_pct=0.2,
      batch_tfms= [*aug_transforms(),Normalize.from_stats(*imagenet_stats)], bs = 64, num_workers=4) 
#TRY THIS FOR LABELS (2/5 Meeting): https://docs.fast.ai/vision.data.html#ImageDataLoaders.from_lists 


# In[12]:


#data.show_batch(nrows=3, figsize=(7,8))
#data.label_list


# In[13]:


#number of classifications
print(data.c)


# In[33]:


learn = cnn_learner(data, models.resnet34, metrics=error_rate) #metrics = [] 


# In[34]:

#print("Reached training")
#for i in range(0, 30):
#  print("Epoch" + str(i))
#  learn.fit_one_cycle(1)
#  learn.save("con_stage-" + str(i))



# In[35]:


#path = learn.save('stage-1')
#print(path)

# In[36]:


#learn.recorder.plot_loss()
#learn.unfreeze()


# In[37]:


#learn.lr_find()


# In[38]:


#figure out better way for lr
#learn.recorder.plot(suggestion = True)
#minlr = learn.recorder.min_grad_lr
#learn.fit_one_cycle(10,10e-5)


# In[39]:


#learn.recorder.plot_loss()


# In[40]:


#learn.save('stage-2')


# In[41]:


#learn.load('stage-2')


# In[42]:

learn = learn.load('con_stage-24', device=torch.device("cuda:0"))

print("Reached interp")
#interp = ClassificationInterpretation.from_learner(learn)


# In[43]:


#interp.plot_confusion_matrix()


# In[31]:


from fastai.vision.widgets import ImageClassifierCleaner


# In[ ]:

#ImageClassifierCleaner(learn)


# In[ ]:

print("Reached Hooks")


class Hook():
    def __init__(self, m):
        self.hook = m.register_forward_hook(self.hook_func)   
    def hook_func(self, m, i, o): self.stored = o.detach().clone()
    def __enter__(self, *args): return self
    def __exit__(self, *args): self.hook.remove()

# Hook class
class HookBwd():
    def __init__(self, m):
        self.hook = m.register_backward_hook(self.hook_func)   
    def hook_func(self, m, gi, go): self.stored = go[0].detach().clone()
    def __enter__(self, *args): return self
    def __exit__(self, *args): self.hook.remove()


def visualize_cnn_by_cam(learn, index):
  x, y = learn.dls.valid_ds[index]
  x.save("./output/image_" + str(index) + ".png")
  x = transforms.ToTensor()(x).unsqueeze(0)
  #for t in tfms:
  #  x = t(x, split_idx=0)
  cls = y
  with HookBwd(learn.model[0]) as hookg:
      with Hook(learn.model[0]) as hook:
          output = learn.model.eval()(x)
          act = hook.stored
      output[0,cls].backward()
      grad = hookg.stored

  # Take the mean of the gradients
  w = grad[0].mean(dim=[1,2], keepdim=True)
  cam_map = (w * act[0]).sum(0)

  # Show the plot
  x_dec = TensorImage(data.train.decode((x,))[0][0])
  _,ax = plt.subplots()
  x_dec.show(ctx=ax)
  ax.imshow(cam_map.detach().cpu(), alpha=0.6, extent=(0,450,450,0),
                interpolation='bilinear', cmap='magma')
  plt.savefig("./output/heatmap" +  str(index) + ".png")
  

visualize_cnn_by_cam(learn, 2)
#num_leuk = len (leuk_data)
#num_no_leuk = len (no_leuk_data)       
for idx in range(10): ## Range can be whatever you want, or select a specific image of interest given its index
    visualize_cnn_by_cam(learn, idx)

