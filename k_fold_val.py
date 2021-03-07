import random
import pandas as pd
leuk_data_rand = random.sample(set(leuk_data), 3390)
no_leuk_data_rand = random.sample(set(no_leuk_data), 3390)
full_data = list(leuk_data_rand) + list(no_leuk_data_rand)
labels_ls = ['all']*3390 + ['hem']*3390

def k_valid(full_data, labels_ls):
    rand_k = list(range(0,10170))
    random.shuffle(rand_k)
    start = 0
    accur = []
    for i in range(10):
        klist = rand_k[start:start+len(rand_k)/10]
        bool_lst = []
        for i in range(len(full_data)):
            if i in klist:
                bool_lst.append(True)
            else:
                bool_lst.append(False)
        dfd = {'col1' : full_data, 'col2' : labels_ls, 'col3' : bool_lst}
        df = pd.DataFrame(data=dfd)
        ImageDataLoaders.from_df(df, fn_col = 0, lab_col = 1, valid_col = 2, bs = 64,batch_tfms= [*aug_transforms(),Normalize.from_stats(*imagenet_stats)],num_workers=4)
        learn.fit_one_cycle(4)
        learn.lr_find()
        lr = learn.recorder.min_grad_lr
        learn.fit_one_cycle(10,lr)
        learn.save('fold_' + str(i))
        preds, trgts = learn.get_preds(ds_type = DatasetType.Valid)
        acy = accuracy(preds, trgts)
        print(acy)
        accur.append(acy)







