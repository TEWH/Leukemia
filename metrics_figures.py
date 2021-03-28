#!/usr/bin/python

RocAUC = []
false_pos = []
true_pos = []

def curve_maker(input, target):
    input, target = input.cpu().numpy()[:,1], target.cpu().numpy()
    fpr, tpr, threshold = metrics.roc_curve(target, input)
    roc_auc = metrics.auc(fpr, tpr)
    #do this if you want to record the false pos and true pos rate each cycle
    false_pos.append(fpr)
    true_pos.append(tpr)
    RocAUC.append(roc_auc)
    return roc_auc
    
# You can call this as a callback_fn, and it will show you the AUROC for every epoch
class AUROC(Callback):
    _order = -20 #Needs to run before the recorder

    def __init__(self, learn, **kwargs): self.learn = learn
    def on_train_begin(self, **kwargs): self.learn.recorder.add_metric_names(['AUROC'])
    def on_epoch_begin(self, **kwargs): self.output, self.target = [], []
    
    def on_batch_end(self, last_target, last_output, train, **kwargs):
        if not train:
            self.output.append(last_output)
            self.target.append(last_target)
            
    def on_epoch_end(self, last_metrics, **kwargs):
        if len(self.output) > 0:
            output = torch.cat(self.output)
            target = torch.cat(self.target)
            preds = F.softmax(output, dim=1)
            metric = auroc_score(preds, target)
            
            curve = curve_maker(preds, target) # will give you a list of lists for fpr and tpr, get fpr[:-1] and tpr[:-1] for final rates
            
            return add_metrics(last_metrics, [curve])
            
                 
def auroc_score(input, target):
    input, target = input.cpu().numpy()[:,1], target.cpu().numpy()
    return roc_auc_score(target, input)
    # (y_test, preds)
    
    
## F1_Score

from sklearn.metrics import f1_score

input, target = preds.cpu().numpy(), y.cpu().numpy()
input = np.argmax(input, axis=1)

f1 = f1_score(target, input)

print(f1)


## Plot the receiver operating characteristic curve

plt.figure(figsize=(10,8))
plt.title('ROC curve')
plt.plot(fpr, tpr, color = 'darkcyan', label='AUROC = {0:0.2f}'
               .format(AUROC)) #save this somewhere or manually put it in variable, AUROC

plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'darkorange')
plt.xlim([0, 1])
plt.ylim([0, 1.05])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()

## Plot the precision recall curve

from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
input, target = input.cpu().numpy()[:,1], target.cpu().numpy()
precision, recall, thresh = precision_recall_curve(target, input)

plt.plot(recall, precision, color = 'darkcyan', label='F1 Score = {0:0.2f}'
               .format(F1_Score)) #save this somewhere or manually put it in variable, F1_Score

plt.legend(loc = 'lower right')
plt.plot([0, 1], [1, 0],'darkorange')
plt.xlim([0, 1])
plt.ylim([0, 1.05])
plt.ylabel('Precision')
plt.xlabel('Recall')
plt.show()

