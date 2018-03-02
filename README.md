
## Dog breed competition with only the pretrained models from imagenet

<font color='blue'>This post describe the method and tips I got from participanting in the Dog Breed challenge[Dog Breed challenge] in Kaggle. I managed to get an final score of 0.13783, which sets me in the 158 postion, considering a lot of competitors are leveraging the 3-rd party dataset(which already contain the test data), I believe my approach worth sharing cause there is nothing else being used except the pretrained imagenet models.(However, there are a ensembles :O)</font>
[Dog Breed challenge]: http://www.kaggle.com/c/dog-breed-identification


```python
import os
os.chdir('D:\Machine Learning\Kaggle\Dog Breed Identification\pytorch')
```

<font color='blue'>I delve into this problem firtsly using keras. However I cannot find powerful pretrained models like nasnet in Keras. Then I found this awesome package of [Pytorch pretrained models], all the models I have tryed are actually coming from this package.</font>
[Pytorch pretrained models]: http://github.com/Cadene/pretrained-models.pytorch


```python
from PIL import  Image
import torch
from torch.utils.data import Dataset,DataLoader,TensorDataset,ConcatDataset
from torchvision import transforms as trans
from torchvision import models,utils
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
%matplotlib inline
from tqdm import tqdm
import pretrainedmodels
from torch import nn
from torch import optim
from torch.autograd import Variable
```

<font color='blue'>A dog image reading class is created using Pytorch Dataset. Please refer to the file for detail.</font>


```python
from dataset.dataset import Dogs
```

<font color='blue'>Setting all the hyperparameters,you can set *batch_size* larger if you have enough GPU memory</font>


```python
work_folder = Path('D:\Machine Learning\Kaggle\Dog Breed Identification')
train_image_folder = work_folder/'train'
test_image_folder = work_folder/'test'
bottlenecks_folder = work_folder/'pytorch'/'bottlenecks'
pred_folder = work_folder/'pred'
df_train = pd.read_csv(work_folder/'labels.csv',index_col=0)
df_test = pd.read_csv(work_folder/'sample_submission.csv',index_col=0)
img_size = 331
batch_size = 4
batch_size_top = 4096
use_cuda = torch.cuda.is_available()
date = '0222'
model_name = 'nasnet'
learning_rate = 0.0001
dropout_ratio = 0.5
input_shape = 331
crop_mode = 'center'
use_bias = True
name = '{}__model={}__lr={}__input_shape={}__drop={}__crop_mode={}__bias={}'.format(date,model_name,learning_rate,input_shape,dropout_ratio,crop_mode,use_bias)
```

<font color='blue'>I found out there 2 ways to preprocess the diffrent size image into same shape, resize and center cropping. The diffrence is subtle between them, hence it become part of hyperparameters, following transforms is tested for image preprocessing, however, after several checks I can say center cropping can gain a better result than resize.It looks like at least in this dataset it's better to keep the original image height and width ratio than keep the image margin.</font>


```python
if crop_mode == 'center':
    transforms = trans.Compose([
        trans.Resize(input_shape),
        trans.CenterCrop(input_shape),
        trans.ToTensor(),
        trans.Normalize([0.5, 0.5, 0.5],[0.5, 0.5, 0.5])])
elif crop_mode == 'resize':
    transforms = trans.Compose([
        trans.Resize((input_shape,input_shape)),
        trans.ToTensor(),
        trans.Normalize([0.5, 0.5, 0.5],[0.5, 0.5, 0.5])])
```

<font color='blue'>Create the corresponding datasets and dataloader</font>


```python
train_dataset = Dogs(train_image_folder,df_train,df_test,is_train=True,resize=False,transforms=transforms)
test_dataset = Dogs(test_image_folder,df_train,df_test,False,resize=False,transforms=transforms)
train_dataset_resize = Dogs(train_image_folder,df_train,df_test,is_train=True,resize=True,transforms=transforms)

train_loader = DataLoader(train_dataset,batch_size,num_workers=0,shuffle=False)
test_loader = DataLoader(test_dataset,batch_size,num_workers=0,shuffle=False)
```

<font color='blue'>We can see the diffrence between center crop and resize here</font>


```python
img_center_crop = train_dataset.__getitem__(0)[0]*0.5 + 0.5

transforms_resize = trans.Compose([
        trans.Resize((input_shape,input_shape)),
        trans.ToTensor(),
        trans.Normalize([0.5, 0.5, 0.5],[0.5, 0.5, 0.5])])

train_dataset_resize = Dogs(train_image_folder,df_train,df_test,is_train=True,resize=False,transforms=transforms_resize)

img_resize = train_dataset_resize.__getitem__(0)[0]*0.5 + 0.5
```


```python
trans.ToPILImage()(img_center_crop)
```




![png](output_15_0.png)




```python
trans.ToPILImage()(img_resize)
```




![png](output_16_0.png)



<font color='blue'>The key to transfer learning is to get the bottleneck outputs.Normally we are using the second last layer before the final softmax classifier.</font> 
<font color='blue'><br>For *__nasnet__* in the predefined model,we can simply realize this by changing the last 2 layers of the original model into an identity mapping.</font> 


```python
def get_extraction_model():
    nasnet = pretrainedmodels.nasnetalarge(num_classes=1000)
    nasnet = nasnet.eval()
    nasnet.avg_pool = nn.AdaptiveAvgPool2d(1)
    del nasnet.dropout
    del nasnet.last_linear
    nasnet.dropout = lambda x:x
    nasnet.last_linear = lambda x:x
    return nasnet
```


```python
extraction_nasnet = get_extraction_model()

if use_cuda:
    extraction_nasnet.cuda()
```

<font color='blue'>function to get the bottleneck output, notice that we keep the dataloader not shuffled so the output is sequential </font> 


```python
def get_bottlenecks(data_loader,extration_model,test_mode=False):
    x_pieces = []
    y_pieces = []
    for x,y in tqdm(iter(data_loader)):
        if use_cuda:
            x = Variable(x)
            y = Variable(y) if not test_mode else y
            x = x.cuda()
            y = y.cuda() if not test_mode else y
        x_pieces.append(extration_model(x).cpu().data.numpy())
        y_pieces.append(y.cpu().data.numpy()) if not test_mode else y_pieces
    bottlenecks_x = np.concatenate(x_pieces)
    bottlenecks_y = np.concatenate(y_pieces) if not test_mode else None
    return bottlenecks_x,bottlenecks_y
```


```python
bottlenecks_x,bottlenecks_y= get_bottlenecks(train_loader,extraction_nasnet)
```


```python
# np.save(bottlenecks_folder/(name+'_x'),bottlenecks_x)
# np.save(bottlenecks_folder/(name+'_y'),bottlenecks_y)

# bottlenecks_x = np.load(bottlenecks_folder/(name + '_x.npy'))
# bottlenecks_y = np.load(bottlenecks_folder/(name + '_y.npy'))
```

<font color='blue'>delete the model to save GPU memory</font> 


```python
del extraction_nasnet
```

<font color='blue'>Create the linear layer whose input is the bottleneck features, output is the 120 classes.</font> 


```python
class TopModule(nn.Module):
    def __init__(self,dropout_ratio):
        super(TopModule, self).__init__()
        self.aff = nn.Linear(4032, 120,bias=use_bias)
        self.dropout_ratio = dropout_ratio
    def forward(self,x):
        x = nn.Dropout(p = dropout_ratio)(x)
        x = self.aff(x)
        return x
```


```python
criterion = nn.CrossEntropyLoss()
criterion = criterion.cuda()
```

<font color='blue'>train and validation data split</font> 


```python
permutation = np.random.permutation(bottlenecks_x.shape[0])

x_train = bottlenecks_x[permutation][:-int(bottlenecks_x.shape[0]//5)]
x_val = bottlenecks_x[permutation][-int(bottlenecks_x.shape[0]//5):]
y_train = bottlenecks_y[permutation][:-int(bottlenecks_y.shape[0]//5)]
y_val = bottlenecks_y[permutation][-int(bottlenecks_y.shape[0]//5):]

top_only_train_dataset = TensorDataset(torch.FloatTensor(x_train),torch.LongTensor(y_train))

top_only_val_dataset = TensorDataset(torch.FloatTensor(x_val),torch.LongTensor(y_val))

top_only_train_loader = DataLoader(top_only_train_dataset,batch_size=batch_size_top,shuffle=True)
top_only_val_loader = DataLoader(top_only_val_dataset,batch_size=batch_size_top,shuffle=True)

total_dataset = ConcatDataset([top_only_train_dataset,top_only_val_dataset])
total_loader = DataLoader(total_dataset,batch_size=batch_size_top,shuffle=True)
```

<font color='blue'>training function.</font> 


```python
def fit(loader,optimizer,criterion,model=top_only_model,epochs=1500,evaluate=True):
    val_loss_history = []
    val_acc_history = []
    for epoch in range(epochs):  
        running_loss = 0.0
        for i, data in enumerate(loader, 0):
        
            inputs, labels = data
            inputs, labels = Variable(inputs), Variable(labels)
            if use_cuda:
                inputs = inputs.cuda()
                labels = labels.cuda()
        
            optimizer.zero_grad()
        
            # forward + backward 
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()   
        
            optimizer.step()
        
        running_loss += loss.data[0]
    
        print('[%d, %5d] Train_loss: %.3f' \% (epoch+1, i+1, running_loss / len(loader)))
    
        if evaluate:
            model.eval()
            outputs = model(Variable(torch.from_numpy(x_val),volatile=True).cuda() if use_cuda else Variable(torch.from_numpy(x_val),volatile=True))
            labels = torch.from_numpy(y_val).cuda() if use_cuda else torch.from_numpy(y_val)
            labels = Variable(labels,volatile=True)
            loss = criterion(outputs,labels)
            x_val_v = Variable(torch.FloatTensor(x_val),volatile=True).cuda() if use_cuda else Variable(torch.FloatTensor(x_val),volatile=True)
            _,pred = torch.max(model(x_val_v),1)
            val_acc = np.mean(pred.cpu().data.numpy() == labels.cpu().data.numpy())
            val_loss_history.append(loss.cpu().data.numpy())
            val_acc_history.append(val_acc)
    
            print('[%d] Val_loss: %.3f'% (epoch+1, loss))
            print('[%d] Val_acc: %.3f'% (epoch+1, val_acc))
            model.train()
    
    print('Finished Training')
    return val_loss_history,val_acc_history 
```


```python
val_loss_history,val_acc_history = fit(top_only_train_loader,optimizer,criterion,top_only_model,epochs=1500,evaluate=True)
```

<font color='blue'>get the best_epochs, and use it to train all the data(yes,I 'd like any tiny bit of improvement :)</font> 


```python
best_epochs = np.argmin(np.array(val_loss_history))
```


```python
best_epochs
```


```python
best_val_loss = min(val_loss_history)
```


```python
best_val_loss
```


```python
top_only_model = TopModule(dropout_ratio)
if use_cuda:
    top_only_model = top_only_model.cuda()
optimizer = optim.Adam(top_only_model.parameters(),lr=learning_rate)
```


```python
fit(total_loader,optimizer,criterion,top_only_model,epochs=best_epochs,evaluate=False)
```


```python
extraction_nasnet = get_extraction_model()

if use_cuda:
    extraction_nasnet.cuda()
```

<font color='blue'>get the test bottleneck features.</font> 


```python
bottlenecks_test_x,test_y = get_bottlenecks(test_loader,extraction_nasnet,True)
```


```python
np.save(bottlenecks_folder/(name+'_test_x'),bottlenecks_test_x)
```


```python
del extraction_nasnet
```

<font color='blue'>remember to switch the top model to eval mode, cause it used dropout</font> 


```python
top_only_model.eval()
```

<font color='blue'>generate the final prediction.</font> 


```python
x_test = Variable(torch.FloatTensor(bottlenecks_test_x),volatile=True).cuda() if use_cuda else Variable(torch.FloatTensor(bottlenecks_test_x),volatile=True)
```


```python
pred_np = (nn.Softmax(1)(top_only_model(x_test))).cpu().data.numpy()
```


```python
df_pred = pd.DataFrame(pred_np,index=df_test.index,columns=df_test.columns)
```


```python
df_pred.to_csv(pred_folder/(name+'.csv'))
```

- <font color='blue'>By simply using nasnet pretrained model, I got a score of 0.157.</font> 
- <font color='blue'>Final score of 0.137 is achieved through [psuedo labeling] and ensemble with results from other models.</font> 

## <font color='blue'>some tips I got:</font> 
- <font color='blue'>center cropping is better than resize</font> 
- <font color='blue'>I tried data augmentation, which is not helping, I think this is because there is already a lot of dog pictures of diffrent dog species in the imagenet, hence the model already learned enough feature format in the upper layer</font> 
- <font color='blue'>I tried *nasnet*,*inceptionv4*,*inceptionresnetv2*,*dpn107*,*xception*,*resnet152*,*inceptionv3* and some other models, the comparison of their performance in this task is identical to their result in the imagenet. Hence, I guess better model in imagenet can get better transfer learning performance, at least in here, fair enough</font>
- <font color='blue'>I also tried to bind the bottleneck features from diffrent models together and train a linear classifier on it,it works, and helped as an important ensemble portion.</font> 

## <font color='blue'>some other things I should try if got more time:</font> 
- <font color='blue'>found out the classes have the highest error rate, do something about it</font>
- <font color='blue'>more playing with the input resolution, I just used the original imagenet input size, I wonder using clearer picture whether would help</font> 
- <font color='blue'>another round of pseudo labeling</font> 
- <font color='blue'>K fold validation</font> 
