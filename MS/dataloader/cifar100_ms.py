import os,sys
import numpy as np
import mindspore as ms
import mindspore.dataset as ds
import mindspore.dataset.vision as vision
from sklearn.utils import shuffle
import pickle

cf100_dir = './data/cifar-100-binary'
file_dir = './data/binary_cifar100'


def write_bin(obj, filename):
    with open(filename, 'wb') as f:
        pickle.dump(obj, f)


def read_bin(filename):
    with open(filename, 'rb') as f:
        obj = pickle.load(f)
    return obj


def get(seed=0, pc_valid=0.10):
    data={}
    taskcla=[]
    size=[3, 32, 32]

    if not os.path.isdir(file_dir):
        os.makedirs(file_dir)

        mean=[x/255 for x in [125.3,123.0,113.9]]
        std=[x/255 for x in [63.0,62.1,66.7]]

        dat={}
        transforms = ds.transforms.Compose([
            vision.ToTensor(),
            vision.Normalize(mean=mean, std=std, is_hwc=False),
        ])
        dat['train'] = ds.Cifar100Dataset(cf100_dir, usage='train', shuffle=False)
        dat['train'] = dat['train'].map(operations=transforms)
        dat['test'] = ds.Cifar100Dataset(cf100_dir, usage='test', shuffle=False)
        dat['test'] = dat['test'].map(operations=transforms)

        for n in range(10):
            data[n]={}
            data[n]['name']='cifar100'
            data[n]['ncla']=10
            data[n]['train']={'x': [],'y': []}
            data[n]['test']={'x': [],'y': []}

        for s in ['train','test']:
            loader=ds.GeneratorDataset(dat[s], shuffle=False, column_names=dat[s].get_col_names())
            loader.batch(batch_size=1)
            for image, label1, label2 in loader:
                n = label2.numpy()
                nn = n // 10
                data[nn][s]['x'].append(image)  
                data[nn][s]['y'].append(n%10)

        for t in data.keys():
            for s in ['train','test']:
                data[t][s]['x'] = ms.ops.stack(data[t][s]['x']).view(-1, size[0], size[1], size[2])
                data[t][s]['y'] = ms.Tensor.int(
                    ms.tensor(np.array(data[t][s]['y'], dtype=int))
                ).view(-1)
                write_bin(data[t][s]['x'], os.path.join(os.path.expanduser(file_dir),'data'+str(t)+s+'x.bin'))
                write_bin(data[t][s]['y'], os.path.join(os.path.expanduser(file_dir),'data'+str(t)+s+'y.bin'))

    data={}
    ids=list(np.arange(10))
    print('Task order =',ids)
    for i in range(10):
        data[i] = dict.fromkeys(['name','ncla','train','test'])
        for s in ['train','test']:
            data[i][s]={'x':[],'y':[]}
            data[i][s]['x']=read_bin(os.path.join(os.path.expanduser(file_dir),'data'+str(ids[i])+s+'x.bin'))
            data[i][s]['y']=read_bin(os.path.join(os.path.expanduser(file_dir),'data'+str(ids[i])+s+'y.bin'))
        data[i]['ncla']=len(np.unique(data[i]['train']['y'].numpy()))
        if data[i]['ncla']==2:
            data[i]['name']='cifar10-'+str(ids[i])
        else:
            data[i]['name']='cifar100-'+str(ids[i])

    for t in data.keys():
        r=np.arange(data[t]['train']['x'].shape[0])
        r=np.array(shuffle(r,random_state=seed),dtype=int)
        nvalid=int(pc_valid*len(r))
        ivalid=ms.Tensor.long(ms.tensor(r[:nvalid]))
        itrain=ms.Tensor.long(ms.tensor(r[nvalid:]))
        data[t]['valid']={}
        data[t]['valid']['x']=data[t]['train']['x'][ivalid].copy()
        data[t]['valid']['y']=data[t]['train']['y'][ivalid].copy()
        data[t]['train']['x']=data[t]['train']['x'][itrain].copy()
        data[t]['train']['y']=data[t]['train']['y'][itrain].copy()

    n=0
    for t in data.keys():
        taskcla.append((t,data[t]['ncla']))
        n+=data[t]['ncla']
    data['ncla']=n

    return data, taskcla, size

if __name__ == "__main__":
    ms.set_context(device_target='GPU', device_id=0)
    get()