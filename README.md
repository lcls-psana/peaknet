# peaknet
A peaknet API with pytorch backbone

## conda environment recipe 

```
conda activate ana-1.4.16
conda install pytorch cudatoolkit=10.1 -c pytorch
conda install tensorboardX
```

or 

```
pip install tensorboardX
```

## Example

```
from peaknet import Peaknet
peaknet = Peaknet() # Init a Peaknet instance
peaknet.loadDefaultCFG() # Load newpeaksv10 network and pretrained weights 
```

## API

### predict
```
peaknet.predict( imgs )
```

`imgs` is a numpy array with dimensions `(n,m,h,w)`. `imgs` will be treated as a stack of `n`x`m` tiles.

### train (for client)
```
peaknet.train( imgs, labels, box_size = 7 )
```

`imgs` is a numpy array with dimensions `(n,m,h,w)`. `imgs` will be treated as a stack of `n`x`m` tiles.
`labels` is a list of tutple of length `n`. Each item in the list is a tutple of three numpy arrays `s`, `r`, `c`, where `s` is an array of integers 0~`(m-1)`.

### model access 
```
peaknet.model
```
returns the current model

### update model 
```
peaknet.updateModel( newModel )
```
replaces the current model with `newModel`, including the network and the weights

### update grad
```
peaknet.updateGrad( newModel )
```
replaces gradients in the current model with that from `newModel`. `newModel` must have same network as current model.

### optimize
```
peaknet.optimize()
```
performs one step of SGD optimization.
 
