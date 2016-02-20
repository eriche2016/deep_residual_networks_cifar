# deep_residual_networks_cifar

## when run this code, you need to update your torch and packages like nn, cutorch, cunn, cudnn to the latest versions

###1 luarocks install torch 
###2 luarocks install nn
###3 may need to run: luarocks install FindCUDA first to avoid the anoying warning message  
###4 luarocks install cutorch 
###5 luarocks install cunn 

## install cudnn

register and download cudnn, then install it(from nvidia website).

then install bindings of cudnn

1. git clone https://github.com/soumith/cudnn.torch
2. cd cudnn.torch 
3. luarocks make 


