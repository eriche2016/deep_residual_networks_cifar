# deep_residual_networks_cifar

1. when run this code, you need to update your torch and packages like nn, cutorch, cunn, cudnn to the latest versions
  1. luarocks install torch 
  2. luarocks install nn
  3. may need to run: luarocks install FindCUDA first to avoid the anoying warning message  
  4. luarocks install cutorch 
  5. luarocks install cunn 
  6. install cudnn
    * register and download cudnn, then install it(from nvidia website).
    * install torch bindings of cudnn
      * git clone https://github.com/soumith/cudnn.torch
      * cd cudnn.torch 
      * luarocks make 

###Note
This repository implements 56 layers(N = 9, 6N+2 layers) residual network.
using gcr's code and model(https://github.com/gcr/torch-residual-networks), can achive  lowest test error rate of value
0.0647. check it in logs/resNet1/  floder. 
using my code, can achieve 0.0668 test error rate.  but with pretrained model of 200 epochs, and then training for 1 epoch with lr = 0.01, all the other stuff are the same. 
