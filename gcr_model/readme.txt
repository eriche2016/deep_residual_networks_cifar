here i transfer the model which is constructed by gcr, 
and modify it to run with the same cifar data interface 
as well as do the same data preprocessing, turns out that 
it will also have the similiar performace. which indicates 
that the data preprocessing and training data sample scheme 
make the difference(my training scheme is that random shuffle
 the data, and everytime samples a batch in the order, while 
gcr's training scheme is that just random sample a batch of data).

note on 2/26:
using gcr's orginal code for training can achieve the best results, i modify 
the code and get the log results in resNet1 directory, check , check it.
note on 2/28:
using random sample a batch everytime for training.

