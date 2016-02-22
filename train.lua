require 'xlua'
require 'optim'
require 'nn'
require 'image' -- for data argumentation 

require 'batch_flip'

dofile './provider.lua'

local c = require 'trepl.colorize'

opt = lapp[[
   -s,--save                  (default "logs")      subdirectory to save logs
   -b,--batchSize             (default 128)          batch size
   -r,--learningRate          (default 0.1)        learning rate
   --learningRateDecay        (default 0)           no learning rate decay at the original paper 
   --weightDecay              (default 0.0001)      weightDecay
   -m,--momentum              (default 0.9)         momentum
   --epoch_step               (default 1000)          epoch step, ie halve the learning rateevery 1000 epochs, currently not use, just set it high
   --model                    (default residual_network)     model name
   --max_epoch                (default 160)           maximum number of epochs, the original paper terminates at 160 epcohs(ie 64k iterations)
   --backend                  (default cudnn)            backend
   --is_batch_norm            (default true)          add batch normalization
   --gpuid                    (default 0)             gpu used for training 
   --seed                     (default 123)           seed for random number generator
   --init_from                (default '')            path to the pretrained model 

]]


print(opt)

torch.manualSeed(opt.seed)

-- currently no gpu 
torch.setdefaulttensortype('torch.FloatTensor')


if opt.gpuid >= 0 then 
    local ok, cunn = pcall(require, 'cunn')
    local ok2, cutorch = pcall(require, 'cutorch')

    if not ok then print('cunn not found') end 
    if not ok2 then print('cutorch not found') end 
    if ok and ok2 then 
        print('using cuda on gpu ' .. opt.gpuid .. '...')
        cutorch.setDevice(opt.gpuid + 1) -- torch is 1-indexed
        cutorch.manualSeed(opt.seed)
    else 
        print('cannnot run in the gpu mode because of cuda is not installed correctly or cunn and cutorch are not installed')
        print('falling back to cpu mode')
        opt.gpuid = -1 
        print('current not support cpu training, so exit')
        os.exit() 
    end 
end 


print(c.blue '==>' ..' configuring model and criterion')


local model = nn.Sequential() 
-- simple data argumentation 
 model:add(nn.BatchFlip():float())

model:add(nn.Copy('torch.FloatTensor','torch.CudaTensor'):cuda())
if string.len(opt.init_from) > 0 then -- load pretrained model 
    print('loading model from checkpoint' .. opt.init_from)
    local checkpoint_model3 = torch.load(opt.init_from) 
    model:add(checkpoint_model3)
    
else -- construct the model from scratch     
    -- load from script 
    sub_model = paths.dofile('models/'..opt.model..'.lua')
    sub_model = sub_model:cuda() 

    model:add(sub_model)
end 

-- when call backward method, this module will call this function
model:get(2).updateGradInput = function(input) return end

-- specify criterion 
criterion = nn.ClassNLLCriterion() 
criterion = criterion:cuda()

if opt.backend == 'cudnn' then
   require 'cudnn'
   -- convert nn model to model with cudnn backend 
   cudnn.convert(model:get(3), cudnn)
end

print(model)

print(c.blue '==>' ..' loading data')
provider = torch.load 'provider.t7'
provider.trainData.data = provider.trainData.data:float()
provider.testData.data = provider.testData.data:float()

confusion = optim.ConfusionMatrix(10)

print('Will save at '..opt.save)
paths.mkdir(opt.save)
testLogger = optim.Logger(paths.concat(opt.save, 'test.log'))
testLogger:setNames{'% mean class accuracy (train set)', '% mean class accuracy (test set)'}
testLogger.showPlot = false

parameters,gradParameters = model:getParameters()

print(c.blue'==>' ..' configuring optimizer')

-- set optimizing hyperparameters accordding to the original paper
optimState = {
  learningRate = opt.learningRate,
  weightDecay = opt.weightDecay,
  momentum = opt.momentum,
  nesterov = true,
  dampening = 0,
  learningRateDecay = opt.learningRateDecay,
}

-- good practice to collect garbage once in a while 
collectgarbage() 

function train()
  model:training()
  epoch = epoch or 1

  --[[
  -- halve learning rate every "epoch_step" epochs
   if epoch % opt.epoch_step == 0 then optimState.learningRate = optimState.learningRate/2 end
  --]]
  -- at epoch 1, we set the learning rate to 0.001 to warm up the training 
  if epoch == 1 then optimState.learningRate = 0.001 end
  
  -- then we raise the learning rate to 0.1 after the first epoch  
  if epoch == 2 then optimState.learningRate = 0.1 end 
  -- according to original paper, we will divide the learning rate by 10 at 80 epochs and 120 epochs
  if epoch == 80 then optimState.learningRate = optimState.learningRate/10 end

  if epoch == 120 then optimState.learningRate = optimState.learningRate/10 end  



  print(c.blue '==>'.." online epoch # " .. epoch .. ' [batchSize = ' .. opt.batchSize .. ']')

  local targets = torch.CudaTensor(opt.batchSize)
  -- split method will split LongTensor(vector) to a table of Tensor of length opt.batchSize 
  -- indices will be a table of LongTensor of length opt.batchSize, each element of the Tensor will be fed to the model all at once 
  local indices = torch.randperm(provider.trainData.data:size(1)):long():split(opt.batchSize)
  -- remove last element so that all the batches have equal size
  -- no a big issue here, coz every time we call train(), we will random shuffle the data inside the function  
  indices[#indices] = nil
 
  -- start timer    
  local tic = torch.tic()

  for t,v in ipairs(indices) do
    xlua.progress(t, #indices)

    -- inputs: batchSize x 3 x 32 x 32  
    local inputs = provider.trainData.data:index(1,v)
    targets:copy(provider.trainData.labels:index(1,v))

    local feval = function(x)
      if x ~= parameters then parameters:copy(x) end
      gradParameters:zero()
      
      -- forward pass 
      local outputs = model:forward(inputs)

      local f = criterion:forward(outputs, targets)

      -- backward pass 
      local df_do = criterion:backward(outputs, targets)
      model:backward(inputs, df_do)

      confusion:batchAdd(outputs, targets)

      return f,gradParameters
    end
    optim.sgd(feval, parameters, optimState)
  end
  -- update related measure of the confusion table 
  confusion:updateValids()

  -- dur_in_sec = torch.toc(tic)  
  print(('Train accuracy: '..c.cyan'%.2f'..' %%\t time: %.2f s'):format(
        confusion.totalValid * 100, torch.toc(tic)))

  train_acc = confusion.totalValid * 100

  confusion:zero()
  epoch = epoch + 1
end

function test()
  -- disable flips, dropouts and batch normalization
  model:evaluate()
  print(c.blue '==>'.." testing")
  local bs = 125
  for i=1,provider.testData.data:size(1),bs do
    local outputs = model:forward(provider.testData.data:narrow(1,i,bs))
    confusion:batchAdd(outputs, provider.testData.labels:narrow(1,i,bs))
  end

  confusion:updateValids()
  print('Test accuracy:', confusion.totalValid * 100)
  
  if testLogger then
    paths.mkdir(opt.save)
    testLogger:add{train_acc, confusion.totalValid * 100}

    testLogger:style{'-','-'}
    testLogger:plot()

    local base64im
    do
      -- mark: convert command is a very useful command in ubuntu terminal(command line)to do 
      -- such operations on images as resize, blur, crop, despeckle, dither, draw on, flip, join, 
      -- re-sample, and much more  
      -- command synopsis: convert input_file [options] output_file 
      os.execute(('convert -density 200 %s/test.log.eps %s/test.png'):format(opt.save,opt.save))
      -- encode the image using base64 encode
      os.execute(('openssl base64 -in %s/test.png -out %s/test.base64'):format(opt.save,opt.save))
      local f = io.open(opt.save..'/test.base64')
      if f then base64im = f:read'*all' end
    end
 
    -- write report.html file 
    local file = io.open(opt.save..'/report.html','w')
    file:write(([[
    <!DOCTYPE html>
    <html>
    <body>
    <title>%s - %s</title>
    <img src="data:image/png;base64,%s">
    <h4>optimState:</h4>
    <table>
    ]]):format(opt.save,epoch,base64im))
    for k,v in pairs(optimState) do
      if torch.type(v) == 'number' then
        file:write('<tr><td>'..k..'</td><td>'..v..'</td></tr>\n')
      end
    end
    file:write'</table><pre>\n'
    file:write(tostring(confusion)..'\n')
    file:write(tostring(model)..'\n')
    file:write'</pre></body></html>'
    file:close()
  end

  -- save model every 20 epochs
  if epoch % 20 == 0 then
    -- before being save, need to convert cudnn to nn, so that we can load the model correctly  
    if opt.backend == 'cudnn' then
         -- convert nn model to model with cudnn backend 
         checkpoint = model:get(3):clone() 
         cudnn.convert(checkpoint, nn)
    end
    

    local filename = paths.concat(opt.save, 'checkpoint.t7')
    print('==> saving model to '..filename)
    torch.save(filename, checkpoint:clearState())
  end

  confusion:zero()
end


for i=1,opt.max_epoch do
    train()
    test()
    -- collect garbage once an epoch, maybe two long, but it is better than nothing, why not
    collectgarbage() 
end
