require 'nn'

-- using initialization 

local nninit = require 'nninit'

-- easy for debugging, coz provide opt.is_batch_norm paramters
if not opt then 
    cmd = torch.CmdLine() 
    cmd:text() 
    cmd:text('residual network')
    cmd:text('Options:')
    cmd:option('-is_batch_norm', true, 'batch norm or not')
    cmd:text() 
    opt = cmd:parse(arg or {}) 
end 


-- nfin: number of input feature maps 
-- nfout: number of output feature maps 
-- n: how many conv layers per residual block 
-- half: need to half the feature map size
function make_residual_block(nfin, nfout, half, is_batch_norm)
    local stride = 1
    if half == true then stride = 2 end  
    
    residual_block_cat = nn.ConcatTable()

    -- for every residual block, there will be n layers 
    residual_block_par1 = nn.Sequential() 
    residual_block_par1:add(nn.SpatialConvolution(nfin, nfout, 3, 3, stride, stride, 1, 1):init('weight', nninit.kaiming, {gain='relu'}) 
                      :init('bias', nninit.constant, 0))

    if opt.is_batch_norm then residual_block_par1:add(nn.SpatialBatchNormalization(nfout):init('weight', nninit.normal, 1.0, 0.002):init('bias', nninit.constant, 0)) end 
   
    residual_block_par1:add(nn.ReLU(true))

    residual_block_par1:add(nn.SpatialConvolution(nfout, nfout, 3, 3, 1, 1, 1, 1):init('weight', nninit.kaiming, {gain='relu'}) 
                      :init('bias', nninit.constant, 0)) 

     -- whether need to add bach normalization 
    if opt.is_batch_norm then residual_block_par1:add(nn.SpatialBatchNormalization(nfout):init('weight', nninit.normal, 1.0, 0.002):init('bias', nninit.constant, 0)) end    
    
    if half == false then 
         residual_block_par2 = nn.Identity() 
    else
         
         -- choice 1: test error can achieve:0.0725 by using pretrained model for epoch 200  
         
         residual_block_par2 = nn.Sequential() 
         residual_block_par2:add(nn.SpatialConvolution(nfin, nfout, 1, 1, 2, 2, 0, 0):init('weight', nninit.kaiming, {gain='relu'}) 
                      :init('bias', nninit.constant, 0))-- no pad

        -- whether need to add bach normalization
        if opt.is_batch_norm then residual_block_par2:add(nn.SpatialBatchNormalization(nfout)) end 
        
         --[[
         -- choice 2: test err can achieve: 0.0804%(using pretrained model) 
         --  but overall, choice 2 leads to slightly worse results 
         residual_block_par2 = nn.Sequential()
         -- what about change SpatialAveragePooling to max Pooling?
         residual_block_par2:add(nn.SpatialAveragePooling(1, 1, 2, 2))

         if nfout > nfin then 
                residual_block_par2:add(nn.Padding(1, (nfout - nfin), 3))
         end 
         --]]
    end 
   
    residual_block_cat:add(residual_block_par1)

    residual_block_cat:add(residual_block_par2)

    -- add F and x together 
    residual_block = nn.Sequential() 
    residual_block:add(residual_block_cat)

    
    residual_block:add(nn.CAddTable())

    residual_block:add(nn.ReLU(true))
    return residual_block 
end 


local model = nn.Sequential() 
-- for cifar data, every images is a rgb images
-- input images: 3 x 32 x 32 for each images 
model:add(nn.SpatialConvolution(3, 16, 3, 3, 1, 1, 1, 1):init('weight', nninit.kaiming, {gain='relu'}) 
                      :init('bias', nninit.constant, 0))

if opt.is_batch_norm then model:add(nn.SpatialBatchNormalization(16)) end 
model:add(nn.ReLU(true))
-- make residual part of the whole model 

n = 9 
fin_table = {16, 32, 64} 

-- 3 types of feature map sizes 
for i = 1, 3 do 
    
    -- how many redisual block per each feature map size 
    if i == 1 then 
        for j = 1, n do 
            model:add(make_residual_block(fin_table[1], fin_table[1], false, true))  -- half = true, is_batch_norm = true 
        end 
    else 
        for j = 1 , n do
            if j == 1 then 
                model:add(make_residual_block(fin_table[i-1], fin_table[i], true, true))
            else 
                model:add(make_residual_block(fin_table[i], fin_table[i], false, true))
            end 
        end 
    end 
end 

-- add avg pooling layer and fc layer 
model:add(nn.SpatialAveragePooling(8, 8))
model:add(nn.Reshape(64))
model:add(nn.Linear(64, 10))
model:add(nn.LogSoftMax())

return model 
