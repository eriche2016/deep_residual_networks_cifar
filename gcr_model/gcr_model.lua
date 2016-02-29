--[[
Copyright (c) 2016 Michael Wilber

This software is provided 'as-is', without any express or implied
warranty. In no event will the authors be held liable for any damages
arising from the use of this software.

Permission is granted to anyone to use this software for any purpose,
including commercial applications, and to alter it and redistribute it
freely, subject to the following restrictions:

1. The origin of this software must not be misrepresented; you must not
   claim that you wrote the original software. If you use this software
   in a product, an acknowledgement in the product documentation would be
   appreciated but is not required.
2. Altered source versions must be plainly marked as such, and must not be
   misrepresented as being the original software.
3. This notice may not be removed or altered from any source distribution.
--]]

require 'nn'
require 'nngraph'
require 'cudnn'
require 'cunn'
local nninit = require 'nninit'

function addResidualLayer2(input,  nChannels, nOutChannels, stride)
   --[[

   Residual layers! Implements option (A) from Section 3.3. The input
   is passed through two 3x3 convolution filters. In parallel, if the
   number of input and output channels differ or if the stride is not
   1, then the input is downsampled or zero-padded to have the correct
   size and number of channels. Finally, the two versions of the input
   are added together.

               Input
                 |
         ,-------+-----.
   Downsampling      3x3 convolution+dimensionality reduction
        |               |
        v               v
   Zero-padding      3x3 convolution
        |               |
        `-----( Add )---'
                 |
              Output
   --]]
   nOutChannels = nOutChannels or nChannels
   stride = stride or 1
   -- Path 1: Convolution
   -- The first layer does the downsampling and the striding
   local net = cudnn.SpatialConvolution(nChannels, nOutChannels,
                                           3,3, stride,stride, 1,1)
                                           :init('weight', nninit.kaiming, {gain = 'relu'})
                                           :init('bias', nninit.constant, 0)(input)
   net = cudnn.SpatialBatchNormalization(nOutChannels)
                                            :init('weight', nninit.normal, 1.0, 0.002)
                                            :init('bias', nninit.constant, 0)(net)
   net = cudnn.ReLU(true)(net)
   net = cudnn.SpatialConvolution(nOutChannels, nOutChannels,
                                      3,3, 1,1, 1,1)
                                      :init('weight', nninit.kaiming, {gain = 'relu'})
                                      :init('bias', nninit.constant, 0)(net)
   -- Should we put Batch Normalization here? I think not, because
   -- BN would force the output to have unit variance, which breaks the residual
   -- property of the network.
   -- What about ReLU here? I think maybe not for the same reason. Figure 2
   -- implies that they don't use it here

   -- Path 2: Identity / skip connection
   local skip = input
   if stride > 1 then
       -- optional downsampling
       skip = nn.SpatialAveragePooling(1, 1, stride,stride)(skip)
   end
   if nOutChannels > nChannels then
       -- optional padding
       skip = nn.Padding(1, (nOutChannels - nChannels), 3)(skip)
   elseif nOutChannels < nChannels then
       -- optional narrow, ugh.
       skip = nn.Narrow(2, 1, nOutChannels)(skip)
       -- NOTE this BREAKS with non-batch inputs!!
   end

   -- Add them together
   net = cudnn.SpatialBatchNormalization(nOutChannels)(net)
   net = nn.CAddTable(){net, skip}
   net = cudnn.ReLU(true)(net)
   -- ^ don't put a ReLU here! see http://gitxiv.com/comments/7rffyqcPLirEEsmpX

   return net
end

N = 9 -- 6 * 9 + 2 = 56 layers

input = nn.Identity()()
------> 3, 32,32
model = cudnn.SpatialConvolution(3, 16, 3,3, 1,1, 1,1)
			:init('weight', nninit.kaiming, {gain = 'relu'})
			:init('bias', nninit.constant, 0)(input)
model = cudnn.SpatialBatchNormalization(16)(model)
model = cudnn.ReLU(true)(model)
------> 16, 32,32   First Group
for i=1,N do   model = addResidualLayer2(model, 16)   end
------> 32, 16,16   Second Group
model = addResidualLayer2(model, 16, 32, 2)
for i=1,N-1 do   model = addResidualLayer2(model, 32)   end
------> 64, 8,8     Third Group
model = addResidualLayer2(model, 32, 64, 2)
for i=1,N-1 do   model = addResidualLayer2(model, 64)   end
------> 10, 8,8     Pooling, Linear, Softmax
model = nn.SpatialAveragePooling(8,8)(model)
model = nn.Reshape(64)(model)
model = nn.Linear(64, 10)(model)
model = nn.LogSoftMax()(model)

model = nn.gModule({input}, {model})

return model


