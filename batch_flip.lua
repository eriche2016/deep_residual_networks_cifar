-- data augmentation module
require 'nn'
require 'image'

local BatchFlip,parent = torch.class('nn.BatchFlip', 'nn.Module')

function BatchFlip:__init()
    parent.__init(self)
    self.train = true
    -- zero padding 4 pixels along each side for simple data argumentation 
    self.module = nn.SpatialZeroPadding(4, 4, 4, 4)
    self.output = torch.Tensor()
end

function BatchFlip:updateOutput(inputs)
    if self.train then
        self.output:resizeAs(inputs):copy(inputs):zero() 
        --[[ method 1, can acheve test err:  
        for i=1,inputs:size(1) do
            if torch.uniform() < 0.5 then
                image.hflip(inputs[i], inputs[i]) 
            end
        end

        self.temp = self.module:forward(inputs)
   
        for i = 1, self.temp:size(1) do
            -- for every image, we need to random crop a 3 x 32 x 32 image
           -- crop out a 3 x 32 x 32 images 
           local start_x = torch.random(4)
           local start_y = torch.random(4)


            image.crop(self.output[i], self.temp[i], start_x, start_y, start_x+32, start_y+32)
        end 
        --]]
        -- [[ method 2
        for i = 1, inputs:size(1) do 
            -- Horizontal flip!!
            -- flip first
            if torch.random(1,2) == 1 then 
                self.output[i]:copy(image.hflip(self.output[i]))
            end

            -- crop the images 
            -- i-th image in the inputs
            local input = inputs[i]
            local xoffs, yoffs = torch.random(-4, 4), torch.random(-4, 4) 
            local input_y = {math.max(1,   1 + yoffs),math.min(32, 32 + yoffs)}
            local data_y = {math.max(1,   1 - yoffs),math.min(32, 32 - yoffs)}

            local input_x = {math.max(1,   1 + xoffs),math.min(32, 32 + xoffs)}
            local data_x = {math.max(1,   1 - xoffs), math.min(32, 32 - xoffs)}

            self.output[i][{{}, input_y, input_x }] = input[{ {}, data_y, data_x }]:clone() 

        end
        collectgarbage() 
   -- ]]
    else -- testing mode, set is okay, because we never modify inputs here 
        self.output:set(inputs)
    end

    return self.output
end
