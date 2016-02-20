-- data augmentation module
require 'nn'

local BatchFlip,parent = torch.class('nn.BatchFlip', 'nn.Module')

function BatchFlip:__init()
    parent.__init(self)
    self.train = true
    -- zero padding 4 pixels along each side for simple data argumentation 
    self.module = nn.SpatialZeroPadding(4, 4, 4, 4)
end

function BatchFlip:updateOutput(input)
    if self.train then
        self.output:resizeAs(input):copy(input)

        local bs = input:size(1)
        local flip_mask = torch.randperm(bs):le(bs/2)
        for i=1,input:size(1) do
            if flip_mask[i] == 1 then image.hflip(input[i], input[i]) end
        end

        self.temp = self.module:forward(input)
   
        -- crop out a 3 x 32 x 32 images 
        local start_x = torch.random(4)
        local start_y = torch.random(4)

        for i = 1, self.temp:size(1) do
            image.crop(self.output[i], self.temp[i], start_x, start_y, start_x+32, start_y+32)
        end 

    else 
        self.output:set(input)
    end 

    return self.output
end
