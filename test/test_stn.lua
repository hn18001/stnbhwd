require 'nn'
require 'cunn'
require 'cudnn'
require 'image'

require 'stn'
require 'spatial_transformer'

torch.setdefaulttensortype("torch.FloatTensor")

model = nn.Sequential()
mdel:add(spanet)
model:cuda()

--img = image.lena()
img = image.load("./1.jpg")
gray = image.rgb2y(img)
input = gray:view(1, 1, gray:size(2), gray:size(3))
output = model:forward(input:cuda())
output = image.toDisplayTensor(output)
image.save("output.jpg", output)
