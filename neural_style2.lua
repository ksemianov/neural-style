require 'torch'
require 'nn'
require 'image'
require 'optim'

require 'loadcaffe'


local cmd = torch.CmdLine()

-- Basic options
cmd:option('-style_image', 'examples/inputs/seated-nude.jpg',
           'Style target image')
cmd:option('-style_blend_weights', 'nil')
cmd:option('-content_image', 'examples/inputs/tubingen.jpg',
           'Content target image')
cmd:option('-image_size', 512, 'Maximum height / width of generated image')
cmd:option('-gpu', '0', 'Zero-indexed ID of the GPU to use; for CPU mode set -gpu = -1')
cmd:option('-multigpu_strategy', '', 'Index of layers to split the network across GPUs')

-- Optimization options
cmd:option('-content_weight', 1e1)
cmd:option('-style_weight', 1e2)
cmd:option('-histogram_weight', 5e2)
cmd:option('-mrf_weight', 1e-4)
cmd:option('-matching_nbins', 64)
cmd:option('-tv_weight', 1e-3)
cmd:option('-num_iterations', 1001)
cmd:option('-normalize_gradients', false)
cmd:option('-init', 'random', 'random|image')
cmd:option('-init_image', '')
cmd:option('-optimizer', 'lbfgs', 'lbfgs|adam')
cmd:option('-learning_rate', 1e1)
cmd:option('-lbfgs_num_correction', 0)

-- Output options
cmd:option('-print_iter', 50)
cmd:option('-save_iter', 100)
cmd:option('-output_image', 'out.png')

-- Other options
cmd:option('-style_scale', 1.0)
cmd:option('-original_colors', 0)
cmd:option('-pooling', 'max', 'max|avg')
cmd:option('-proto_file', 'models/VGG_ILSVRC_19_layers_deploy.prototxt')
cmd:option('-model_file', 'models/VGG_ILSVRC_19_layers.caffemodel')
cmd:option('-backend', 'nn', 'nn|cudnn|clnn')
cmd:option('-cudnn_autotune', false)
cmd:option('-seed', -1)

cmd:option('-content_layers', 'relu4_2', 'layers for content')
cmd:option('-style_layers', 'relu1_1,relu2_1,relu3_1,relu4_1,relu5_1', 'layers for style')
cmd:option('-histogram_layers', 'relu1_1,relu5_1', 'layers for histogram')
cmd:option('-mrf_layers', 'relu5_4', 'layers for MRF')

cmd:option('-mrf_patch_size', 3)
cmd:option('-target_sample_stride', 2)
cmd:option('-source_sample_stride', 2)
cmd:option('-mrf_confidence_threshold', 0)

cmd:option('-gpu_chunck_size_1', 256)
cmd:option('-gpu_chunck_size_2', 16)


local function main(params)
  local dtype, multigpu = setup_gpu(params)

  local loadcaffe_backend = params.backend
  if params.backend == 'clnn' then loadcaffe_backend = 'nn' end
  local cnn = loadcaffe.load(params.proto_file, params.model_file, loadcaffe_backend):type(dtype)

  local content_image = image.load(params.content_image, 3)
  content_image = image.scale(content_image, params.image_size, 'bilinear')
  local content_image_caffe = preprocess(content_image):float()
    
  local hist_image_list = params.style_image:split(',')
  local hist_image = image.load(hist_image_list[1], 3)
  hist_image = image.scale(hist_image, (#content_image)[3], (#content_image)[2])
  local hist_image_caffe = preprocess(hist_image):float()

  local style_size = math.ceil(params.style_scale * params.image_size)
  local style_image_list = params.style_image:split(',')
  local style_images_caffe = {}
  for _, img_path in ipairs(style_image_list) do
    local img = image.load(img_path, 3)
    img = image.scale(img, style_size, 'bilinear')
    local img_caffe = preprocess(img):float()
    table.insert(style_images_caffe, img_caffe)
  end
  
  

  local init_image = nil
  if params.init_image ~= '' then
    init_image = image.load(params.init_image, 3)
    local H, W = content_image:size(2), content_image:size(3)
    init_image = image.scale(init_image, W, H, 'bilinear')
    init_image = preprocess(init_image):float()
  end

  -- Handle style blending weights for multiple style inputs
  local style_blend_weights = nil
  if params.style_blend_weights == 'nil' then
    -- Style blending not specified, so use equal weighting
    style_blend_weights = {}
    for i = 1, #style_image_list do
      table.insert(style_blend_weights, 1.0)
    end
  else
    style_blend_weights = params.style_blend_weights:split(',')
    assert(#style_blend_weights == #style_image_list,
      '-style_blend_weights and -style_images must have the same number of elements')
  end
  -- Normalize the style blending weights so they sum to 1
  local style_blend_sum = 0
  for i = 1, #style_blend_weights do
    style_blend_weights[i] = tonumber(style_blend_weights[i])
    style_blend_sum = style_blend_sum + style_blend_weights[i]
  end
  for i = 1, #style_blend_weights do
    style_blend_weights[i] = style_blend_weights[i] / style_blend_sum
  end

  local content_layers = params.content_layers:split(",")
  local style_layers = params.style_layers:split(",")
  local histogram_layers = params.histogram_layers:split(",")
  local mrf_layers = params.mrf_layers:split(",")

  -- Set up the network, inserting style and content loss modules
  local content_losses, style_losses, histogram_losses, mrf_losses = {}, {}, {}, {}
  local next_content_idx, next_style_idx, next_histogram_idx, next_mrf_idx = 1, 1, 1, 1
  local mrf_idx = {}
  local net = nn.Sequential()
  if params.tv_weight > 0 then
    local tv_mod = nn.TVLoss(params.tv_weight):type(dtype)
    net:add(tv_mod)
  end



     -- --------------------------------------------------------------------------------------------------------
  -- -- local function for adding a mrf layer, with image rotation andn scaling
  -- --------------------------------------------------------------------------------------------------------

  local function build_mrf(id_mrf)
    --------------------------------------------------------
    -- deal with target
    --------------------------------------------------------
    local target_image_caffe = style_images_caffe[1]:cuda()
    
    -- compute the coordinates on the pixel layer
    local target_x
    local target_y
    local target_x_per_image = {}
    local target_y_per_image = {}
    local target_imageid

      net:forward(target_image_caffe)
      local target_feature_map = net:get(mrf_idx[id_mrf] - 1).output:float()

      if params.mrf_patch_size > target_feature_map:size()[2] or params.mrf_patch_size > target_feature_map:size()[3] then
        print('target_images is not big enough for patch')
        print('target_images size: ')
        print(target_feature_map:size())
        print('patch size: ')
        print(params.mrf_patch_size)
        do return end
      end
      local target_x_, target_y_ = drill_computeMRFfull(target_feature_map,  params.mrf_patch_size, params.target_sample_stride, -1)


      local x = torch.Tensor(target_x_:nElement() * target_y_:nElement())
      local y = torch.Tensor(target_x_:nElement() * target_y_:nElement())
      local target_imageid_ = torch.Tensor(target_x_:nElement() * target_y_:nElement()):fill(1)
      local count = 1
      for i_row = 1, target_y_:nElement() do
        for i_col = 1, target_x_:nElement() do
          x[count] = target_x_[i_col]
          y[count] = target_y_[i_row]
          count = count + 1
        end
      end

        target_x = x:clone()
        target_y = y:clone()
        target_imageid = target_imageid_:clone()

      table.insert(target_x_per_image, x)
      table.insert(target_y_per_image, y)

    local num_channel_mrf = net:get(mrf_idx[id_mrf] - 1).output:size()[1]
    local target_mrf = torch.Tensor(target_x:nElement(), num_channel_mrf * params.mrf_patch_size * params.mrf_patch_size)
    local tensor_target_mrf = torch.Tensor(target_x:nElement(), num_channel_mrf, params.mrf_patch_size, params.mrf_patch_size)
    local count_mrf = 1

      net:forward(target_image_caffe)
      -- sample mrf on mrf_layers
      local tensor_target_mrf_, target_mrf_ = sampleMRFAndTensorfromLocation2(target_x_per_image[1], target_y_per_image[1], net:get(mrf_idx[id_mrf] - 1).output:float(), params.mrf_patch_size)
      target_mrf[{{count_mrf, count_mrf + target_mrf_:size()[1] - 1}, {1, target_mrf:size()[2]}}] = target_mrf_:clone()
      tensor_target_mrf[{{count_mrf, count_mrf + target_mrf_:size()[1] - 1}, {1, tensor_target_mrf:size()[2]}, {1, tensor_target_mrf:size()[3]}, {1, tensor_target_mrf:size()[4]}}] = tensor_target_mrf_:clone()
      count_mrf = count_mrf + target_mrf_:size()[1]
      tensor_target_mrf_ = nil
      target_mrf_ = nil
      collectgarbage()

    local target_mrfnorm = torch.sqrt(torch.sum(torch.cmul(target_mrf, target_mrf), 2)):resize(target_mrf:size()[1], 1, 1)

    --------------------------------------------------------
    -- process source
    --------------------------------------------------------
    -- print('*****************************************************')
    -- print(string.format('process source image'));
    -- print('*****************************************************')

        net:forward(content_image_caffe:cuda())

    local source_feature_map = net:get(mrf_idx[id_mrf] - 1).output:float()
    if params.mrf_patch_size > source_feature_map:size()[2] or params.mrf_patch_size > source_feature_map:size()[3] then
      print('source_image_caffe is not big enough for patch')
      print('source_image_caffe size: ')
      print(source_feature_map:size())
      print('patch size: ')
      print(params.mrf_patch_size)
      do return end
    end
    local source_xgrid, source_ygrid = drill_computeMRFfull(source_feature_map:float(), params.mrf_patch_size, params.source_sample_stride, -1)
    local source_x = torch.Tensor(source_xgrid:nElement() * source_ygrid:nElement())
    local source_y = torch.Tensor(source_xgrid:nElement() * source_ygrid:nElement())
    local count = 1
    for i_row = 1, source_ygrid:nElement() do
      for i_col = 1, source_xgrid:nElement() do
        source_x[count] = source_xgrid[i_col]
        source_y[count] = source_ygrid[i_row]
        count = count + 1
      end
    end
    -- local tensor_target_mrfnorm = torch.repeatTensor(target_mrfnorm:float(), 1, net:get(mrf_layers[id_mrf] - 1).output:size()[2] - (params.mrf_patch_size[id_mrf] - 1), net:get(mrf_layers[id_mrf] - 1).output:size()[3] - (params.mrf_patch_size[id_mrf] - 1))

    -- print('*****************************************************')
    -- print(string.format('call layer implemetation'));
    -- print('*****************************************************')
    local nInputPlane = target_mrf:size()[2] / (params.mrf_patch_size * params.mrf_patch_size)
    local nOutputPlane = target_mrf:size()[1]
    local kW = params.mrf_patch_size
    local kH = params.mrf_patch_size
    local dW = 1
    local dH = 1
    local input_size = source_feature_map:size()

    local source_xgrid_, source_ygrid_ = drill_computeMRFfull(source_feature_map:float(), params.mrf_patch_size, 1, -1)
    local response_size = torch.LongStorage(3)
    response_size[1] = nOutputPlane
    response_size[2] = source_ygrid_:nElement()
    response_size[3] = source_xgrid_:nElement()
    net:get(mrf_idx[id_mrf]):implement(params.mode, target_mrf, tensor_target_mrf, target_mrfnorm, source_x, source_y, input_size, response_size, nInputPlane, nOutputPlane, kW, kH, 1, 1, params.mrf_confidence_threshold, params.mrf_weight, params.gpu_chunck_size_1, params.gpu_chunck_size_2, params.backend, params.gpu)
    target_mrf = nil
    tensor_target_mrf = nil
    source_feature_map = nil
    collectgarbage()
  end

  for i = 1, #cnn do
    if next_content_idx <= #content_layers or next_style_idx <= #style_layers or next_histogram_idx <= #histogram_layers or next_mrf_idx <= #mrf_layers then
      local layer = cnn:get(i)
      local name = layer.name
      local layer_type = torch.type(layer)
      local is_pooling = (layer_type == 'cudnn.SpatialMaxPooling' or layer_type == 'nn.SpatialMaxPooling')
      if is_pooling and params.pooling == 'avg' then
        assert(layer.padW == 0 and layer.padH == 0)
        local kW, kH = layer.kW, layer.kH
        local dW, dH = layer.dW, layer.dH
        local avg_pool_layer = nn.SpatialAveragePooling(kW, kH, dW, dH):type(dtype)
        local msg = 'Replacing max pooling at layer %d with average pooling'
        print(string.format(msg, i))
        net:add(avg_pool_layer)
      else
        net:add(layer)
      end
      if name == content_layers[next_content_idx] then
        print("Setting up content layer", i, ":", layer.name)
        local norm = params.normalize_gradients
        local loss_module = nn.ContentLoss(params.content_weight, norm):type(dtype)
        net:add(loss_module)
        table.insert(content_losses, loss_module)
        next_content_idx = next_content_idx + 1
      end
      if name == histogram_layers[next_histogram_idx] then
        print("Setting up histogram layer", i, ":", layer.name)
        local input_features = net:forward(content_image_caffe:type(dtype)):clone():float()
        local target_features = net:forward(hist_image_caffe:type(dtype)):clone():float()
        sizes = #input_features
        input_features = input_features:reshape(sizes[1],sizes[2]*sizes[3])
        target_features = target_features:reshape(sizes[1],sizes[2]*sizes[3])

        local input_features_remaped = input_features:clone()
          for j=1,sizes[1] do
            local nbins = params.matching_nbins
            local input_hist = torch.cumsum(torch.histc(input_features[{j}], nbins, 0, 1):div(nbins))
            local target_hist = torch.cumsum(torch.histc(target_features[{j}], nbins, 0, 1):div(nbins))

            local function Match(x)
              local input_bin = math.min(math.max(math.floor(x * nbins), 1), nbins)
              local input_dens = input_hist[input_bin]
              local l = 1
              local r = nbins + 1
              local m
              while (r - l > 1) do
                m = math.floor((l + r) / 2)
                if (target_hist[m] <= input_dens) then
                  l = m
                else
                  r = m
                end
              end

              return l / nbins
            end
            
            input_features_remaped[{j}] = input_features_remaped[{j}]:apply(Match)
          end

        local loss_module = nn.HistogramLoss(params.histogram_weight,
	input_features_remaped:type(dtype):resize(sizes[1],sizes[2],sizes[3]), false):type(dtype)
        
        net:add(loss_module)
        table.insert(histogram_losses, loss_module)
        next_histogram_idx = next_histogram_idx + 1
      end
      if name == style_layers[next_style_idx] then
        print("Setting up style layer  ", i, ":", layer.name)
        local norm = params.normalize_gradients
        local loss_module = nn.StyleLoss(params.style_weight, norm):type(dtype)
        net:add(loss_module)
        table.insert(style_losses, loss_module)
        next_style_idx = next_style_idx + 1
      end
      if name == mrf_layers[next_mrf_idx] then
        print("Setting up MRF layer  ", i, ":", layer.name)
        local loss_module = nn.MRFMM()
        loss_module:cuda()
        net:add(loss_module)
        table.insert(mrf_losses, loss_module)
        table.insert(mrf_idx, #net)
        next_mrf_idx = next_mrf_idx + 1
      end
    end
  end
  if multigpu then
    net = setup_multi_gpu(net, params)
  end
  net:type(dtype)

  -- Capture content targets
  for i = 1, #content_losses do
    content_losses[i].mode = 'capture'
  end
  print 'Capturing content targets'
  print(net)
  content_image_caffe = content_image_caffe:type(dtype)
  net:forward(content_image_caffe:type(dtype))
    
  -- Capture style targets
  for i = 1, #content_losses do
    content_losses[i].mode = 'none'
  end
  for i = 1, #style_images_caffe do
    print(string.format('Capturing style target %d', i))
    for j = 1, #style_losses do
      style_losses[j].mode = 'capture'
      style_losses[j].blend_weight = style_blend_weights[i]
    end
    net:forward(style_images_caffe[i]:type(dtype))
  end

  print 'Building MRF loss'
  for i = 1, #mrf_layers do
    build_mrf(i)
  end

  -- Set all loss modules to loss mode
  for i = 1, #content_losses do
    content_losses[i].mode = 'loss'
  end
  for i = 1, #style_losses do
    style_losses[i].mode = 'loss'
  end

  -- We don't need the base CNN anymore, so clean it up to save memory.
  cnn = nil
  for i=1, #net.modules do
    local module = net.modules[i]
    if torch.type(module) == 'nn.SpatialConvolutionMM' then
        -- remove these, not used, but uses gpu memory
        module.gradWeight = nil
        module.gradBias = nil
    end
  end
  collectgarbage()

  -- Initialize the image
  if params.seed >= 0 then
    torch.manualSeed(params.seed)
  end
  local img = nil
  if params.init == 'random' then
    img = torch.randn(content_image:size()):float():mul(0.001)
  elseif params.init == 'image' then
    if init_image then
      img = init_image:clone()
    else
      img = content_image_caffe:clone()
    end
  else
    error('Invalid init type')
  end
  img = img:type(dtype)

  -- Run it through the network once to get the proper size for the gradient
  -- All the gradients will come from the extra loss modules, so we just pass
  -- zeros into the top of the net on the backward pass.
  local y = net:forward(img)
  local dy = img.new(#y):zero()

  -- Declaring this here lets us access it in maybe_print
  local optim_state = nil
  if params.optimizer == 'lbfgs' then
    optim_state = {
      maxIter = params.num_iterations,
      verbose=true,
      tolX=-1,
      tolFun=-1,
    }
    if params.lbfgs_num_correction > 0 then
      optim_state.nCorrection = params.lbfgs_num_correction
    end
  elseif params.optimizer == 'adam' then
    optim_state = {
      learningRate = params.learning_rate,
    }
  else
    error(string.format('Unrecognized optimizer "%s"', params.optimizer))
  end

  local function maybe_print(t, loss)
    local verbose = (params.print_iter > 0 and t % params.print_iter == 0)
    if verbose then
      print(string.format('Iteration %d / %d', t, params.num_iterations))
      for i, loss_module in ipairs(content_losses) do
        print(string.format('  Content %d loss: %f', i, loss_module.loss))
      end
      for i, loss_module in ipairs(histogram_losses) do
        print(string.format('  Histogram %d loss: %f', i, loss_module.loss))
      end
      for i, loss_module in ipairs(style_losses) do
        print(string.format('  Style %d loss: %f', i, loss_module.loss))
      end
      for i, loss_module in ipairs(mrf_losses) do
        print(string.format('  MRF %d loss: %f', i, loss_module.loss))
      end
      print(string.format('  Total loss: %f', loss))
    end
  end

  local function maybe_save(t)
    local should_save = params.save_iter > 0 and t % params.save_iter == 0
    should_save = should_save or t == params.num_iterations
    if should_save then
      local disp = deprocess(img:double())
      disp = image.minmax{tensor=disp, min=0, max=1}
      local filename = build_filename(params.output_image, t)
      if t == params.num_iterations then
        filename = params.output_image
      end

      -- Maybe perform postprocessing for color-independent style transfer
      if params.original_colors == 1 then
        disp = original_colors(content_image, disp)
      end

      image.save(filename, disp)
    end
  end

  -- Function to evaluate loss and gradient. We run the net forward and
  -- backward to get the gradient, and sum up losses from the loss modules.
  -- optim.lbfgs internally handles iteration and calls this function many
  -- times, so we manually count the number of iterations to handle printing
  -- and saving intermediate results.
  local num_calls = 0
  local function feval(x)
    num_calls = num_calls + 1
    net:forward(x)
    local grad = net:updateGradInput(x, dy)
    local loss = 0
    for _, mod in ipairs(content_losses) do
      loss = loss + mod.loss
    end
    for _, mod in ipairs(style_losses) do
      loss = loss + mod.loss
    end
    for _, mod in ipairs(histogram_losses) do
      loss = loss + mod.loss
    end
    for _, mod in ipairs(mrf_losses) do
      loss = loss + mod.loss
    end
    maybe_print(num_calls, loss)
    maybe_save(num_calls)

    collectgarbage()
    -- optim.lbfgs expects a vector for gradients
    return loss, grad:view(grad:nElement())
  end

  -- Run optimization.
  if params.optimizer == 'lbfgs' then
    print('Running optimization with L-BFGS')
    local x, losses = optim.lbfgs(feval, img, optim_state)
  elseif params.optimizer == 'adam' then
    print('Running optimization with ADAM')
    for t = 1, params.num_iterations do
      local x, losses = optim.adam(feval, img, optim_state)
    end
  end
end


function setup_gpu(params)
  local multigpu = false
  if params.gpu:find(',') then
    multigpu = true
    params.gpu = params.gpu:split(',')
    for i = 1, #params.gpu do
      params.gpu[i] = tonumber(params.gpu[i]) + 1
    end
  else
    params.gpu = tonumber(params.gpu) + 1
  end
  local dtype = 'torch.FloatTensor'
  if multigpu or params.gpu > 0 then
    if params.backend ~= 'clnn' then
      require 'cutorch'
      require 'cunn'
      if multigpu then
        cutorch.setDevice(params.gpu[1])
      else
        cutorch.setDevice(params.gpu)
      end
      dtype = 'torch.CudaTensor'
    else
      require 'clnn'
      require 'cltorch'
      if multigpu then
        cltorch.setDevice(params.gpu[1])
      else
        cltorch.setDevice(params.gpu)
      end
      dtype = torch.Tensor():cl():type()
    end
  else
    params.backend = 'nn'
  end

  if params.backend == 'cudnn' then
    require 'cudnn'
    if params.cudnn_autotune then
      cudnn.benchmark = true
    end
    cudnn.SpatialConvolution.accGradParameters = nn.SpatialConvolutionMM.accGradParameters -- ie: nop
  end
  return dtype, multigpu
end


function setup_multi_gpu(net, params)
  local DEFAULT_STRATEGIES = {
    [2] = {3},
  }
  local gpu_splits = nil
  if params.multigpu_strategy == '' then
    -- Use a default strategy
    gpu_splits = DEFAULT_STRATEGIES[#params.gpu]
    -- Offset the default strategy by one if we are using TV
    if params.tv_weight > 0 then
      for i = 1, #gpu_splits do gpu_splits[i] = gpu_splits[i] + 1 end
    end
  else
    -- Use the user-specified multigpu strategy
    gpu_splits = params.multigpu_strategy:split(',')
    for i = 1, #gpu_splits do
      gpu_splits[i] = tonumber(gpu_splits[i])
    end
  end
  assert(gpu_splits ~= nil, 'Must specify -multigpu_strategy')
  local gpus = params.gpu

  local cur_chunk = nn.Sequential()
  local chunks = {}
  for i = 1, #net do
    cur_chunk:add(net:get(i))
    if i == gpu_splits[1] then
      table.remove(gpu_splits, 1)
      table.insert(chunks, cur_chunk)
      cur_chunk = nn.Sequential()
    end
  end
  table.insert(chunks, cur_chunk)
  assert(#chunks == #gpus)

  local new_net = nn.Sequential()
  for i = 1, #chunks do
    local out_device = nil
    if i == #chunks then
      out_device = gpus[1]
    end
    new_net:add(nn.GPU(chunks[i], gpus[i], out_device))
  end

  return new_net
end


function build_filename(output_image, iteration)
  local ext = paths.extname(output_image)
  local basename = paths.basename(output_image, ext)
  local directory = paths.dirname(output_image)
  return string.format('%s/%s_%d.%s',directory, basename, iteration, ext)
end


-- Preprocess an image before passing it to a Caffe model.
-- We need to rescale from [0, 1] to [0, 255], convert from RGB to BGR,
-- and subtract the mean pixel.
function preprocess(img)
  local mean_pixel = torch.DoubleTensor({103.939, 116.779, 123.68})
  local perm = torch.LongTensor{3, 2, 1}
  img = img:index(1, perm):mul(256.0)
  mean_pixel = mean_pixel:view(3, 1, 1):expandAs(img)
  img:add(-1, mean_pixel)
  return img
end


-- Undo the above preprocessing.
function deprocess(img)
  local mean_pixel = torch.DoubleTensor({103.939, 116.779, 123.68})
  mean_pixel = mean_pixel:view(3, 1, 1):expandAs(img)
  img = img + mean_pixel
  local perm = torch.LongTensor{3, 2, 1}
  img = img:index(1, perm):div(256.0)
  return img
end


-- Combine the Y channel of the generated image and the UV channels of the
-- content image to perform color-independent style transfer.
function original_colors(content, generated)
  local generated_y = image.rgb2yuv(generated)[{{1, 1}}]
  local content_uv = image.rgb2yuv(content)[{{2, 3}}]
  return image.yuv2rgb(torch.cat(generated_y, content_uv, 1))
end


-- Define an nn Module to compute content loss in-place
local ContentLoss, parent = torch.class('nn.ContentLoss', 'nn.Module')

function ContentLoss:__init(strength, normalize)
  parent.__init(self)
  self.strength = strength
  self.target = torch.Tensor()
  self.normalize = normalize or false
  self.loss = 0
  self.crit = nn.MSECriterion()
  self.mode = 'none'
end

function ContentLoss:updateOutput(input)
  if self.mode == 'loss' then
    self.loss = self.crit:forward(input, self.target) * self.strength
  elseif self.mode == 'capture' then
    self.target:resizeAs(input):copy(input)
  end
  self.output = input
  return self.output
end

function ContentLoss:updateGradInput(input, gradOutput)
  if self.mode == 'loss' then
    if input:nElement() == self.target:nElement() then
      self.gradInput = self.crit:backward(input, self.target)
    end
    if self.normalize then
      self.gradInput:div(torch.norm(self.gradInput, 1) + 1e-8)
    end
    self.gradInput:mul(self.strength)
    self.gradInput:add(gradOutput)
  else
    self.gradInput:resizeAs(gradOutput):copy(gradOutput)
  end
  return self.gradInput
end


-- Define an nn Module to compute histogram loss in-place
local HistogramLoss, parent = torch.class('nn.HistogramLoss', 'nn.Module')

function HistogramLoss:__init(strength, target, normalize)
  parent.__init(self)
  self.strength = strength
  self.target = target
  self.normalize = normalize or false
  self.loss = 0
  self.crit = nn.MSECriterion()
end

function HistogramLoss:updateOutput(input)
  if input:nElement() == self.target:nElement() then
    self.loss = self.crit:forward(input, self.target) * self.strength
  else
    print('WARNING: Skipping histogram loss')
    print(#input, #self.target)
  end
  self.output = input
  return self.output
end

function HistogramLoss:updateGradInput(input, gradOutput)
  if input:nElement() == self.target:nElement() then
    self.gradInput = self.crit:backward(input, self.target)
  end
  if self.normalize then
    self.gradInput:div(torch.norm(self.gradInput, 1) + 1e-8)
  end
  self.gradInput:mul(self.strength)
  self.gradInput:add(gradOutput)
  return self.gradInput
end


local Gram, parent = torch.class('nn.GramMatrix', 'nn.Module')

function Gram:__init()
  parent.__init(self)
end

function Gram:updateOutput(input)
  assert(input:dim() == 3)
  local C, H, W = input:size(1), input:size(2), input:size(3)
  local x_flat = input:view(C, H * W)
  self.output:resize(C, C)
  self.output:mm(x_flat, x_flat:t())
  return self.output
end

function Gram:updateGradInput(input, gradOutput)
  assert(input:dim() == 3 and input:size(1))
  local C, H, W = input:size(1), input:size(2), input:size(3)
  local x_flat = input:view(C, H * W)
  self.gradInput:resize(C, H * W):mm(gradOutput, x_flat)
  self.gradInput:addmm(gradOutput:t(), x_flat)
  self.gradInput = self.gradInput:view(C, H, W)
  return self.gradInput
end


-- Define an nn Module to compute style loss in-place
local StyleLoss, parent = torch.class('nn.StyleLoss', 'nn.Module')

function StyleLoss:__init(strength, normalize)
  parent.__init(self)
  self.normalize = normalize or false
  self.strength = strength
  self.target = torch.Tensor()
  self.mode = 'none'
  self.loss = 0

  self.gram = nn.GramMatrix()
  self.blend_weight = nil
  self.G = nil
  self.crit = nn.MSECriterion()
end

function StyleLoss:updateOutput(input)
  self.G = self.gram:forward(input)
  self.G:div(input:nElement())
  if self.mode == 'capture' then
    if self.blend_weight == nil then
      self.target:resizeAs(self.G):copy(self.G)
    elseif self.target:nElement() == 0 then
      self.target:resizeAs(self.G):copy(self.G):mul(self.blend_weight)
    else
      self.target:add(self.blend_weight, self.G)
    end
  elseif self.mode == 'loss' then
    self.loss = self.strength * self.crit:forward(self.G, self.target)
  end
  self.output = input
  return self.output
end

function StyleLoss:updateGradInput(input, gradOutput)
  if self.mode == 'loss' then
    local dG = self.crit:backward(self.G, self.target)
    dG:div(input:nElement())
    self.gradInput = self.gram:backward(input, dG)
    if self.normalize then
      self.gradInput:div(torch.norm(self.gradInput, 1) + 1e-8)
    end
    self.gradInput:mul(self.strength)
    self.gradInput:add(gradOutput)
  else
    self.gradInput = gradOutput
  end
  return self.gradInput
end


local TVLoss, parent = torch.class('nn.TVLoss', 'nn.Module')

function TVLoss:__init(strength)
  parent.__init(self)
  self.strength = strength
  self.x_diff = torch.Tensor()
  self.y_diff = torch.Tensor()
end

function TVLoss:updateOutput(input)
  self.output = input
  return self.output
end

-- TV loss backward pass inspired by kaishengtai/neuralart
function TVLoss:updateGradInput(input, gradOutput)
  self.gradInput:resizeAs(input):zero()
  local C, H, W = input:size(1), input:size(2), input:size(3)
  self.x_diff:resize(3, H - 1, W - 1)
  self.y_diff:resize(3, H - 1, W - 1)
  self.x_diff:copy(input[{{}, {1, -2}, {1, -2}}])
  self.x_diff:add(-1, input[{{}, {1, -2}, {2, -1}}])
  self.y_diff:copy(input[{{}, {1, -2}, {1, -2}}])
  self.y_diff:add(-1, input[{{}, {2, -1}, {1, -2}}])
  self.gradInput[{{}, {1, -2}, {1, -2}}]:add(self.x_diff):add(self.y_diff)
  self.gradInput[{{}, {1, -2}, {2, -1}}]:add(-1, self.x_diff)
  self.gradInput[{{}, {2, -1}, {1, -2}}]:add(-1, self.y_diff)
  self.gradInput:mul(self.strength)
  self.gradInput:add(gradOutput)
  return self.gradInput
end

function computeMRF(input, size, stride, gpu, backend)
  local coord_x, coord_y = computegrid(input:size()[3], input:size()[2], size, stride)
  local dim_1 = input:size()[1] * size * size
  local dim_2 = coord_y:nElement()
  local dim_3 = coord_x:nElement()
  local t_feature_mrf = torch.Tensor(dim_2 * dim_3, input:size()[1], size, size)

  t_feature_mrf = t_feature_mrf:cuda()
  local count = 1
  for i_row = 1, dim_2 do
    for i_col = 1, dim_3 do
      t_feature_mrf[count] = input[{{1, input:size()[1]}, {coord_y[i_row], coord_y[i_row] + size - 1}, {coord_x[i_col], coord_x[i_col] + size - 1}}]
      count = count + 1
    end
  end
  local feature_mrf = t_feature_mrf:resize(dim_2 * dim_3, dim_1)

  return t_feature_mrf, feature_mrf, coord_x, coord_y
end


function computeMRFnoTensor(input, size, stride, backend)
  local coord_x, coord_y = computegrid(input:size()[3], input:size()[2], size, stride)
  local dim_1 = input:size()[1] * size * size
  local dim_2 = coord_y:nElement()
  local dim_3 = coord_x:nElement()
  local t_feature_mrf = torch.Tensor(dim_2 * dim_3, input:size()[1], size, size)

  t_feature_mrf = t_feature_mrf:cuda()
  local count = 1
  for i_row = 1, dim_2 do
    for i_col = 1, dim_3 do
      t_feature_mrf[count] = input[{{1, input:size()[1]}, {coord_y[i_row], coord_y[i_row] + size - 1}, {coord_x[i_col], coord_x[i_col] + size - 1}}]
      count = count + 1
    end
  end
  local feature_mrf = t_feature_mrf:resize(dim_2 * dim_3, dim_1)

    t_feature_mrf = nil
    collectgarbage()
  return feature_mrf, coord_x, coord_y
end


function drill_computeMRFfull(input, size, stride, gpu)
  local coord_x, coord_y = computegrid(input:size()[3], input:size()[2], size, stride, 1)
  local dim = torch.Tensor(2)
  return coord_x, coord_y
end


function sampleMRFAndTensorfromLocation2(coord_x, coord_y, input, size, gpu)
  local t_feature_mrf = torch.Tensor(coord_x:nElement(), input:size()[1], size, size)
  for i_patch = 1, coord_x:nElement() do
    t_feature_mrf[i_patch] = input[{{1, input:size()[1]}, {coord_y[i_patch], coord_y[i_patch] + size - 1}, {coord_x[i_patch], coord_x[i_patch] + size - 1}}]
  end
    local feature_mrf = t_feature_mrf:reshape(coord_x:nElement(), input:size()[1] * size * size)
  return t_feature_mrf, feature_mrf
end


function computeBB(width, height, alpha)
  local min_x, min_y, max_x, max_y
  local x1 = 1
  local y1 = 1
  local x2 = width
  local y2 = 1
  local x3 = width
  local y3 = height
  local x4 = 1
  local y4 = height
  local x0 = width / 2
  local y0 = height / 2

  local x1r = x0+(x1-x0)*math.cos(alpha)+(y1-y0)*math.sin(alpha)
  local y1r = y0-(x1-x0)*math.sin(alpha)+(y1-y0)*math.cos(alpha)

  local x2r = x0+(x2-x0)*math.cos(alpha)+(y2-y0)*math.sin(alpha)
  local y2r = y0-(x2-x0)*math.sin(alpha)+(y2-y0)*math.cos(alpha)

  local x3r = x0+(x3-x0)*math.cos(alpha)+(y3-y0)*math.sin(alpha)
  local y3r = y0-(x3-x0)*math.sin(alpha)+(y3-y0)*math.cos(alpha)

  local x4r = x0+(x4-x0)*math.cos(alpha)+(y4-y0)*math.sin(alpha)
  local y4r = y0-(x4-x0)*math.sin(alpha)+(y4-y0)*math.cos(alpha)

  -- print(x1r .. ' ' .. y1r .. ' ' .. x2r .. ' ' .. y2r .. ' ' .. x3r .. ' ' .. y3r .. ' ' .. x4r .. ' ' .. y4r)
  if alpha > 0 then
    -- find intersection P of line [x1, y1]-[x4, y4] and [x1r, y1r]-[x2r, y2r]
    local px1 = ((x1 * y4 - y1 * x4) * (x1r - x2r) - (x1 - x4) * (x1r * y2r - y1r * x2r)) / ((x1 - x4) * (y1r - y2r) - (y1 - y4) * (x1r - x2r))
    local py1 = ((x1 * y4 - y1 * x4) * (y1r - y2r) - (y1 - y4) * (x1r * y2r - y1r * x2r)) / ((x1 - x4) * (y1r - y2r) - (y1 - y4) * (x1r - x2r))
    local px2 = px1 + 1
    local py2 = py1
    -- print(px1 .. ' ' .. py1)
    -- find the intersection Q of line [px1, py1]-[px2, py2] and [x2r, y2r]-[x3r][y3r]

    local qx = ((px1 * py2 - py1 * px2) * (x2r - x3r) - (px1 - px2) * (x2r * y3r - y2r * x3r)) / ((px1 - px2) * (y2r - y3r) - (py1 - py2) * (x2r - x3r))
    local qy = ((px1 * py2 - py1 * px2) * (y2r - y3r) - (py1 - py2) * (x2r * y3r - y2r * x3r)) / ((px1 - px2) * (y2r - y3r) - (py1 - py2) * (x2r - x3r))  
    -- print(qx .. ' ' .. qy)

    min_x = width - qx
    min_y = qy
    max_x = qx
    max_y = height - qy
  else if alpha < 0 then
    -- find intersection P of line [x2, y2]-[x3, y3] and [x1r, y1r]-[x2r, y2r]
    local px1 = ((x2 * y3 - y2 * x3) * (x1r - x2r) - (x2 - x3) * (x1r * y2r - y1r * x2r)) / ((x2 - x3) * (y1r - y2r) - (y2 - y3) * (x1r - x2r))
    local py1 = ((x2 * y3 - y1 * x3) * (y1r - y2r) - (y2 - y3) * (x1r * y2r - y1r * x2r)) / ((x2 - x3) * (y1r - y2r) - (y2 - y3) * (x1r - x2r))
    local px2 = px1 - 1
    local py2 = py1
    -- find the intersection Q of line [px1, py1]-[px2, py2] and [x1r, y1r]-[x4r][y4r]
    local qx = ((px1 * py2 - py1 * px2) * (x1r - x4r) - (px1 - px2) * (x1r * y4r - y1r * x4r)) / ((px1 - px2) * (y1r - y4r) - (py1 - py2) * (x1r - x4r))
    local qy = ((px1 * py2 - py1 * px2) * (y1r - y4r) - (py1 - py2) * (x1r * y4r - y1r * x4r)) / ((px1 - px2) * (y1r - y4r) - (py1 - py2) * (x1r - x4r))  
    min_x = qx
    min_y = qy
    max_x = width - min_x
    max_y = height - min_y
    else
      min_x = x1
      min_y = y1
      max_x = x2
      max_y = y3
    end
  end

  return math.max(math.floor(min_x), 1), math.max(math.floor(min_y), 1), math.floor(max_x), math.floor(max_y)
end

function computegrid(width, height, block_size, block_stride, flag_all)
  local coord_block_y = torch.range(1, height - block_size + 1, block_stride)
  if flag_all == 1 then
    if coord_block_y[#coord_block_y] < height - block_size + 1 then
      local tail = torch.Tensor(1)
      tail[1] = height - block_size + 1
      coord_block_y = torch.cat(coord_block_y, tail)
    end
  end
  local coord_block_x = torch.range(1, width - block_size + 1, block_stride)
  if flag_all == 1 then
    if coord_block_x[#coord_block_x] < width - block_size + 1 then
      local tail = torch.Tensor(1)
      tail[1] = width - block_size + 1
      coord_block_x = torch.cat(coord_block_x, tail)
    end
  end
  return coord_block_x, coord_block_y
end

local MRFMM, parent = torch.class('nn.MRFMM', 'nn.Module')

function MRFMM:__init()
   parent.__init(self)
   self.loss = 0
end

function MRFMM:implement(mode, target_mrf, tensor_target_mrf, target_mrfnorm, source_x, source_y, input_size, response_size, nInputPlane, nOutputPlane, kW, kH, dW, dH, threshold_conf, strength, gpu_chunck_size_1, gpu_chunck_size_2, backend, gpu)
  self.target_mrf = target_mrf:clone()
  self.target_mrfnorm = target_mrfnorm:clone()
  self.source_x = source_x
  self.source_y = source_y
  self.input_size = input_size
  self.nInputPlane = nInputPlane
  self.nOutputPlane = nOutputPlane
  self.kW = kW
  self.kH = kH
  self.dW = dW
  self.dH = dH
  self.threshold_conf = threshold_conf
  self.strength = strength
  self.padW = padW or 0
  self.padH = padH or self.padW
  self.bias = torch.Tensor(nOutputPlane):fill(0)
  self.backend = backend
  self.gpu = gpu
  self.bias = self.bias:cuda()
  self.gradTO = torch.Tensor(input_size[1], input_size[2], input_size[3])
  self.gradTO_confident = torch.Tensor(input_size[2], input_size[3])
  self.response = torch.Tensor(response_size[1], response_size[2], response_size[3])
  self.gpu_chunck_size_1 = gpu_chunck_size_1
  self.gpu_chunck_size_2 = gpu_chunck_size_2
  self.tensor_target_mrfnorm = torch.repeatTensor(target_mrfnorm, 1, self.gpu_chunck_size_2, input_size[3] - (kW - 1))
  
  self.target_mrf = self.target_mrf:cuda()
  self.target_mrfnorm = self.target_mrfnorm:cuda()
  self.tensor_target_mrfnorm = self.tensor_target_mrfnorm:cuda()
  self.gradTO = self.gradTO:cuda()
  self.gradTO_confident = self.gradTO_confident:cuda()
  self.response = self.response:cuda()
end


local function makeContiguous(self, input, gradOutput)
   if not input:isContiguous() then
    print('not contiguous, make it so')
      self._input = self._input or input.new()
      self._input:resizeAs(input):copy(input)
      input = self._input
   end
   if gradOutput then
      if not gradOutput:isContiguous() then
   self._gradOutput = self._gradOutput or gradOutput.new()
   self._gradOutput:resizeAs(gradOutput):copy(gradOutput)
   gradOutput = self._gradOutput
      end
   end
   return input, gradOutput
end

function MRFMM:updateOutput(input)
    input = makeContiguous(self, input)
    self.output = input:clone()
    return self.output
end

function MRFMM:updateGradInput(input, gradOutput)
  input = makeContiguous(self, input)
  self.gradTO = self.gradTO:fill(0)
  self.gradTO_confident = self.gradTO_confident:fill(0) + 1e-10
  local source_mrf, x, y = computeMRFnoTensor(input:float(), self.kW, 1, self.backend)
  local source_mrfnorm = torch.Tensor(source_mrf:size()[1])
  source_mrfnorm = torch.sqrt(torch.sum(torch.cmul(source_mrf, source_mrf), 2)):resize(1, y:nElement(), x:nElement())
  local tensor_source_mrfnorm = torch.repeatTensor(source_mrfnorm, self.gpu_chunck_size_1, 1, 1)
  tensor_source_mrfnorm = tensor_source_mrfnorm:cuda()
  local nOutputPlane_all = self.nOutputPlane -- hacked for memory safety
  local num_chunk = math.ceil(nOutputPlane_all / self.gpu_chunck_size_1) 

  for i_chunk = 1, num_chunk do
    local i_start = (i_chunk - 1) * self.gpu_chunck_size_1 + 1
    local i_end = math.min(i_start + self.gpu_chunck_size_1 - 1, nOutputPlane_all)

    self.weight = self.target_mrf[{{i_start, i_end}, {1, self.target_mrf:size()[2]}}]

    self.nOutputPlane = i_end - i_start + 1

    local subBias = self.bias:sub(i_start, i_end)
    if self.gpu < 0 then
      self.finput = torch.Tensor()
      self.fgradInput = torch.Tensor()
    end

    input.THNN.SpatialConvolutionMM_updateOutput(
      input:cdata(),
      self.output:cdata(),
      self.weight:cdata(),
      subBias:cdata(),
      self.finput:cdata(),
      self.fgradInput:cdata(),
      self.kW, self.kH,
      self.dW, self.dH,
      self.padW, self.padH
    )
    local temp = self.output

    -- normalize w.r.t source_mrfnorm
    if i_chunk < num_chunk then
        temp = temp:cdiv(tensor_source_mrfnorm)
    else
        temp = temp:cdiv(tensor_source_mrfnorm[{{1, i_end - i_start + 1}, {1, temp:size()[2]}, {1, temp:size()[3]}}])
    end

    self.response[{{i_start, i_end}, {1, self.response:size()[2]}, {1, self.response:size()[3]}}] = temp
  end

  local num_chunk_2 = math.ceil(self.response:size()[2] / self.gpu_chunck_size_2) 
  for i_chunk_2 = 1, num_chunk_2 do
    local i_start = (i_chunk_2 - 1) * self.gpu_chunck_size_2 + 1
    local i_end = math.min(i_start + self.gpu_chunck_size_2 - 1, self.response:size()[2])
      if i_chunk_2 < num_chunk_2 then
        self.response[{{1, self.response:size()[1]}, {i_start, i_end}, {1, self.response:size()[3]}}] = self.response[{{1, self.response:size()[1]}, {i_start, i_end}, {1, self.response:size()[3]}}]:cdiv(self.tensor_target_mrfnorm)
      else
        self.response[{{1, self.response:size()[1]}, {i_start, i_end}, {1, self.response:size()[3]}}] = self.response[{{1, self.response:size()[1]}, {i_start, i_end}, {1, self.response:size()[3]}}]:cdiv(self.tensor_target_mrfnorm[{{1, self.response:size()[1]}, {1, i_end - i_start + 1}, {1, self.response:size()[3]}}])
      end
  end

  local max_response, max_id = torch.max(self.response, 1)
  source_mrf = source_mrf:resize(source_mrf:size()[1], self.nInputPlane, self.kW, self.kH)
  self.target_mrf = self.target_mrf:resize(self.target_mrf:size()[1], self.nInputPlane, self.kW, self.kH)
  for i_patch = 1, self.source_x:nElement() do
      local sel_response = max_response[1][self.source_y[i_patch]][self.source_x[i_patch]]
      if sel_response >= self.threshold_conf then
        local sel_idx = max_id[1][self.source_y[i_patch]][self.source_x[i_patch]]
        local source_idx = (self.source_y[i_patch] - 1) * x:nElement() + self.source_x[i_patch]        
        self.gradTO[{{1, self.nInputPlane}, {self.source_y[i_patch], self.source_y[i_patch] + self.kH - 1}, {self.source_x[i_patch], self.source_x[i_patch] + self.kW - 1}}]:add(self.target_mrf[sel_idx] - source_mrf[source_idx])
        self.gradTO_confident[{{self.source_y[i_patch], self.source_y[i_patch] + self.kH - 1}, {self.source_x[i_patch], self.source_x[i_patch] + self.kW - 1}}]:add(1)    
      end
  end
  self.gradTO:cdiv(torch.repeatTensor(self.gradTO_confident, self.nInputPlane, 1, 1))
  self.nOutputPlane = nOutputPlane_all
  self.target_mrf = self.target_mrf:resize(self.target_mrf:size()[1], self.nInputPlane * self.kW * self.kH)

  if gradOutput:size()[1] == input:size()[1] then
    self.gradInput = gradOutput:clone() + self.gradTO:cuda() * self.strength * (-1)
  else
    self.gradInput = self.gradTO * self.strength * (-1)
  end
  source_mrf = nil
  source_mrfnorm = nil
  tensor_source_mrfnorm = nil
  collectgarbage()
  return self.gradInput
end

function MRFMM:type(type)
   self.finput = torch.Tensor()
   self.fgradInput = torch.Tensor()
   return parent.type(self,type)
end




local params = cmd:parse(arg)
main(params)
