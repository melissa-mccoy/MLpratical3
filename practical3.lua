---------------------------------------------------------------------------------------
-- Practical 3 - Learning to use different optimizers with logistic regression
--
-- to run: th -i practical3.lua
-- or:     luajit -i practical3.lua
---------------------------------------------------------------------------------------

require 'torch'
require 'math'
require 'nn'
require 'optim'
require 'gnuplot'
require 'dataset-mnist'
gnuplot.raw('gnuplot --persist')

------------------------------------------------------------------------------
-- INITIALIZATION AND DATA
------------------------------------------------------------------------------

torch.manualSeed(1)    -- fix random seed so program runs the same every time

-- TODO: play with these optimizer options for the second handin item, as described in the writeup
-- NOTE: see below for optimState, storing optimiser settings
local opt = {}         -- these options are used throughout
opt.optimization = 'adagrad'
opt.batch_size = 5
opt.train_size = 8000  -- set to 0 or 60000 to use all 60000 training data
opt.test_size = 0      -- 0 means load all data
opt.epochs = 3         -- **approximate** number of passes through the training data (see below for the `iterations` variable, which is calculated from this)

--Best Adagrad--
-- The training error is:
-- 0.840625
-- The test error is:
-- 0.837
-- opt.optimization = 'adagrad'
-- opt.batch_size = 3
-- opt.train_size = 8000  -- set to 0 or 60000 to use all 60000 training data
-- opt.test_size = 0      -- 0 means load all data
-- opt.epochs = 3         -- **approximate** number of passes through the training data (see below for the `iterations` variable, which is calculated from this)

--Best SGD--
-- The training error is:
-- 0.840625
-- The test error is:
-- 0.837
-- opt.batch_size = 3
-- opt.train_size = 8000  -- set to 0 or 60000 to use all 60000 training data
-- opt.test_size = 0      -- 0 means load all data
-- opt.epochs = 15         -- **approximate** number of passes through the training data (see below for the `iterations` variable, which is calculated from this)

-- NOTE: the code below changes the optimization algorithm used, and its settings
local optimState       -- stores a lua table with the optimization algorithm's settings, and state during iterations
local optimMethod      -- stores a function corresponding to the optimization routine
-- remember, the defaults below are not necessarily good
if opt.optimization == 'lbfgs' then
  optimState = {
    learningRate = 1e-1,
    maxIter = 2,
    nCorrection = 10
  }
  optimMethod = optim.lbfgs
elseif opt.optimization == 'sgd' then
  optimState = {
    learningRate = 1e-1,
    weightDecay = 0,
    momentum = 0,
    learningRateDecay = 1e-7
  }
  optimMethod = optim.sgd
elseif opt.optimization == 'adagrad' then
  optimState = {
    learningRate = 1e-1,
  }
  optimMethod = optim.adagrad
else
  error('Unknown optimizer')
end

mnist.download()       -- download dataset if not already there

-- load dataset using dataset-mnist.lua into tensors (first dim of data/labels ranges over data)
local function load_dataset(train_or_test, count)
    -- load
    local data
    if train_or_test == 'train' then
        data = mnist.loadTrainSet(count, {32, 32})
    else
        data = mnist.loadTestSet(count, {32, 32})
    end

    -- shuffle the dataset
    local shuffled_indices = torch.randperm(data.data:size(1)):long()
    -- creates a shuffled *copy*, with a new storage
    data.data = data.data:index(1, shuffled_indices):squeeze()
    data.labels = data.labels:index(1, shuffled_indices):squeeze()

    -- TODO: (optional) UNCOMMENT to display a training example
    -- for more, see torch gnuplot package documentation:
    -- https://github.com/torch/gnuplot#plotting-package-manual-with-gnuplot
    -- gnuplot.imagesc(data.data[10])

    -- vectorize each 2D data point into 1D
    data.data = data.data:reshape(data.data:size(1), 32*32)

    print('--------------------------------')
    print(' loaded dataset "' .. train_or_test .. '"')
    print('inputs', data.data:size())
    print('targets', data.labels:size())
    print('--------------------------------')

    return data
end

local train = load_dataset('train', opt.train_size)
local test = load_dataset('test', opt.test_size)

------------------------------------------------------------------------------
-- MODEL
------------------------------------------------------------------------------
-- Train Data Model
local n_train_data = train.data:size(1) -- number of training data
local n_inputs = train.data:size(2)     -- number of cols = number of dims of input
local n_outputs = train.labels:max()    -- highest label = # of classes

print(train.labels:max())
print(train.labels:min())

local lin_layer = nn.Linear(n_inputs, n_outputs)
local softmax = nn.LogSoftMax()
local model = nn.Sequential()
model:add(lin_layer)
model:add(softmax)

-- Test Data Model
local n_test_data = test.data:size(1) -- number of testing data
local n_test_inputs = test.data:size(2)     -- number of cols = number of dims of input
local n_test_outputs = test.labels:max()    -- highest label = # of classes

print(test.labels:max())
print(test.labels:min())

local lin_layer_test = nn.Linear(n_test_inputs, n_test_outputs)
local softmax_test = nn.LogSoftMax()
local model_test = nn.Sequential()
model_test:add(lin_layer)
model_test:add(softmax)

------------------------------------------------------------------------------
-- LOSS FUNCTION
------------------------------------------------------------------------------

local criterion = nn.ClassNLLCriterion()
local criterion_test = nn.ClassNLLCriterion()

------------------------------------------------------------------------------
-- TRAINING
------------------------------------------------------------------------------

local parameters, gradParameters = model:getParameters()
local testParameters, testGradParameters = model_test:getParameters()

------------------------------------------------------------------------
-- Define closure with mini-batches
------------------------------------------------------------------------

local counter = 0
local feval = function(x)
  if x ~= parameters then
    parameters:copy(x)
  end

  -- get start/end indices for our minibatch (in this code we'll call a minibatch a "batch")
  --           -------
  --          |  ...  |
  --        ^ ---------<- start index = i * batchsize + 1
  --  batch | |       |
  --   size | | batch |
  --        v |   i   |<- end index (inclusive) = start index + batchsize
  --          ---------                         = (i + 1) * batchsize + 1
  --          |  ...  |                 (except possibly for the last minibatch, we can't
  --          --------                   let that one go past the end of the data, so we take a min())
  local start_index = counter * opt.batch_size + 1
  local end_index = math.min(n_train_data, (counter + 1) * opt.batch_size + 1)
  if end_index == n_train_data then
    counter = 0
  else
    counter = counter + 1
  end

  local batch_inputs = train.data[{{start_index, end_index}, {}}]
  local batch_targets = train.labels[{{start_index, end_index}}]
  gradParameters:zero()

  -- In order, these lines compute:
  -- 1. compute outputs (log probabilities) for each data point
  local batch_outputs = model:forward(batch_inputs)
  -- 2. compute the loss of these outputs, measured against the true labels in batch_target
  local batch_loss = criterion:forward(batch_outputs, batch_targets)
  -- 3. compute the derivative of the loss wrt the outputs of the model
  local dloss_doutput = criterion:backward(batch_outputs, batch_targets)
  -- 4. use gradients to update weights, we'll understand this step more next week
  model:backward(batch_inputs, dloss_doutput)

  -- optim expects us to return
  --     loss, (gradient of loss with respect to the weights that we're optimizing)
  return batch_loss, gradParameters
end

local counterTest = 0
local fevalTest = function(x)
  -- if x ~= testParameters then
  --   testParameters:copy(x)
  -- end

  -- get start/end indices for our minibatch (in this code we'll call a minibatch a "batch")
  --           -------
  --          |  ...  |
  --        ^ ---------<- start index = i * batchsize + 1
  --  batch | |       |
  --   size | | batch |
  --        v |   i   |<- end index (inclusive) = start index + batchsize
  --          ---------                         = (i + 1) * batchsize + 1
  --          |  ...  |                 (except possibly for the last minibatch, we can't
  --          --------                   let that one go past the end of the data, so we take a min())
  local start_index = counterTest * opt.batch_size + 1
  local end_index = math.min(n_test_data, (counterTest + 1) * opt.batch_size + 1)
  if end_index == n_test_data then
    counterTest = 0
  else
    counterTest = counterTest + 1
  end

  local batch_inputs = test.data[{{start_index, end_index}, {}}]
  local batch_targets = test.labels[{{start_index, end_index}}]
    testGradParameters:zero()

  -- In order, these lines compute:
  -- 1. compute outputs (log probabilities) for each data point
  local batch_outputs = model_test:forward(batch_inputs)
  -- 2. compute the loss of these outputs, measured against the true labels in batch_target
  local batch_loss = criterion_test:forward(batch_outputs, batch_targets)
  -- 3. compute the derivative of the loss wrt the outputs of the model
  local dloss_doutput = criterion_test:backward(batch_outputs, batch_targets)
  -- 4. use gradients to update weights, we'll understand this step more next week
  model_test:backward(batch_inputs, dloss_doutput)

  -- optim expects us to return
  --     loss, (gradient of loss with respect to the weights that we're optimizing)
  return batch_loss, testGradParameters
end
------------------------------------------------------------------------
-- OPTIMIZE: FIRST HANDIN ITEM
------------------------------------------------------------------------
local losses = {}          -- training losses for each iteration/minibatch
local testLosses = {}          -- training losses for each iteration/minibatch
local epochs = opt.epochs  -- number of full passes over all the training data
local iterations = epochs * math.ceil(n_train_data / opt.batch_size) -- integer number of minibatches to process
-- (note: number of training data might not be divisible by the batch size, so we round up)

-- In each iteration, we:
--    1. call the optimization routine, which
--      a. calls feval(parameters), which
--          i. grabs the next minibatch
--         ii. returns the loss value and the gradient of the loss wrt the parameters, evaluated on the minibatch
--      b. the optimization routine uses this gradient to adjust the parameters so as to reduce the loss.
--    3. then we append the loss to a table (list) and print it
for i = 1, iterations do
  -- optimMethod is a variable storing a function, either optim.sgd or optim.adagrad or ...
  -- see documentation for more information on what these functions do and return:
  --   https://github.com/torch/optim
  -- it returns (new_parameters, table), where table[0] is the value of the function being optimized
  -- and we can ignore new_parameters because `parameters` is updated in-place every time we call
  -- the optim module's function. It uses optimState to hide away its bookkeeping that it needs to do
  -- between iterations.
  local _, minibatch_loss = optimMethod(feval, parameters, optimState)

  -- Our loss function is cross-entropy, divided by the number of data points,
  -- therefore the units (units in the physics sense) of the loss is "loss per data sample".
  -- Since we evaluate the loss on a different minibatch each time, the loss will sometimes
  -- fluctuate upwards slightly (i.e. the loss estimate is noisy).
  if i % 10 == 0 then -- don't print *every* iteration, this is enough to get the gist
    local _, minibatch_loss_test = optimMethod(fevalTest, parameters, optimState)
    testLosses[#testLosses + 1] = minibatch_loss_test[1] -- append the new loss
    print(string.format("minibatches processed: %6s, train loss = %6.6f, test loss = %6.6f", i, minibatch_loss[1],minibatch_loss_test[1]))
    -- print(string.format("minibatches processed: %6s, train loss = %6.6f", i, minibatch_loss[1]))
  end
  -- TIP: use this same idea of not saving the test loss in every iteration if you want to increase speed.
    losses[#losses + 1] = minibatch_loss[1] -- append the new loss
  -- Then you can get, 10 (for example) times fewer values than the training loss. If you do this,
  -- you just have to be careful to give the correct x-values to the plotting function, rather than
  -- Tensor{1,2,...,#losses}. HINT: look up the torch.linspace function, and note that torch.range(1, #losses)
  -- is the same as torch.linspace(1, #losses, #losses).

end

-- local testLosses = {}          -- training losses for each iteration/minibatch
-- local testEpochs = opt.epochs  -- number of full passes over all the training data
-- local iterations = testEpochs * math.ceil(n_test_data / opt.batch_size) -- integer number of minibatches to process
-- -- (note: number of training data might not be divisible by the batch size, so we round up)

-- -- In each iteration, we:
-- --    1. call the optimization routine, which
-- --      a. calls feval(parameters), which
-- --          i. grabs the next minibatch
-- --         ii. returns the loss value and the gradient of the loss wrt the parameters, evaluated on the minibatch
-- --      b. the optimization routine uses this gradient to adjust the parameters so as to reduce the loss.
-- --    3. then we append the loss to a table (list) and print it
-- for i = 1, iterations do
--   -- optimMethod is a variable storing a function, either optim.sgd or optim.adagrad or ...
--   -- see documentation for more information on what these functions do and return:
--   --   https://github.com/torch/optim
--   -- it returns (new_parameters, table), where table[0] is the value of the function being optimized
--   -- and we can ignore new_parameters because `parameters` is updated in-place every time we call
--   -- the optim module's function. It uses optimState to hide away its bookkeeping that it needs to do
--   -- between iterations.
--   local _, minibatch_loss = optimMethod(fevalTest, testParameters, optimState)

--   -- Our loss function is cross-entropy, divided by the number of data points,
--   -- therefore the units (units in the physics sense) of the loss is "loss per data sample".
--   -- Since we evaluate the loss on a different minibatch each time, the loss will sometimes
--   -- fluctuate upwards slightly (i.e. the loss estimate is noisy).
--   if i % 10 == 0 then -- don't print *every* iteration, this is enough to get the gist
--       -- print(string.format("test minibatches processed: %6s, loss = %6.6f", i, minibatch_loss[1]))
--   end
--   -- TIP: use this same idea of not saving the test loss in every iteration if you want to increase speed.
--   -- Then you can get, 10 (for example) times fewer values than the training loss. If you do this,
--   -- you just have to be careful to give the correct x-values to the plotting function, rather than
--   -- Tensor{1,2,...,#losses}. HINT: look up the torch.linspace function, and note that torch.range(1, #losses)
--   -- is the same as torch.linspace(1, #losses, #losses).

--   testLosses[#testLosses + 1] = minibatch_loss[1] -- append the new loss
-- end

-- TODO: for the first handin item, evaluate test loss above, and add to the plot below
--       see TIP/HINT above if you want to make the optimization loop faster

-- Turn table of losses into a torch Tensor, and plot it
gnuplot.plot({ 'Train Data Loss',
  torch.range(1, #losses),        -- x-coordinates for data to plot, creates a tensor holding {1,2,3,...,#losses}
  torch.Tensor(losses),           -- y-coordinates (the training losses)
  '-'},
  { 'Test Data Loss',
  torch.linspace(1, #testLosses*10, #testLosses),        -- x-coordinates for data to plot, creates a tensor holding {1,2,3,...,#losses}
  torch.Tensor(testLosses),           -- y-coordinates (the training losses)
  '-'})

------------------------------------------------------------------------------
-- TESTING THE LEARNED MODEL: 2ND HANDIN ITEM
------------------------------------------------------------------------------

local logProbs = model:forward(train.data)
local classProbabilities = torch.exp(logProbs)
local _, classPredictions = torch.max(classProbabilities, 2)
local badPredictionsCount = 0
for i=1,classPredictions:size()[1] do
  if(classPredictions[{i,1}] ~= train.labels[{i}]) then
    badPredictionsCount = badPredictionsCount + 1
  end
end
local trainError = badPredictionsCount/(classPredictions:size()[1])
print("The training error is:")
print(trainError)

local logProbsTest = model:forward(test.data)
local classProbabilitiesTest = torch.exp(logProbsTest)
local _, classPredictionsTest = torch.max(classProbabilitiesTest, 2)
local badPredictionsCountTest = 0
for i=1,classPredictionsTest:size()[1] do
  if(classPredictionsTest[{i,1}] ~= test.labels[{i}]) then
    badPredictionsCountTest = badPredictionsCountTest + 1
  end
end
local testError = badPredictionsCountTest/(classPredictionsTest:size()[1])
print("The test error is:")
print(testError)

-- classPredictions holds predicted classes from 1-10

