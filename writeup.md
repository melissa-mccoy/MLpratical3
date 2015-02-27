Melissa McCoy

***********************Part 1: Add the test data loss************************
"Modify the code so that it evaluates its performance on the test set after every minibatch
(or epoch if you prefer), then plot both the test loss and training loss on the same plot,
rather than just the training set loss as the code does now. Handin: the plot, and a brief
explanation of the code required to do this"

With the if statement that evaluates only tenth data pt, the code I added was:
    local outputs = model:forward(test.data)
    local test_loss = criterion:forward(outputs, test.labels)
    testLosses[#testLosses + 1] = test_loss -- append the new loss
which calculates the loss for all the test data with the given status of the model.
I also had to change the plot to accomodate the every 10 aspect:
{ 'Test Data Loss',
  torch.linspace(1, #testLosses*10, #testLosses),
  torch.Tensor(testLosses),
  '-'}

************Part 2: Choose best optimization approach & Calculate the test & trade data**************
"Find a configuration that works well (for the optimizer and perhaps mini-batch size and
others). The classification error is the percentage of instances that are misclassified, in either the training or test set. Handin: Find a configuration that predicts well, and
report your training set and test set classification error as we just defined. Explain
your findings about which optimizers were easier to configure than others. Very briefly
explain why the final model you pick solution is “good”. Show your code for computing
the classification error, which should be short."

The configuration that works the best were:
--For Adagrad--
The training error is:
0.04825
The test error is:
0.1208
opt.optimization = 'adagrad'
opt.batch_size = 3
opt.train_size = 8000
opt.test_size = 0
opt.epochs = 15
learningRate = 3e-1

--For SGD--
The training error is:
0.065
The test error is:
0.1236
opt.optimization = 'sgd'
opt.batch_size = 3
opt.train_size = 8000
opt.test_size = 0
opt.epochs = 10

--For LBFGS--
The training error is:
0.572
The test error is:
0.5793
opt.optimization = 'lbfgs'
opt.batch_size = 3
opt.train_size = 8000
opt.test_size = 0
opt.epochs = 15

SVG was the easiest to configure as it seemed to be the most sensitive to option changes so I could use trial & error fairly effectively. LBFGS was the hardest to configure as it many scenarios it would not learn and it has a small sweetspot to find - because it's like Newtons method. I picked Agradad finally because it yielded the lowest classification errors for both test & training data. I calculated the error with this code:
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
