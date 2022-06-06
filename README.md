## DogImages Classification using AWS SageMaker
In this project we were asking to use AWS Sagemaker to train a pretrained model that can perform image classification by using the Sagemaker profiling, debugger, hyperparameter tuning and other good ML engineering practices. This can be done on either the provided dog breed classication data if you want to practice with this code you can set or one of your choice.

## Dataset
The provided dataset is the dogbreed classification dataset which can be found in the classroom. The project is designed to be dataset independent so if there is a dataset that is more interesting or relevant to your work, you are welcome to use it.

# Report: DogImages Classification using AWS SageMaker
### Philippe Jean Mith

## Hyperparameter Tuning
### What kind of model did you choose for this experiment and why? Give an overview of the types of parameters and their ranges used for the hyperparameter search.
For this experimentation i used the resnet152 pretrained model because chieves the best accuracy among Resnet family members. And i used three hyperparameters.

| Hyperparameter | Range |
| -------------- | ----- |
| Learning rate  | (0.001, 0.1) |
| Batch Size     | [8, 16, 32, 64] |
| Epochs         | (2, 10) |

After the training jobs here is the results.
![trainJobs](https://github.com/PedroToto/dogImage-classification/blob/main/training_job.PNG)

![best](https://github.com/PedroToto/dogImage-classification/blob/main/bet_training_job.PNG)

## Debugging and Profiling
For model Debugging i configured the hook in SageMaker's python SDK using the Estimator class and instantiate it with with `create_from_json_file()` from `smdebug.pytorch.Hook`. I added the *SMDebug hook* for PyTorch with TRAIN mode in the train function, EVAL mode in the test function and specified the debugger rules and configs in what i set a save interval of 100 and 10 for training and testing respectively. For the Profiling, i specified the metrics to track and create the profiler rules.

## Results
They is no anomalous behaviour in the debugging output.

## Model Deployment
First time i deployed the model i call the `deploy` method from the estimator with the instance type with the parameter `instance_type` and the number of instances with the oarameter `initial_instance_count`. But, when i tried to make a prediction i got the `ModuleNotFoundError: No module named 'smdebug' error`. To resolve this issue i created a `Pytorch model object` by fallowing the recommendation [here](https://knowledge.udacity.com/questions/775344) and then call the `deploy` method from the `Pytorch model object` with the instance type with the parameter `instance_type` and the number of instances with the oarameter `initial_instance_count`. 

![Endpoints](https://github.com/PedroToto/dogImage-classification/blob/main/endpoints.PNG)


## Standout Suggestions
After the model deployement i had some issue to predict on the endpoint.
The first error was `ModuleNotFoundError: No module named 'smdebug'`. To work around this is issue, i created a different script for deployment which was recommended as a solution for the issue. After that there was no more `ModuleNotFoundError: No module named 'smdebug'` error. However, i couldn't still have the result from the prediction. When i check in the Log groups what i found is in the picture bellow.

![error](https://github.com/PedroToto/dogImage-classification/blob/main/Error3.png)

The status code 200 means that the request is fulfilled but i wanted the prediction result. After doing more research i found that i can use the invokeEndpoint, and i did that by fallowing some documentations which can be found:
* [InvokeEndpoint](https://docs.aws.amazon.com/sagemaker/latest/APIReference/API_runtime_InvokeEndpoint.html)
* [Deploy PyTorch Models](https://sagemaker.readthedocs.io/en/stable/frameworks/pytorch/using_pytorch.html#deploy-pytorch-models)
* [sagemaker-tensorflow-serving-container](https://github.com/aws/sagemaker-tensorflow-serving-container#prepost-processing)


