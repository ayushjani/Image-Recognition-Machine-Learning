# Image recognition using machine learning

Classifies the input image into 1 of the 10 classes of possible objects.

```
Dataset used:https://www.cs.toronto.edu/~kriz/cifar.html
```

**USAGE INSTRUCTIONS**
```
Dependancy installation instructions located in the requirements.txt
```

```Python
Default model is LeNet5

If you would like to test LeNet5, copy/paste "model = LeNet5().to(device)" over line 164 and copy/paste "learnRate = 0.001" underneath that line

If you would like to test Net_9ine.py, copy/paste "model = Net_9ine.py().to(device)" over line 164 and copy/paste "learnRate = 0.001" underneath that line

If you would like to test AlexNet, copy/paste "model = Alexnet().to(device)" over line 164 and copy/paste "learnRate = 0.0005" underneath that line

If you would like to test VGG_16, copy/paste "model = VGG_16().to(device)" over line 164 and copy/paste "learnRate = 0.0001" underneath that line
```

```
If you would like to save the trained model, set the variable save_model located on line 140 to true
```

