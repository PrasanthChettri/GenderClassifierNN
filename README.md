# GenderClassifierNN
A GenderClassifier built with python, served using FastAPI 

# Installation 
- Clone the project
- For installing dependencies, run ```pip install -r requirements.txt```
- set path variable for python using ```export PYTHONPATH=$PWD``` (linux)
- Now, we need to train the model, for doing so run ```python3 classifier/train.py```
- For starting the uvicorn server, run ```uvicorn web.main:app```
- If we go to the localhost:8000, a frontend is present to use the classifier

![picture alt](img.png)

# Usage
- Using frontend : 
  - A basic frontend for typing in names and getting classifications
- Using REST API 
  - for single name classification ```classify?name=<name>```

    **returns**
    ```
    {
      'name' : name being classified 
      'male'   :  Confidence/Uncertainty of being male
      'female' :  Confidence/Uncertainty of being female 
    }
    ```
  - for multiple name classification ```bulk_classify?names=<name1>&names=<name2>```
    ```
    [{
      'name' : name being classified 
      'male'   :  Confidence/Uncertainty of being male
      'female' :  Confidence/Uncertainty of being female 
    }]
    ```


# Model 

- The names are one hot encoded and fed to the neural net 
- The model is for now is a bidirectional stacked LSTM followed by a dense layer 
 
   ![Operation-of-two-stacked-bidirectional-LSTM-RNN-model](https://user-images.githubusercontent.com/35295049/120662870-1ffa1e00-c4a7-11eb-8247-48e0fc2819f5.png)
  
  
  here input for the dense layer concat(h<sup>b2</sup><sub>t-1</sub> , h<sup>f2</sup><sub>t+1</sub>)
- The output is passed through a sigmoid function such that outputs a confidence for male (Zero begin female, One being male)
- The test accuracy of the model currently is roughly **87%** where accuracy = ``` (tn + tp)/ total ``` for a confusion matrix 

TODO :
* Work on automating best config for training the nn for hardware
* Try out other approaches with more complex archetecture
* Host the website
