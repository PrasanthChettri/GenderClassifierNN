# GenderClassifierNN
A API backend serving, a gender classifier

# API 
- Implemented with Fastapi
- Takes in lists of names and returns, runs the classifier and returns confidence (List[probab_for_male , probab_for_female])

# Classifer 
- Takes in names, tokenises the names and runs a LSTM model
- softmax the output
- returns the probability of the name being male or female
```
    {
      'male'   :  P(male)
      'female" : P(female) 
    }
```
  st : 
  
      P(female) = 1 - P(male) 
