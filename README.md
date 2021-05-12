# Neural-Estimator-of-Information-non-i.i.d

## CMI Estimation
To run the experiment of estimating CMI I(X;Y|Z) you can run the command: 
python CMI_Markov.py

## DI Estimation
To run the experiment of estimating DI terms such I(X->Y), run the command:
python DI_Markov.py --link X_Y

Other DI terms such as I(Z->Y) can be run by changing the argument of the command (Z_Y in this case)

## CDI Estimation
To run the experiment of estimating CDI terms such I(X->Y||Z), run the command:
python DI_Markov.py --link X_Y-Z

Other DI terms such as I(Z->Y||X) can be run by changing the argument of the command (Z_Y-X in this case)



