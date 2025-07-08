# bts_algo

BTS Deep Hedging Model:


Framework:
- Strategy/Model
  - computes value of signal at each time
  - compute returns 
- Data
- Optimizer
  - define objective functions (profit factor, sharpe ratio)
  - test various parameters for best historical objective optimum (e.g. window, or hyperparameters)

1. In-Sample excellence
   - Trade dependence
   - Overfitting and data mining bias? or actual performance
2. In-Sample MC Permutation Test
   - Null hypothesis is the strategy is overfitting
   - Compare strategy against noise
   - Test against various permutations using MC (n = 1000)
   - Long memory and volatility clusterin concerns
   - test p value 
1. Walk forward test
2. Walk forward permutation Test

Testing and Tuning Market Trading Systems: https://amzn.to/3DfAaVm
https://github.com/Apress/testing-and-tuning-market-trading-systems

