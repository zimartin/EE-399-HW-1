# EE-399-HW-1

## Abstract
Through implementng basic machine learning techniques, I fit various models to a given data set. Using the least-squared error function, I measured the performance of the optimized models I generated and compared them.
To better visualize the minima of the models, I generated a number of 2D loss landscapes. 

## Introduction and Overview
Machine learning models rely on minimizing the difference between the model and the training data to make accurate predictions. To achieve this, we used the least-squares error method to optimize our models. In this homework assignment, we worked with a dataset of 31 data points, exploring different models such as a trigonometric function with linear and constant terms, as well as a line, parabola, and 19th degree polynomial.

Using the minimize function provided by the Scipy library, we found the parameters that minimized the least-squares error for each model on the training data. We then evaluated the performance of each model on the test data and compared their least-squares errors. We discovered that the choice of model and the size of the dataset had a significant impact on the accuracy of the predictions.

Through this exercise, we gained insights into the challenges of model fitting and error analysis, as well as the importance of model selection and evaluation for real-world applications. By optimizing models with the least-squares error method, we were able to improve their accuracy and create reliable predictions.

As a fun aside, much of the code for this project was generated using ChatGPT

## Theoretical Background
In the field of machine learning, a common approach for modeling a system or phenomenon is to use the Least-Squares fitting method. This algorithm involves defining a function or set of functions that maps inputs to outputs by optimizing the parameters of the function(s) to minimize the sum of the squared errors between the predicted and actual outputs. The least-squares error is evaluated at each point by finding the square of the difference between the model and true data. The goal is to minimize this error through various solutions of the model, which give different errors at each point.

When using the least-squares approach, it's important to consider the complexity of the model and how it affects the performance on both training and test data. A model that is too simple may underfit the data and not capture the underlying patterns and relationships, while a model that is too complex may overfit the data and memorize the noise instead of learning the general pattern. To avoid underfitting or overfitting, the dataset is split into a training and test set, and the model is evaluated on the test set.

Python provides various optimization functions to find the optimal solution through minimizing the least-squares error. For example, the np.minimize() function can be used to specify the objective function and the parameters to be optimized. Once the model has been trained on the training data, it can be used to predict values for new data points in the test dataset. The error between the model's expected value and the true value at each data point is then evaluated using the same method as before, and the model's overall performance is represented by a single scalar that is the square root of the average of the squared differences between the predicted and actual values.

## Algorithm Implementation and Development
This assignment started with a data set of test data:
```
X = np.arange(0,31)
Y = np.array([30, 35, 33, 32, 34, 37, 39, 38, 36, 36, 37, 39, 42, 45, 45, 41,
40, 39, 42, 44, 47, 49, 50, 49, 46, 48, 50, 53, 55, 54, 53])
```

2.i

In the first part of the assignment we were tasked with fitting the given model: ```f(x) = A*cos(B*x) + C*x + D ``` to the data set using the least squares error optimization. Using the least squares, I was able to determine values for the coeffecients A, B, C, and D that corresponded to a minima for the error function. First, I defined the error function and the starting parameters, then used the SciPy optimization function to minimize the error.

```
def fit_err(c, x, y):
    model = c[0]*np.cos(c[1]*x)+c[2]*x + c[3]
    er = np.sqrt(np.sum((model-y)**2)/len(x))
    return er

# initial guesses for params of function
v0 = np.array([3, 1*np.pi/4, 2/3, 32])

# perform optimization
res = opt.minimize(fit_err, v0, args=(x, y))

# store optimized params
c = res.x
```

2.ii

In the second part, I fixed two of the four parameters (A, B, C, D) and swept across the other two parameters to create a 2D loss landscape of the error function. I then repeated this process for all six possible combinations of fixed and swept parameters.

```
# find the error with 2 values locked, sweep the other 2
def find_err(param, sweep):
  a, b, c, d = param

  # store param values
  cfix = np.zeros((4, L))

  # store calculated error
  errcalc = np.zeros((L,L))
  
  # possible combinations: ab, ac, ad, bc, bd, cd
  # returns 2d array of calculated err
  for i in range(0, L):
    for j in range(0, L):
      cfix[3] = d[j]
      if (sweep == 0): # sweep a
        cfix[0], cfix[1], cfix[2] = a[i], b[j], c[j]
      elif (sweep == 1): # sweep b
        cfix[0], cfix[1], cfix[2] = a[j], b[i], c[j]
      elif (sweep == 2): # sweep c
        cfix[0], cfix[1], cfix[2] = a[j], b[j], c[i]
      errcalc[i][j] = fit_err(cfix, x, y)
  return errcalc
```

2.iii

For the third task, I started by splitting the data as outlined in the assignment.

```
# separate training data
xtrain = x[:20]
ytrain = y[:20]

# separate test data
xtest = x[-11:]
ytest = y[-11:]
```

I then fit the three models (Linear, Parabolic, and 19th Degree Polynomial) to the training data (the first 20 data points). For the Linear and Parabolic models I was able to use the same process as seen in 2.i but fot the polynomial I used `np.polyfit`.

```
# set the initial guess for the params on line and parabola functions
line0 = np.array([1, 20])
parab0 = np.array([1, 5, 20])

# perform optimization and save params
resline = opt.minimize(fitline, line0, args=(xtrain, ytrain))
resparab = opt.minimize(fitparab, parab0, args=(xtrain, ytrain))
respoly = np.polyfit(xtrain, ytrain, 19, full=True)

# get the optimized params
minsline = resline.x
minsparab = resparab.x
minspoly = respoly[0]

# generate data for plotting
yline = (minsline[0] * xtrain + minsline[1])
yparab = (minsparab[0] * xtrain ** 2 + minsparab[1] * xtrain + minsparab[2])
ypoly = np.polyval(minspoly, xtrain)
```

2.iv

For the final section of the assignmnet, I just repeated 2.iii but with the first 10 and last 10 data points.

## Computational Results

2.i

In the first optimization problem, I was able to determine the coeffecients and minimum error of the model ```f(x) = A*cos(B*x) + C*x + D ``` to be:

```
A = 2.171726965658166
B = 0.9093254571756474
C = 0.7324879682604329
D = 31.452780916275167
Minimum Error: 1.5927258502884714
```
I then plotted the newly optimized model over the test data to verify visually that the fit was successful.

![download](https://user-images.githubusercontent.com/129991497/231077260-20d24977-446c-406f-9f5c-fc617517ee8d.png)

2.ii

For the second part of the assignment, I plotted the six loss landscapes and calculated the number of local minima as well as the value of the local minima as show below:

![download](https://user-images.githubusercontent.com/129991497/231078327-939ea374-9d8e-4abb-9fcf-734961528c6e.png)

```
Number of minima and average value:
AB: 68  Min Err: 1.7541558854533992
AC: 195  Min Err: 2.156075563725492
AD: 97  Min Err: 2.510106041519397
BC: 209  Min Err: 1.684055925011477
BD: 93  Min Err: 1.964311651010509
CD: 148  Min Err: [2.16936261 2.17050672 2.17173451]
```

2.iii

For the third part of the assignment, I started by fitting the three models (Linear, Parabolic, and 19th Degree Polynomial) to the training data.

![download](https://user-images.githubusercontent.com/129991497/231078624-291db146-64c9-47b0-9ead-3afae2455b8f.png)

I then fit the three models to the test data.

![download](https://user-images.githubusercontent.com/129991497/231078895-6ca6ffd0-2bc2-4f30-9a96-94258bf28a17.png)

Finally, I calculated the least-square error (LSE) for both the training data and the test data for each ofthe three models.

```
Line LSE:
Training: 2.2427493868088466
Test: 3.3636415371732986 

Parabola LSE:
Training: 2.1255393482814218
Test: 8.713676162354922 

Polynomial LSE:
Training: 0.02835145876672336
Test: 28621872795.094692
```

2.iv

For the final part of the assignment, I repeated the optimization process as seen in 2.iii with the new data set.

![download](https://user-images.githubusercontent.com/129991497/231079328-50d8d055-d85e-42a6-a8d3-40c1d5fe6550.png)

I also repeated the least-square error calculations.

```
Line LSE:
Training: 1.8516699043294016
Test: 2.940307974977315 

Parabola LSE:
Training: 1.85083641178975
Test: 2.905825819297066 

Polynomial LSE:
Training: 0.16382490406567501
Test: 507.5418394864483
```

## Summary and Conclusions
After optimizing and fitting different models to a given dataset of 31 points, it became evident that the choice of model and training data directly impacts the accuracy of the predictions with relatively low error. By minimizing the least-squares error and sweeping through parameter combinations, it was discovered that using just the minimize function may not be sufficient and should be used in combination with other optimization methods.

In addition, when training the models on different subsets of the data, it was observed that the line fit and parabola fit models had low errors and showed little deviation between training and test errors. However, the 19th degree polynomial fit model had a significantly lower training error but a much higher test error, indicating overfitting. This implies that the model was not able to learn the general pattern of the data but rather memorized the training set.

Therefore, when working with datasets, it is crucial to identify overfitting or underfitting, choose the right model that fits the data, and select the appropriate subset of data to train and test the model. The success of the model largely depends on these factors, and a thorough understanding of them can lead to accurate predictions with relatively low error.
