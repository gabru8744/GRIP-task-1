#!/usr/bin/env python
# coding: utf-8

# ## GRIP :Data Science & Business Analytics Tasks

# BY: ABHILASH BHARDWAJ.

# # TASK 1
# **In this regression task we will predict the percentage of marks that a student is expected to score based upon the number of hours they studied**

# **What will be predicted score if a student study for 9.25 hrs in a day?**

# **Reading the data**

# In[1]:


# importing Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns


# In[2]:


# Reading data from remote link
url = "http://bit.ly/w-data"
data = pd.read_csv(url)
print("Data imported successfully")

data


# In[3]:


#checking for null values
data.isnull().sum()


# **DATA VISUALIZATION**

# In[4]:


# Plotting the distribution of scores
data.plot(x='Hours', y='Scores', style='o')  
plt.title('Study Hours vs Percentage Scores')  
plt.xlabel('Hours Studied')  
plt.ylabel('Percentage Score')  
plt.show()


# In[5]:


#plotting regressor plot to determine the relationship between feature and target
sns.regplot(x=data['Hours'],y=data['Scores'],data=data)
plt.title('Study Hours vs Percentage Scores')
plt.xlabel('Study Hours')
plt.ylabel('Percentage')
plt.show()


# From the graph above, it can be clearly seen that there is a positive linear relation between the number of hours studied and percentage of score.

# Preparing our data

# Next is to define our "attributes"(input) variable and "labels"(output)

# In[8]:


X = data.iloc[:, :-1].values  #Attribute
y = data.iloc[:, 1].values    #Labels


# Now that we have the attributes and labels defined, the next step is to split this data into training and test sets.

# In[10]:


# Using Scikit-Learn's built-in train_test_split() method:

from sklearn.model_selection import train_test_split  
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)


# We have the training and testing sets ready for training our model.

# # Training the Algorithm

# First I will be making our linear regression algorithm from scratch and then I will compare it with the built-in function sklearn.linear_model.LinearRegression()

# Making the linear regression from scratch

# In[11]:


y_train_new = y_train.reshape(-1,1)  
ones = np.ones([X_train.shape[0], 1]) # create a array containing only ones 
X_train_new = np.concatenate([ones, X_train],1) # concatenate the ones to X matrix


# In[12]:


# creating the theta matrix
# notice small alpha value
alpha = 0.01
iters = 5000

theta = np.array([[1.0, 1.0]])
print(theta)


# In[13]:



# Cost Function
def computeCost(X, y, theta):
    inner = np.power(((X @ theta.T) - y), 2)
    return np.sum(inner) / (2 * len(X))


# In[14]:


computeCost(X_train_new, y_train_new, theta)


# The value of 1259.1955 is the initial value. The aim will be to minimise this to as small as possible

# In[16]:


# Gradient Descent
def gradientDescent(X, y, theta, alpha, iters):
    m = len(X)
    for i in range(iters):
        theta = theta - (alpha/m) * np.sum(((X @ theta.T) - y) * X, axis=0)
        cost = computeCost(X, y, theta)
        #if i % 10 == 0:
            #print(cost)
    return (theta, cost)


# In[17]:


g, cost = gradientDescent(X_train_new, y_train_new, theta, alpha, iters)  
print("Intercept -", g[0][0])
print("Coefficient- ", g[0][1])
print("The final cost obtained after optimisation - ", cost)


# Now Let's Plot our result

# In[18]:



# Plotting scatter points
plt.scatter(X, y, label='Scatter Plot')
axes = plt.gca()

# Plotting the Line
x_vals = np.array(axes.get_xlim()) 
y_vals = g[0][0] + g[0][1]* x_vals #the line equation

plt.plot(x_vals, y_vals, color='red', label='Regression Line')
plt.legend()
plt.show()


# So, the above method was building our regression algorithm from scratch and implementing it on the data-set. However, as you can see this is in crude form, a more elegant would be to make a class of it.
# 
# The hyper-parameters such "alpha" also known as the learning rate is optimised by hit and trial method, which should not be the practice.
# 
# So, instead of writing this long code, python has an-inbuilt library for the same. Let's try that too.

# Using Scikit-Learn library

# In[19]:


from sklearn.linear_model import LinearRegression  
regressor = LinearRegression()  
regressor.fit(X_train, y_train) 

print("Training complete.")


# In[21]:


print ("Coefficient -", regressor.coef_)
print ("Intercept - ", regressor.intercept_)


# In[22]:


# Plotting the regression line
line = regressor.coef_*X + regressor.intercept_

# Plotting for the test data
plt.scatter(X, y)
plt.plot(X, line,color='red', label='Regression Line')
plt.legend()
plt.show()


# As we can see that both the graph are identical and even the intercepts and coefficient of the line are same. So my effort of making linear regression algorithm from scratch has been a success!!
# 
# But as you can see, it is also pretty easy to use the built-in function. Few lines of code and your work is done.Viola!!

# # Making Predictions

# Now that we have trained our algorithm, it's time to make some predictions.

# In[23]:


print(X_test) # Testing data - In Hours
y_pred = regressor.predict(X_test) # Predicting the scores


# In[24]:



# Comparing Actual vs Predicted
df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})  
df


# In[25]:


#Estimating training and test score
print("Training Score:",regressor.score(X_train,y_train))
print("Test Score:",regressor.score(X_test,y_test))


# In[26]:


#plotting the grid to depict the actual and predicted value
df.plot(kind='bar',figsize=(7,7))
plt.grid(which='major', linewidth='0.5', color='green')
plt.grid(which='minor', linewidth='0.5', color='black')
plt.show()


# In[27]:


# Testing with some new data
hours = 9.25
test = np.array([hours])
test = test.reshape(-1, 1)
own_pred = regressor.predict(test)
print("No of Hours = {}".format(hours))
print("Predicted Score = {}".format(own_pred[0]))


# # Evaluating the model

# The final step is to evaluate the performance of algorithm. This step is particularly important to compare how well different algorithms perform on a particular dataset.

# In[28]:



from sklearn import metrics  
print('Mean Absolute Error:',metrics.mean_absolute_error(y_test, y_pred)) 
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
print('R-2:', metrics.r2_score(y_test, y_pred))


# R-squared gives the goodness of the fit and as can be seen it is 96.7% which is really good. This means that the algorithm has proven to be good for the given data.
# 
# It can also be said that the model's accuracy is 96.78%
# 
# NOTE - In practice, you’ll never see a regression model with an R2 of 100%.
# 
# This concludes this Notebook.

# # THANK YOU

# In[ ]:




