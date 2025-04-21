This repo introduces Linear Regression, one of the most basic and widely used methods in statistics and machine learning. It's all about finding the best-fitting straight line through a bunch of data points. This line helps us understand and predict relationships between variables.

🧠 What's Happening Conceptually?
Let’s say you want to predict something (like price, weight, or sales) based on one or more input factors (like time, height, or advertising spend). Linear regression tries to draw a straight line that best explains this relationship.

The relationship is often written as:

𝑦
=
𝑎
+
𝑏
𝑥
y=a+bx
Where:

y is what we’re trying to predict.

x is the input.

a is the intercept (where the line hits the y-axis).

b is the slope (how much y changes for each unit of x).

🧪 What We Did in the Code
Generated Fake Data
We created 500 data points that roughly follow a line (y = 1 + 2x) but with some random noise added in, just like real-life data that isn’t perfect.

Estimated the Line (Closed-form Solution)
We used a mathematical shortcut called the Normal Equation to calculate the best values of a and b that minimize the error. This is known as Ordinary Least Squares (OLS).

Did the Same with Scikit-Learn
Instead of calculating by hand, we used Python’s scikit-learn library to do the same thing using its LinearRegression tool.

Visualized the Results
We plotted:

The original data (blue dots),

The predicted line (red),

And the true underlying line we used to create the data (green).

Evaluated Performance We calculated:

R² Score – tells us how much of the variation in y is explained by x. We got 0.797, which means ~80% is explained.

Pearson Correlation – a measure of how strong the relationship is. We got ~0.89, which means a strong positive relationship.

🛤️ Where This Leads
This basic example sets the stage for more advanced topics like:

Multiple Linear Regression (more than one x),

Bayesian Linear Regression (adds probability),

And Regularization (handles complex real-world data better with techniques like Ridge and Lasso regression).
