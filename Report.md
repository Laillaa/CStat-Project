# From Features to Forecasts: Predicting Home Prices Using Computational Statistics 
**Angeliki Andreadi, Laila Ibrahim, Salsabil Mtiraoui**
## Abstract
Summary of the report
## Introduction
- Subject

Ask a home buyer to describe their dream house, they rarely start with its price. Instead, they focus on features like the number of rooms, proximity to transportation, or size of the garden. However, these characteristics ultimately shape the property's market value.

- Objective
  
Though this project, our goal is to first identify the features that truly influence a home's price and subsequently build a reliable model capable of accuratly predicting housing prices in Ames, Iowa (USA).

- Hypothesis?
  
  idk if there needs to be a hypothesis
## Data Overview
The data used in this study originates from the Kaggle competition [House Prices: Advanced Regression Techniques](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/data)
. It represents a sample of residential property sales recorded in Ames, Iowa. While not exhaustive of the entire housing market, it offers a representative snapshot of housing transactions in the area. The dataset contains 79 explanatory variables that capture a wide range of features related to the properties, such as lot dimensions, room counts, building characteristics, and neighborhood information.

## Methodology

All coding was conducted using Google Colab (short for Colaboratory), a free, cloud-based development environment that supports collaborative programming in a Jupyter notebook interface. Its seamless integration with Python libraries and ease of sharing made it particularly well-suited for our project.

### Preprocessing
  Our first step was to separate our data set into three subsets: binary, quantitative and categarical, because each data type requires a different statstical treatment treatment to be made useful. 
  
  In summary, the conditions and statements we used to classify the variables are: 
  
- If a column's `dtype` is either a `int64`(integer) or `float64` and it has exactly 2 unique values, it is appended to the `binary_var` list.
- If the column is numerical(`int64` or `float64` and has more than `quant_threshold` unique values, it is considered a quantitative variables and appended to `quant_var`. **de base 10 mais on a changé à 2, ajoutter ?Trial and error**
- If the column is numerical but has less or equal to `quant_threshold` unique values, we treat it as a categorical variables, and it is added to `categorical_var`.
- If the column's `dtype` is `object`, `category` or `bool`:
  - And it has exactly 2 unique values:
    - If it's of type `object`, we convert it to binary by mapping its two unique values to 0 and 1 and then append it to `binary_var`.
  - If it has more than 2 unique values, we append it to `categorical_var`
  
### Feature selection

#### Quantitative variables
To begin our exploration of the quantitative variables, we conducted a comprehensive descriptive statistical analysis. This step aimed to summarize the key characteristics of each variable in our datset `df_quant`.

We created a function called `descriptive_stats`, which builds on Python's built-in `.describe()` method by adding 95% confidence intervals for the mean of each variable. For each column, it computes: 
- The count, check if there are any missing values
- The mean
- The standard deviation
- The minimum and maximum value
- The 25th, 50th and 75th percentiles
- The lower and upper bounds of a 95% Confidence Interval, using Student's t-distribution formula

To assess the linear relationship between each quantiative feature and the target variable we implemented a function called `hypothesis_test` which computes the Pearson correlation coefficient, which measures the strength and direction of a linear association, and the corresponding p-value, which helps determine the statistical significance of that correlation. 

Next, to identify the most relevant featurs for predicting housing prices, we implemented a custom ranking function. It combines both descriptive statistics and results from hypothesis testing to score each feature's usefulness in relation to the target variable. 
This ranking process is based on three key criteria: 
- Correlation strength with target (using absolute correlation values to capture both positive and negative relationships)
- P-values from hypothesis testing (where lower values indicate stronger statistical significance)
- Variance of each feature (under the assumption that a higher spread may carry more informative value)

Each of the criteria were assigned a cutstom (and perhaps arbitrary) weight in the final ranking formula. Features were then sorted by their total score, allowing us to focus on those with the greatest potential predictive power. 

#### Binary variables
For the binary variables, we started by collecting some insights via descriptive statistics by using the '.describe()' function. Then, we computed their correlation with the 'SalePrice' variable, which represents the focal point of our project/analysis. The resulting correlations were all negative, therefore we decided not to use those variables for our analysis.
#### Categorical variables
## Model Construction and Diagnostic Analysis

After preprocessing and feature selection, we built a multiple linear regression model to predict house prices using both quantitative variables and the top 10 categorical variables selected via ANOVA. These categorical variables were transformed using one-hot encoding, excluding the first category to avoid multicollinearity.

All variables were explicitly converted to numeric type, and any rows containing missing values were removed to ensure a clean dataset for model training. An intercept term was also added to the design matrix.

We applied an Ordinary Least Squares (OLS) regression using the `statsmodels` library. The regression model produced a summary output including R-squared values, coefficient estimates, and significance levels for each predictor.

### Regression Summary Interpretation

The regression summary provided several key insights:

- **R-squared = 0.716**: The model explains 71.6% of the variability in SalePrice.
- **Adjusted R-squared = 0.699**: This slightly lower value accounts for the number of variables in the model.
- **F-statistic p-value ≈ 0**: The model is globally statistically significant.
- **Individual p-values**: These help identify which variables have a significant influence on house price. Variables with a p-value below 0.05 are considered statistically significant.

For example, neighborhood categories such as `Neighborhood_Crawfor` and `Neighborhood_ClearCr` showed strong positive effects on price, while others like `Neighborhood_MeadowV` or `Exterior1st_ImStucc` had a strong negative effect.

### Categorical Coefficients Visualization

To better interpret the impact of categorical variable modalities (i.e., specific categories within each variable), we generated a horizontal bar chart displaying the estimated regression coefficients:

- **Bars to the right (red)** indicate a **positive influence** on price.
- **Bars to the left (blue)** indicate a **negative influence**.
- The **longer the bar**, the greater the impact (in dollars) on predicted SalePrice.

This visualization reveals that some categories, such as houses located in `Neighborhood_NoRidge` or with an `Exterior1st_Stone` finish, are associated with significantly higher prices. Conversely, categories like `Exterior1st_ImStucc` or properties in `Neighborhood_MeadowV` reduce the predicted price.

### Residual Analysis

To evaluate the validity of our model and verify that the assumptions of linear regression were met, we conducted a residual analysis using three standard plots.

#### Residuals vs Fitted Values

This scatter plot checks the assumptions of linearity and homoscedasticity:

- The x-axis shows predicted values from the model.
- The y-axis displays residuals (actual − predicted prices).
- A random scatter around zero with no visible pattern supports the assumptions of a linear model with constant variance.

In our plot, the residuals were tightly clustered around zero with no apparent pattern, indicating good model behavior, although their very small magnitude suggests that SalePrice might have been normalized or scaled.

#### Histogram of Residuals

This plot checks whether the residuals are normally distributed:

- A bell-shaped curve centered around zero is expected.
- In our case, the distribution was symmetric and bell-shaped, supporting the normality assumption of residuals.

#### Q-Q Plot

The Q-Q plot compares residual quantiles to a theoretical normal distribution:

- If residuals are normally distributed, points will lie along the red reference line.
- Our plot showed good alignment, with only slight deviation at the tails, indicating minor outliers but general normality.


### Model
## Analysis
## Discussion
## Conclusion
## Annex
---
University of Neuchâtel - This work was done as part of the "Computational Statistics" course.

