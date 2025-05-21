# From Features to Forecasts: Predicting Home Prices Using Computational Statistics 
**Angeliki Andreadi, Laila Ibrahim, Salsabil Mtiraoui**

## Abstract
Summary of the report

## Introduction
Ask a home buyer to describe their dream house, they rarely start with its price. Instead, they focus on features like the number of rooms, proximity to transportation, or size of the garden. However, these characteristics ultimately shape the property's market value.  
Throughout this project, our goal is to first identify the features that truly influence a home's price and subsequently build a reliable model capable of accuratly predicting housing prices in Ames, Iowa (USA).

- Hypothesis?
  idk if there needs to be a hypothesis

Our hypothesis is that the variables on which people focus the most are the most likely to affect the SalesPrice.

*people look for a good environment (neighborhood...)*

## Data Overview
The data used in this study originates from the Kaggle competition [House Prices: Advanced Regression Techniques](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/data). It represents a sample of residential property sales recorded in Ames, Iowa. While not exhaustive of the entire housing market, it offers a representative snapshot of housing transactions in the area. The dataset contains 79 explanatory variables that capture a wide range of features related to the properties, such as lot dimensions, room counts, building characteristics, and neighborhood information.

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
For the binary variables, we started by collecting some insights via descriptive statistics by using the `.describe()` function. Then, we computed their correlation with the 'SalePrice' variable, representing our analysis's focal point. The resulting correlations were all negative, therefore, we decided not to use those variables for our analysis.

#### Categorical variables
We started the exploration of our 3rd category also by collecting some insights via descriptive statistics by using the `.describe()` function. What caught our attention the most was the value count (how many observations a variable has). A variable should normally have 1460 observations, but not all do. To deal with those missing values, we decided to keep only the variables with a count of 1460 observations, with which we will continue our analysis.

Then we built a function by the name of `rank_categorical_vars` to rank the filter categorical variables. It ranks them according to their p-value, from the lowest to the highest. Variables with lower p-values (**lower than 5%?**) are more likely to affect the SalesPrice.

We visualized those rankings in the 3rd graph, `Ranking of Categorical Variables Based on Final Score`. The visualized order is the actual order; the lower we go, the lower the ranking order. Ex: Neighborhood is ranked 1st .... **ranking issues**

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

### Times Series Analysis
Before applying any form of preprocessing or modeling, we first examined the time-related variables in the dataset. Among these, we identified four of particular interest: YrSold (the year the house was sold), MoSold (the month of sale), YearBuilt (the original construction year), and YearRemodAdd (the year of remodeling or renovation). We chose to focus primarily on the combination of YrSold and MoSold, as we expected this pair to reveal potential seasonal patterns or market trends over time more clearly than the others.

By first plotting the data, the monthly average housing price against the months of sale, we observed a unclear seasonality and not a distrinct long term trend. We will first attempt to remove these short-term flucturations to highlight long-term trends. 

#### Smoothing (Moving Average)
We will first attempt to remove these short-term flucturations to highlight long-term trends through smoothing. We used a rolling window method (?). We inplemented two windows, one of 6 months and one of 12 months

*INSERT IMAGE OF GRAPH*

Both curves show a general decline, in blue the MA6 and in orange the MA12. The MA6 still shows us some general seasonality.

Our dates range from 01.2006 and 07.2010, which could match with the american Subprime Mortgage Crisis of 2008. In short, in the early 2000s U.S. banks game out lots of home loans to individuals who couldn't really afford them (subprime loans). Around 2006 the housing prices stopped rising and started to fall. People with subprime loans couldn't repay them and since their homes had dropped in value they couldn't sell them either. Even Iowa, which wasn't at the center of the crisis could have felt these effects. 

#### Stationarity Test
To further plot our data, we tested its stationarity test with the ADF statitic (Augmented Dickey-Fuller test) as to further fit it with any model. 
That value is smaller 0.5 (0.0036) meaning we can process to model it directly with no differenciation. 

#### ACF/PACF
By plotting the ACF we will measure the degree of similarity between our time series and a lagged version of it, in this case `lag=20`, to measure the correlation with a little less than two year's time. 
The PACF, by regressing a target against its previous value, 

### Model
## Analysis
## Discussion
## Conclusion
## Annex
---
University of Neuchâtel - This work was done as part of the "Computational Statistics" course.

