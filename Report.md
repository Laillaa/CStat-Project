# From Features to Forecasts: Predicting Home Prices Using Computational Statistics 
**Angeliki Andreadi, Laila Ibrahim, Salsabil Mtiraoui**

## Abstract
Summary of the report

## Introduction
Ask a home buyer to describe their dream house, they rarely start with its price. Instead, they focus on features like the number of rooms, proximity to transportation, or size of the garden. However, these characteristics ultimately shape the property's market value.  
Throughout this project, our goal is to first identify the features that truly influence a home's price and subsequently build a reliable model capable of accuratly predicting housing prices in Ames, Iowa (USA).

We hypothesize that the variables on which people focus the most are more likely to affect the SalesPrice.

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
For the binary variables, we started by collecting some insights via descriptive statistics by using the `.describe()` function. Then, we computed their correlation with the 'SalePrice' variable, representing our analysis's focal point. Despite the negative results, we kept them for further possible uses.

#### Categorical variables
We started the exploration of our 3rd category also by collecting some insights via descriptive statistics by using the `.describe()` function. What caught our attention the most was the value count (how many observations a variable has). A variable should normally have 1460 observations, but not all do. To deal with those missing values, we decided to keep only the variables with a count of 1460 observations, with which we will continue our analysis.

Then we built a function by the name of `rank_categorical_vars` to rank the filter categorical variables. It ranks them according to their p-value, from the lowest to the highest. Variables with lower p-values (**lower than 5%?**) are more likely to affect the SalesPrice.

We visualized those rankings in the 3rd graph, `Ranking of Categorical Variables Based on Final Score`. The visualized order is the actual order; the lower we go, the lower the ranking order. Ex: Neighborhood is ranked 1st .... **ranking issues**


## $2^k$ fractional factorial design
To demonstrate our understanding of full factorial design, we applied a $2^3$ factorial model using the three binary variables we extracted from our dataset: `Street`, `Utilities`, and `CentralAir`. The model included all main effects and interaction terms.

Despite being theoretically sound, this approach proved to be of limited practical value in our context. The model yielded a low R² of 0.064, indicating that it explains only 6.4% of the variance in sale price. Moreover, most of the interaction terms were statistically insignificant or even unestimable (`nan` coefficients), due to a lack of variability in the combinations of certain factors (nearly all observations have the same value for `Utilities`). Only the variable `CentralAir` showed a statistically significant effect.

This exercise, although not useful for improving our predictive performance, allowed us to grasp the logic of factorial designs and how interaction terms are interpreted in linear modeling.

We considered applying factorial design to a larger subset of variables, but this approach quickly becomes computationally infeasible and statistically unstable. A full factorial design with 10 variables would involve testing $2^{10} = 1024$ combinations, which is unrealistic given our sample size and the potential for multicollinearity and overfitting.

In addition, many categorical variables in the dataset have more than two levels, making them incompatible with the binary factor structure required for standard factorial designs. This would require either binarizing variables in an arbitrary way (risking loss of information) or using alternative designs beyond the scope of a $2^k$ factorial approach.

Instead, we chose to focus our modeling efforts on variable selection methods that are more suited to predictive modeling and high-dimensional data, such as stepwise regression and regularized models.

## Model Construction and Diagnostic Analysis

After preprocessing and feature selection, we built a multiple linear regression model to predict house prices using both quantitative variables and the top 10 categorical variables selected via ANOVA. These categorical variables were transformed using one-hot encoding, excluding the first category to avoid multicollinearity.

All variables were explicitly converted to numeric type, and any rows containing missing values were removed to ensure a clean dataset for model training. An intercept term was also added to the design matrix.

We applied an Ordinary Least Squares (OLS) regression using the `statsmodels` library. The regression model produced a summary output including R-squared values, coefficient estimates, and significance levels for each predictor.

### Regression Summary Interpretation

The regression summary provided several key insights:
We developed a multiple linear regression model aimed at predicting housing prices (`SalePrice`) based on a subset of carefully selected variables. This final feature set includes:
- The top 10 **categorical variables**, selected using ANOVA-based statistical ranking and encoded using **one-hot encoding**, and
- A selection of **quantitative variables** identified via their statistically significant Pearson correlation coefficients and low p-values relative to the target.

The model was estimated using the **Ordinary Least Squares (OLS)** method implemented via the `statsmodels` library, ensuring interpretability of coefficient estimates and access to comprehensive diagnostic statistics.

###  Model Summary and Global Performance

The model achieved an **R-squared value of 0.843**, meaning that approximately 84.3% of the variance in the house prices is explained by the selected predictors. This is a strong result, particularly given the heterogeneity of the data and the relatively large number of predictors included. The **adjusted R-squared**, which corrects for the number of predictors and penalizes the inclusion of less informative variables, remains high at **0.833**, further validating the relevance of the chosen features.
The **F-statistic value of 78.21** with a p-value essentially equal to 0 provides strong evidence that the overall model is statistically significant, rejecting the null hypothesis that all regression coefficients are simultaneously equal to zero.
The dataset used includes **1460 observations**, and the model comprises **94 degrees of freedom** (after encoding and cleaning), indicating a balance between model complexity and data availability.

###  Residual Diagnostics and Assumption Testing

To assess the validity of the classical assumptions underpinning linear regression, we reviewed several key statistics included in the model’s summary output.
- **Omnibus test = 486.533 (p < 0.001)** and **Jarque-Bera = 52,401.61 (p < 0.001)**: These jointly test whether the residuals are normally distributed in terms of skewness and kurtosis. The extremely low p-values strongly indicate that the residuals deviate from normality.
- **Skewness = –0.505**: Suggests a moderate left-tail asymmetry in the distribution of residuals.
- **Kurtosis = 32.332**: Far above the normal distribution benchmark of 3, implying the presence of heavy tails and outliers—consistent with the Jarque-Bera test.
- **Durbin-Watson = 1.912**: A value close to 2 confirms that there is **no evidence of autocorrelation** in the residuals, which supports the assumption of independent errors.
- **Condition Number = 1.01e+16**: This extremely high value is a strong red flag for **severe multicollinearity**, indicating that some predictors are highly linearly dependent. Such multicollinearity can inflate the standard errors of the coefficients, potentially destabilizing the interpretation of individual predictors.
These diagnostics suggest that while the model performs well overall, its inferential reliability could be enhanced by addressing multicollinearity (e.g., via dimensionality reduction or regularization techniques such as Ridge regression).


###  Interpretation of Categorical Coefficients
We extracted and visualized the 20 most influential categorical modalities based on the **absolute values of their estimated coefficients**. These coefficients quantify the marginal impact of each category on the predicted sale price, controlling for all other variables in the model.

- **Positive coefficients (e.g., `Neighborhood_NoRidge`, `SaleType_New`)** imply that the presence of these categories increases the predicted house price—sometimes by tens of thousands of dollars.
- **Negative coefficients (e.g., `KitchenQual_Fa`, `Exterior1st_ImStucc`)** indicate a downward impact on price, often signaling poorer quality or less desirable characteristics.

This visualization enhances the interpretability of the regression output by translating abstract statistics into concrete economic meaning.

### Graphical Residual Diagnostics

#### 1. **Residuals vs Fitted Plot**

This scatter plot shows the distribution of residuals against the model’s predicted values and is used to assess the assumptions of:
- **Linearity**: Residuals should be symmetrically distributed around zero.
- **Homoscedasticity**: The variance of the residuals should remain constant across fitted values.
**Observations**: The residuals are mostly randomly scattered around the zero line, suggesting that the model captures linear trends well. However, there is slight evidence of increasing variance at higher fitted values, suggesting **moderate heteroscedasticity**.

#### 2. **Histogram of Residuals**

The histogram allows us to visually inspect the **distribution shape** of residuals. Ideally, we expect a bell-shaped curve centered around zero.
**Observations**: The residuals are centered and exhibit symmetry, but the peak is sharp and the tails are long—confirming **leptokurtic behavior**, in line with the kurtosis and Jarque-Bera statistics.

#### 3. **Q-Q Plot (Quantile-Quantile Plot)**

This plot compares the quantiles of the residuals with those of a theoretical normal distribution.
**Observations**: The residuals generally follow the normal distribution line, but deviate at the tails. This further supports the conclusion that while residuals are **approximately normal**, there are **outliers and tail deviations**.

The multiple linear regression model constructed here:
- Demonstrates **strong explanatory power**,
- Is **statistically significant** overall,
- Identifies a **coherent set of key predictors** that significantly impact housing prices.

However, there are caveats:
- The residuals show **departures from normality** and mild **heteroscedasticity**,
- The extremely high condition number reveals **severe multicollinearity**, which undermines the stability of coefficient estimates.

To enhance the model's robustness in future iterations, we recommend:
- Investigating **regularized regression methods** (Ridge, Lasso) to mitigate multicollinearity,
- Considering **transformations** of skewed variables or the target (e.g., log transformations),
- Exploring **nonlinear models** or tree-based ensembles if predictive accuracy is prioritized over interpretability.


### Times Series Analysis
Prior to engaging in any modeling or advanced preprocessing, we conducted an exploratory analysis of the time-related variables in the dataset. Among these, four variables stood out due to their potential relevance in capturing housing market dynamics:

- YrSold: the year the house was sold
- MoSold: the month of sale
- YearBuilt: the year the house was originally constructed-
- YearRemodAdd: the year of the most recent remodeling or renovation

While YearBuilt and YearRemodAdd are valuable for understanding the property’s age and condition, they are static in nature and not directly tied to temporal price evolution. Consequently, our focus shifted to YrSold and MoSold, whose combination forms a true time series capable of capturing seasonal patterns and cyclical market behavior.

We aggregated housing prices by sale month and plotted the average monthly sale price across the dataset’s full time span (January 2006 to July 2010). The resulting plot exhibited no immediately clear*long-term upward or downward trend, and only weak indicators of seasonal structure. These preliminary observations motivated the use of smoothing techniques to suppress short-term noise and highlight broader trends.

#### Smoothing via Moving Averages

To attenuate local volatility and extract more stable trends, we applied a moving average (MA) smoothing technique with two different window sizes: 6 months (MA6)** and 12 months (MA12). This method replaces each data point with the average of its surrounding values within the defined window, effectively reducing the impact of transient fluctuations.

*INSERT IMAGE OF GRAPH*

The smoothed series revealed a clear downward trajectory across the period of interest, with the A6 curve (blue) capturing finer-grained seasonal oscillations, while the MA12 curve(orange) offered a more pronounced view of the underlying long-term trend.

This overall decline in housing prices coincides with the 2008 U.S. Subprime Mortgage Crisis. During the early 2000s, financial institutions increasingly issued high-risk subprime mortgages to borrowers with limited repayment capacity. When housing prices peaked and began declining around 2006, many borrowers defaulted, triggering widespread market instability. While Iowa was not among the epicenters of the crisis, its housing market may have nevertheless experienced secondary effects, consistent with the observed downward pressure on sale prices during this period.

#### Stationarity Assessment with the Augmented Dickey-Fuller Test

To determine the appropriateness of time series modeling approaches that assume stationarity (e.g., ARIMA), we conducted the Augmented Dickey-Fuller (ADF) test. The ADF test evaluates the null hypothesis that the time series contains a unit root, implying non-stationarity.

Our test yielded a p-value of 0.0036, well below the conventional 5% significance level. This provides strong statistical evidence to reject the null hypothesis, indicating that the series is stationary and can be modeled directly without differencing.

#### Autocorrelation and Partial Autocorrelation Analysis

We further analyzed the temporal dependencies in the series using Autocorrelation Function (ACF) and Partial Autocorrelation Function (PACF) plots, computed up to lag = 20 months, sufficient to capture dependencies across nearly two years.

- The ACF quantifies the correlation between the time series and its own lagged values, providing insight into repeated cycles and temporal structure. Strong autocorrelation at specific lags may suggest seasonality or persistence.

- The PACF, in contrast, measures the direct correlation between the time series and its lagged values, after controlling for the influence of intermediate lags. This allows us to isolate the effect of a specific lag while adjusting for shorter-term correlations.

### Model
## Analysis
## Discussion
## Conclusion
## Annex
## Sources
---
University of Neuchâtel - This work was done as part of the "Computational Statistics" course.

