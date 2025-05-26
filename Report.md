# From Features to Forecasts: Predicting Home Prices Using Computational Statistics 
**Angeliki Andreadi, Laila Ibrahim, Salsabil Mtiraoui**

## Abstract
We analyzed a real-world housing dataset using diverse statistical methods, including descriptive statistics, correlation tests, and a $2^k$ factorial design. Variables were categorized as quantitative, binary, or categorical and ranked by significance to identify those most predictive of housing prices. While the $2^3$ factorial design helped us understand interaction effects, its predictive power was limited due to low variance and multicollinearity. Ultimately, our focus shifted toward more scalable variable selection techniques better suited for high-dimensional data analysis.

A multiple linear regression model using selected quantitative and categorical variables (via Pearson correlation and ANOVA) showed strong performance (R² = 0.843), but diagnostic tests revealed non-normal residuals, heteroscedasticity, and multicollinearity—suggesting some instability in coefficient estimates. Still, several categorical features had interpretable effects on price.
A complementary time series analysis of monthly sale prices (2006–2010) showed a downward trend aligned with the 2008 housing crisis. After confirming stationarity (ADF test, p = 0.0036), we modeled the trend using ARIMA(4,0,0), which offered the best fit with no detected seasonality.

## Introduction
Ask a home buyer to describe their dream house, they rarely start with its price. Instead, they focus on features like the number of rooms, proximity to transportation, or size of the garden. However, these characteristics ultimately shape the property's market value.  

The main goal of this project is to apply and gain insight into various statistical techniques within a practical context. While the Kaggle competition on Ames housing prices provides the framework, our focus lies in implementing and understanding methods such as 2ᵏ factorial designs, ANOVA, time series analysis, and statistical inference. Rather than purely striving for predictive performance, we use the modeling process as a means to explore and solidify our grasp of these core statistical tools.

We hypothesize that the most intuitive and surface-level features—those that non-expert buyers tend to focus on, such as the number of rooms, presence of a garden, or proximity to the city—are also the ones that have the strongest influence on a home's selling price. While these variables may not capture all the technical nuances of property valuation, they likely reflect the criteria most salient to buyers during the decision-making process.

We believe it aligns with real-world buyer behavior: individuals often rely on easily observable characteristics when assessing the value of a home, and these features are commonly emphasized in real estate listings. By comparing the predictive power of such “intuitive” variables with more technical or structural features, we can better understand the role that perceived value plays in actual sales outcomes.

## Data Overview
The data used in this study originates from the Kaggle competition [House Prices: Advanced Regression Techniques](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/data). It represents a sample of residential property sales recorded in Ames, Iowa. While not exhaustive of the entire housing market, it offers a representative snapshot of housing transactions in the area. The dataset contains 79 explanatory variables that capture a wide range of features related to the properties, such as lot dimensions, room counts, building characteristics, and neighborhood information.

All coding was conducted using Google Colab, a free, cloud-based development environment that supports collaborative programming in a Jupyter notebook interface. Its seamless integration with Python libraries and ease of sharing made it particularly well-suited for our project.

## Preprocessing
  Our first step was to separate our data set into three subsets: binary, quantitative and categarical, because each data type requires a different statstical treatment to be made useful. 
  
  In summary, the conditions and statements we used to classify the variables are: 
  
- If a column's `dtype` is either a `int64`(integer) or `float64` and it has exactly 2 unique values, it is appended to the `binary_var` list.
- If the column is numerical(`int64` or `float64` and has more than `quant_threshold` unique values, it is considered a quantitative variables and appended to `quant_var`.
- If the column is numerical but has less or equal to `quant_threshold` unique values, we treat it as a categorical variables, and it is added to `categorical_var`. 
- If the column's `dtype` is `object`, `category` or `bool`:
  - And it has exactly 2 unique values:
    - If it's of type `object`, we convert it to binary by mapping its two unique values to 0 and 1 and then append it to `binary_var`.
  - If it has more than 2 unique values, we append it to `categorical_var`

Following that, for each of our data frames we dropped the `ID` column as we had no use for it and dropped the variables that were incomplete, that had less than 1460 complete rows. 

## Feature selection

### Quantitative variables
To begin our exploration of the quantitative variables, we conducted a comprehensive descriptive statistical analysis. This step aimed to summarize the key characteristics of each variable in our datset `df_quant`.

We created a function called `descriptive_stats`, which builds on Python's built-in `.describe()` method by adding 95% confidence intervals for the mean of each variable. For each column, it computes: 
- The count, check if there are any missing values
- The mean
- The standard deviation
- The minimum and maximum value
- The 25th, 50th and 75th percentiles
- The lower and upper bounds of a 95% Confidence Interval, using Student's t-distribution formula

To assess the linear relationship between each quantiative feature and the target variable we implemented a function called `hypothesis_test` which computes the Pearson correlation coefficient, which measures the strength and direction of a linear association, and the corresponding p-value, which helps determine the statistical significance of that correlation. 

Next, to identify the most relevant features for predicting housing prices, we implemented a custom ranking function. It combines both descriptive statistics and results from hypothesis testing to score each feature's usefulness in relation to the target variable. 
This ranking process is based on three key criteria: 
- Correlation strength with target (using absolute correlation values to capture both positive and negative relationships)
- P-values from hypothesis testing (where lower values indicate stronger statistical significance)
- Variance of each feature (under the assumption that a higher spread may carry more informative value)

The p-values is the only criteria we took into consideration in the final ranking formula, since it's the one that expresses the most the significance of the variables. Features were then sorted by their total score, allowing us to focus on those with the greatest potential predictive power. 

### Binary variables
For the binary variables, we started by collecting some insights via descriptive statistics by using, same as before, the `.describe()` function. Then, we computed their correlation with the 'SalePrice' variable, our analysis's focal point, to get an idea of their impact on it. Despite the negative results, we kept them for further possible uses.

### Categorical variables
We started the exploration of our 3rd category also by collecting some insights via descriptive statistics with the `.describe()` function. In opposite to the two precious variables, the given insights were on the frequency of the variables and not the mean, standard deviation, etc. For each column, we got:

- The count, to check if there are any missing values 
- The number of possible observations (`unique`)
- The observation that has the highest frequency (`top`)
- The frequency of this observation (`freq`)

Then we created a function named `rank_categorical_vars` to evaluate and rank the filtered categorical variables. The ranking is based on two criteria: the p-value (ranked in ascending order) and the mean range (ranked in descending order). Variables with smaller p-values and larger mean ranges are considered more influential on the SalesPrice.

These rankings are visualized in the following graph,`Ranking of Categorical Variables Based on Final Score`. The final score combines two components: p_rank, which reflects the significance of the p-value (lower values indicate greater significance), and range_rank, which captures the magnitude of the mean range (higher values denote greater differences between group means). This score serves as our selection criterion, where a lower final score indicates a more impactful variable.

![image](https://github.com/user-attachments/assets/d4dfab0a-c0d6-44ee-9cb7-abe563e697e8)


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
For this part we did multiple regression tests to find what keysights we can get from our data. We did multiple linear regression model to predict the housing prices (`SalePrice`) based on the choosen variables. After some testing, we finally settled for the top 10 categorical variables using ANOVA-based statistical ranking and a selection of quantative variables selected for their Pearson correlation coefficients and low p-values.

### Global Performance

The model gave us an R-squared value of 0.843, showing that approximately 84.3% of the variance in the house prices is explained by the selected predictors. It's quite a strong result, sèecially with the heterogeneity of the data and the large number of predictors. The adjusted R-squared remains high at 0.833, which validates the importance of the choosen variables.
The F-statistic value of 78.21 with a p-value closely equal to 0. it shows a strong evidence that the model is statistically significant, rejecting the null hypothesis that all regression coefficients are simultaneously equal to zero.
The dataset used includes 1460 observations and ended up with 94 degrees of freedom after processing the data, which means the model is detailed but still well supported by the amount of data."

###  Residual Diagnostics and Assumption Testing

After that we checked several key statistics from the model summary to see if the main assumptions of a linear regression was true:
- Omnibus test = 486.533 (p < 0.001) and Jarque-Bera = 52,401.61 (p < 0.001): The residuals are normally distributed in terms of skewness and kurtosis. The extremely low p-values strongly indicate that the residuals deviate from normality.
- Skewness = –0.505: Could mean a slight left-tail asymmetry in the distribution of residuals.
- Kurtosis = 32.332: Above the normal distribution benchmark of 3, meaning the presence of heavy tails and outliers—consistent with the Jarque-Bera test.
- Durbin-Watson = 1.912: The value is close to 2, confirming that there is no evidence of autocorrelation in the residuals, supporting the assumption of independent errors.
- Condition Number = 1.01e+16: This very large value raises a serious concern. It suggests that some predictors are too closely related to one another, creating multicollinearity. As a result, the standard errors of the coefficients may increase, making it harder to trust the individual estimates.
Overall, the model performs well. However, these results suggest a weakness beneath the surface. To enhance the reliability of the conclusions we draw from it, we should address the multicollinearity—possibly by reducing the number of overlapping variables or applying regularization methods such as Ridge regression.

###  Interpretation of Categorical Coefficients
We took a closer look at the 20 categories that had the biggest impact on the predicted house prices. To do that, we checked which ones had the highest or lowest coefficients in the model. These results help show which features really matter when it comes to house pricing.
Categories with positive coefficients (like Neighborhood_NoRidge or SaleType_New) tend to increase the estimated price — sometimes by several tens of thousands of dollars.
On the other hand, negative coefficients (such as KitchenQual_Fa or Exterior1st_ImStucc) are linked to lower prices, often because they reflect features seen as lower quality or less appealing.

### Graphical Residual 

#### 1. Residuals vs Fitted Plot

This scatter plot shows the distribution of residuals against the model’s predicted values and is used to explain the assumptions of:
- Linearity: Residuals should be symmetrically distributed around zero.
- Homoscedasticity: The variance of the residuals should remain constant across fitted values.

In our case, the residuals are mostly randomly scattered around the zero line, suggesting that the model captures linear trends well. But there is a slight evidence of increasing variance at higher fitted values, suggesting moderate heteroscedasticity.

#### 2. Histogram of Residuals

The histograms gives the shape of the distribution of the residuals. The ideal is a bell-shaped curved centered at 0. In our figure the residuals are centered and exhibit symmetry, but the peak is sharp and the tails are long—confirming leptokurtic behavior, in line with the kurtosis and Jarque-Bera statistics.

#### 3. **Q-Q Plot (Quantile-Quantile Plot)**

This plot compares the quantiles of the residuals with those of a theoretical normal distribution. The residuals generally follow the normal distribution line, but deviate at the tails. This further supports the conclusion that while residuals are approximately normal, there are outliers and tail deviations.

## Times Series Analysis
Prior to engaging in any modeling or advanced preprocessing, we conducted an exploratory analysis of the time-related variables in the dataset. Among these, four variables stood out due to their potential relevance in capturing housing market dynamics:

- YrSold: the year the house was sold
- MoSold: the month of sale
- YearBuilt: the year the house was originally constructed-
- YearRemodAdd: the year of the most recent remodeling or renovation

While YearBuilt and YearRemodAdd are valuable for understanding the property’s age and condition, they are static in nature and not directly tied to temporal price evolution. Consequently, our focus shifted to YrSold and MoSold, whose combination forms a true time series capable of capturing seasonal patterns and cyclical market behavior.

We aggregated housing prices by sale month and plotted the average monthly sale price across the dataset’s full time span (January 2006 to July 2010). The resulting plot exhibited only a slight downward trend, and only weak indicators of seasonal structure. These preliminary observations motivated the use of smoothing techniques to suppress short-term noise and highlight broader trends

![image](https://github.com/user-attachments/assets/892daf62-8f2c-4851-9344-4d70bd8b7478)

### Smoothing via Moving Averages

To attenuate local volatility and extract more stable trends, we applied a moving average (MA) smoothing technique with two different window sizes: 6 months (MA6) and 12 months (MA12). This method replaces each data point with the average of its surrounding values within the defined window, effectively reducing the impact of transient fluctuations.

![image](https://github.com/user-attachments/assets/22c01e41-adf2-48d2-bf29-274f9a97b57d)

The smoothed series revealed a clear downward trajectory across the period of interest, with the A6 curve (blue) capturing finer-grained seasonal oscillations, while the MA12 curve(orange) offered a more pronounced view of the underlying long-term trend.

This overall decline in housing prices coincides with the 2008 U.S. Subprime Mortgage Crisis. During the early 2000s, financial institutions increasingly issued high-risk subprime mortgages to borrowers with limited repayment capacity. When housing prices peaked and began declining around 2006, many borrowers defaulted, triggering widespread market instability. While Iowa was not among the epicenters of the crisis, its housing market may have nevertheless experienced secondary effects, consistent with the observed downward pressure on sale prices during this period.

### Stationarity Assessment with the Augmented Dickey-Fuller Test

To determine the appropriateness of time series modeling approaches that assume stationarity (e.g., ARIMA), we conducted the Augmented Dickey-Fuller (ADF) test. The ADF test evaluates the null hypothesis that the time series contains a unit root, implying non-stationarity.

Our test yielded a p-value of 0.0036, well below the conventional 5% significance level. This provides strong statistical evidence to reject the null hypothesis, indicating that the series is stationary and can be modeled directly without differencing.

### Autocorrelation and Partial Autocorrelation Analysis

We further analyzed the temporal dependencies in the series using Autocorrelation Function (ACF) and Partial Autocorrelation Function (PACF) plots, computed up to lag = 20 months, sufficient to capture dependencies across nearly two years.

- The ACF quantifies the correlation between the time series and its own lagged values, providing insight into repeated cycles and temporal structure. Strong autocorrelation at specific lags may suggest seasonality or persistence.

- The PACF, in contrast, measures the direct correlation between the time series and its lagged values, after controlling for the influence of intermediate lags. This allows us to isolate the effect of a specific lag while adjusting for shorter-term correlations.

![image](https://github.com/user-attachments/assets/a861c8b6-1f03-4425-b516-e52beefa7555)

![image](https://github.com/user-attachments/assets/17707220-8905-4d35-99c1-68a91d7cc11c)

The ACF and PACF plots did not reveal any clear tapering patterns or seasonal spikes—only a noticeable spike at lag 4 in both. Based on this, we compared the AIC and BIC values of three potential models:
- An AR(4)
- An MA(4)
- An ARIMA (4,0,0) which yielded the following results: 

|Model    |AIC      | BIC    |
|---------|---------|--------|
|AR(4)    | 1226.85 | 1238.91|
|MA(4)    | 1229.55 | 1241.61|
|ARMA(4,4)| 1226.85 | 1238.91|

Since both AR(4) and ARMA(4,4) yielded the lowest AIC and BIC values, and the AR model is simpler, we selected AR(4) to reduce the risk of overfitting.
While the AR(4) model emerged as the most statistically efficient among the initial candidates, we recognized the importance of validating this choice against slightly more general and structured alternatives. To this end, we fitted and compared three additional models:

1. ARMA(4,0,0): equivalent to the previously selected AR(4), included here as a benchmark.
2. ARIMA(0,0,4): a purely moving average model to test whether the structure of the time series was better captured by lagged errors rather than lagged values.
3. SARIMAX(1,0,1)(1,0,0,12): a seasonal model incorporating a yearly seasonality component (lag 12), to test for subtle recurring patterns not immediately visible in the ACF/PACF plots.

![image](https://github.com/user-attachments/assets/8e0f4cf0-bf7f-4d2f-b9f0-2f0deab6d261)

We visualized and compared each model’s fit to the observed data to assess how well each captured the temporal dynamics. In addition to visual inspection, we compared their AIC and BIC scores:
|Model       |AIC      |BIC      |
|------------|---------|---------|
|ARIMA(4,0,0)| 1226.86 | 1238.90 |
|ARIMA(0,0,4)| 1229.56 | 1241.60 |
|SARIMAX     | 1248.11 | 1256.14 |

The ARIMA(4,0,0) model again showed the lowest AIC and BIC values, reinforcing our earlier conclusion. While the SARIMAX model was included to test for seasonality, its higher AIC/BIC scores — combined with no visual evidence of seasonal spikes in the ACF/PACF — confirmed that seasonality likely wasn’t a strong driver in this dataset. Similarly, the moving average model underperformed both in terms of fit and complexity.

By systematically comparing these models, we validated our initial AR(4) choice not just through information criteria, but also through model diagnostics and practical visual fit. This stepwise approach gave us confidence that the AR(4) structure is the most appropriate, simple, and robust representation of the data’s temporal dependencies.

## Discussion

Looking back on our project, there are several aspects we would approach differently if given the chance. One key limitation was our lack of experience, which made it difficult to build a precise and accurate model. This was particularly evident during feature selection, where we faced challenges in ranking the quantitative variables. Although we had initially defined three important criteria to guide this process, we encountered difficulties in assigning meaningful weights to each one. The weighting formula did not yield the expected results, and due to time constraints, we ultimately based our ranking solely on p-values, which may have oversimplified the selection process.

In terms of linear regression, while our model showed decent performance, we noticed issues such as multicollinearity and non-normal residuals. With more time, we would have explored regularization techniques like Ridge or Lasso regression to mitigate multicollinearity, applied log transformations to skewed variables for better normality, and even experimented with nonlinear models such as decision trees to improve predictive power, even if it meant sacrificing some interpretability.

For the time series analysis, our ARIMA model was based on manually selected parameters. In future iterations, we would opt for automated hyperparameter tuning using tools like `auto_arima()` from the `pmdarima` library. This would allow for a more objective and statistically grounded selection of (p, d, q) parameters, ultimately improving the forecasting accuracy.

Finally, our approach to evaluating prediction accuracy could have been improved. We initially planned to use the `test.csv` file to evaluate our model, expecting it to contain house prices for future years. However, we found that the test set covered the same years (2006–2010) as the training data and lacked target values. In retrospect, a better strategy would have been to split the original training data chronologically, using the last 20% of observations as a proper test set to assess how well our model predicts unseen, more recent data.

## Conclusion

Originally, our ambition was to create a complete pipeline including a test phase on future data. As we progressed, we had to adapt our choices and objectives to the time constraints and limitations of the dataset.

We cleaned and structured the data, then selected the most relevant variables. Our final model is based on a combination of the ten best categorical variables, selected using an ANOVA test, and the quantitative variables most correlated with sales price according to Pearson's coefficient and statistical significance. This approach enabled us to build a multiple linear regression model achieving an R² of 0.843, demonstrating very good explanatory power. That said, analysis of the residuals and statistical diagnostics revealed some important weaknesses, such as strong multicollinearity, a non-normal distribution of errors, and a slightly unstable variance.

In parallel, we explored the temporal dimension of the data. Through time series analysis, we observed price variations over several years, applied smoothing techniques to better visualize trends, and tested several models. The AR(4) model proved to be the most suitable, both simple and efficient, while capturing the visible effects of the 2008 subprime crisis.

All in all, even if not everything went according to plan, we feel we achieved our objectives. We were able to build a relevant model, adapt our methods to the nature of the data, and take a step back from the tools we used. Over and above the application of theoretical concepts, this project taught us to plan our work better, to keep a clear thread running through it, and to always think several steps ahead. Last but not least, it showed us how much the choice of method depends on the context, and that each statistical technique has its advantages but also its limitations.

## References

1. Kaggle. *Time Series: Interpreting ACF and PACF*. https://www.kaggle.com/code/iamleonie/time-series-interpreting-acf-and-pacf/notebook
2. Kaggle. *Comprehensive data exploration with Python*. https://www.kaggle.com/code/pmarcelino/comprehensive-data-exploration-with-python
3. Kaggle. *Price Prediction Train & Test Data Analysis*. https://www.kaggle.com/code/sonalisingh1411/price-prediction-train-test-data-analysis
4. Investopedia (2024). *Stock Market Crash of 2008*. https://www.investopedia.com/articles/economics/09/subprime-market-2008.asp
5. Data Heroes (2024). *Complete Time Series Analysis and Forecasting with Python*. https://www.youtube.com/watch?v=eKiXtGzEjos
6. W3schools *Machine Learning - Linear Regression:* https://www.w3schools.com/python/python_ml_linear_regression.asp

---
University of Neuchâtel - This work was done as part of the "Computational Statistics" course.
