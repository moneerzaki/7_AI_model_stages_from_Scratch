# 7_AI_model_stages_from_Scratch
7 stages to create your own AI model to solve any kind of problems: Regression, classification, and anomanly detection

### project 1: Data Manipulation and Visulaization 
Perform basic data manipulation: filtering, sorting, grouping, and aggregating. Create visualizations using matplotlib and seaborn to understand data distributions and relationships.
<br>
<div align="center">
  <img src="https://github.com/user-attachments/assets/b2e4325a-f6ee-4324-aace-9c35746075ec" width="400">
  <p><em>Figure 1: Univariate Analysis</em></p>
</div>
<div align="center">
  <img src="https://github.com/user-attachments/assets/0ebc271e-d878-4738-8565-34c1fb7a40d9" width="600">
  <p><em>Figure 2: Heat map</em></p>
</div>




### project 2: Data Preprocessing
Handle missing values using various techniques (mean/mode imputation, forward fill, etc.). Encode categorical variables (one-hot encoding, label encoding). Normalize/standardize numerical features. Identify and remove outliers if necessary

<div align="center">
  <img src="https://github.com/user-attachments/assets/f26abf8a-5956-4ba8-8d9c-3a4f9a521845" width="400">
  <p><em>Figure : Null percentage representation </em></p>
</div>

### project 3: Exploratory Data Analysis
Calculate and interpret summary statistics. Create visualizations to explore relationships between variables (scatter plots, box plots, histograms, heatmaps).Identify and analyze key patterns and insights from the data.

<div align="center">
  <img src="https://github.com/user-attachments/assets/1ba58ea9-93a6-429c-8008-bdb5e30f96af" width="400">
  <p><em>Figure : Intensive univariate anlaysis </em></p>
</div>

<div align="center">
  <img src="https://github.com/user-attachments/assets/e544db5d-8cf3-4b8a-9f76-67959d1cfb8a" width="400">
  <p><em>Figure : Intensive multivariate analysis </em></p>
</div>


### project 4: Supervised Learning (Regression)
Train a linear regression model and evaluate its performance using metrics like RMSE, MAE, and R^2. Experiment with other regression algorithms (Ridge, Lasso, Decision Tree Regressor). Perform hyperparameter tuning to improve model performance.

<div align="center">
  <img src="https://github.com/user-attachments/assets/3d8f50f3-498c-41b6-8443-d2a17cb9ae40" width="400">
  <p><em>Figure : Linear regression model before hyperparameter tuning </em></p>
</div>

<div align="center">
  <img src="https://github.com/user-attachments/assets/a4fdeaef-a2b2-436a-9061-469e9064c9f0" width="400">
  <p><em>Figure : Linear regression model after hyperparameter tuning </em></p>
</div>


<!-- ![image](https://github.com/user-attachments/assets/3d8f50f3-498c-41b6-8443-d2a17cb9ae40)
![image](https://github.com/user-attachments/assets/a4fdeaef-a2b2-436a-9061-469e9064c9f0) -->



### project 5: Supervised Learning (Classification)
Train a logistic regression model and evaluate its performance using metrics like accuracy, precision, recall, and F1-score. Experiment with other classification algorithms (Random Forest, SVM, KNN). Perform hyperparameter tuning to improve model performance.

<div align="center">
  <img src="https://github.com/user-attachments/assets/cc54fd8b-4393-4d9c-aaeb-7a52bbfe8c69" width="400">
  <p><em>Figure : Logistic Regression </em></p>
</div>

<div align="center">
  <img src="https://github.com/user-attachments/assets/eb64708f-1cef-4f77-89ab-ca1e6fb22b2e" width="400">
  <p><em>Figure : Random Forest </em></p>
</div>

<div align="center">
  <img src="https://github.com/user-attachments/assets/8ca327ce-44eb-4aea-af76-9a2fc6645033" width="400">
  <p><em>Figure : SVM </em></p>
</div>

<div align="center">
  <img src="https://github.com/user-attachments/assets/7ec3609e-a442-4b53-b474-fc667d3d06e7" width="400">
  <p><em>Figure : KNN </em></p>
</div>

<div align="center">
  <img src="https://github.com/user-attachments/assets/fd8c97f3-b46d-4144-9e46-a9371d9f5154" width="400">
  <p><em>Figure : All model comparison </em></p>
</div>


### project 6: Unsupervised Learning (Clustering)
Preprocess the dataset (normalize features if necessary). Implement K-means clustering and determine the optimal number of clusters using the elbow method. Visualize clusters and interpret the results. Experiment with hierarchical clustering and compare results.


<div align="center">
  <img src="https://github.com/user-attachments/assets/55a555e2-7aef-438c-ad29-aea3b9318870" width="400">
  <p><em>Figure : Elbow method </em></p>
</div>

<div align="center">
  <img src="https://github.com/user-attachments/assets/dfdf9b8a-a9d1-4b64-9389-7c6ce11b5d8a" width="400">
  <p><em>Figure : k-means clustering visualized 3d </em></p>
</div>

<div align="center">
  <img src="https://github.com/user-attachments/assets/0d853090-77fb-4ad3-928d-0dec07009fbf" width="400">
  <p><em>Figure : k-means clustering visualized 2d </em></p>
</div>

<div align="center">
  <img src="https://github.com/user-attachments/assets/84879e99-5828-4197-8c1f-3cd49a249db2" width="400">
  <p><em>Figure : k-means clustering visualized 3d </em></p>
</div>

<div align="center">
  <img src="https://github.com/user-attachments/assets/f114353f-6a75-4758-a219-5065bbcca200" width="400">
  <p><em>Figure : Hierarchical clustering Dendrogram  </em></p>
</div>

<div align="center">
  <img src="https://github.com/user-attachments/assets/77a92392-62f6-48e9-9246-4a9042c5066a" width="400">
  <p><em>Figure : k-Means++ Clustering  </em></p>
</div>


<div align="center">
  <img src="https://github.com/user-attachments/assets/33fae14b-1652-4b5c-913f-e4a52448aa62" width="400">
  <p><em>Figure : Number of clusters metric </em></p>
</div>


### project 7: Deep Learning (Neural Network)
Preprocess the dataset (normalize pixel values, one-hot encode labels). Build a neural network using TensorFlow/Keras. Train the network and evaluate its performance on the test set. Experiment with different network architectures and hyperparameters.


<div align="center">
  <img src="https://github.com/user-attachments/assets/605d7470-ddf8-4951-abf6-234f96a7ab64" width="400">
  <p><em>Figure : WandB visutlaization for hyperparameter tuning </em></p>
</div>


