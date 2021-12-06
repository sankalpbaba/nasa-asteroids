# nasa-asteroids
** SMALL DESCRIPTION- **
My model basically predict wether the asteriods are hazardous or non hazardous.
I made two models , one using catboost and other using xgboost.Basically my model has one of the best accuracy with least deviation right now compare to models avaliable all over the models present right mow respective to the dataset i have used.

** THE datset has imbalanced distribution and unneccesary features. But it has been reduced and implmented in my model.It comprises of around 1512 values(small dataset).**

** INTRODUCTION **
FOR MODEL 1 - 
We Used catboost .
what is catboost?
CatBoost is an algorithm for gradient boosting on decision trees.
Robust
CatBoost can improve the performance of the model while reducing overfitting and the time spent on tuning.  CatBoost has several parameters to tune. Still, it reduces the need for extensive hyper-parameter tuning because the default parameters produce a great result.
Accuracy
The CatBoost algorithm is a high performance and greedy novel gradient boosting implementation. Hence, CatBoost (when implemented well) either leads or ties in competitions with standard benchmarks.
Categorical Features Support
The key features of CatBoost is one of the significant reasons why it was selected by many boosting algorithms such as LightGBM,  XGBoost algorithm ..etc With other machine learning algorithms. After preprocessing and cleaning your data, the data has to be converted into numerical features so that the machine can understand and make predictions.This is same like, for any text related models we convert the text data into to numerical data it is know as word embedding techniques.This process of encoding or conversion is time-consuming. CatBoost supports working with non-numeric factors, and this saves some time plus improves your training results.

These features are easy to tune and are well-explained in the CatBoost documentation. Here are some of the parameters that can be optimized for a better result;
cat_ features, one_hot_max_size, learning_rate & n_estimators,max_depth, subsample, colsample_bylevel, colsample_bytree,  colsample_bynode,  l2_leaf_reg, random_strength.
The default settings of the parameters in CatBoost would do a good job.CatBoost produces good results without extensive hyper-parameter tuning.


CATBoost works excelently with categorical features, while XGBoost only accepts numeric inputs.
accuracy-99.33%
Precision: 0.987
Recall: 0.987


![Screenshot (459)](https://user-images.githubusercontent.com/90260133/144815490-724569cf-a1fd-41f9-a572-9a79e7e9d599.png)

after k folds
Accuracy: 99.43 %
Standard Deviation: 0.97 %
![Screenshot (464)](https://user-images.githubusercontent.com/90260133/144816874-17cdae81-59e2-4dd9-b71c-be00705f247f.png)



** FOR MY MODEL 2- **
** XGBOOST **
what is xg boost?
XGBoost is a decision-tree-based ensemble Machine Learning algorithm that uses a gradient boosting framework. In prediction problems involving unstructured data (images, text, etc.) artificial neural networks tend to outperform all other algorithms or frameworks. However, when it comes to small-to-medium structured/tabular data, decision tree based algorithms are considered best-in-class right now. THE PROCEDURE FOR XGBOOST ALGORITHM IS- 1 Decision Tree: Every decision tree has a set of criteria which are features and parameters like for example for teacher hiring criteria is education level, number of years of experience, interview performance. A decision tree is analogous to a teacher hiring manger interviewing candidates based on his or her own criteria. 2 Bagging: Now imagine instead of a single teacher hiring interviewer , now there is an whole panel where each interviewer has a vote. Bagging or bootstrap aggregating involves combining inputs from all interviewers for the final decision through a democratic voting process. 3 Random Forest: It is a bagging-based algorithm with a key difference wherein only a subset of features is selected at random. In other words, every interviewer will only test the interviewee on certain randomly selected qualifications (e.g. a english interview for testing english speaking skills and a behavioral interview for evaluating non-technical skills). 4 Boosting: This is an alternative approach where each interviewer alters the evaluation criteria based on feedback from the previous interviewer. This ‘boosts’ the efficiency of the interview process by deploying a more dynamic evaluation process. 5 Gradient Boosting: A special case of boosting where errors are minimized by gradient descent algorithm e.g. the aptitude skills firms will leverage by using case interviews to filter out less qualified candidates. 6 XGBoost: Think of XGBoost as gradient boosting on ‘steroids’ (thats why it is called extreme gradient boosting). It is a perfect combination of software and hardware optimization techniques to yield superior results using less computing resources in the shortest amount of time.

Why does XGBoost perform so well?
 However, XGBoost improves upon the base GBM framework through systems optimization and algorithmic enhancements. The library provides a system for use in a range of computing environments, not least: Parallelization of tree construction using all of your CPU cores during training. Distributed Computing for training very large models using a cluster of machines. Out-of-Core Computing for very large datasets that don’t fit into memory. Cache Optimization of data structures and algorithm to make best use of hardware. System Optimization: 1 Parallelization: XGBoost approaches the process of sequential tree building using parallelized implementation. This is possible due to the interchangeable nature of loops used for building base learners; the outer loop that enumerates the leaf nodes of a tree, and the second inner loop that calculates the features. This nesting of loops limits parallelization because without completing the inner loop (more computationally demanding of the two), the outer loop cannot be started. Therefore, to improve run time, the order of loops is interchanged using initialization through a global scan of all instances and sorting using parallel threads. This switch improves algorithmic performance by offsetting any parallelization overheads in computation. 2 Tree Pruning: The stopping criterion for tree splitting within GBM framework is greedy in nature and depends on the negative loss criterion at the point of split. XGBoost uses ‘max_depth’ parameter as specified instead of criterion first, and starts pruning trees backward. This ‘depth-first’ approach improves computational performance significantly. 3 Hardware Optimization: This algorithm has been designed to make efficient use of hardware resources. This is accomplished by cache awareness by allocating internal buffers in each thread to store gradient statistics. Further enhancements such as ‘out-of-core’ computing optimize available disk space while handling big data-frames that do not fit into memory.
99.5% accuracy 
Avg Bias: 0.007
Avg Variance: 0.005



![Screenshot (460)](https://user-images.githubusercontent.com/90260133/144815714-0c96babe-73ad-468d-b4ab-631c12595418.png)


Accuracy: 99.43 %
Standard Deviation: 0.97 %

![Screenshot (463)](https://user-images.githubusercontent.com/90260133/144816764-e000b4aa-a38a-4a04-b45e-5ce1e7e148c0.png)


