2 b) From the file coeffs.py, we print the coefficients learnt using L1 regularisation. We set lambda
as 10(high value). This makes all coefficients except 7th, 14th, 23rd, 29th and 32nd approximately zero.
Hence the ones mentioned here are the important attributes.

3 c) Looking at the heatmap, we see that 8 gets confused the most, along with 3.
From the heatmap again, the digit that gets the least confused is 0, since other than the diagonal entry, its entire row is dark on the heatmap.

3 d) From the PCA plot, we can make out that most numbers are separable, but as we see in the bottom centre section, these numbers will tend to get mixed up by the model, because they have similar projected attributes, so they will likely have similar attributes.

4) The time complexity of logistic regression is the same as that of Linear regression.
Let t be number of iterations, n be the number of samples and m be the number of features
Time complexity of Fitting: O(t * n * m)
Time complexity of Predicting: O(n * m) (n stands for number of prediction samples), if there are k classes, O(n x m x k), since we have to choose the highest probability out of k classes.


