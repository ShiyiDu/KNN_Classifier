# EECS4404_KNN_Project
This is the project we did for EECS4404 machine learning course. Here we played around with KNN model on the dataset we downloaded from [here](https://archive.ics.uci.edu/ml/datasets/Adult).

## Data Preprocessing
The original data consists of both numeric values and categorical string values. I mapped string values to numbers, removed all the entries with empty values(2399 of them), split the data 80/20 for training/testing, and balanced the training data to make sure the training data has 50% of each class. The processed training data is at [Main/trainingAdult.data](https://github.com/ShiyiDu/EECS4404_KNN_Project/blob/main/Main/trainingAdult.data), the processed test data is at [Main/testingAdult.test](https://github.com/ShiyiDu/EECS4404_KNN_Project/blob/main/Main/testingAdult.test)

## Distance Function

Since the KNN model doesn't really do much in the training process, it just remembers all the training data and tries to make predictions from them. We wanted to explore what kind of parameters we can change to improve the prediction accuracy. The most obvious way is to play around with the distance function; Luckily, SKlearn allows us to specify our own distance function.

### Euclidean Distance
Euclidean distance is the first distance function we tried, by applying the Genetic Algorithm(details are described later), we managed to hit accuracy of nearly 80%; the complete prediction result from this approach can be found here at [PredictionResults/KNN_euclid_dist.result](https://github.com/ShiyiDu/EECS4404_KNN_Project/blob/main/PredictionResults/KNN_euclid_dist.result)

### Hamming+Euclidean Distance
The second distance function we tried is a mixture of both Hamming distance and Euclidean distance. We wanted to try hamming distance because 8 out of 14 features are categorical values. When it comes to categorical values, you either belong to this category or don't; Thus, we guessed that Hamming distance might have a better prediction. However, we still have 4 features with numeric values that are not suitable for hamming distance, so we decided to mix them. Take an example of input `x = [1,2,3,4]; y = [5,6,7,8]`, assuming the first 2 values are numeric value and the rest are categorical values, the exact procedure is as follows:

1. extract numeric values from x and y, we get x_n = [1,2], y_n=[5,6]
2. extract categorical values from x and y, we get x_c = [3,4], y_c = [7,8]
3. calculate Euclidean distance between x_n and y_n; in this case, we get 5.66
4. calculate the Hamming distance between x_c and y_c; in this case, we get 2
5. add them up to get the result; in this case, we get 5.66 + 2 = 7.66

#### Coefficient
>Note that in the above procedure, we assumed that each feature contributed equally, and each distance function contributed equally; but in reality, some features might have a more significant influence on the result than the others, and one of the distance function might also have more significant influence than another. Therefore, we introduced weights for each feature as well as each distance function. I will refer to them as coefficients since 'weight' actually means differently in KNN of SKlearn.

>The coefficient for 2 distance function is easy, just multiply the result of each with the coreesponding coeeficient. For example, `Euclid(x,y) + Ham(x,y)` is applied with coeeficient `[a,b]`: `a * Euclid(x,y) + b * Ham(x,y)`

>For the input features' coefficient, we handled them differently for each distance function; for euclidean distance, we simply apply the coefficient on the input features before calculating the result. However, doing so doesn't make any sense for Hamming distance since hamming distance only has 2 result values 0/1 for each feature; therefore, we applied the coefficient during the calculation for hamming distance. For example, `(1, 2, 3) (3, 2, 4)` have a hamming distance of `1 + 0 + 1 = 2`, after apply the coefficient `[a, b, c]`, we get the following result instead: `1 * a + 0 * b + 1 * c`

Combining with Genetic Algorithm, we hit an accuracy of around 80%, the complete result is here at [PredictionResults/KNN_comp_dist.result](https://github.com/ShiyiDu/EECS4404_KNN_Project/blob/main/PredictionResults/KNN_comp_dist.result) (Yeah, the accuracy of prediction is basically the same as just using Euclidean distance)

---

## Genetic Algorithm

Genetic algorithm is something I have always wanted to try ever since the first time I learned it. GA is typically used to optimize discrete parameters; in this case, we have 14 coefficients, a k value for KNN, and 2 coefficients for 2 distance function. The implementation of GA is at [Main/GA.py](https://github.com/ShiyiDu/EECS4404_KNN_Project/blob/main/Main/GA.py); no third-party GA implementation is used.

### Result
Before I talk about the exact procedure, I want to present you with my conclusion: the GA algorithm did help with the model's accuracy, but the improvement is marginal after the first couple of generations. Each generation's population size is 20 individuals; the best individuals for each generation are recorded at [Docs/GA_result_1.txt](https://github.com/ShiyiDu/EECS4404_KNN_Project/blob/main/Docs/GA_result_1.txt) for Euclidean, and [Docs/GA_result_2.txt](https://github.com/ShiyiDu/EECS4404_KNN_Project/blob/main/Docs/GA_result_2.txt) for Complex distance.

### Genes
The Gene pool consists of value ranging \[0,9\], both inclusive. Therefore we have 10 possible values for each gene. The chromosomes for the first generation are randomly selected from the gene pool

### Chromosome
The Chromosome of a single individual looks like this:

`[0, 2, 0, 3, 8, 7, 0, 9, 4, 0, 3, 6, 1, 1, 9, 2, 1]`

As I discussed earlier, I tried both Euclidean distance and complex distance; Therefore, the chromosome was encoded differently, although they both have 17 genes. For Euclidean distance, the first 14 genes are the coefficient for 14 features. The last 3 genes are added up to get K's value; I did this because I wanted to determine the optimum value for k within a relatively large range. However, for complex distance, I only used the last gene to determine K value, the 15th and 16th gene is used to determine the 2 distance function's coefficient.

### Fitness
One essential component for GA is how fitness value for each individual is evaluated. In this case, I simply applied the genes from individuals' chromosomes on the KNN model, as I described above, and assess the model's correct rate on a randomly selected subset of training data.

### Selection
The selection is implemented on a probability basis. During the selection phase, half of the population will be selected to survive; the higher the fitness, the higher the chance. The exact probability of survival is calculated as `(fitness - lowest_fitness)/ (highest_fitness - lowest_fitness)`

### Crossover / Recombination
The next phase of GA is the crossover; During this stage, the other half of the population is filled by crossover. The individuals being selected for crossover are also selected on a probability base; just like selection, the higher the fitness, the higher the chance of being selected to mate. in this project, I used single-point crossover described in [Wikipedia](https://en.wikipedia.org/wiki/Crossover_(genetic_algorithm))

### Mutation
During the mutation stage, all the individuals selected to go to the next generation have a 10% chance of mutation. If it mutates, one of its genes is chosen randomly, and its value is randomly selected from the gene pool.

### Conclusion
Although after applying GA, the accuracy improvement is marginal, GA did help us identify which features are entirely useless. For example, the third gene is almost always 0 for every generation's best individual, which tells us that this feature does not contribute to its accuracy. If we want to simplify the model, these features with a coefficient of 0 might be selected to be removed.
