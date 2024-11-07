# Hyperparameter-Optimization-Genetic-Algorithm

## Steps of Implementation : 

1. Define the hyperparameters’ search space
The search space refers to the possible range of values that each defined hyperparameter can take at any given time. These hyperparameters mostly have integer values, float values, boolean values, or string values. It is essential to select a search space that is broad enough to incorporate potentially good combinations that can produce the best performance but not so broad that the search becomes inefficient.

Here are a few examples of hyperparameters from the KNN algorithm available on the scikit-learn library (KNeighborsClassifier()):

n_neighbors: This is the number of neighbors to use.

weights: This is the weight function used in prediction.

algorithm: This is the algorithm used to compute the nearest neighbors.

leaf_size: This is the leaf size passed to BallTree or KDTree.

2. Define chromosomes
The next step is to define chromosomes, which are different combinations of the hyperparameters’ values, forming the solution to the problem (classification or regression).

The chromosomes have genes, which refer to hyperparameters of the algorithm with specific values.

Some examples of chromosomes are listed below based on the selected hyperparameters from the KNeighborsClassifier()

3. Set the termination criteria
We have to define the criteria for stopping the genetic algorithm, such as reaching a certain number of generations or achieving a satisfactory level of performance.

For example, having a larger number of generations, such as 20 or 50, is better because the genetic algorithm will try different combinations before selecting the best one. However, it can take a lot of time to optimize because it might require more computational power if the model is complex and the dataset is large.

4. Initialize the population
The fourth step is to generate an initial population of chromosomes randomly. This will be the first generation. This population serves as the starting point for the optimization process.

![image](https://github.com/user-attachments/assets/8179ba2a-3f55-43f0-b4d7-354068c6698f)

5. Evaluate the fitness (performance)
Evaluate the performance of each chromosome (ML model) in the population using a fitness function. The fitness function quantifies how well the corresponding set of hyperparameters performs on the ML task.

The fitness function can be any ML evaluation metric depending on the type of problem to be solved, i.e., classification or regression.

When working on the ML project for classification problems, we can use the following evaluation metrics:

Accuracy: This is the number of correct predictions made by an ML model divided by the total number of predictions made.

F1 score: This is the harmonic mean of precision and recall.

When working on an ML project for regression problems, we can use the following evaluation metrics:

Mean absolute error (MAE): This is the absolute value of the difference between the predicted value and the actual value.

Root mean squared error (RMSE): This is the square root of the value calculated from the mean square error (MSE).

6. Select the chromosomes
We select the chromosomes from the current population based on their fitness scores. The idea is to give preference to better-performing solutions. Some common selection methods are explained below, which include roulette wheel selection, tournament selection, and rank-based selection:

Roulette wheel selection: ML models in a population are assigned probabilities of being chosen based on their fitness. Selection occurs by spinning a virtual roulette wheel, with fitter ML models having higher chances of being selected.

Tournament selection: Tournament selection involves randomly selecting a subset (tournament) of chromosomes from the population and choosing the fittest chromosome from this subset. This process is repeated to form the next generation, providing a balance between exploration and exploitation in the search for optimal solutions.

Rank-based selection: ML models in the population are sorted based on their fitness, and each chromosome is assigned a probability of selection proportional to its rank. This method helps maintain diversity by giving lower-ranked individuals a chance to be selected, promoting exploration. It ensures that even less fit chromosomes contribute to the genetic diversity of the population, potentially preventing premature convergence to suboptimal solutions.

7. Conduct the reproduction (crossover and mutation)
Following the selection process, the next step involves generating offspring through the reproduction process. Within this step, two variations of operators are applied to the parent population. The two operators utilized in the reproduction step are explained below:

(a) Crossover
Crossover means combining the genetic material of selected chromosomes to create new offspring. In the context of hyperparameter optimization, crossover involves exchanging hyperparameter values between two parent chromosomes to produce one or more child chromosomes.

![image](https://github.com/user-attachments/assets/2f91d43c-fe3b-4aed-87a2-1498a879e8be)

The parents’ genes (hyperparameter’s values) are swapped or exchanged until they reach the crossover point. From the example above, we can see that new offspring has both n_neigbors=5 and leaf_size=30 from Parent 1, as well as both weights='distance and algorithm='ball_tree from Parent 2.

(b) Mutation
The mutation operator introduces random genes (hyperparameter values) into the offspring (new machine learning model) to uphold diversity in the population. This involves making random changes to certain genes in the chromosomes (ML models), aiding in the exploration of diverse regions within the hyperparameter space that could potentially lead to better solutions.

![image](https://github.com/user-attachments/assets/467e9be9-45b6-4c23-bfee-38bb351bcf5e)

From the above example, we can see that after mutation, n_neighbors=7 and leaf-size=33 are added to the population.

8. Create a new generation
In this step, the genetic algorithm replaces the old population with the newly created offspring from the reproduction step. This step ensures that the population evolves over generations, favoring chromosomes with higher fitness values, which means ML models that have the potential to produce the best performance based on the selected evaluation metric are preferred.

9. Iterate the process
The steps from 5 to 8 are repeated for a predefined number of generations or until the termination criteria are met. The genetic algorithm will continue to evolve the population over time, adapting to the characteristics of the search space defined.

10. Terminate the process
Finally, the process is terminated when a stopping criterion is met, such as when a maximum number of generations has been reached or when the improvement of the criteria defined is above a certain threshold. The genetic algorithm will then identify and extract the best-performing set of hyperparameters from the final population as the optimized solution for the given problem.

Implementing these steps effectively requires careful consideration of the specific ML problem, the choice of hyperparameters, and the design of the fitness function or evaluation metric. Adjusting parameters such as number of generations, population size, crossover rate, and mutation rate can also impact the algorithm’s performance.

The various parameters of the genetic algorithm are discussed in detail below:

Number of generations: The number of generations determines how many iterations of the genetic algorithm will be executed. A larger number of generations allows for a more thorough exploration of the search space, which can lead to better solutions.

Population size: The population size refers to the number of individuals in each generation. A larger population size leads to a better exploration of the search space, as it increases the diversity of solutions and reduces the risk of premature convergence to suboptimal solutions.

Crossover rate: The crossover rate determines the probability that crossover will be applied to pairs of parent individuals to produce offspring. A higher crossover rate encourages exploration, while a lower crossover rate promotes exploitation of existing solutions.

Mutation rate: The mutation rate specifies the probability that mutation will be applied to each gene (or parameter) in an individual. A higher mutation rate increases exploration but might slow convergence, while a lower mutation rate might lead to premature convergence.

It is recommended to perform numerous experiments to fine-tune the process and achieve optimal results.

## Implementation example : Binary Classification using KNN model is optimized using Genetic algorithm


