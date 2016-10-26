# ID3-Decision-Tree

## Implementation
The decision-learner is implemented according to the following guidelines:

- Candidate splits for nominal features should have one branch per value of the nominal feature. The branches should be ordered according to the order of the feature values listed in the ARFF file.
- Candidate splits for numeric features should use thresholds that are midpoints betweeen values in the given set of instances. The left branch of such a split should represent values that are less than or equal to the threshold.
- Splits should be chosen using information gain. If there is a tie between two features in their information gain, you should break the tie in favor of the feature listed first in the header section of the ARFF file. If there is a tie between two different thresholds for a numeric feature, you should break the tie in favor of the smaller threshold.
- The stopping criteria (for making a node into a leaf) are that (i) all of the training instances reaching the node belong to the same class, or (ii) there are fewer than m training instances reaching the node, where m is provided as input to the program, or (iii) no feature has positive information gain, or (iv) there are no more remaining candidate splits at the node.
- If the classes of the training instances reaching a leaf are equally represented, the leaf should predict the most common class of instances reaching the parent node.
- If the number of training instances that reach a leaf node is 0, the leaf should predict the the most common class of instances reaching the parent node.

## Running the program
dt-learn train-set-file test-set-file m
