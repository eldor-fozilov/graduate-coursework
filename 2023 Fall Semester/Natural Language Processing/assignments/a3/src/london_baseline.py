# Calculate the accuracy of a baseline that simply predicts "London" for every
#   example in the dev set.
# Hint: Make use of existing code.
# Your solution here should only be a few lines.

import utils

def main():

    eval_path = "birth_dev.tsv"
    eval = open(eval_path, "r")
    predictions = ["London"] * len(eval.readlines())
    total, correct = utils.evaluate_places(eval_path, predictions)
    print('Correct: {} out of {}: {}%'.format(correct, total, correct/total*100))  
    
if __name__ == "__main__":
    main()