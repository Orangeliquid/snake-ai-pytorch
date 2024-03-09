import matplotlib.pyplot as plt
from IPython import display
import os

plt.ion()


def plot(scores, mean_scores):
    display.clear_output(wait=True)
    display.display(plt.gcf())
    plt.clf()
    plt.title('Training...')
    plt.xlabel('Number of Games')
    plt.ylabel('Score')
    plt.plot(scores)
    plt.plot(mean_scores)
    plt.ylim(ymin=0)
    plt.text(len(scores)-1, scores[-1], str(scores[-1]))
    plt.text(len(mean_scores)-1, mean_scores[-1], str(mean_scores[-1]))
    plt.show(block=False)
    plt.pause(.1)


def get_top_score():
    top_score_file = "top_score.txt"

    if os.path.exists(top_score_file):
        with open(top_score_file, "r") as file:
            try:
                top_score = int(file.read().strip())
                return top_score
            except ValueError:
                print("Error reading top score. Resetting to 0.")
                return 0
    else:
        print("Top score file not found. Creating a new one.")
        with open(top_score_file, "w") as file:
            file.write("0")
        return 0


def update_top_score(new_score):
    top_score_file = "top_score.txt"
    current_top_score = get_top_score()

    if new_score > current_top_score:
        with open(top_score_file, "w") as file:
            file.write(str(new_score))
        print(f"New top score: {new_score}")
    else:
        print(f"Current top score: {current_top_score}")
