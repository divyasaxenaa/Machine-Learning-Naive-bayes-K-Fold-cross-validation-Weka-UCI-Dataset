import Naive as nb
import Graphplot as gr
import argparse
#python main.py --dataset cancer
#python main.py --dataset iris
#python main.py --dataset hayes
#python main.py --dataset car
def main():
   # construct the argument parse and parse the arguments
   ap = argparse.ArgumentParser()
   ap.add_argument("-i", "--dataset", required=True, help="select dataset")
   args = vars(ap.parse_args())
   dataset_selected = args["dataset"]
   nb.main(dataset_selected)

main()