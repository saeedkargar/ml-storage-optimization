from utilscopy import * 
import sys
# command line guide:
# note to self: run "conda activate base" before running the code to switch environment to where tensorflow is installed
'''
python3 main.py <experiment-code>


experiment-code: 
- "segment-count": the effect of changing segment count on accuracy and latency
    python3 main.py segment-count <segment-count>

    - segment-count: max number of segmnets in experiment

    example:
        python3 main.py segment-count 16 

- "root-size": the effect of changing root size on accuracy and latency
    python3 main.py root-size <segment-count>
    - segment-count: the number of segments to run this experiment on

    example:
        python3 main.py root-size 6


'''

def main():
    print('Number of arguments:', len(sys.argv), 'arguments')

    if len(sys.argv) < 2:
        print("please pass arguments to the command")
        exit()


    experiment_code = sys.argv[1]
    print('Experiment:', experiment_code)
    models = ['vgg'] 
    datasets = ['cifar10'] 


    if experiment_code == "segment-count":
        print('Segment count:', sys.argv[2])

        # create a split to divide the dataset into a number "segment_count" of even training segments
        segment_list  = list(range(2, int(sys.argv[2]) + 1)) #the range of segment counts needed for experiment
        results = dict()
        root_percent = 0.50

        for (model, ds_name) in [(x, y) for x in models for y in datasets]:
            model_key = model+f'-{root_percent:.2f}' # f strings are used to format strings easily (new feature upgrade from .format())
            result, ds_info, ds_test = get_latency_acc_dict(model, ds_name, segment_list, experiment_code, root_percent)

            if result == None:
                continue
            else:
                results[model_key] = result

            printdict(results)

    else:
        print("There was an error with the argumnets passed to the script")



if __name__ == "__main__":
    main()
    # alert when experiments conclude successfully
    alert()
