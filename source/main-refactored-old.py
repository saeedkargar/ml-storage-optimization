from utilsrefactored import * 
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
        ds_test, ds_info = tfds.load(
            'mnist', 
            split='test',
            as_supervised=True,
            with_info=True
        )

        learning_rate = 0.25 #, 'Learning rate for training')
        noise_multiplier = 0.1 #'Ratio of the standard deviation to the clipping norm')
        l2_norm_clip = 1.5 #, 'Clipping norm')
        microbatches = batch_size #, 'Number of microbatches '(must evenly divide batch_size)')

        ds_test = test_pipeline(ds_test)
        basic_mnist_hush_dp_model = tf.keras.models.load_model('/Users/samaagazzaz/Desktop/UCSC_Courses/IPPML/experiments/source/hush_dp/mnist/25-04-2023_22:19:48.h5', compile=False)
        opt = DPKerasAdamOptimizer(l2_norm_clip= l2_norm_clip, noise_multiplier= noise_multiplier, num_microbatches= microbatches, learning_rate= learning_rate)
        basic_mnist_hush_dp_model.compile(
            optimizer=opt,
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=[tf.keras.metrics.SparseCategoricalAccuracy()]
        )

        loss, acc = basic_mnist_hush_dp_model.evaluate(ds_test)
        print(loss, acc)
        print("please pass arguments to the command")
        exit()


    experiment_code = sys.argv[1]
    print('Experiment:', experiment_code)
    models = ['mnist-basic'] # ['mnist-basic', 'vgg', 'resnet', mobilenet]
    datasets = ['mnist'] # ['mnist', 'cifar10', 'imagenet_v2', 'open_images_v4']


    if experiment_code == "segment-count":
        print('Segment count:', sys.argv[2])

        seg_count = int(sys.argv[2])
        results = dict()
        root_percent = 0.50

        for (model, ds_name) in [(x, y) for x in models for y in datasets]:
            model_key = "OURS-"+model+f'-{root_percent:.2f}' # f strings are used to format strings easily (new feature upgrade from .format())
            result = {}

            approach = "ours"
            result, ds_info, ds_test = get_latency_acc_dict(model, ds_name, seg_count, experiment_code, root_percent)
            
            if result == None:
                continue
            else:
                results[model_key] = result
                saveresultsfile(sys.argv, model, results, approach)
            
            # approach = "sisa"            
            # model_key = "SISA-"+model+f'-{root_percent:.2f}' # f strings are used to format strings easily (new feature upgrade from .format())

            # result, ds_info, ds_test = get_latency_acc_dict_SISA(model, ds_name, seg_count, experiment_code, root_percent)

            if result == None:
                continue
            else:
                results[model_key] = result

            results[model_key]['benchmark'] = get_latency_acc_dict_benchmark(model, ds_name, ds_info, ds_test, differential_privacy = False)
            # results[model_key]['benchmark_DP'] = get_latency_acc_dict_benchmark(model, ds_name, ds_info, ds_test, differential_privacy = True)
            printdict(results)


        saveresultsfile(sys.argv, model, results, approach)

    # TODO refactor the root-size elif    
    elif experiment_code == "root-size":
        percents = [x * 0.1 for x in range(1, 10)]
        seg_count = int(sys.argv[2])
        results = dict()

        for (model, ds_name) in [(x, y) for x in models for y in datasets]:
            model_key = f"{model}-{ds_name}"
            model_results = {model_key: dict()}

            for p in percents:
                
                result, ds_info, ds_test = get_latency_acc_dict(model, ds_name, seg_count, experiment_code, p)
                
                if result == None:
                    print(f"experiment {experiment_code}: model {model}, dataset {ds_name} and percent {p} did not return any result")
                else:
                    merge(model_results[model_key], result)
            
            merge(results, model_results)
                
        printdict(results)
        saveresultsfile(sys.argv, model, results, "HUSH")

    elif experiment_code == "":
        # TODO
        pass

    elif experiment_code == "":
        # TODO
        pass

    else:
        print("There was an error with the argumnets passed to the script")



if __name__ == "__main__":
    main()
