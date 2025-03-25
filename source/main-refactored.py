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
        print("please pass arguments to the command")
        exit()


    experiment_code = sys.argv[1]
    print('Experiment:', experiment_code)
    models = ['vgg'] # ['mnist-basic', 'vgg', 'resnet', mobilenet]
    datasets = ['cifar10'] # ['mnist', 'cifar10', 'imagenet_v2', 'open_images_v4']
    exp = 'reversed' # [HUSH, HUSH_DP, reveresed_HUSH_DP, SISA, baseline, baseline_DP, other, reversed]

    if experiment_code == "segment-count":
        print('Segment count:', sys.argv[2])

        seg_count = int(sys.argv[2])
        results = dict()
        root_percent = 0.5

        for (model, ds_name) in [(x, y) for x in models for y in datasets]:
            result = {}
            model_key = exp+"-"+model+"-"+str(seg_count) # f strings are used to format strings easily (new feature upgrade from .format())

            if exp == 'HUSH':
                result = get_latency_acc_dict(model, ds_name, seg_count, experiment_code, root_percent)
                if result == None:
                    continue
                else:
                    results[model_key] = result
            
            elif exp == 'HUSH_DP':
                result = get_latency_acc_dict_HUSHDP(model, ds_name, seg_count, experiment_code, root_percent)
                if result == None:
                    continue
                else:
                    results[model_key] = result

            elif exp == 'reveresed_HUSH_DP':
                result = get_latency_acc_dict_HUSHD_reveresed(model, ds_name, seg_count, experiment_code, root_percent)
                if result == None:
                    continue
                else:
                    results[model_key] = result
    
            elif exp == 'SISA':           
                result = get_latency_acc_dict_SISA(model, ds_name, seg_count, experiment_code, root_percent)
                if result == None:
                    continue
                else:
                    results[model_key] = result
            
            elif exp == 'baseline':
                results['benchmark'] = get_latency_acc_dict_benchmark(model, ds_name, differential_privacy = False)
            
            elif exp == 'baseline_DP':
                results['benchmark_DP'] = get_latency_acc_dict_benchmark(model, ds_name, differential_privacy = True)
            
            elif exp == 'other':
                # logic for evaluating saved model "baseline_DP"
                basic_mnist_hush_dp_model = tf.keras.models.load_model('/Users/samaagazzaz/Desktop/UCSC_Courses/IPPML/experiments/source/hush_dp/mnist/25-04-2023_22:19:48.h5', compile=False)
                ds_test, ds_info = getDsTestDsInfo('mnist')
                ds_test = test_pipeline(ds_test) #pipeline for the test data

                
                basic_mnist_hush_dp_model.compile(
                    # optimizer=get_optimizer('DP'),
                    optimizer= tf.keras.optimizers.Adam(0.001),
                    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                    metrics=[tf.keras.metrics.SparseCategoricalAccuracy()]
                )
                loss, acc = basic_mnist_hush_dp_model.evaluate(ds_test)
                print('model accuracy: ', acc)

                for i in range(ceil(len(basic_mnist_hush_dp_model.layers)/4)):
                    basic_mnist_hush_dp_model.layers[i].trainable = False

                ds_train = load_data(seg_count, range(seg_count), ds_name)
                ds_train = segment_train_pipeline(range(seg_count), ds_train) #pipeline for the training data segments
      

                basic_mnist_hush_dp_model.fit(
                    ds_train[0],
                    epochs=20,
                    verbose = 1
                )
                loss, acc = basic_mnist_hush_dp_model.evaluate(ds_test)
                print('model accuracy: ', acc)

            elif exp == 'reversed':
                #TODO add logic to implement multiple roots and one branch experiment
                # logic for evaluating saved model "baseline_DP"
                basic_mnist_hush_dp_model = tf.keras.models.load_model('/Users/samaagazzaz/Desktop/UCSC_Courses/IPPML/experiments/source/hush_dp/mnist-basic/model.h5', compile=False)
                ds_test, ds_info = getDsTestDsInfo('mnist')
                ds_test = test_pipeline(ds_test) #pipeline for the test data

                
                basic_mnist_hush_dp_model.compile(
                    # optimizer=get_optimizer('DP'),
                    optimizer= tf.keras.optimizers.Adam(0.001),
                    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                    metrics=[tf.keras.metrics.SparseCategoricalAccuracy()]
                )
                loss, acc = basic_mnist_hush_dp_model.evaluate(ds_test)
                print('model accuracy: ', acc)

                basic_mnist_hush_dp_model.trainable = False
                for i in range(3):
                    basic_mnist_hush_dp_model.layers[i].trainable = True

                ds_train = load_data(seg_count, range(seg_count), ds_name)
                ds_train = segment_train_pipeline(range(seg_count), ds_train) #pipeline for the training data segments
      
                for i in range(len(ds_train)):
                    basic_mnist_hush_dp_model.fit(
                        ds_train[i],
                        epochs=20,
                        verbose = 0
                    )
                    loss, acc = basic_mnist_hush_dp_model.evaluate(ds_test)
                    print('model accuracy: ', acc)

            printdict(results)
            saveresultsfile(sys.argv, model, results, exp)

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
