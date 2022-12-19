# TRAF2-11215

This repo contains the files used for solving the TRAF2-11215 task. The requirement was to classify individual tokens to their respective category.

TL;DR: There are four versions of the model. The last one is the best and the results are lower in the README.

There are two versions for this task: legacy and bert.

## Legacy
The legacy model is trained from scratch using a Bidirectional LSTM as the main learning layer. The results were lacking and we felt that an upgrade was required. The legacy pipeline is described in the file `ML_pipeline.png`, in `/utils/`. 

The second attempt tried to simplify the vocabulary by grouping words that contained numbers together. The groups were: `[DATE]` for date-related words, `[QUANTITY]` for words that could be converted to Integer, `[UNITPRICEAMOUNT]` for words that could be converted to Float and `[INCOTERMS]` for all the other words that contained numbers, but couldn't be converted into one of the other types. This process is documented in `/notebooks/preprocessing/better_tokenisation.ipynb`.

In the end, this pre-processing method was removed, as the pre-processing and post-processing were done manually in both Python and C#, this adding room for errors. Another problem was converting the tags back to the original words after running the sentence through the model. Because of these reasons, we decided the best course of action was to remove this step.

## Bert
Using a pre-trained model we were able to improve the performance of the model, in some areas with 40% improvement. We've taken advantage of a library that has a lot of features built on top of Python, `transformers`.`

### Requirements
The requirements folder contains the necessary libraries to run this project. To install the necesarry libraries, use the command `pip install -r requirements.txt`.
### Data
The data consists of 1660 sentences divided into 3 parts (70,15,15): train, test and validation. 

The training data is used for training the model, the testing dataset is used for fine-tuning and reasoning and the validation set is used as a sanity check before deployment of the model.

We tried to use `over-sampling` as a means to increase the dataset, but the results after over-sampling were lack-luster. This is because over-sampling does not add new data, it just copies and pastes the same data. Also, because we're dealing with token-classification, when adding copying and pasting an under-sampled class, it's highly probable to also add a over-sampled class again. Because of this reason, and because the improvements weren't noticeable, we decided to remove this step as well.

### Models: architecture, versioning, training, ONNX and output
There are currently 4 version of the models, 2 legacy ones and 2 bert-based ones. The last one, version 4 is the only one currently in use. 

The architecture uses Bert as a base and attaches a Dropout layer, followed by a Linear layer, so the final architecure will be `BERT -> Dropout layer -> Linear layer`. 

The output has a softmax function applied to it, thus we take the maximum argument (`argmax`) to get the predictions.

The training is done locally on a Virtual Machine, after creating a `venv` or a conda env with the required libraries. The library will also require a `huggingface` account and a token for authentification. You will need to change the directories to your own folders. 
After the training is done, you can safely remove the newly created folder with the tokenizer and model, since the model is automatically pushed to the `huggingface hub`. 

After training, the last step required is to convert the model to ONNX format for the C# back-end. These blogs [1](https://rubikscode.net/2021/10/25/using-huggingface-transformers-with-ml-net/), [2](https://rubikscode.net/2022/09/13/bert-tokenizers-for-ml-net/) were really helpful.

After converting to ONNX using the command:

```python -m transformers.onnx -m ClaudiuFilip1100/bert-ner-conpend-v4 --feature token-classification output```

you will need to check if the models' inputs and output shapes are recognised. Plug the ONNX model into [this tool](https://netron.app), click Properties and make sure the inputs are of type `[batch, sequence]` and the outputs are of type `[batch, sequence, 8]`. 

The final step requires you to send the tokenizer's vocabulary (in a .txt file) to the back-end team, along with the converted ONNX model.
### Results

The results are based on the validation set.
```json
{
  'HSCode': {
    'precision': 0.0,
    'recall': 0.0,
    'f1': 0.0,
    'number': 6
  },
  'Incoterms': {
    'precision': 0.7165354330708661,
    'recall': 0.7711864406779662,
    'f1': 0.7428571428571428,
    'number': 354
  },
  'UnitPriceAmount': {
    'precision': 0.7106382978723405,
    'recall': 0.6720321931589537,
    'f1': 0.6907962771458118,
    'number': 497
  },
  'Tolerance': {
    'precision': 0.0,
    'recall': 0.0,
    'f1': 0.0,
    'number': 19
  },
  'GoodsDescription': {
    'precision': 0.4713375796178344,
    'recall': 0.3425925925925926,
    'f1': 0.3967828418230563,
    'number': 432
  },
  'GoodsOrigin': {
    'precision': 0.0,
    'recall': 0.0,
    'f1': 0.0,
    'number': 15
  },
  'Quantity': {
    'precision': 0.6342857142857142,
    'recall': 0.5842105263157895,
    'f1': 0.6082191780821918,
    'number': 380
  },
  'overall_precision': 0.6440342781806196,
  'overall_recall': 0.5736934820904287,
  'overall_f1': 0.606832298136646,
  'overall_accuracy': 0.9046435431024743
}
```
As we can see from the model, the overall F1-score is at 60%, which is a good result, considering the small size of the dataset and the limited samples. For example, for HSCode, Tolerance and GoodsOrigin we couldn't determine the score, since there are too few datapoints. 

I suggest we add more data to improve results.

### Inference
When we want to check the model or improve upon it we must use the huggingface repository. 


### Further improvements
Remove the split between train, test and validation and train on the whole dataset, but I don't beleive the improvements will be huge. We also won't have any way of measuring our and won't know how well we're doing until we get results back from the clients.