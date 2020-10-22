from classifiers.cnn import CNN
from os.path import join
import pandas as pd
import pickle
import torch
import classifiers.data_processing as dp
import multiprocessing as mp
from classifiers.data_processing import get_policy_of_interest_tokens, get_tokens
from classifiers.privacy_policies_dataset import PrivacyPoliciesDataset as PPD
from multiprocessing import Pool, cpu_count

def predictAttributeProbabilities(dicty, segments):
    oldSegments = segments
    allMightyPredictions = []
    for query, rowIndeces in dicty.items():
        print( " query ",query)
        print("rowIndeces", rowIndeces)
        lookupForLater = [(oldSegments[rowIndex], rowIndex) for rowIndex in rowIndeces]
        segments = [pair[0] for pair in lookupForLater]
        print('ich predicte', segments)
        labelsDict = dp.attr_value_labels(query)

        labels = []
        if labelsDict:
            for (key, value) in enumerate(labelsDict.items()):
                labels.append(value[0])

            path = 'classifiers/trained_models'
            model_file = join(path, f'cnn_100_200_[100]_{len(labels)}_[3]_{query}_zeros_60-20-20_polisis_attr_state.pt')
            params_file = join(path, f'cnn_100_200_[100]_{len(labels)}_[3]_{query}_zeros_60-20-20_polisis_attr_params.pkl')

            with open(params_file, 'rb') as f:
                params = pickle.load(f)
            model = CNN(**params)
            model.load_state_dict(torch.load(model_file))
            model.eval()


            all_tokens = get_policy_of_interest_tokens(segments, "classifiers/embeddings_data")
            dictionary = get_tokens(f"classifiers/{query}", "classifiers/embeddings_data/attr", read = True)
            segments_tensor = dp.process_policy_of_interest(dictionary, segments)

            predictions = model(segments_tensor)

            # print(f'PREDICTIONS: {predictions}')
            y_pred = predictions > 0.5
            # print(f'Y PRED: {y_pred}')
            predictionsList = []
            for row in range(len(segments)):
                predictedValues = y_pred[row, :]
                otherPredictedValues = predictions[row, :]
                for label in range(len(labels)):
                    if predictedValues[label] == 1:
                        # print("paragraph " + str(row) + " : " + labels[label])
                        # print('--------------------------------------')

                        result = (query, labels[label])
                        secondPair = (result, otherPredictedValues[label].data.item())
                        predictionsList.append([lookupForLater[row][1], secondPair])
        else:
            predictionsList = []

        allMightyPredictions.append(predictionsList)

    flat_list = []
    for sublist in allMightyPredictions:
        for item in sublist:
            flat_list.append(item)

    print(flat_list)
    print("SIND ES HIER SCHON ZU VIELE", flat_list)
    return flat_list

def predictProbabilities(query, segments, rowIndex):
    print('ich predicte', query)
    labelsDict = dp.attr_value_labels(query)

    labels = []
    if labelsDict:
        for (key, value) in enumerate(labelsDict.items()):
            labels.append(value[0])

        path = 'classifiers/trained_models'
        model_file = join(path, f'cnn_100_200_[100]_{len(labels)}_[3]_{query}_zeros_60-20-20_polisis_attr_state.pt')
        params_file = join(path, f'cnn_100_200_[100]_{len(labels)}_[3]_{query}_zeros_60-20-20_polisis_attr_params.pkl')

        with open(params_file, 'rb') as f:
            params = pickle.load(f)
        model = CNN(**params)
        model.load_state_dict(torch.load(model_file))
        model.eval()


        all_tokens = get_policy_of_interest_tokens(segments, "classifiers/embeddings_data")
        dictionary = get_tokens(f"classifiers/{query}", "classifiers/embeddings_data/attr", read = True)
        segments_tensor = dp.process_policy_of_interest(dictionary, segments)

        predictions = model(segments_tensor)

        # print(f'PREDICTIONS: {predictions}')
        y_pred = predictions > 0.5
        # print(f'Y PRED: {y_pred}')
        predictionsList = []
        for row in range(len(segments)):
            predictedValues = y_pred[row, :]
            otherPredictedValues = predictions[row, :]
            for label in range(len(labels)):
                if predictedValues[label] == 1:
                    # print("paragraph " + str(row) + " : " + labels[label])
                    # print('--------------------------------------')
                    if rowIndex is not None:
                        result = (query, labels[label])
                        secondPair = (result, otherPredictedValues[label].data.item())
                        predictionsList.append([rowIndex, secondPair])
                    else:
                        result = (query, labels[label])
                        secondPair = (result, otherPredictedValues[label].data.item())
                        predictionsList.append([row, secondPair])
    else:
        predictionsList = []

    return predictionsList

def mergeArray(array, length):
    array = list(array)
    # print("ANFANGSARRAY", array)
    toTransform = array[1]
    currentIndex = 0
    index = 0
    if toTransform:
        while index < length:
            # print("index", index)
        #    print("totransform", toTransform)
            currentIndex = toTransform[index][0]
        #    print("currentIndex", currentIndex)
            if index+1 == len(toTransform):
                break
            nextIndex = toTransform[index+1][0]
            # print("nextIndex", nextIndex)
            if currentIndex == nextIndex:
                toRemove = toTransform[index+1]
                toTransform.remove(toRemove)
                toTransform[index].append(toRemove[1])

                # print("toTransform",toTransform)

            else:
                index = index + 1
    array[1] = toTransform
    return array

def predict(segments):
    listOfPredictions = []
    predictions = predictProbabilities("Main", segments, None)
    mainPredictions = ("MainPredictions", predictions)
    print("mainPredictions", mainPredictions)
    listOfPredictions.append(mainPredictions)
    attributePredictions = []
    probabilities = []

    dictionary = {}
    for prediction in predictions:
        rowIndexList = []
        attributes = dp.get_possible_attributes(prediction[1][0][1])
        rowIndexList.append(prediction[0])
        for attribute in attributes:
            # {"Purpose": [[segments],purpose[0]}
            rowIndexList = list(set(rowIndexList))
            # dictionary[attribute] = rowIndexList
            if attribute in dictionary:
                oldList = dictionary[attribute]
                print("oldList", oldList)
                print("type of oldList", type(oldList))
                # make entries unique
                newList = list(set(oldList + [prediction[0]]))
                print("newList", newList)
                dictionary[attribute] = newList
            else:
                dictionary[attribute] = [prediction[0]]
            print("dictionary",dictionary)
        #     probabilities = predictProbabilities(attribute, [segments[prediction[0]]], prediction[0])
        #     print("attribute", attribute)
        #     print("semgents[prediction[0]]", [segments[prediction[0]]])
        #     print("prediction[0]", prediction[0])
        #     if probabilities:
        #         attributePredictions.append(probabilities[0]
    attributePredictions = predictAttributeProbabilities(dictionary, segments)
    print("sorted", attributePredictions.sort())
    attributePredictionsToAppend =  ("AttributePredictions", attributePredictions)
    print("attributePredictionsToAppend", attributePredictionsToAppend)
    listOfPredictions.append(attributePredictionsToAppend)
    mergeArray(listOfPredictions[0], len(segments))
    mergeArray(listOfPredictions[1], len(segments))
    print("list of predictions:", listOfPredictions)
    return (listOfPredictions,segments)
