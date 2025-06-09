import os
from ast import literal_eval
from typing import List, Optional, Tuple, Union, Dict

from hmmlearn import hmm
from Algorithms import Algorithms
from HMM import HMM
import pandas as pd
import numpy as np
from itertools import product, chain
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
from Plots import Plots
from typeguard import typechecked

@typechecked
class Testing:
    __testExperiments: int = 30
    __presetExperiments: int = 45
    __testType: List[str] = ["Ambiguous", "Cyclic", "Sparse", "Uniform", "Diagonal", "Regular"]
    __genType: List[str] = HMM.getDistributions()
    __seqLengths = np.array([3e2, 1e3, 5e3, 1e4], dtype=np.uint32)
    __seqNum = np.array([1, 5, 10, 50, 100], dtype=np.uint32)

    def __init__(self):
        pass

    @staticmethod
    def __prepareTests(elements: Optional[List] = None, fileName: str = "tasks.txt") -> Dict[str, Union[List[Tuple], Dict]]:
        if elements is None:
            elements = [Testing.__testType, Testing.__genType, Testing.__seqLengths, Testing.__seqNum]

        folder = f"{os.getcwd()}/testsCases"
        cartesian = list(chain.from_iterable(  # Flatten everything into one array of tuples
            (name + (int(fileName.split('.')[0]),) for fileName in os.listdir(f"{folder}/{name[0]}"))
            # Add the number of states (name of the file without the extension) to the name of the test
            for name in list(product(*elements))
            # Generate a cartesian product based on the sets above
        ))
        dummyHmm = HMM()

        # Do not run the test if there is already data stored in the file for this key
        runTests = Testing.__readRunTests(fileName)
        cartesian = list(set(cartesian) - set(list(runTests.keys())))

        matrices = dict({name: None for name in cartesian})
        result = {
            name: None for name in cartesian}
        result.update(runTests)  # Add the already run tests from the file in order to show them on the plots as well

        for name in cartesian:
            matrices[name] = dummyHmm.readMatrices(f"{folder}/{name[0]}/{name[-1]}.txt", True)
            matrices[name]["nStates"] = matrices[name]["logP"].shape[0]
            matrices[name]["nOutputs"] = matrices[name]["logB"].shape[1]

        return dict({
            "matrices": matrices,
            "result": result,
            "cartesian": cartesian,
        })

    @staticmethod
    def baumWelchTesting() -> None:
        aux = Testing.__prepareTests()
        matrices, result, cartesian = aux["matrices"], aux["result"], aux["cartesian"]

        hmmModel = HMM()
        for name in cartesian:
            hmmParams = matrices[name]
            interResults = dict({
                "alg": dict({"P": np.empty(Testing.__testExperiments), "A": np.empty(Testing.__testExperiments),
                             "B": np.empty(Testing.__testExperiments)}),
                "hmm": dict({"P": np.empty(Testing.__testExperiments), "A": np.empty(Testing.__testExperiments),
                             "B": np.empty(Testing.__testExperiments)}),
            })
            for i in range(Testing.__testExperiments):
                hmmModel.setParams(**hmmParams)
                nStates, nOutputs = hmmParams["logP"].shape[0], hmmParams["logB"].shape[1]
                expP, expA, expB = np.exp(hmmParams["logP"]), np.exp(hmmParams["logA"]), np.exp(hmmParams["logB"])
                subResult = Algorithms.baumWelch(hmmModel, name[1], name[2], name[3], useModelSeq=False)

                interResults["alg"]["P"][i] = np.linalg.norm(expP - np.exp(subResult["logP"]))
                interResults["alg"]["A"][i] = np.linalg.norm(expA - np.exp(subResult["logA"]))
                interResults["alg"]["B"][i] = np.linalg.norm(expB - np.exp(subResult["logB"]))

                model = hmm.MultinomialHMM(n_components=nStates, n_iter=Algorithms.learnIterations, init_params='')
                model.startprob_ = subResult["initP"]
                model.transmat_ = subResult["initA"]
                model.emissionprob_ = subResult["initB"]

                sequences = np.concatenate(
                    [np.eye(nOutputs, dtype=np.uint32)[sequence["emissions"] - 1] for sequence in subResult["sequences"]])
                length = [name[2]] * name[3]
                model.fit(sequences, length)

                # Aligning the result
                estP, estA, estB = Algorithms.resolveLabelSwitching(np.exp(hmmParams["logB"]), model.emissionprob_,
                                                                    model.transmat_, model.startprob_)

                interResults["hmm"]["P"][i] = np.linalg.norm(expP - estP)
                interResults["hmm"]["A"][i] = np.linalg.norm(expA - estA)
                interResults["hmm"]["B"][i] = np.linalg.norm(expB - estB)

            result[name] = dict({
                key: {subkey: np.mean(arr) for subkey, arr in subdict.items()}
                for key, subdict in interResults.items()
            })
            Testing.__writeRunTests(name, result[name])

        result = Testing.__regroupTests(result)
        Plots.compareAlgAndHmm(result)
        Plots.showTestCasesPlot(result)

        Testing.runRegression(result)

    @staticmethod
    def __regroupTests(results: Dict[Tuple, Dict[str, Dict[str, float]]]) -> Dict[str, Dict[int, Dict[Tuple, Dict[str, Dict[str, float]]]]]:
        regrouped = dict({})
        for name, value in results.items():
            first, *middle, last = name
            regrouped.setdefault(first, {}).setdefault(last, {})[tuple(middle)] = value

        return regrouped

    @staticmethod
    def __readRunTests(fileName: str = "tasks.txt") -> Dict[Tuple, Dict[str, Dict[str, float]]]:
        result = dict({})
        if not os.path.exists(fileName):
            with open(fileName, 'w') as f: pass

        with open(fileName, 'r') as file:
            while name := file.readline():
                name = literal_eval(name.strip())
                result[name] = dict({
                    "alg": dict({
                        'P': float(file.readline()),
                        'A': float(file.readline()),
                        'B': float(file.readline())
                    }),
                    "hmm": dict({
                        'P': float(file.readline()),
                        'A': float(file.readline()),
                        'B': float(file.readline())
                    })
                })

        return result

    @staticmethod
    def runRegression(results: Dict[str, Dict[int, Dict[Tuple, Dict[str, Dict[str, float]]]]], addNames: Optional[Dict[str, Dict[str, Union[bool, str]]]] = None, isSub: bool = False) -> None:
        names = dict({
            "matrixType": {"skip": True, "name": "matrix type"},
            "nStates": {"skip": True, "name": "number of states"},
            "init": {"skip": True},
            "seqLen": {"skip": isSub, "name": "length of sequences"},
            "seqNum": {"skip": isSub, "name": "number of sequences"}
        })
        if addNames is not None:
            names.update(addNames)

        names.update({
            "errorP": {"skip": True, "name": "Error in P"},
            "errorA": {"skip": True, "name": "Error in A"},
            "errorB": {"skip": True, "name": "Error in B"},
        })

        # 1. Flatten results into rows
        rows = []
        for matrixType, typeDict in results.items():
            for nStates, stateDict in typeDict.items():
                for values, res in stateDict.items():
                    rows.append(dict(
                        zip(names, [matrixType, nStates, *values, res['alg']['P'], res['alg']['A'], res['alg']['B']])))

        names.update({
            f"init_{value}": {"name": f"{value.lower()} sampling", "skip": False} for value in Testing.__genType[1:]
        })
        df = pd.DataFrame(rows)

        # 2. Compute original std dev BEFORE standardization
        stdSeqLen = df['seqLen'].std()
        stdSeqNum = df['seqNum'].std()

        if df['init'].nunique() > 1:
            # 3. One-hot encode categorical feature
            df = pd.get_dummies(df, columns=['init'])
            df = df.drop(columns=[f"init_{Testing.__genType[0]}"])
        else:
            names["init"]["skip"] = True
            df.drop(columns=['init'])

        # 4. Standardize
        scaler = StandardScaler()
        df[['seqLen', 'seqNum']] = scaler.fit_transform(df[['seqLen', 'seqNum']])

        # 5. Run Ridge Regression and store results
        records = []
        alphas = np.logspace(-3, 3, 13)

        toSkip = set({})
        for key, value in names.items():
            if value["skip"]: toSkip.add(key)

        featureCols = [col for col in df.columns if col not in toSkip]
        for matrixType, groupByType in df.groupby('matrixType'):
            for nStates, group in groupByType.groupby('nStates'):
                X = group[featureCols].values

                for target in ('errorP', 'errorA', 'errorB'):
                    y = group[target].values
                    model = RidgeCV(alphas=alphas, store_cv_results=True)
                    model.fit(X, y)

                    for feat, coef in zip(featureCols, model.coef_):
                        records.append({
                            'matrixType': matrixType,
                            'nStates': nStates,
                            'target': target,
                            'feature': names[feat]["name"],
                            'coefficient': coef,
                            'alpha': model.alpha_,
                            'stdSeqLen': stdSeqLen,
                            'stdSeqNum': stdSeqNum
                        })

        ridgecvCoeffDf = pd.DataFrame(records)
        Plots.plotRegressionCoeff(ridgecvCoeffDf, names, isSub)

    @staticmethod
    def baumWelchPresetTesting() -> None:
        testType = ["Sparse"]
        genType = ["Normal"]
        seqLen = np.array([Testing.__seqLengths.max(), Testing.__seqLengths.min()], dtype=np.uint32)
        seqLen = np.append(seqLen, [np.uint32(np.average(seqLen))])

        seqNum = np.array([Testing.__seqNum.min(), np.median(Testing.__seqNum), Testing.__seqNum.max()], dtype=np.uint32)
        presets = [tuple([]), tuple(['A']), tuple(['B']), tuple(['P', 'A']), tuple(['P', 'B']), tuple(['A', 'B']),
                   tuple(['P', 'A', 'B'])]

        aux = Testing.__prepareTests([testType, genType, seqLen, seqNum, presets], "preset.txt")
        matrices, result, cartesian = aux["matrices"], aux["result"], aux["cartesian"]

        hmmModel = HMM()
        for name in cartesian:
            hmmParams = matrices[name]
            interResults = dict({
                "alg": dict({"P": np.empty(Testing.__presetExperiments), "A": np.empty(Testing.__presetExperiments),
                             "B": np.empty(Testing.__presetExperiments)}),
                "hmm": dict({"P": np.empty(Testing.__presetExperiments), "A": np.empty(Testing.__presetExperiments),
                             "B": np.empty(Testing.__presetExperiments)}),
            })
            for i in range(Testing.__presetExperiments):
                hmmModel.setParams(**hmmParams)
                nStates, nOutputs = hmmParams["logP"].shape[0], hmmParams["logB"].shape[1]
                expP, expA, expB = np.exp(hmmParams["logP"]), np.exp(hmmParams["logA"]), np.exp(hmmParams["logB"])
                subResult = Algorithms.baumWelch(hmmModel, name[1], name[2], name[3], useModelSeq=False,
                                                 preset=np.array(name[4]))

                interResults["alg"]["P"][i] = np.linalg.norm(expP - np.exp(subResult["logP"]))
                interResults["alg"]["A"][i] = np.linalg.norm(expA - np.exp(subResult["logA"]))
                interResults["alg"]["B"][i] = np.linalg.norm(expB - np.exp(subResult["logB"]))

                model = hmm.MultinomialHMM(n_components=nStates, n_iter=Algorithms.learnIterations, startprob_prior=1.0,
                                           transmat_prior=1.0,
                                           tol=1e-4, init_params='')
                model.startprob_ = subResult["initP"]
                model.transmat_ = subResult["initA"]
                model.emissionprob_ = subResult["initB"]

                sequences = np.concatenate(
                    [np.eye(nOutputs, dtype=np.uint32)[sequence["emissions"] - 1] for sequence in subResult["sequences"]])
                length = [name[2]] * name[3]
                model.fit(sequences, length)

                # Aligning the result
                estP, estA, estB = Algorithms.resolveLabelSwitching(np.exp(hmmParams["logB"]), model.emissionprob_,
                                                                    model.transmat_, model.startprob_)

                interResults["hmm"]["P"][i] = np.linalg.norm(expP - estP)
                interResults["hmm"]["A"][i] = np.linalg.norm(expA - estA)
                interResults["hmm"]["B"][i] = np.linalg.norm(expB - estB)

            result[name] = dict({
                key: {subkey: np.mean(arr) for subkey, arr in subdict.items()}
                for key, subdict in interResults.items()
            })
            Testing.__writeRunTests(name, result[name], "preset.txt")

        regTests = Testing.__readRunTests("preset.txt")

        Plots.showTestCasesPlot(Testing.__regroupTests(regTests), True)

        Testing.runRegression(Testing.__regroupPresetTests(result), dict({
            "P": {"skip": False, "name": "ground-truth P"},
            "A": {"skip": False, "name": "ground-truth A"},
            "B": {"skip": False, "name": "ground-truth B"},
        }), True)

    @staticmethod
    def __regroupPresetTests(results: Dict[Tuple, Dict[str, Dict[str, float]]]) -> Dict[str, Dict[int, Dict[Tuple, Dict[str, Dict[str, float]]]]]:
        regrouped = dict({})
        for name, value in results.items():
            first, *middle, matrices, last = name
            values = []
            matrices = np.array(matrices)
            for i in ['P', 'A', 'B']: values.append(0 if i not in matrices else 1)
            middle = [*middle, *values]

            regrouped.setdefault(first, {}).setdefault(last, {})[tuple(middle)] = value

        return regrouped

    @staticmethod
    def __writeRunTests(name: Tuple, values: Dict[str, Dict[str, float]], fileName: str = "tasks.txt") -> None:
        with open(fileName, 'a+') as file:
            file.write(str(name) + '\n')
            if "alg" in values:
                file.write(str(values["alg"]["P"]) + '\n')
                file.write(str(values["alg"]["A"]) + '\n')
                file.write(str(values["alg"]["B"]) + '\n')

            if "hmm" in values:
                file.write(str(values["hmm"]["P"]) + '\n')
                file.write(str(values["hmm"]["A"]) + '\n')
                file.write(str(values["hmm"]["B"]) + '\n')
