import copy
from typing import Tuple, Union, Dict, Optional, List
import numpy as np
import numpy.typing as npt
from hmmlearn import hmm
from scipy.special import logsumexp
import os
from scipy.optimize import linear_sum_assignment
from HMM import HMM
from joblib import Parallel, delayed
from numba import njit
import Helpers
from typeguard import typechecked

# typeguard will not work here because of Numba, so the types here are primarily for manual control
@njit
def forwardNumba(nStates: int, seq: npt.NDArray[np.uint32], logP: npt.NDArray[np.float64],
                 logA: npt.NDArray[np.float64], logB: npt.NDArray[np.float64]) -> Tuple[npt.NDArray[np.float64], float]:
    T = seq.shape[0]
    forwardMatrix = np.full((nStates, T), -np.inf, dtype=np.float64)
    # Initialization of the first observation
    forwardMatrix[:, 0] = logP + logB[:, seq[0]]

    for t in range(1, T):
        emissionInd = seq[t]
        prevColumn = forwardMatrix[:, t - 1]

        # Computing prevColumn[:, None] + logA in a way that can be optimized
        temp = np.empty((nStates, nStates))
        for i in range(nStates):
            for j in range(nStates):
                temp[i, j] = prevColumn[i] + logA[i, j]

        # Computing logsumexp over the rows for each column
        tempLogSum = Helpers.logsumexp2dNumba(temp, 0)

        # Add emission log-probabilities for current observation
        for j in range(nStates):
            forwardMatrix[j, t] = tempLogSum[j] + logB[j, emissionInd]

    # Final probability is logsumexp over the last column:
    prob = Helpers.logsumexp1dNumba(forwardMatrix[:, T - 1])
    return forwardMatrix, prob

# typeguard will not work here because of Numba, so the types here are primarily for manual control
@njit
def backwardNumba(nStates: int, seq: npt.NDArray[np.uint32], logP: npt.NDArray[np.float64],
                  logA: npt.NDArray[np.float64], logB: npt.NDArray[np.float64]) -> Tuple[
    npt.NDArray[np.float64], float]:
    T = seq.shape[0]
    backwardMatrix = np.full((nStates, T), -np.inf, dtype=np.float64)
    # Final time step initialization
    backwardMatrix[:, -1] = 0

    # Recursion:
    for t in range(T - 2, -1, -1):
        emissionInd = seq[t + 1]
        nextColumn = backwardMatrix[:, t + 1] + logB[:, emissionInd]

        # Computing nextColumn[None, :] + logA in a way that can be optimized
        temp = np.empty((nStates, nStates))
        for i in range(nStates):
            for j in range(nStates):
                temp[i, j] = nextColumn[j] + logA[i, j]

        # Compute logsumexp over the columns for each row
        backwardMatrix[:, t] = Helpers.logsumexp2dNumba(temp, 1)

    # Final probability
    prob = Helpers.logsumexp1dNumba(logP + logB[:, seq[0]] + backwardMatrix[:, 0])
    return backwardMatrix, prob

@typechecked
class Algorithms:
    learnIterations = 100
    __epsilon = 1e-4

    def __init__(self):
        pass

    @staticmethod
    def forward(hmmModel: HMM) -> Dict[str, Union[npt.NDArray[np.float64], float]]:
        data = hmmModel.getHMMParams()
        nStates, seq, logP, logA, logB = data["states"], data["seq"] - 1, data["logP"], data["logA"], data["logB"]

        fwdMatrix, prob = forwardNumba(nStates, seq, logP, logA, logB)
        return {
            "matrix": fwdMatrix,
            "prob": prob,
        }

    @staticmethod
    def forwardHmm(hmmModel: HMM) -> float:
        data = hmmModel.getHMMParams()

        model = hmm.MultinomialHMM(n_components=data["states"], init_params='')
        model.startprob_ = np.exp(data["logP"])
        model.transmat_ = np.exp(data["logA"])
        model.emissionprob_ = np.exp(data["logB"])
        sequence = np.eye(data["outputs"], dtype=np.uint32)[data["seq"] - 1]
        model.n_trials = 1

        return model.score(sequence)

    @staticmethod
    def backward(hmmModel: HMM) -> Dict[str, Union[npt.NDArray[np.float64], float]]:
        data = hmmModel.getHMMParams()
        nStates, seq, logP, logA, logB = data["states"], data["seq"] - 1, data["logP"], data["logA"], data["logB"]

        bwdMatrix, prob = backwardNumba(nStates, seq, logP, logA, logB)
        return {"matrix": bwdMatrix, "prob": prob}

    @staticmethod
    def getBaumWelchParams(hmmModel: HMM, sequence: Dict[str, npt.NDArray[np.uint32]]) -> Dict[
        str, npt.NDArray[np.float64]]:
        hmmModel.setSequence(sequence)
        logAlpha = Algorithms.forward(hmmModel)["matrix"]
        logBeta = Algorithms.backward(hmmModel)["matrix"]

        data = hmmModel.getHMMParams()
        T, nStates, seq, logP, logA, logB = data["seq"].size, data["states"], data["seq"] - 1, data["logP"], data[
            "logA"], data["logB"]

        logGamma: npt.NDArray[np.float64] = logAlpha + logBeta
        logGamma -= logsumexp(logGamma, axis=0, keepdims=True)

        # Get emission indices for all the time periods but the last one + turning into zero-indexing
        emissionIndices = seq[1:]

        # Calculating emission and beta part of the numerator
        betaB = logB[:, emissionIndices] + logBeta[:, 1:]

        # Ksi has tree dimensions - (curState, nextState, time)
        # In order to calculate the numerator, we need to reshape each of terms (log space)
        # logA matrix determines the "relationship" between the curState and nextState, hence its dimensions are (states, states, 1)
        # logAlpha is defined for the curState and time period, hence its dimensions are (states, 1, T-1)
        # betaB is defined for the nextState and time period, hence it has (1, states, T-1) dimensions
        logKsi: npt.NDArray[np.float64] = (
                logAlpha[:, :T - 1].reshape(nStates, 1, T - 1) +
                logA[:, :, None] +
                betaB.reshape(1, nStates, T - 1)
        )

        # Normalize each time slice so that probabilities sum to 1 in log space
        logKsi -= logsumexp(logKsi, axis=(0, 1), keepdims=True)

        return dict({
            "ksi": logKsi,
            "gamma": logGamma
        })

    @staticmethod
    def viterbiPlain(hmmModel: HMM) -> Dict[str, Union[npt.NDArray[np.uint32], float]]:
        data = hmmModel.getHMMParams()
        viterbiMatrix = np.zeros((data["states"], data["seq"].size))
        backpointers = np.zeros((data["states"], data["seq"].size), dtype=np.uint32)

        emissionInd = data['seq'][0] - 1
        viterbiMatrix[:, 0] = data["logP"] + data["logB"][:, emissionInd]
        backpointers[:, 0] = 0

        for time in range(1, data["seq"].size):
            emissionInd = data['seq'][time] - 1
            # Compute a temporary matrix of shape (nStates, nStates) where each element is:
            # temp[i, j] = viterbiMatrix[i, t-1] + data["logA"][i, j]
            temp = viterbiMatrix[:, time - 1][:, None] + data["logA"]
            viterbiMatrix[:, time] = np.max(temp, axis=0) + data["logB"][:, emissionInd]
            # Record the best previous state for each current state
            backpointers[:, time] = np.argmax(temp, axis=0)

        states = np.empty(data["seq"].size, dtype=np.uint32)
        states[-1] = np.argmax(viterbiMatrix[:, -1])
        for time in range(data["seq"].size - 1, 0, -1):
            states[time - 1] = backpointers[states[time], time]

        hammingAlg = np.sum(states != data["path"])
        return dict({
            "estimatedAlg": states + 1,
            "actual": data["path"] + 1,
            "hammingAlg": hammingAlg,
            "errorRateAlg": np.round(hammingAlg / data["seq"].size, 3),
        })

    @staticmethod
    def viterbiHmm(hmmModel: HMM) -> Dict[str, Union[npt.NDArray[np.uint32], float]]:
        data = hmmModel.getHMMParams()

        model = hmm.MultinomialHMM(n_components=data["states"], init_params='')
        model.startprob_ = np.exp(data["logP"])
        model.transmat_ = np.exp(data["logA"])
        model.emissionprob_ = np.exp(data["logB"])

        # One-hot code each emission
        sequence = np.eye(data["outputs"], dtype=np.uint32)[data["seq"] - 1]
        model.n_trials = 1
        prediction = model.predict(sequence)

        hammingHmm = np.sum(prediction != data["path"])
        return dict({
            "estimatedHmm": prediction + 1,
            "hammingHmm": hammingHmm,
            "errorRateHmm": np.round(hammingHmm / sequence.shape[0], 3),
        })

    @staticmethod
    def baumWelch(hmmModel: HMM, genType: str = "Normal", seqLen: Union[int, np.uint32] = int(1e4), seqNum: Union[int, np.uint32] = 10,
                  useModelSeq: bool = True, preset: Optional[Union[List[str], npt.NDArray[str]]] = None) -> Dict[
        str, Union[npt.NDArray[np.uint32], npt.NDArray[np.float64]]
    ]:
        if preset is None:
            preset = []
        data = hmmModel.getHMMParams()

        # initialize a training model
        trainModel = HMM()
        trainModel.generateDimensions(data['states'], data['states'], data['outputs'], data['outputs'])
        trainModel.generateHMMParams(genType)
        # if there are preset matrices, use them while leaving the remaining ones unchange from the previous step
        if len(preset) != 0: trainModel.setParams(**{f"log{key}": data[f"log{key}"] for key in preset})

        # If new sequences are needed to be generated instead of using one sequence set in the model
        if not useModelSeq:
            sequences = [hmmModel.generateModelSequence(seqLen, isSave=False) for _ in range(seqNum)]
        else:
            sequences = [{'emissions': data['seq'], 'path': data['path']}]
            seqNum = 1

        # Store initial params for calculating distance between corresponding matrices
        trainModel.doSmoothing()
        trainParams = trainModel.getHMMParams()
        initA = np.exp(copy.deepcopy(trainParams['logA']))
        initB = np.exp(copy.deepcopy(trainParams['logB']))
        initP = np.exp(copy.deepcopy(trainParams['logP']))

        prob, newParams = [], {}
        while len(prob) < 2 or (
                len(prob) < Algorithms.learnIterations and abs(prob[-1] - prob[-2]) > Algorithms.__epsilon):
            trainModel.setParams(**newParams)

            # Analyze sequences in parallel
            results = Parallel(n_jobs=min(seqNum, os.cpu_count() - 1))(
                delayed(Algorithms.getBaumWelchParams)(trainModel, seq) for seq in sequences
            )

            # Aggregate ksi and gamma across all sequences for each state at each point in time
            ksiTotal = logsumexp([r['ksi'] for r in results], axis=0)
            gammaTotal = logsumexp(np.stack([r['gamma'] for r in results], axis=0), axis=0)
            ksiTotal -= logsumexp(ksiTotal, axis=(0, 1), keepdims=True)
            gammaTotal -= logsumexp(gammaTotal, axis=0, keepdims=True)

            # New logP matrix is equal to the first column of gamma as it represents the normalized number of times (a probability)
            # of being at state X at time 0
            newParams["logP"] = gammaTotal[:, 0]

            # We aggregate the normalized number of transitions from the curState over all periods of time but last (we do not transition from the last state)
            # Gamma for the state i at time t is equal to the sum of all transitions from the state i to any other state at time t (ksi),
            # so we can avoid summing the number of transitions
            transFromCurState = logsumexp(gammaTotal[:, :-1], axis=1)
            newParams["logA"] = np.where(transFromCurState[:, None] > -np.inf,
                                         logsumexp(ksiTotal, axis=2) - transFromCurState[:, None],
                                         -np.inf)

            # Aggregate all the gammas for the curState across all time periods and all sequences
            transFromState = logsumexp(np.hstack([res["gamma"] for res in results]), axis=1)
            newParams["logB"] = np.full((data["states"], data["outputs"]), -np.inf)
            seqIndices = [
                {output: np.where(seq["emissions"] == output + 1)[0] for output in range(data["outputs"])}
                for seq in sequences
            ]
            for curState in range(data["states"]):
                for output in range(data["outputs"]):
                    if transFromCurState[curState] > -np.inf:
                        numerator = logsumexp(np.hstack([
                            results[i]["gamma"][curState, seqIndices[i][output]]
                            for i in range(seqNum)
                        ]))
                        newParams["logB"][curState][output] = numerator - transFromState[curState]
                    else:
                        newParams["logB"][curState][output] = -np.inf

            newLogP, newLogA, newLogB = Helpers.smoothing(newParams["logP"], newParams["logA"], newParams["logB"])
            newParams = {'logA': newLogA, 'logB': newLogB, 'logP': newLogP}

            trainModel.setParams(**newParams)
            logLikelihoods = Parallel(n_jobs=min(seqNum, os.cpu_count() - 1))(
                delayed(Algorithms.__computeLogLikelihood)(trainModel, seq) for seq in sequences
            )
            prob.append(np.average(logLikelihoods))

            print(f"Iteration {len(prob)}")

        # Perform label switching
        P, A, B = Algorithms.resolveLabelSwitching(
            np.exp(data['logB']), np.exp(newParams['logB']),
            np.exp(newParams['logA']), np.exp(newParams['logP'])
        )
        newParams['logP'] = np.log(P, where=(P > 0), out=np.full_like(P, -np.inf))
        newParams['logA'] = np.log(A, where=(A > 0), out=np.full_like(A, -np.inf))
        newParams['logB'] = np.log(B, where=(B > 0), out=np.full_like(B, -np.inf))
        return {**newParams, 'sequences': sequences, 'initA': initA, 'initB': initB, 'initP': initP}

    @staticmethod
    def __computeLogLikelihood(model: HMM, sequence: Dict[str, npt.NDArray[np.uint32]]) -> float:
        model.setSequence(sequence)
        return Algorithms.forward(model)["prob"]

    @staticmethod
    def resolveLabelSwitching(B: npt.NDArray[np.float64], estB: npt.NDArray[np.float64], estA: npt.NDArray[np.float64],
                              estP: npt.NDArray[np.float64]) -> Tuple[
        npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        # Number of states
        states = B.shape[0]

        # Build cost matrix using L1 norm between rows of emission matrices
        costMat = np.zeros((states, states))
        for i in range(states):
            for j in range(states):
                costMat[i, j] = np.linalg.norm(B[i] - estB[j], ord=1)

        # Solve assignment problem (Hungarian algorithm)
        _, colInd = linear_sum_assignment(costMat)

        # Permute estimated parameters
        alignedB = estB[colInd, :]
        alignedA = estA[np.ix_(colInd, colInd)]
        alignedP = estP[colInd]

        return alignedP, alignedA, alignedB
