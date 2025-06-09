import os
import random
from typing import List, Optional, Dict, Union
import numpy as np
import re
import Helpers
import numpy.typing as npt
from typeguard import typechecked

@typechecked
class HMM:
    __distributions: List[str] = ['Random', "Gamma", "Normal"]

    def __init__(self):
        self.__curSequence: Optional[npt.NDArray[np.uint32]] = None
        self.__nStates: Optional[int] = None
        self.__nOutputs: Optional[int] = None
        self.__path: Optional[npt.NDArray[np.uint32]] = None
        self.__logP: Optional[npt.NDArray[np.float64]] = None
        self.__logA: Optional[npt.NDArray[np.float64]] = None
        self.__logB: Optional[npt.NDArray[np.float64]] = None

    @staticmethod
    def getDistributions() -> List[str]:
        return HMM.__distributions

    def generateRandom(self, states: int, outputs: int) -> None:
        self.__nStates = states
        self.__nOutputs = outputs

        self.__logP = np.random.dirichlet(alpha=np.full(states, 1.0))
        self.__logA = np.array([np.random.dirichlet(alpha=np.full(states, 1.0)) for _ in range(states)])
        self.__logB = np.array([np.random.dirichlet(alpha=np.full(outputs, 1.0)) for _ in range(states)])

        self.__convertToLog()

    def __convertToLog(self) -> None:
        self.__logP = np.log(self.__logP, where=(self.__logP > 0), out=np.full_like(self.__logP, -np.inf))
        self.__logA = np.log(self.__logA, where=(self.__logA > 0), out=np.full_like(self.__logA, -np.inf))
        self.__logB = np.log(self.__logB, where=(self.__logB > 0), out=np.full_like(self.__logB, -np.inf))

    def generateNormal(self, states: int, outputs: int) -> None:
        self.__nStates = states
        self.__nOutputs = outputs

        # Normal distribution within each row with applied softmax for getting a valid probability distribution
        def softmax(x: np.ndarray, axis: int = 0, keepdims: bool = True) -> np.ndarray:
            expX = np.exp(x - np.max(x, axis=axis, keepdims=keepdims))
            return expX / np.sum(expX, axis=axis, keepdims=keepdims)

        self.__logP = softmax(np.random.normal(loc=0, scale=1, size=self.__nStates))
        self.__logA = softmax(np.random.normal(loc=0, scale=1, size=(self.__nStates, self.__nStates)), 1)
        self.__logB = softmax(np.random.normal(loc=0, scale=1, size=(self.__nStates, self.__nOutputs)), 1)

        self.__convertToLog()

    def generateGamma(self, states: int, outputs: int, shape: float = 1.0, scale: float = 1.0) -> None:
        self.__nStates = states
        self.__nOutputs = outputs

        rawP = np.random.gamma(shape=shape, scale=scale, size=self.__nStates)
        self.__logP = rawP / np.sum(rawP)

        rawA = np.random.gamma(shape=shape, scale=scale, size=(self.__nStates, self.__nStates))
        self.__logA = rawA / np.sum(rawA, axis=1, keepdims=True)

        rawB = np.random.gamma(shape=shape, scale=scale, size=(self.__nStates, self.__nOutputs))
        self.__logB = rawB / np.sum(rawB, axis=1, keepdims=True)

        self.__convertToLog()

    def setParams(self, nStates: Optional[int] = None, nOutputs: Optional[int] = None,
                  logP: Optional[npt.NDArray[np.float64]] = None, logA: Optional[npt.NDArray[np.float64]] = None,
                  logB: Optional[npt.NDArray[np.float64]] = None) -> None:
        if nStates is not None: self.__nStates = nStates
        if nOutputs is not None: self.__nOutputs = nOutputs
        if logP is not None: self.__logP = logP
        if logA is not None: self.__logA = logA
        if logB is not None: self.__logB = logB

    def readMatrices(self, fileName: str = "matrices.txt", isReturn: bool = False) -> Optional[
        Dict[str, npt.NDArray[np.float64]]
    ]:
        if not os.path.exists(fileName):
            with open(fileName, 'w') as f: pass

        with open(fileName, 'r') as file:
            dim = [int(elem) for elem in re.sub("[ ]+", ' ', file.readline()).split()]
            self.__nStates, self.__nOutputs = dim[0], dim[1]

            self.__logP = np.zeros(self.__nStates)
            for i in range(self.__nStates):
                self.__logP[i] = eval(file.readline())

            self.__logA = np.zeros((self.__nStates, self.__nStates))
            for i in range(self.__nStates):
                for j, value in enumerate(re.sub("[ ]+", ' ', file.readline()).split()):
                    self.__logA[i][j] = eval(value)

            self.__logB = np.zeros((self.__nStates, self.__nOutputs))
            for i in range(self.__nStates):
                for j, value in enumerate(re.sub("[ ]+", ' ', file.readline()).split()):
                    self.__logB[i][j] = eval(value)

            self.__convertToLog()
            if isReturn:
                return dict({
                    "logP": self.__logP,
                    "logA": self.__logA,
                    "logB": self.__logB
                })

    def generateDimensions(self, sFrom: int = 2, sTo: int = 10, oFrom: int = 2, oTo: int = 10) -> None:
        self.__nStates = random.randint(sFrom, sTo)
        self.__nOutputs = random.randint(oFrom, oTo)

    def generateHMMParams(self, genType: str = "Random") -> None:
        if genType == 'Random':
            self.generateRandom(self.__nStates, self.__nOutputs)
        elif genType == 'Normal':
            self.generateNormal(self.__nStates, self.__nOutputs)
        elif genType == "Gamma":
            self.generateGamma(self.__nStates, self.__nOutputs)

    def generateModelSequence(self, seqLen: Union[int, np.uint32], isSave: bool = True) -> Optional[Dict[str, npt.NDArray[np.uint32]]]:
        P, A, B = np.exp(self.__logP), np.exp(self.__logA), np.exp(self.__logB)
        path = np.empty(seqLen, dtype=np.uint32)
        emissions = np.empty(seqLen, dtype=np.uint32)

        # While less efficient, multinomial is tolerant of floating-point errors which are going to happen during normalization and smoothing
        curState = np.argmax(np.random.multinomial(1, P))
        for i in range(seqLen):
            path[i] = curState
            emissions[i] = np.argmax(np.random.multinomial(1, B[curState, :]))

            curState = np.argmax(np.random.multinomial(1, A[curState, :]))

        if isSave:
            self.__curSequence = emissions + 1
            self.__path = path
        else:
            return dict({
                "emissions": emissions + 1,
                "path": path
            })

    def doSmoothing(self, params: Optional[List[str]] = None) -> None:
        if params is None:
            params = ["A", "B", "P"]

        if "P" in params:
            self.__logP, _, _ = Helpers.smoothing(P=self.__logP)
        if "A" in params:
            _, self.__logA, _ = Helpers.smoothing(A=self.__logA)
        if "B" in params:
            _, _, self.__logB = Helpers.smoothing(B=self.__logB)

    def setSequence(self, sequence: Dict[str, npt.NDArray[np.uint32]]) -> None:
        self.__curSequence = sequence["emissions"]
        self.__path = sequence["path"]

    def getNOutputs(self) -> int:
        return self.__nOutputs

    def getHMMParams(self) -> Dict[str, Union[
        npt.NDArray[np.float64],
        npt.NDArray[np.uint32],
        int,
    ]]:
        return dict({
            "logP": self.__logP,
            "logA": self.__logA,
            "logB": self.__logB,
            'seq': self.__curSequence,
            'path': self.__path,
            'states': self.__nStates,
            'outputs': self.__nOutputs
        })
