import sys
import random
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout,
    QPushButton, QLabel, QStackedWidget, QSizePolicy, QScrollArea,
    QInputDialog, QMessageBox
)
from PySide6.QtCore import Qt, QThreadPool, QRunnable, QObject, Signal, QTimer
from Algorithms import Algorithms
from HMM import HMM
from Testing import Testing
import numpy as np

np.set_printoptions(precision=5, suppress=True)

class Manager(QMainWindow):
    class LoaderOverlay(QWidget):
        def __init__(self, parent=None):
            super().__init__(parent)
            self.setStyleSheet("background-color: rgba(0, 0, 0, 180);")
            self.setAttribute(Qt.WA_TransparentForMouseEvents, False)
            self.setVisible(False)

            layout = QVBoxLayout(self)
            layout.setAlignment(Qt.AlignCenter)

            self.spinner = QLabel("Processing...", self)
            self.spinner.setStyleSheet("color: white; font-size: 18px; font-weight: bold;")
            layout.addWidget(self.spinner)

    class WorkerSignals(QObject):
        finished = Signal(object)

    class BackgroundTask(QRunnable):
        def __init__(self, fn, *args, **kwargs):
            super().__init__()
            self.fn = fn
            self.args = args
            self.kwargs = kwargs
            self.signals = Manager.WorkerSignals()

        def run(self):
            try:
                result = self.fn(*self.args, **self.kwargs)
                # mark successful
                self.signals.finished.emit((True, result))
            except Exception as e:
                # propagate exception
                self.signals.finished.emit((False, e))

    def __init__(self):
        super().__init__()
        self.__currentTask = None
        self.setWindowTitle("HMM manager")
        self.setFixedSize(500, 400)

        self.hmmModel = HMM()
        self.paramsReady = False
        self.sequenceReady = False

        self.threadPool = QThreadPool()

        self.stack = QStackedWidget()
        self.setCentralWidget(self.stack)

        self.loader = Manager.LoaderOverlay(self)
        self.loader.setGeometry(self.rect())
        self.loader.raise_()
        self.__backgroundTasks = []

        self.buildTopMenu()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.loader.setGeometry(self.rect())

    def showLoader(self):
        QTimer.singleShot(0, lambda: self.loader.setVisible(True))

    def hideLoader(self):
        QTimer.singleShot(0, lambda: self.loader.setVisible(False))

    def runInBackground(self, fn, callback):
        self.showLoader()
        task = Manager.BackgroundTask(fn)
        self.__backgroundTasks.append(task)  # Store reference

        def handle_result(result):
            self.__onTaskFinished(result, callback)
            if task in self.__backgroundTasks:
                self.__backgroundTasks.remove(task)

        task.signals.finished.connect(handle_result)
        self.threadPool.start(task)

    def __onTaskFinished(self, successAndPayload, callback):
        self.hideLoader()
        success, payload = successAndPayload
        if not success:
            # payload is the Exception instance
            dlg = QMessageBox(self)
            dlg.setWindowTitle("Error")
            dlg.setIcon(QMessageBox.Critical)
            dlg.setText("An error occurred:")
            dlg.setInformativeText(str(payload))
            dlg.exec()
            return

        # otherwise payload is the normal result
        try:
            callback(payload)
        except Exception as e:
            # if your callback itself throws
            dlg = QMessageBox(self)
            dlg.setWindowTitle("Error")
            dlg.setIcon(QMessageBox.Critical)
            dlg.setText("Error while processing result:")
            dlg.setInformativeText(str(e))
            dlg.exec()

    def clearStack(self):
        while self.stack.count():
            widget = self.stack.widget(0)
            self.stack.removeWidget(widget)
            widget.deleteLater()

    def buildTopMenu(self):
        self.clearStack()
        menu = {
            "Baum‑Welch testing": self.buildBaumWelchMenu,
            "Proceed to HMM tasks": self.buildMenu
        }
        self.createMenuPage(menu, "Main Menu")

    def buildBaumWelchMenu(self):
        self.clearStack()
        menu = {
            "Run Baum‑Welch features testing": self.baumWelchTesting,
            "Run Baum-Welch ground-truth initialization testing": self.baumWelchPresetTesting,
            "Back": self.buildTopMenu
        }
        self.createMenuPage(menu, "Baum‑Welch Menu")

    def buildMenu(self):
        self.clearStack()
        if not self.paramsReady:
            menu = {
                "Load from file": self.loadMatrixFile,
                "Specify the number of states and emissions": self.specifyDimsParams,
                "Random generation in [2, 10] range": self.__params,
            }
            title = "Generate HMM parameters"
        elif not self.sequenceReady:
            menu = {
                "Specify sequence length": self.specifySequence,
                "Random generation in [1, 1e6] range": self.randomSequence,
            }
            title = "Generate sequence"
        else:
            menu = {
                "Change the sequence": self.changeSequence,
                "Compute likelihood of the sequence": self.computeLikelihood,
                "Decode sequence (the most likely sequence of states)": self.decodeSequence,
                "Learn HMM parameters based on the sequence": self.learnParams,
                "Exit": self.exitApp,
            }
            title = "Actions"

        self.createMenuPage(menu, title)

    def createMenuPage(self, menu, title):
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setAlignment(Qt.AlignTop)

        titleLabel = QLabel(title)
        titleLabel.setAlignment(Qt.AlignCenter)
        titleLabel.setStyleSheet("font-size: 18px; font-weight: bold; margin: 10px;")
        layout.addWidget(titleLabel)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        content = QWidget()
        scrollLayout = QVBoxLayout(content)
        scrollLayout.setAlignment(Qt.AlignTop)
        scroll.setWidget(content)

        for label, func in menu.items():
            btn = QPushButton(label)
            btn.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            btn.clicked.connect(func)

            scrollLayout.addWidget(btn)

        layout.addWidget(scroll)
        self.stack.addWidget(page)
        self.stack.setCurrentWidget(page)

    def loadMatrixFile(self):
        self.hmmModel.readMatrices()
        data = self.hmmModel.getHMMParams()

        self.paramsReady = True
        self.showMessage(
            "Parameters has been loaded from file\n\n"
            f"Matrix P:\n{np.exp(data["logP"])}\n\n"
            f"Matrix A:\n{np.exp(data["logA"])}\n\n"
            f"Matrix B:\n{np.exp(data["logB"])}\n\n"
        )
        self.buildMenu()

    def specifyDimsParams(self):
        states, ok1 = QInputDialog.getInt(self, "States", "Number of states:", 2, 1, 10)
        if not ok1:
            return
        outputs, ok2 = QInputDialog.getInt(self, "Outputs", "Number of outputs:", 2, 1, 10)
        if not ok2:
            return

        self.__params(states, outputs)

    def __params(self, states=None, outputs=None):
        if states is None or outputs is None:
            self.hmmModel.generateDimensions()
        else:
            self.hmmModel.generateDimensions(states, states, outputs, outputs)

        self.hmmModel.generateHMMParams()
        self.paramsReady = True

        data = self.hmmModel.getHMMParams()
        self.showMessage(
            f"Random parameters has been generated. The model has {data["states"]} states and {data["outputs"]} outputs.\n"
            f"Matrix P:\n{np.exp(data["logP"])}\n\n"
            f"Matrix A:\n{np.exp(data["logA"])}\n\n"
            f"Matrix B:\n{np.exp(data["logB"])}\n\n"
        )

        self.buildMenu()

    def specifySequence(self):
        length, ok = QInputDialog.getInt(
            self, "Sequence Length", "Enter sequence length:", 100, 1, int(1e6)
        )
        if not ok:
            return

        def task():
            self.hmmModel.generateModelSequence(length)
            return self.hmmModel.getHMMParams()['seq']

        def callback(seq):
            self.sequenceReady = True
            self.showMessage(f"Sequence generated (length={len(seq)}):\n{seq}")
            self.buildMenu()

        self.runInBackground(task, callback)

    def randomSequence(self):
        length = random.randint(1, int(1e6))

        def task():
            self.hmmModel.generateModelSequence(length)
            return self.hmmModel.getHMMParams()['seq']

        def callback(seq):
            self.sequenceReady = True
            self.showMessage(f"Sequence generated (length={len(seq)}):\n{seq}")
            self.buildMenu()

        self.runInBackground(task, callback)

    def _onSequenceGenerated(self, seq):
        self.sequenceReady = True

        self.showMessage(
            f"Sequence generated (length={len(seq)}):\n{seq}"
        )
        self.buildMenu()

    def changeSequence(self):
        self.sequenceReady = False
        self.buildMenu()

    def computeLikelihood(self):
        def task():
            result = Algorithms.forward(self.hmmModel)
            result["probHmm"] = Algorithms.forwardHmm(self.hmmModel)
            return result

        def callback(result):
            seq = self.hmmModel.getHMMParams()['seq']
            msg = (
                f"\nLog-likelihood of the sequence (custom implementation)\n{seq}\nis {result['prob']}\n"
                f"\nLog-likelihood of the sequence (hmmlearn)\n{seq}\nis {result['probHmm']}\n"
            )
            self.showMessage(msg)

        self.runInBackground(task, callback)

    def decodeSequence(self):
        def task():
            result = Algorithms.viterbiPlain(self.hmmModel)
            result.update(**Algorithms.viterbiHmm(self.hmmModel))

            return result

        def callback(values):
            seq = self.hmmModel.getHMMParams()['seq']
            msg = (
                f"Emissions:\n{seq}\n"
                f"The most likely sequence of states (custom implementation) is\n{values['estimatedAlg']}\n"
                f"The most likely sequence of states (hmmlearn) is\n{values['estimatedHmm']}\n"
                f"The actual sequence of states is\n{values['actual']}\n"
                f"Hamming distance (custom implementation):\n{values['hammingAlg']}\n"
                f"Hamming distance (hmmlearn):\n{values['hammingHmm']}\n"
                f"Error rate (custom implementation):\n{values['errorRateAlg']}\n"
                f"Error rate (hmmlearn):\n{values['errorRateHmm']}\n"
            )

            QTimer.singleShot(0, lambda: self.showMessage(msg))

        self.runInBackground(task, callback)

    def learnParams(self):
        def task():
            return Algorithms.baumWelch(self.hmmModel)

        def callback(results):
            data = self.hmmModel.getHMMParams()
            msg = (
                f"Learned P:\n{np.exp(results['logP'])}\n"
                f"Actual P:\n{np.exp(data['logP'])}\n"
                f"Frobenius norm of P:\n{np.linalg.norm(np.exp(data['logP']) - np.exp(results['logP']))}\n\n"
                f"Learned A:\n{np.exp(results['logA'])}\n"
                f"Actual A:\n{np.exp(data['logA'])}\n"
                f"Frobenius norm of A:\n{np.linalg.norm(np.exp(data['logA']) - np.exp(results['logA']))}\n\n"
                f"Learned B:\n{np.exp(results['logB'])}\n"
                f"Actual B:\n{np.exp(data['logB'])}\n"
                f"Frobenius norm of B:\n{np.linalg.norm(np.exp(data['logB']) - np.exp(results['logB']))}\n\n"
            )
            self.showMessage(msg)

        self.runInBackground(task, callback)

    def baumWelchTesting(self):
        self.runInBackground(Testing.baumWelchTesting, lambda _: self.showMessage("Feature testing has been completed."))

    def baumWelchPresetTesting(self):
        self.runInBackground(Testing.baumWelchPresetTesting, lambda _: self.showMessage("Ground-truth initialization testing has been completed."))

    def exitApp(self):
        QApplication.instance().quit()

    def showMessage(self, text):
        dlg = QMessageBox(self)
        dlg.setWindowTitle("Result")
        dlg.setText(text)
        dlg.exec()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyleSheet("""
        QPushButton { background-color: #007ACC; color: white; padding: 8px; border-radius: 4px; font-size:14px; }
        QPushButton:hover { background-color: #005F99; }
        QLabel { font-size:13px; }
    """)
    window = Manager()
    window.show()
    sys.exit(app.exec())