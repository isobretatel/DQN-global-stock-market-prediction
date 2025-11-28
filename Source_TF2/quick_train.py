
import DataPPRL as DRL
import train as TR

# Quick training parameters
maxiter = 100000  # Reduced for demo (paper uses 5M)
print(f"Training for {maxiter:,} iterations...")

DRead = DRL.DataReaderRL()
Model = TR.trainModel(1.0, 0.1, maxiter, 32, 10, 1000, 0.00001, 0)

filepathX = '../Sample_Training/WH32_32_2017_2018/inputX.txt'
filepathY = '../Sample_Training/WH32_32_2017_2018/inputY.txt'

XData = DRead.readRaw_generate_X(filepathX, 32, 32)
YData = DRead.readRaw_generate_Y(filepathY, len(XData), len(XData[0]))

Model.set_Data(XData, YData)
Model.trainModel(32, 32, 5, 2, 2, 3, 1000, 0.99)
