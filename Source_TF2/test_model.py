"""
Test Script - DQN Global Stock Market Prediction
Tests the trained model on all available test datasets
"""
import DataPPRL as DRL
import train as TR

# Parameters matching the paper
FSize = 5
PSize = 2
PStride = 2
NumAction = 3
W = 32

# Initialize
DRead = DRL.DataReaderRL()
Model = TR.trainModel(1.0, 0.1, 5000000, 32, 10, 1000, 0.00001, 0)

# Get test folders
folderlist = DRead.get_filelist('../Sample_Testing/')
sess, saver, state, isTrain, rho_eta = Model.TestModel_ConstructGraph(W, W, FSize, PSize, PStride, NumAction)

print("\n" + "="*70)
print("Testing on {} datasets...".format(len(folderlist)))
print("="*70 + "\n")

for i in range(len(folderlist)):
    print("="*70)
    print("Testing: {}".format(folderlist[i]))
    print("="*70)
    
    filepathX = folderlist[i] + 'inputX.txt'
    filepathY = folderlist[i] + 'inputY.txt'
    
    XData = DRead.readRaw_generate_X(filepathX, W, W)
    YData = DRead.readRaw_generate_Y(filepathY, len(XData), len(XData[0]))
    
    Model.set_Data(XData, YData)
    
    print("\nNeutralized Portfolio Strategy:")
    Model.Test_Neutralized_Portfolio(sess, saver, state, isTrain, rho_eta, W, W, NumAction)
    
    print("\nTop/Bottom 20% Portfolio Strategy:")
    Model.Test_TopBottomK_Portfolio(sess, saver, state, isTrain, rho_eta, W, W, NumAction, 0.2)
    print()

print("\n" + "="*70)
print("Testing Complete! Results saved to TestResult.txt")
print("="*70)

