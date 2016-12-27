import org.deeplearning4j.api.storage.StatsStorage;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.*;
import org.deeplearning4j.nn.conf.layers.GravesLSTM;
import org.deeplearning4j.nn.conf.layers.RnnOutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.ui.api.UIServer;
import org.deeplearning4j.ui.stats.StatsListener;
import org.deeplearning4j.ui.storage.InMemoryStatsStorage;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction;

import java.io.File;
import java.io.IOException;
import java.util.Random;

public class Lstm {
    private CharIterator charIterator;
    private MultiLayerNetwork net;
    private Random rng;

    public Lstm(CharIterator charIterator) {
        this.charIterator = charIterator;

        int nIn = charIterator.inputColumns();
        int nOut = charIterator.totalOutcomes();

        rng = new Random(12345);

//        DataTypeUtil.setDTypeForContext(DataBuffer.Type.HALF);
//        CudaEnvironment.getInstance().getConfiguration()
//                .setMaximumDeviceCacheableLength(1024*1024*1024L)
//                .setMaximumDeviceCache(8L*1024*1024*1024L)
//                .setMaximumHostCacheableLength(1024*1024*1024L)
//                .setMaximumHostCache(8L*1024*1024*1024L);

        UIServer uiServer = UIServer.getInstance();
        StatsStorage statsStorage = new InMemoryStatsStorage();
        uiServer.attach(statsStorage);

        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT).iterations(1)
                .learningRate(0.1)
                .rmsDecay(0.95)
                .seed(12345)
                .regularization(true)
                .l2(0.001)
                .weightInit(WeightInit.XAVIER)
                .updater(Updater.RMSPROP)
                .list()
                .layer(0, new GravesLSTM.Builder()
                        .nIn(nIn).nOut(LstmTrain.LSTM_LAYER_SIZE)
                        .activation("tanh")
                        .build())
                .layer(1, new GravesLSTM.Builder()
                        .nIn(LstmTrain.LSTM_LAYER_SIZE).nOut(LstmTrain.LSTM_LAYER_SIZE)
                        .activation("tanh")
                        .build())
                .layer(2, new RnnOutputLayer.Builder(LossFunction.MCXENT)
                        .nIn(LstmTrain.LSTM_LAYER_SIZE).nOut(nOut)
                        .activation("softmax")
                        .build())
                .backpropType(BackpropType.TruncatedBPTT).tBPTTForwardLength(LstmTrain.T_BP_T_T_LENGTH).tBPTTBackwardLength(LstmTrain.T_BP_T_T_LENGTH)
                .pretrain(false).backprop(true)
                .build();

        net = new MultiLayerNetwork(conf);

        net.init();
        net.setListeners(new StatsListener(statsStorage));
    }

    public void train(File netDir) throws IOException {
        for (int i = 1; i <= LstmTrain.EPOCHS_COUNT; i++) {
            while (charIterator.hasNext()) {
                net.fit(charIterator.next());
            }

            System.out.println("Epoch " + i + ":");
            for (int j = 0; j < 10; j++) {
                String sample[] = sampleCharactersFromNetwork(LstmTrain.SAMPLES_LENGTH, LstmTrain.SAMPLES_COUNT);
                System.out.println(sample[0]);
            }
            System.out.println();

            save(netDir, i);

            charIterator.reset();
        }
    }

    public void save(File file, int epoch) throws IOException {
        ModelSerializer.writeModel(net, new File(file.getPath() + "/net_" + epoch + ".zip"), false);
    }

    private String[] sampleCharactersFromNetwork(int charactersToSample, int numSamples) {
        String initialization = String.valueOf(charIterator.getRandomCharacter());
        INDArray initializationInput = Nd4j.zeros(numSamples, charIterator.inputColumns(), initialization.length());
        char[] init = initialization.toCharArray();

        for (int i = 0; i < init.length; i++) {
            int idx = charIterator.convertCharacterToIndex(init[i]);
            for (int j = 0; j < numSamples; j++) {
                initializationInput.putScalar(new int[]{j, idx, i}, 1f);
            }
        }

        StringBuilder[] sb = new StringBuilder[numSamples];
        for (int i = 0; i < numSamples; i++)
            sb[i] = new StringBuilder(initialization);

        net.rnnClearPreviousState();
        INDArray output = net.rnnTimeStep(initializationInput);
        output = output.tensorAlongDimension(output.size(2) - 1, 1, 0);

        for (int i = 0; i < charactersToSample; i++) {
            INDArray nextInput = Nd4j.zeros(numSamples, charIterator.inputColumns());

            for (int s = 0; s < numSamples; s++) {
                double[] outputProbDistribution = new double[charIterator.totalOutcomes()];
                for (int j = 0; j < outputProbDistribution.length; j++)
                    outputProbDistribution[j] = output.getDouble(s, j);
                int sampledCharacterIdx = sampleFromDistribution(outputProbDistribution);

                nextInput.putScalar(new int[]{s, sampledCharacterIdx}, 1f);
                sb[s].append(charIterator.convertIndexToCharacter(sampledCharacterIdx));
            }

            output = net.rnnTimeStep(nextInput);    //Do one time step of forward pass
        }

        String[] out = new String[numSamples];
        for (int i = 0; i < numSamples; i++) out[i] = sb[i].toString();
        return out;
    }

    public int sampleFromDistribution(double[] distribution) {
        double d = rng.nextDouble();
        double sum = 0.0;

        double temperature = 0.2;

        for (int i = 0; i < distribution.length; i++) {
            distribution[i] = Math.exp(Math.log(distribution[i])/temperature);
            sum += distribution[i];
        }

        for (int i = 0; i < distribution.length; i++) {
            distribution[i] = distribution[i]/sum;
        }

        sum = 0.0;

        for (int i = 0; i < distribution.length; i++) {
            sum += distribution[i];
            if (d <= sum) return i;
        }

        throw new IllegalArgumentException("Distribution is invalid.");
    }
}