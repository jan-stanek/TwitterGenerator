package cz.fit.pdd.TwitterPredict;

import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.io.File;
import java.io.IOException;
import java.util.IllegalFormatException;
import java.util.Random;

public class Lstm {

    private MultiLayerNetwork net;
    private Random rng;

    public Lstm(File netFile) throws IOException {
        rng = new Random(System.currentTimeMillis());
        net = ModelSerializer.restoreMultiLayerNetwork(netFile);
    }

    public String generate() {
        String[] samples = sampleCharactersFromNetwork(140, 1);
        return samples[0];
    }

    private String[] sampleCharactersFromNetwork(int charactersToSample, int numSamples) {
            Character[] startCharacters = new Character[LstmBean.startCharactersSet.size()];
            int k = 0;
            for (Character c : LstmBean.startCharactersSet) {
                startCharacters[k++] = c;
            }
            int random = rng.nextInt(startCharacters.length);

            String initialization = String.valueOf(startCharacters[random]);
            INDArray initializationInput = Nd4j.zeros(numSamples, LstmBean.characterSet.size(), initialization.length());
            char[] init = initialization.toCharArray();

            for (int i = 0; i < init.length; i++) {
                int idx = LstmBean.charToIndexMap.get(init[i]);
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

            INDArray nextInput;
            for (int i = 0; i < charactersToSample; i++) {
                nextInput = Nd4j.zeros(numSamples, LstmBean.characterSet.size());

                for (int s = 0; s < numSamples; s++) {
                    double[] outputProbDistribution = new double[LstmBean.characterSet.size()];
                    for (int j = 0; j < outputProbDistribution.length; j++)
                        outputProbDistribution[j] = output.getDouble(s, j);
                    int sampledCharacterIdx = sampleFromDistribution(outputProbDistribution);

                    nextInput.putScalar(new int[]{s, sampledCharacterIdx}, 1f);
                    Character c = LstmBean.indexToCharArray[sampledCharacterIdx];
                    sb[s].append(LstmBean.indexToCharArray[sampledCharacterIdx]);
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

        double temperature = 0.1;

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