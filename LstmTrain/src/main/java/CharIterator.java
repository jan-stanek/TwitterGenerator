import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.DataSetPreProcessor;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;

import java.io.IOException;
import java.util.*;

public class CharIterator implements DataSetIterator {
    private Random rng;

    private LinkedList<Integer> examplesStarts;
    private char[] characters;
    private int charsetSize;
    private Map<Character, Integer> charToIndexMap;
    private Character[] indexToCharArray;

    public CharIterator(String data, int charsetSize, Map<Character, Integer> charToIndexMap, Character[] indexToCharArray) throws IOException {
        this.rng = new Random();

        String[] dataLines = data.split("\0");

        char[][] dataLinesChar = new char[dataLines.length][LstmTrain.EXAMPLES_LENGTH];

        for (int i = 0; i < dataLines.length; i++) {
            char[] line = dataLines[i].toCharArray();
            for (int j = 0; j < line.length; j++) {
                dataLinesChar[i][j] = line[j];
            }
        }

        characters = new char[dataLines.length * LstmTrain.EXAMPLES_LENGTH];

        for (int i = 0; i < dataLines.length; i++) {
            for (int j = 0; j < LstmTrain.EXAMPLES_LENGTH; j++) {
                characters[i * LstmTrain.EXAMPLES_LENGTH + j] = dataLinesChar[i][j];
            }
        }

        this.charsetSize = charsetSize;
        this.charToIndexMap = charToIndexMap;
        this.indexToCharArray = indexToCharArray;

        initializeOffsets();
    }

    private void initializeOffsets() {
        examplesStarts = new LinkedList<>();

        int examplesCount = characters.length / LstmTrain.EXAMPLES_LENGTH;

        for (int i = 0; i < examplesCount; i++) {
            examplesStarts.add(i * LstmTrain.EXAMPLES_LENGTH);
        }

        Collections.shuffle(examplesStarts, rng);
    }

    public boolean hasNext() {
        return examplesStarts.size() > 0;
    }

    public DataSet next() {
        return next(batch());
    }

    public DataSet next(int num) {
        if (examplesStarts.size() == 0)
            throw new NoSuchElementException();

        int currentMinibatchSize = Math.min(num, examplesStarts.size());
        INDArray input = Nd4j.create(new int[]{currentMinibatchSize, charsetSize, LstmTrain.EXAMPLES_LENGTH}, 'f');
        INDArray labels = Nd4j.create(new int[]{currentMinibatchSize, charsetSize, LstmTrain.EXAMPLES_LENGTH}, 'f');

        for (int i = 0; i < currentMinibatchSize; i++) {
            int start = examplesStarts.removeFirst();
            int end = start + LstmTrain.EXAMPLES_LENGTH;
            int currentChar = charToIndexMap.get(characters[start]);

            int c = 0;
            for (int j = start + 1; j < end; j++, c++) {
                int nextChar = charToIndexMap.get(characters[j]);
                input.putScalar(new int[]{i, currentChar, c}, 1.0);
                labels.putScalar(new int[]{i, nextChar, c}, 1.0);
                currentChar = nextChar;
            }
        }

        return new DataSet(input, labels);
    }

    public char convertIndexToCharacter(int index) {
        return indexToCharArray[index];
    }

    public int convertCharacterToIndex(char c) {
        return charToIndexMap.get(c);
    }

    public char getRandomCharacter() {
        return indexToCharArray[rng.nextInt(charsetSize)];
    }

    public int totalExamples() {
        return characters.length / batch();
    }

    public int inputColumns() {
        return charsetSize;
    }

    public int totalOutcomes() {
        return charsetSize;
    }

    public void reset() {
        examplesStarts.clear();
        initializeOffsets();
    }

    public boolean resetSupported() {
        return true;
    }

    public boolean asyncSupported() {
        return true;
    }

    public int batch() {
        return LstmTrain.MINI_BATCH_SIZE;
    }

    public int cursor() {
        return totalExamples() - examplesStarts.size();
    }

    public int numExamples() {
        return totalExamples();
    }

    public void setPreProcessor(DataSetPreProcessor preProcessor) {
        throw new UnsupportedOperationException("Not implemented");
    }

    public DataSetPreProcessor getPreProcessor() {
        throw new UnsupportedOperationException("Not implemented");
    }

    public List<String> getLabels() {
        throw new UnsupportedOperationException("Not implemented");
    }

    public void remove() {
        throw new UnsupportedOperationException("Not implemented");
    }
}