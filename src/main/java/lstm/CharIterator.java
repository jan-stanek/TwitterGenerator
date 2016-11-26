package lstm;

import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.DataSetPreProcessor;
import org.nd4j.linalg.factory.Nd4j;

import java.io.IOException;
import java.util.*;

public class CharIterator implements DataSetIterator {

    private int exampleLength;
    private int miniBatchSize;

    private Random rng;

    private Map<Character, Integer> charToIndexMap;
    private Character[] indexToCharArray;
    private LinkedList<Integer> examplesStarts;

    private Set<Character> validChars;

    private char[] characters;

    public CharIterator(List<String> texts, Set<Character> validChars, int miniBatchSize, int exampleLength) throws IOException {
        this.exampleLength = exampleLength;
        this.miniBatchSize = miniBatchSize;
        this.rng = new Random();
        this.validChars = validChars;

        int totalLength = 0;
        for (String text : texts) {
            totalLength += text.length();
        }

        characters = new char[totalLength];
        int index = 0;

        for (String s : texts) {
            char[] text = s.toCharArray();
            for (char character : text) {
                characters[index++] = character;
            }
        }

        charToIndexMap = new HashMap<>();
        indexToCharArray = new Character[validChars.size()];
        int i = 0;
        for (Character c : validChars) {
            charToIndexMap.put(c, i);
            indexToCharArray[i] = c;
            i++;
        }

        initializeOffsets();
    }

    private void initializeOffsets() {
        examplesStarts = new LinkedList<>();

        int examplesCount = (characters.length - 1) / exampleLength - 2;

        for (int i = 0; i < examplesCount; i++) {
            examplesStarts.add(i * exampleLength);
        }

        Collections.shuffle(examplesStarts, rng);
    }

    public boolean hasNext() {
        return examplesStarts.size() > 0;
    }

    public DataSet next() {
        return next(miniBatchSize);
    }

    public DataSet next(int num) {
        if (examplesStarts.size() == 0)
            throw new NoSuchElementException();

        int currentMinibatchSize = Math.min(num, examplesStarts.size());
        INDArray input = Nd4j.create(new int[]{currentMinibatchSize, validChars.size(), exampleLength}, 'f');
        INDArray labels = Nd4j.create(new int[]{currentMinibatchSize, validChars.size(), exampleLength}, 'f');

        for (int i = 0; i < currentMinibatchSize; i++) {
            int start = examplesStarts.removeFirst();
            int end = start + exampleLength;
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

    public char convertIndexToCharacter(int idx) {
        return indexToCharArray[idx];
    }

    public int convertCharacterToIndex(char c) {
        return charToIndexMap.get(c);
    }

    public char getRandomCharacter() {
        return indexToCharArray[(int) (rng.nextDouble() * validChars.size())];
    }

    public int totalExamples() {
        return (characters.length - 1) / miniBatchSize - 2;
    }

    public int inputColumns() {
        return validChars.size();
    }

    public int totalOutcomes() {
        return validChars.size();
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
        return miniBatchSize;
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